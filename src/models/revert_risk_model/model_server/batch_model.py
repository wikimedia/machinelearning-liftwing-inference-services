import asyncio
import json
import logging
from collections.abc import Sequence
from http import HTTPStatus
from typing import Any, Optional

import mwapi
from base_model import RevisionRevertRiskModel
from fastapi import HTTPException
from knowledge_integrity.mediawiki import Error, get_revision
from knowledge_integrity.schema import InvalidJSONError, Revision
from kserve.errors import InferenceError, InvalidInput

from python import events
from python.preprocess_utils import (
    check_input_param,
    get_lang,
    get_rev_id,
    is_domain_wikipedia,
    validate_json_input,
)


class RevisionRevertRiskModelBatch(RevisionRevertRiskModel):
    def __init__(
        self,
        name: str,
        module_name: str,
        model_path: str,
        wiki_url: str,
        aiohttp_client_timeout: int,
        force_http: bool,
        allow_revision_json_input: bool,
        eventgate_url: str,
        eventgate_stream: str,
    ) -> None:
        super().__init__(
            name,
            module_name,
            model_path,
            wiki_url,
            aiohttp_client_timeout,
            force_http,
            allow_revision_json_input,
        )
        self.event_key = "event"
        self.eventgate_url = eventgate_url
        self.eventgate_stream = eventgate_stream
        self.tls_cert_bundle_path = "/etc/ssl/certs/wmf-ca-certificates.crt"
        self.custom_user_agent = "WMF ML Team revert-risk model inference (LiftWing)"

    async def get_revisions(
        self, session: mwapi.AsyncSession, rev_ids: list[int], lang: str
    ) -> Optional[Sequence[Revision]]:
        tasks = [get_revision(session, rev_id, lang) for rev_id in rev_ids]
        return await asyncio.gather(*tasks)

    def get_lang(self, lang_lst: list[str]) -> str:
        lang_set = set(lang_lst)
        if len(lang_set) > 1:
            logging.error("More than one language in the request.")
            raise InvalidInput(
                "Requesting multiple revisions should have the same language."
            )
        return lang_set.pop()

    def parse_input_data(self, inputs: dict[str, Any]) -> tuple[list[int], str]:
        """
        Parse batch, event-based, and single input data that will be used to get
        revision revert-risk predictions.

        This method handles multiple input payload formats including:
        - Batch inputs via the "instances" key (expects a list of revision objects,
          each containing at least 'lang' and 'rev_id'. A maximum of 20 revisions
          is allowed per request).
        - Event-based inputs via the event key (e.g {"event": {...}}). In this case,
          it ensures the event originates from Wikipedia and extracts the appropriate
          language and revision identifier using helper functions.
        - Single revision inputs provided with direct keys "rev_id" and "lang".
        """
        wiki_ids = []
        rev_ids = []
        if "instances" in inputs:
            # payload like {"instances": [...]}
            for input in inputs["instances"]:
                check_input_param(lang=input.get("lang"), rev_id=input.get("rev_id"))
                wiki_ids.append(input.get("lang"))
                rev_ids.append(input.get("rev_id"))
                if len(rev_ids) > 20:
                    logging.error("Received request more than 20 rev_ids.")
                    raise InvalidInput(
                        "Only accept a maximum of 20 rev_ids in the request."
                    )
            # For now only accept the same language in batch inputs.
            # This allows us to use the same async session.
            lang = self.get_lang(wiki_ids)
            logging.info(f"Received request for revision {rev_ids} ({lang}).")
        elif self.event_key in inputs:
            # payload like {"event": {...}}
            source_event = inputs.get(self.event_key)
            if not is_domain_wikipedia(source_event):
                error_message = "This model is not recommended for use in projects outside of Wikipedia (e.g. Wiktionary, Wikinews, etc)"
                logging.error(error_message)
                raise InvalidInput(error_message)
            lang = get_lang(inputs, self.event_key)
            rev_id = get_rev_id(inputs, self.event_key)
            check_input_param(lang=lang, rev_id=rev_id)
            rev_ids.append(rev_id)
        else:
            # payload like {"rev_id": 12345, "lang": "en"}
            lang = inputs.get("lang")
            rev_id = inputs.get("rev_id")
            check_input_param(lang=lang, rev_id=rev_id)
            rev_ids.append(rev_id)
        return rev_ids, lang

    def get_revision_from_input(self, inputs) -> dict[str, Any]:
        revision_json = json.dumps(inputs["revision_data"])
        try:
            rev = Revision.from_json(revision_json)
        except InvalidJSONError:
            logging.error("Missing some required fields.")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=(
                    "Missing some required fields. Required fields"
                    " are `id`, `lang`, `text`, `timestamp`, `bytes`,"
                    " `page.id`, `page.title`, `page.first_edit_timestamp`,"
                    " `parent.id`, `parent.lang`, `parent.text`,"
                    " `parent.timestamp`, `parent.bytes`, `user.id`"
                ),
            )
        lang = inputs["revision_data"]["lang"]
        # We assign rev_id as -1 if it is a user-provided pre-saved edit that
        # has not been saved in MediaWiki. This way we can distinguish it in
        # the output from the requests for existing revisions.
        rev_id = -1
        inputs["revision"] = {(rev_id, lang): rev}
        return inputs

    async def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        inputs = validate_json_input(inputs)
        if "revision_data" in inputs and self.allow_revision_json_input:
            logging.info(
                "Received revision data. Bypassing data retrieval from the MW API."
            )
            return self.get_revision_from_input(inputs)
        rev_ids, lang = self.parse_input_data(inputs)
        self.check_wiki_suffix(lang)
        self.check_canonical_wikis(lang)
        self.check_supported_wikis(lang)
        mw_host = self.get_mediawiki_host(lang)
        session = mwapi.AsyncSession(
            # Host is set to http://api-ro.discovery.wmnet within WMF
            # network in Lift Wing. Alternatively, it can be set to
            # https://{lang}.wikipedia.org to call MW API publicly.
            host=self.wiki_url or mw_host,
            user_agent=self.custom_user_agent,
            session=self.get_http_client_session("mwapi"),
        )
        # Additional HTTP Host header must be set if the host is http://api-ro.discovery.wmnet
        session.headers["Host"] = mw_host.replace("https://", "").replace("http://", "")
        try:
            revisions = await self.get_revisions(session, rev_ids, lang)
        except Exception as e:
            logging.error(
                "An error has occurred while fetching revisions: "
                f" {rev_ids} ({lang}). Reason: {e}"
            )
            raise InferenceError(
                "An error happened while fetching info for revision "
                "from the MediaWiki API, please contact the ML-Team "
                "if the issue persists."
            )
        # {(rev_id, lang): revision, (rev_id, lang): revision, ...}
        inputs["revision"] = {
            (rev_ids[i], lang): revisions[i] for i in range(len(revisions))
        }
        return inputs

    async def predict(
        self, request: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        valid_rev = {
            k: v for k, v in request["revision"].items() if not isinstance(v, Error)
        }
        if len(valid_rev) == 0:
            # all requests fail
            rev_lang = request["revision"].keys()
            error_msg = [e.code.value for e in request["revision"].values()]
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=(
                    f"Could not make prediction for revisions {rev_lang}."
                    f" Reason: {error_msg}"
                ),
            )
        num_rev = len(request["revision"])
        logging.info(f"Getting {num_rev} rev_ids in the request")
        results = self.ModelLoader.classify(self.model, valid_rev.values())
        rev_pred = {k: results[i] for i, k in enumerate(valid_rev.keys())}
        predictions = []
        for (rev_id, lang), result in rev_pred.items():
            predictions.append(
                {
                    "model_name": self.name,
                    "model_version": str(self.model.model_version),
                    "wiki_db": lang + "wiki",
                    "revision_id": rev_id,
                    "output": {
                        "prediction": result.prediction,
                        "probabilities": {
                            "true": result.probability,
                            "false": 1 - result.probability,
                        },
                    },
                }
            )
        if len(valid_rev) < num_rev:
            # some requests succeed and others fail
            error_msg = []
            for (rev_id, lang), rev in request["revision"].items():
                if isinstance(rev, Error):
                    error_msg.append(
                        f"Could not make prediction for revision {rev_id} ({lang})."
                        f" Reason: {rev.code.value}"
                    )
            return {"predictions": predictions, "errors": error_msg}
        if self.event_key in request:
            # when the input contains an event, add predictions to output event
            prediction_results = {
                "predictions": [str(predictions[0]["output"]["prediction"]).lower()],
                "probabilities": predictions[0]["output"]["probabilities"],
            }
            await self.send_event(
                request[self.event_key],
                prediction_results,
                predictions[0]["model_version"],
            )
        if "instances" not in request:
            # response for requests like {"rev_id": 12345, "lang": "en"}
            return predictions[0]
        return {"predictions": predictions}

    async def send_event(
        self,
        page_change_event: dict[str, Any],
        prediction_results: dict[str, Any],
        model_version: str,
    ) -> None:
        """
        Send a revision revert risk language-agnostic prediction classification change event to EventGate
        generated from the page_change event and prediction_results passed as input.
        """
        revertrisk_language_agnostic_event = events.generate_prediction_classification_event(
            page_change_event,
            self.eventgate_stream,
            "revertrisk-language-agnostic",  # same name is used in LW endpoint host header and will be used in changeprop for consistency
            model_version,
            prediction_results,
        )
        await events.send_event(
            revertrisk_language_agnostic_event,
            self.eventgate_url,
            self.tls_cert_bundle_path,
            self.custom_user_agent,
            self.get_http_client_session("eventgate"),
        )

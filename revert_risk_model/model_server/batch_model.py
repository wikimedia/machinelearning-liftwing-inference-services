import asyncio
import logging
from http import HTTPStatus
from typing import Any, List, Dict, Tuple, Sequence, Optional

import mwapi
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from base_model import RevisionRevertRiskModel
from knowledge_integrity.mediawiki import get_revision, Error
from knowledge_integrity.schema import Revision
from kserve.errors import InferenceError, InvalidInput

from python.preprocess_utils import check_input_param, validate_json_input


class RevisionRevertRiskModelBatch(RevisionRevertRiskModel):
    def __init__(
        self,
        name: str,
        module_name: str,
        model_path: str,
        wiki_url: str,
        aiohttp_client_timeout: int,
        force_http: bool,
    ) -> None:
        super().__init__(
            name,
            module_name,
            model_path,
            wiki_url,
            aiohttp_client_timeout,
            force_http,
        )

    async def get_revisions(
        self, session: mwapi.AsyncSession, rev_ids: List[int], lang: str
    ) -> Optional[Sequence[Revision]]:
        tasks = [get_revision(session, rev_id, lang) for rev_id in rev_ids]
        return await asyncio.gather(*tasks)

    def get_lang(self, lang_lst: List[str]) -> str:
        lang_set = set(lang_lst)
        if len(lang_set) > 1:
            logging.error("More than one language in the request.")
            raise InvalidInput(
                "Requesting multiple revisions should have the same language."
            )
        return lang_set.pop()

    def prepare_batch_input(self, inputs: Dict[str, Any]) -> Tuple[List[int], str]:
        wiki_ids = []
        rev_ids = []
        if "instances" not in inputs:
            # payload like {"rev_id": 12345, "lang": "en"}
            lang = inputs.get("lang")
            rev_id = inputs.get("rev_id")
            check_input_param(lang=lang, rev_id=rev_id)
            rev_ids.append(rev_id)
        else:
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
        return rev_ids, lang

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        rev_ids, lang = self.prepare_batch_input(inputs)
        self.check_supported_wikis(lang)
        mw_host = self.get_mediawiki_host(lang)
        session = mwapi.AsyncSession(
            # Host is set to http://api-ro.discovery.wmnet within WMF
            # network in Lift Wing. Alternatively, it can be set to
            # https://{lang}.wikipedia.org to call MW API publicly.
            host=self.wiki_url or mw_host,
            user_agent="WMF ML Team revert-risk-model isvc",
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

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
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
        results = self.ModelLoader.classify_batch(self.model, valid_rev.values())
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
            return JSONResponse(
                status_code=HTTPStatus.MULTI_STATUS,
                content={"predictions": predictions + error_msg},
            )
        if "instances" not in request:
            # response for requests like {"rev_id": 12345, "lang": "en"}
            return predictions[0]
        return {"predictions": predictions}

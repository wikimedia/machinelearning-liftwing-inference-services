import importlib
import json
import logging
import os
from http import HTTPStatus
from typing import Any

import aiohttp
import kserve
import mwapi
import pandas as pd
from fastapi import HTTPException
from knowledge_integrity.mediawiki import Error, get_revision
from knowledge_integrity.schema import InvalidJSONError, Revision
from kserve.errors import InferenceError, InvalidInput

from python.config_utils import get_config
from python.preprocess_utils import check_input_param, validate_json_input

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class RevisionRevertRiskModel(kserve.Model):
    def __init__(
        self,
        name: str,
        module_name: str,
        model_path: str,
        wiki_url: str,
        aiohttp_client_timeout: int,
        force_http: bool,
        allow_revision_json_input: bool,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.ModelLoader = importlib.import_module(
            f"knowledge_integrity.models.{module_name}"
        )
        self.ready = False
        self.model_path = model_path
        self.wiki_url = wiki_url
        self.force_http = force_http
        self.allow_revision_json_input = allow_revision_json_input
        self.aiohttp_client_timeout = aiohttp_client_timeout
        self.host_rewrite_config = get_config(key="mw_host_replace")
        self._http_client_session = {}
        self.device = None
        self.wp_language_codes = set()
        self.load()

    def get_http_client_session(self, endpoint):
        """Returns a aiohttp session for the specific endpoint passed as input.
        We need to do it since sharing a single session leads to unexpected
        side effects (like sharing headers, most notably the Host one)."""
        timeout = aiohttp.ClientTimeout(total=self.aiohttp_client_timeout)
        if (
            self._http_client_session.get(endpoint, None) is None
            or self._http_client_session[endpoint].closed
        ):
            logging.info(f"Opening a new Asyncio session for {endpoint}.")
            self._http_client_session[endpoint] = aiohttp.ClientSession(
                timeout=timeout, raise_for_status=True
            )
        return self._http_client_session[endpoint]

    def get_mediawiki_host(self, lang):
        protocol = "http" if self.force_http else "https"
        if self.name == "revertrisk-wikidata":
            return f"{protocol}://www.wikidata.org"
        else:
            updated_lang = self.host_rewrite_config.get(lang, lang)
            return f"{protocol}://{updated_lang}.wikipedia.org"

    def check_canonical_wikis(self, lang):
        # Validate lang parameter against canonical wikis.
        if (
            lang == "test"
        ):  # Treat 'testwiki' as 'enwiki' on the fly for testing purposes.
            lang = "en"

        if lang not in self.wp_language_codes:
            error_message = f"lang '{lang}' does not exist in the canonical Wikipedia language list."
            logging.error(error_message)
            raise InvalidInput(error_message)

    def check_supported_wikis(self, lang):
        if (
            hasattr(self.model, "supported_wikis")
            and lang not in self.model.supported_wikis
        ):
            logging.info(f"Unsupported lang: {lang}.")

    def check_wiki_suffix(self, lang):
        if lang.endswith("wiki"):
            raise InvalidInput(
                "Language field should not have a 'wiki' suffix, i.e. use 'en', not 'enwiki'"
            )

    def load_canonical_wikis(self) -> None:
        """
        Loads and processes the canonical wikis list
        """
        try:
            wikis_tsv_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "wikis.tsv"
            )
            canonical_wikis = pd.read_csv(
                wikis_tsv_path,
                sep="\t",
                usecols=[
                    "database_group",
                    "language_code",
                    "status",
                    "visibility",
                    "editability",
                ],
            )

            wp_codes = canonical_wikis[
                (canonical_wikis["database_group"] == "wikipedia")
                & (canonical_wikis["status"] == "open")
                & (canonical_wikis["visibility"] == "public")
                & (canonical_wikis["editability"] == "public")
            ]["language_code"]

            # Store the list in a set for fast O(1) average time complexity lookups.
            self.wp_language_codes = set(wp_codes.unique())
            logging.info(
                f"Successfully loaded {len(self.wp_language_codes)} canonical wiki languages."
            )

        except FileNotFoundError as e:
            logging.error(
                "'wikis.tsv' not found. This file is required for canonical wiki validation."
            )
            raise e

    def load(self) -> None:
        self.model = self.ModelLoader.load_model(self.model_path)
        self.load_canonical_wikis()
        self.ready = True

    def get_revision_from_input(self, inputs) -> dict[str, Any]:
        revision_json = json.dumps(inputs["revision_data"])
        try:
            inputs["revision"] = Revision.from_json(revision_json)
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
        inputs["lang"] = inputs["revision_data"]["lang"]
        # We assign rev_id as -1 if it is a user-provided pre-saved edit that
        # has not been saved in MediaWiki. This way we can distinguish it in
        # the output from the requests for existing revisions.
        inputs["rev_id"] = -1
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
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        logging.info(f"Received request for revision {rev_id} ({lang}).")

        check_input_param(lang=lang, rev_id=rev_id)

        self.check_wiki_suffix(lang)
        self.check_canonical_wikis(lang)
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
            rev = await get_revision(session, rev_id, lang)
        except Exception as e:
            logging.error(
                "An error has occurred while fetching info for revision: "
                f" {rev_id} ({lang}). Reason: {e}"
            )
            raise InferenceError(
                "An error happened while fetching info for revision "
                "from the MediaWiki API, please contact the ML-Team "
                "if the issue persists."
            )
        inputs["revision"] = rev
        return inputs

    def check_wikidata_result(self, result, edit_summary):
        if not result:
            logging.info(f"Edit type {edit_summary} is not supported at the moment.")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=(
                    "Prediction for this type of edit is not supported "
                    "at the moment. Currently only 'claim' edits and "
                    "'description' edits are supported."
                ),
            )

    def predict(
        self, request: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        rev_id = request.get("rev_id")
        lang = request.get("lang")
        rev = request.get("revision")
        if isinstance(rev, Error):
            logging.info(f"revision {rev_id} ({lang}): {rev.code.value}")
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=(
                    f"Could not make prediction for revision {rev_id} ({lang})."
                    f" Reason: {rev.code.value}"
                ),
            )
        else:
            result = self.ModelLoader.classify(self.model, rev)
            if self.name == "revertrisk-wikidata":
                edit_summary = request["revision"].comment
                self.check_wikidata_result(result, edit_summary)
            output = {
                "prediction": result.prediction,
                "probabilities": {
                    "true": result.probability,
                    "false": 1 - result.probability,
                },
            }
        return {
            "model_name": self.name,
            "model_version": str(self.model.model_version),
            "wiki_db": lang + "wiki",
            "revision_id": rev_id,
            "output": output,
        }

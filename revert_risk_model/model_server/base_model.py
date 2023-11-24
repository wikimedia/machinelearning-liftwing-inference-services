import logging
import importlib
from typing import Any, Dict

import aiohttp
import kserve
import mwapi

from knowledge_integrity.revision import get_current_revision
from kserve.errors import InvalidInput, InferenceError
from python.preprocess_utils import check_input_param, validate_json_input
from http import HTTPStatus
from fastapi import HTTPException

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class RevisionRevertRiskModel(kserve.Model):
    def __init__(
        self,
        name: str,
        module_name: str,
        model_path: str,
        wiki_url: str,
        aiohttp_client_timeout: int,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.ModelLoader = importlib.import_module(
            f"knowledge_integrity.models.{module_name}"
        )
        self.ready = False
        self.model_path = model_path
        self.wiki_url = wiki_url
        self.aiohttp_client_timeout = aiohttp_client_timeout
        self._http_client_session = {}
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
        if self.name == "revertrisk-wikidata":
            return "https://www.wikidata.org"
        else:
            # See https://phabricator.wikimedia.org/T340830
            if lang == "be-x-old":
                return "https://be-tarask.wikipedia.org"
            else:
                return f"https://{lang}.wikipedia.org"

    def validate_inputs(self, lang, rev_id):
        check_input_param(lang=lang, rev_id=rev_id)
        if (
            hasattr(self.model, "supported_wikis")
            and lang not in self.model.supported_wikis
        ):
            logging.error(f"Unsupported lang: {lang}.")
            raise InvalidInput(f"Unsupported lang: {lang}.")

    def load(self) -> None:
        self.model = self.ModelLoader.load_model(self.model_path)
        self.ready = True

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        self.validate_inputs(lang, rev_id)
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
        session.headers["Host"] = mw_host.replace("https://", "")
        try:
            rev = await get_current_revision(session, rev_id, lang)
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
        if rev is None:
            logging.error(
                "get_current_revision returned empty results "
                f"for revision {rev_id} ({lang})"
            )
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=(
                    "The necessary features cannot be obtained from the "
                    "MediaWiki API. It can be the revision, parent revision, "
                    "page information, or user information. This could be "
                    "because the data does not exist or has been deleted."
                ),
            )
        inputs["revision"] = rev
        return inputs

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        result = self.ModelLoader.classify(self.model, request["revision"])
        edit_summary = request["revision"].comment
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
            "wiki_db": request.get("lang") + "wiki",
            "revision_id": request.get("rev_id"),
            "output": output,
        }

import importlib
import logging
from http import HTTPStatus
from typing import Any, Dict

import aiohttp
import kserve
import mwapi
from fastapi import HTTPException
from knowledge_integrity.mediawiki import get_revision, Error
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
        self.aiohttp_client_timeout = aiohttp_client_timeout
        self.host_rewrite_config = get_config(key="mw_host_replace")
        self._http_client_session = {}
        self.device = None
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

    def check_supported_wikis(self, lang):
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
        logging.info(f"Received request for revision {rev_id} ({lang}).")
        check_input_param(lang=lang, rev_id=rev_id)
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
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
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

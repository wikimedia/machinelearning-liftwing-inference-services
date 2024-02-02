import importlib
import logging
from http import HTTPStatus
from typing import Any, Dict

import aiohttp
import kserve
import mwapi
import torch
from fastapi import HTTPException
from knowledge_integrity.revision import get_current_revision
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

    def check_gpu(self):
        """
        Loads the model in the GPU's memory and updates its reference.
        This function needs to run after the webserver's initialization
        (that forks and creates new processes, see https://github.com/pytorch/pytorch/issues/83973).
        """
        if not self.device and self.name == "revertrisk-multilingual":
            # The cuda keyword is internally translated to hip and rocm is used if available.
            # https://pytorch.org/docs/stable/notes/hip.html
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")
            # loading model to GPU
            self.model.title_model.model.to(self.device)
            self.model.insert_model.model.to(self.device)
            self.model.remove_model.model.to(self.device)
            self.model.change_model.model.to(self.device)
            # changing the device of the pipeline
            self.model.title_model.device = self.device
            self.model.insert_model.device = self.device
            self.model.remove_model.device = self.device
            self.model.change_model.device = self.device

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        self.check_gpu()
        inputs = validate_json_input(inputs)
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        logging.info(f"Received request for revision {rev_id} ({lang}).")
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
        session.headers["Host"] = mw_host.replace("https://", "").replace("http://", "")
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
        result = self.ModelLoader.classify(self.model, request["revision"])
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
            "wiki_db": request.get("lang") + "wiki",
            "revision_id": request.get("rev_id"),
            "output": output,
        }

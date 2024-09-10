import os
import logging
import aiohttp
import kserve
import mwapi
from typing import Any, Dict
from distutils.util import strtobool

from knowledge_integrity.mediawiki import get_revision
from knowledge_integrity.models.reference_need import load_model, classify
from kserve.errors import InferenceError

from python.config_utils import get_config
from python.preprocess_utils import (
    validate_json_input,
    check_input_param,
    check_wiki_suffix,
    check_supported_wikis,
)

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ReferenceNeedModel(kserve.Model):
    def __init__(self, name: str, model_path: str, force_http: bool) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.force_http = force_http
        self.host_rewrite_config = get_config(key="mw_host_replace")
        self.aiohttp_client_timeout = 5
        self._http_client_session = {}
        self.ready = False
        self.load()

    def load(self) -> None:
        self.model = load_model(self.model_path)
        logging.info(f"Supported wikis: {self.model.supported_wikis}.")
        self.ready = True

    def get_mediawiki_host(self, lang):
        protocol = "http" if self.force_http else "https"
        updated_lang = self.host_rewrite_config.get(lang, lang)
        return f"{protocol}://{updated_lang}.wikipedia.org"

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

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        logging.info(f"Received request for revision {rev_id} ({lang}).")
        check_input_param(lang=lang, rev_id=rev_id)
        check_wiki_suffix(lang)
        check_supported_wikis(self.model, lang)
        session = mwapi.AsyncSession(
            host=self.get_mediawiki_host(lang),
            user_agent="WMF ML Team reference-need isvc",
            session=self.get_http_client_session("mwapi"),
        )
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

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        result = classify(self.model, request["revision"])
        return result


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "reference-need")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/model.pkl")
    force_http = strtobool(os.environ.get("FORCE_HTTP", "False"))

    model = ReferenceNeedModel(
        name=model_name, model_path=model_path, force_http=force_http
    )
    kserve.ModelServer(workers=1).start([model])

import os
import logging
from typing import Any, Dict

import aiohttp
import kserve
import mwapi

from knowledge_integrity.revision import get_current_revision
from kserve.errors import InvalidInput, InferenceError

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class RevisionRevertRiskModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self._http_client_session = {}
        self.WIKI_URL = os.environ.get("WIKI_URL")
        self.AIOHTTP_CLIENT_TIMEOUT = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
        self._http_client_session = {}
        self.load()

    def get_http_client_session(self, endpoint):
        """Returns a aiohttp session for the specific endpoint passed as input.
        We need to do it since sharing a single session leads to unexpected
        side effects (like sharing headers, most notably the Host one)."""
        timeout = aiohttp.ClientTimeout(total=self.AIOHTTP_CLIENT_TIMEOUT)
        if (
            self._http_client_session.get(endpoint, None) is None
            or self._http_client_session[endpoint].closed
        ):
            logging.info(f"Opening a new Asyncio session for {endpoint}.")
            self._http_client_session[endpoint] = aiohttp.ClientSession(
                timeout=timeout, raise_for_status=True
            )
        return self._http_client_session[endpoint]

    def load(self) -> None:
        self.model = KI_module.load_model("/mnt/models/model.pkl")
        self.ready = True

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        if lang is None:
            logging.error("Missing lang in input data.")
            raise InvalidInput("The parameter lang is required.")
        if rev_id is None:
            logging.error("Missing rev_id in input data.")
            raise InvalidInput("The parameter rev_id is required.")
        if model_name != "revertrisk-wikidata":
            if lang not in self.model.supported_wikis:
                logging.error(f"Unsupported lang: {lang}.")
                raise InvalidInput(f"Unsupported lang: {lang}.")
            mw_host = f"https://{lang}.wikipedia.org"
        else:
            mw_host = "http://www.wikidata.org"
        session = mwapi.AsyncSession(
            # host is set to http://api-ro.discovery.wmnet
            # for accessing MediaWiki APIs within WMF networks
            # in Lift Wing. But it can also be set to URLs like
            # https://en.wikipedia.org in non-LW environment
            # that call MediaWiki APIs in a public manner.
            host=self.WIKI_URL or mw_host,
            user_agent="WMF ML Team revert-risk-model isvc",
            session=self.get_http_client_session("mwapi"),
        )
        # an additional HTTP Host header must be set for the session
        # when the host is set to http://api-ro.discovery.wmnet
        session.headers["Host"] = mw_host
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
        inputs["revision"] = rev
        return inputs

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        if request["revision"] is None:
            # return empty score if missing revision required
            return {
                "lang": request.get("lang"),
                "rev_id": request.get("rev_id"),
                "score": {},
            }
        result = KI_module.classify(self.model, request["revision"])
        return {
            "lang": request.get("lang"),
            "rev_id": request.get("rev_id"),
            "score": {
                "prediction": result.prediction,
                "probability": {
                    "true": result.probability,
                    "false": 1 - result.probability,
                },
            },
        }


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    if model_name == "revertrisk-wikidata":
        import knowledge_integrity.models.revertrisk_wikidata as KI_module
    elif model_name == "revertrisk-multilingual":
        import knowledge_integrity.models.revertrisk_multilingual as KI_module
    else:
        import knowledge_integrity.models.revertrisk as KI_module
    model = RevisionRevertRiskModel(model_name)
    kserve.ModelServer(workers=1).start([model])

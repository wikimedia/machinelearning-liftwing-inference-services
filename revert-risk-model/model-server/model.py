import os
import asyncio
import atexit
import logging
from typing import Any, Dict
from http import HTTPStatus

import aiohttp
import kserve
import mwapi
import tornado.web

from knowledge_integrity.models.revertrisk import load_model, classify
from knowledge_integrity.revision import get_current_revision

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class RevisionRevertRiskModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self._http_client_session = None
        self.WIKI_URL = os.environ.get("WIKI_URL")
        atexit.register(self._shutdown)

    @property
    def http_client_session(self):
        if self._http_client_session is None or self._http_client_session.closed:
            logging.info("Opening a new Asyncio session.")
            self._http_client_session = aiohttp.ClientSession()
        return self._http_client_session

    def _shutdown(self):
        if self._http_client_session and not self._http_client_session.closed:
            logging.info("Closing asyncio session")
            asyncio.run(self._http_client_session.close())

    def load(self) -> None:
        self.model = load_model("/mnt/models/model.pkl")
        self.ready = True

    async def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        if lang is None:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="The parameter lang is required.",
            )
        if lang not in self.model.supported_wikis:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason=f"Unsupported lang: {lang}.",
            )
        if rev_id is None:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="The parameter rev_id is required.",
            )
        if self.WIKI_URL is None:
            self.WIKI_URL = f"https://{lang}.wikipedia.org"
        elif self.WIKI_URL.endswith("wmnet"):
            # access MediaWiki API from internal networks
            # e.g. https://api-ro.discovery.wmnet
            self.http_client_session.headers.update({"Host": f"{lang}.wikipedia.org"})
        session = mwapi.AsyncSession(
            self.WIKI_URL,
            user_agent="WMF ML Team revert-risk-model isvc",
            session=self.http_client_session,
        )
        try:
            rev = await get_current_revision(session, rev_id, lang)
            if rev is None:
                raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason=(
                        f"Revision with lang: {lang} rev_id: {rev_id} "
                        "missing information required for inference."
                    ),
                )
        except Exception as e:
            logging.error(
                "An error has occurred while fetching info for revision: "
                f" {rev_id} ({lang}). Reason: {e}"
            )
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                reason=(
                    "An error happened while fetching info for revision "
                    "from the MediaWiki API, please contact the ML-Team "
                    "if the issue persists."
                ),
            )
        inputs["revision"] = rev
        return inputs

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        result = classify(self.model, request["revision"])
        return {
            "prediction": result.prediction,
            "probability": {
                "true": result.probability,
                "false": 1 - result.probability,
            },
        }


if __name__ == "__main__":
    model = RevisionRevertRiskModel("revert-risk-model")
    model.load()
    kserve.ModelServer(workers=1).start([model])

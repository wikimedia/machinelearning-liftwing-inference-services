import logging
import os
from http import HTTPStatus
from typing import Dict

import kserve
import mwapi
import requests
import tornado.web

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ArticleQualityTransformer(kserve.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        """Fetch article text"""
        rev_id = self._get_rev_id(inputs)
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")
        if wiki_host:
            s = requests.Session()
            s.headers.update({"Host": wiki_host})
        else:
            s = None
        self.session = mwapi.Session(
            wiki_url, user_agent="WMF ML team articlequality transformer", session=s
        )
        return self._fetch_articlequality_text(rev_id)

    def _get_rev_id(self, inputs: Dict) -> Dict:
        try:
            rev_id = inputs["rev_id"]
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Expected "rev_id" in input data.',
            )
        if not isinstance(rev_id, int):
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Expected "rev_id" to be an integer.',
            )
        return rev_id

    def _fetch_articlequality_text(self, rev_id: int) -> Dict:
        """Retrieve article text features."""
        doc = self.session.get(
            action="query",
            prop="revisions",
            revids=[rev_id],
            rvprop=["ids", "content"],
            rvslots=["main"],
            formatversion=2,
        )

        try:
            page_doc = doc["query"]["pages"][0]
        except (KeyError, IndexError) as e:
            logging.error("No Pages found", str(e))
            raise  # propagate exception again
        try:
            rev_doc = page_doc["revisions"][0]
        except (KeyError, IndexError) as e:
            logging.error("No revisions matched", str(e))
            raise  # propagate exception again

        content = rev_doc["slots"]["main"].get("content")
        return {"article_text": content}

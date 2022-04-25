import articlequality
import kserve
import os
from revscoring import Model
from typing import Dict
import requests
import mwapi
import tornado.web
from http import HTTPStatus
import logging

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ArticlequalityModel(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        with open("/mnt/models/model.bin") as f:
            self.model = Model.load(f)
        self.ready = True

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

    def predict(self, request: Dict) -> Dict:
        inputs = request["article_text"]
        results = articlequality.score(self.model, inputs)
        return {"predictions": results}

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
            rev_doc = doc["query"]["pages"][0]["revisions"][0]
        except (KeyError, IndexError) as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Revision {} not found.".format(rev_id),
            )
        content = rev_doc["slots"]["main"].get("content")
        return {"article_text": content}


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = ArticlequalityModel(inference_name)
    model.load()
    kserve.KFServer(workers=1).start([model])

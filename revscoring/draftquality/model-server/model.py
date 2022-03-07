import bz2
import os
from http import HTTPStatus
from typing import Dict

import kserve
import mwapi
import requests
import tornado.web
from revscoring import Model
from revscoring.errors import RevisionNotFound
from revscoring.extractors import api


class DraftqualityModel(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.FEATURE_VAL_KEY = "feature_values"

    def load(self):
        with bz2.open("/mnt/models/model.bz2") as f:
            self.model = Model.load(f)
        self.ready = True

    def preprocess(self, inputs: Dict) -> Dict:
        """Retrieve features from mediawiki."""
        rev_id = self._get_rev_id(inputs)
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")
        if wiki_host:
            s = requests.Session()
            s.headers.update({"Host": wiki_host})
        else:
            s = None
        ua = "WMF ML team draftquality model"
        self.extractor = api.Extractor(
            mwapi.Session(wiki_url, user_agent=ua, session=s)
        )
        inputs[self.FEATURE_VAL_KEY] = self._fetch_draftquality_features(rev_id)
        return inputs

    def predict(self, request: Dict) -> Dict:
        feature_values = request[self.FEATURE_VAL_KEY]
        results = self.model.score(feature_values)
        return {"predictions": results}

    def _fetch_draftquality_features(self, rev_id: int) -> Dict:
        """Retrieve draftquality features."""
        try:
            feature_values = list(self.extractor.extract(rev_id, self.model.features))
        except RevisionNotFound:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Revision {} not found".format(rev_id),
            )
        return feature_values

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


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = DraftqualityModel(inference_name)
    model.load()
    kserve.KFServer(workers=1).start([model])

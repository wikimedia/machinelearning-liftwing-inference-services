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
from revscoring.features import trim


class DrafttopicModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.FEATURE_VAL_KEY = "feature_values"
        self.EXTENDED_OUTPUT_KEY = "extended_output"

    def load(self):
        with open("/mnt/models/model.bin") as f:
            self.model = Model.load(f)
        self.ready = True

    def preprocess(self, inputs: Dict) -> Dict:
        """Use MW API session and Revscoring API to extract feature values
        of edit text based on its revision id"""
        rev_id = self._get_rev_id(inputs)
        extended_output = inputs.get("extended_output", False)
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")
        if wiki_host:
            s = requests.Session()
            s.headers.update({"Host": wiki_host})
        else:
            s = None
        self.extractor = api.Extractor(
            mwapi.Session(wiki_url, user_agent="WMF ML Team topic model", session=s)
        )
        inputs[self.FEATURE_VAL_KEY] = self._fetch_topic_features(rev_id)
        if extended_output:
            base_feature_values = self.extractor.extract(
                rev_id, list(trim(self.model.features))
            )
            inputs[self.EXTENDED_OUTPUT_KEY] = {
                str(f): v
                for f, v in zip(list(trim(self.model.features)), base_feature_values)
            }
        return inputs

    def predict(self, request: Dict) -> Dict:
        feature_values = request.get(self.FEATURE_VAL_KEY)
        extended_output = request.get(self.EXTENDED_OUTPUT_KEY)
        results = self.model.score(feature_values)
        if extended_output:
            # add extended output to reach feature parity with ORES, like:
            # https://ores.wikimedia.org/v3/scores/enwiki/1083325118/articletopic?features
            # If only rev_id is given in input.json, only the prediction results
            # will be present in the response. If the extended_output flag is true,
            # features output will be included in the response.
            return {"predictions": results, "features": extended_output}
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

    def _fetch_topic_features(self, rev_id: int) -> Dict:
        """Retrieve topic features."""
        try:
            feature_values = list(self.extractor.extract(rev_id, self.model.features))
        except RevisionNotFound:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Revision {} not found".format(rev_id),
            )

        return feature_values


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = DrafttopicModel(inference_name)
    model.load()
    kserve.ModelServer(workers=1).start([model])

import bz2
import os
from typing import Dict

import kserve
import mwapi
import requests
from revscoring import Model
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
        rev_id = inputs.get("rev_id")
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
        feature_values = list(self.extractor.extract(rev_id, self.model.features))
        return feature_values


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = DraftqualityModel(inference_name)
    model.load()
    kserve.KFServer(workers=1).start([model])

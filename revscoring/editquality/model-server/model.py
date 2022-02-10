import kserve
import mwapi
import os
import requests
from revscoring import Model
from revscoring.extractors import api
from typing import Dict


class EditQualityModel(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        with open("/mnt/models/model.bin") as f:
            self.model = Model.load(f)
        self.ready = True

    def preprocess(self, inputs: Dict) -> Dict:
        """Use MW API session and Revscoring API to extract feature values
        of edit text based on its revision id"""
        rev_id = inputs.get("rev_id")
        wiki_url = os.environ.get("WIKI_URL")
        wiki_host = os.environ.get("WIKI_HOST")

        if wiki_host:
            s = requests.Session()
            s.headers.update({"Host": wiki_host})
        else:
            s = None

        self.extractor = api.Extractor(
            mwapi.Session(
                wiki_url, user_agent="WMF ML Team editquality model", session=s
            )
        )

        return self.fetch_editquality_features(rev_id)

    def fetch_editquality_features(self, rev_id: int) -> Dict:
        """Retrieve editquality features."""
        feature_values = list(self.extractor.extract(rev_id, self.model.features))
        return feature_values

    def predict(self, request: Dict) -> Dict:
        feature_values = self.preprocess(request)
        results = self.model.score(feature_values)
        return {"predictions": results}


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = EditQualityModel(inference_name)
    model.load()
    kserve.KFServer(workers=1).start([model])

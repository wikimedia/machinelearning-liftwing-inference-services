import kfserving
import mwapi
import os
from revscoring import Model
from revscoring.extractors import api
from typing import Dict


class RevscoringModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        with open("/mnt/models/model.bin") as f:
            self.model = Model.load(f)
        wiki_url = os.environ.get("WIKI_URL")
        self.extractor = api.Extractor(
            mwapi.Session(wiki_url, user_agent="KFServing revscoring demo")
        )
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = request["rev_id"]
        feature_values = list(self.extractor.extract(inputs, self.model.features))
        results = self.model.score(feature_values)
        return {"predictions": results}


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = RevscoringModel(inference_name)
    model.load()
    kfserving.KFServer(workers=1).start([model])

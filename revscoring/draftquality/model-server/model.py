import bz2
import kserve
import os
from revscoring import Model
from typing import Dict


class DraftqualityModel(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        with bz2.open("/mnt/models/model.bz2") as f:
            self.model = Model.load(f)
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        feature_values = request["feature_values"]
        results = self.model.score(feature_values)
        return {"predictions": results}


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = DraftqualityModel(inference_name)
    model.load()
    kserve.KFServer(workers=1).start([model])

import articlequality
import kserve
import os
from revscoring import Model
from typing import Dict


class ArticlequalityModel(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        with open("/mnt/models/model.bin") as f:
            self.model = Model.load(f)
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = request["article_text"]
        results = articlequality.score(self.model, inputs)
        return {"predictions": results}


if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model = ArticlequalityModel(inference_name)
    model.load()
    kserve.KFServer(workers=1).start([model])

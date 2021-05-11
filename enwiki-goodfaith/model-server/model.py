import kfserving
from revscoring import Model
import mwapi
from revscoring.extractors import api
from typing import Dict


class EnWikiGoodfaithModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        with open("enwiki.goodfaith.gradient_boosting.model") as f:
            self.model = Model.load(f)
        self.extractor = api.Extractor(mwapi.Session(
            "https://en.wikipedia.org",
            user_agent="KFServing revscoring demo"))
        self.ready = True

    def predict(self, request: Dict) -> Dict:

        inputs = request["rev_id"]
        feature_values = list(
            self.extractor.extract(inputs, self.model.features))
        results = self.model.score(feature_values)
        return {"predictions": results}


if __name__ == "__main__":
    model = EnWikiGoodfaithModel("enwiki-goodfaith")
    model.load()
    kfserving.KFServer(workers=1).start([model])

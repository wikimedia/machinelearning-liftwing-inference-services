from kserve import Model, ModelServer
from kserve.errors import InvalidInput
from typing import Dict
import fasttext
import logging


class LanguageIdentificationModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        self.model = fasttext.load_model("/mnt/models/lid201-model.bin")
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        text = payload.get("text", None)

        if text is None:
            logging.error("Missing text in input data.")
            raise InvalidInput("The parameter text is required.")

        label, score = self.model.predict(text)
        return {"language": label[0].replace("__label__", ""), "score": score[0]}


if __name__ == "__main__":
    model = LanguageIdentificationModel("langid-model")
    model.load()
    ModelServer(workers=1).start([model])

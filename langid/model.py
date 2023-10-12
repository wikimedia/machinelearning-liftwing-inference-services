import csv
import logging
import os
from typing import Dict

from fasttext.FastText import _FastText
from kserve import Model, ModelServer
from kserve.errors import InvalidInput
from python.preprocess_utils import validate_input


class LanguageIdentificationModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.languages: Dict[str, str] = self.create_language_lookup()
        self.model = self.load()

    def create_language_lookup(self) -> Dict[str, str]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "languages.tsv")
        csv_reader = csv.DictReader(open(csv_path), delimiter="\t")
        languages: Dict[str, str] = {}
        for index, row in enumerate(csv_reader):
            if index == 0:
                continue  # Skip header
            languages[row.get("lid_language")] = {
                "wikicode": row.get("wikicode"),
                "languagename": row.get("language_name"),
            }
        return languages

    def load(self) -> _FastText:
        model = _FastText(os.environ.get("MODEL_PATH", "/mnt/models/lid201-model.bin"))
        self.ready = True
        return model

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        payload = validate_input(payload)
        text = payload.get("text", None)

        if text is None:
            logging.error("Missing text in input data.")
            raise InvalidInput("The parameter text is required.")

        label, score = self.model.predict(text)
        language = label[0].replace("__label__", "")

        return {
            "language": language,
            "wikicode": self.languages.get(language).get("wikicode"),
            "languagename": self.languages.get(language).get("languagename"),
            "score": score[0],
        }


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model = LanguageIdentificationModel(model_name)
    ModelServer(workers=1).start([model])

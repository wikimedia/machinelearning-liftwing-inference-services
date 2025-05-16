import csv
import logging
import os
import re
from typing import Any

from fasttext.FastText import _FastText
from kserve import Model, ModelServer
from kserve.errors import InvalidInput

from python.preprocess_utils import check_input_param, validate_json_input


class LanguageIdentificationModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.max_text_length = int(os.environ.get("MAX_TEXT_LENGTH", 100))
        self.ready = False
        self.languages: dict[str, str] = self.create_language_lookup()
        self.model = self.load()

    def create_language_lookup(self) -> dict[str, str]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "languages.tsv")
        csv_reader = csv.DictReader(open(csv_path), delimiter="\t")
        languages: dict[str, str] = {}
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

    def normalize_text(self, text: str) -> str:
        """
        Replaces newlines (line feed, carriage return), tabs (vertical tab, normal tab),
        and multiple consecutive spaces with a single space then removes leading/trailing spaces
        and finally truncates the string keeping only the first {max_text_length} characters.

        This resolves T377751 by enabling passing of a single line string to the fasttext model as per:
        https://github.com/facebookresearch/fastText/issues/1079#issuecomment-637440314
        """

        return re.sub(r"[\n\r\v\t ]+", " ", text).strip()[: self.max_text_length]

    def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        inputs = validate_json_input(inputs)
        text = inputs.get("text", None)
        check_input_param(text=text)

        if not isinstance(text, str):
            error_message = "The input 'text' should be a string."
            logging.error(error_message)
            raise InvalidInput(error_message)
        else:
            normalized_text = self.normalize_text(text)

        return normalized_text

    def predict(
        self, normalized_text: str, headers: dict[str, str] = None
    ) -> dict[str, Any]:
        label, score = self.model.predict(normalized_text)
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

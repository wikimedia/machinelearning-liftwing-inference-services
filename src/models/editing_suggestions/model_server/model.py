import logging
import os
from typing import Any

import kserve
import pandas as pd
from kserve import ModelServer
from kserve.errors import InvalidInput

from python.preprocess_utils import check_input_param, validate_json_input

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


def to_dict(df: pd.DataFrame) -> dict[str, dict[str, list[dict[str, Any]]]]:
    editing_suggestions = {}
    for row in df.to_dict(orient="records"):
        page_title = row["page_title"]
        wiki_id = row["wiki_id"]
        suggestions = editing_suggestions.setdefault(wiki_id, {})
        page_suggestions = suggestions.setdefault(page_title, [])
        page_suggestions.append(row)
    return editing_suggestions


class EditingSuggestionsModel(kserve.Model):
    def __init__(self, name: str, model_path: str) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.editing_suggestions: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self.ready = False
        self.load()

    def load(self) -> None:
        logging.info("Loading suggestions from %s", self.model_path)
        df = pd.read_csv(self.model_path)
        self.editing_suggestions = to_dict(df)
        self.ready = True

    def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, str]:
        inputs = validate_json_input(inputs)
        wiki_id = inputs.get("wiki_id")
        page_title = inputs.get("page_title")
        check_input_param(wiki_id=wiki_id, page_title=page_title)

        if not isinstance(wiki_id, str):
            error_message = "The input 'wiki_id' should be a string."
            logging.error(error_message)
            raise InvalidInput(error_message)

        if not isinstance(page_title, str):
            error_message = "The input 'page_title' should be a string."
            logging.error(error_message)
            raise InvalidInput(error_message)

        return {"wiki_id": wiki_id, "page_title": page_title}

    def predict(
        self, inputs: dict[str, str], headers: dict[str, str] = None
    ) -> dict[str, list[dict[str, Any]]]:
        wiki_id = inputs["wiki_id"]
        page_title = inputs["page_title"]
        suggestions = self.editing_suggestions.get(wiki_id, {}).get(page_title, [])
        return {"suggestions": suggestions}


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "editing-suggestions")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/suggestions.csv")
    model = EditingSuggestionsModel(model_name, model_path)
    ModelServer(workers=1).start([model])

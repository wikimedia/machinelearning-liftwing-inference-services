import csv
import logging
import os
from typing import Any

import kserve
from kserve import ModelServer
from kserve.errors import InvalidInput

from python.preprocess_utils import check_input_param, validate_json_input

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


def parse_page_id(page_id: Any) -> int:
    if isinstance(page_id, bool):
        error_message = "The input 'page_id' should be an integer."
        logging.error(error_message)
        raise InvalidInput(error_message)

    if isinstance(page_id, int):
        return page_id

    if isinstance(page_id, str):
        try:
            return int(page_id)
        except ValueError:
            error_message = "The input 'page_id' should be an integer."
            logging.error(error_message)
            raise InvalidInput(error_message)

    error_message = "The input 'page_id' should be an integer."
    logging.error(error_message)
    raise InvalidInput(error_message)


def load_from_csv(path: str) -> dict[str, dict[int, list[dict[str, Any]]]]:
    editing_suggestions: dict[str, dict[int, list[dict[str, Any]]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wiki_id = row["wiki_id"]
            page_id = int(row["page_id"])
            suggestion = {
                **row,
                "revision_id": int(row["revision_id"]),
                "page_id": page_id,
            }
            pages = editing_suggestions.setdefault(wiki_id, {})
            pages.setdefault(page_id, []).append(suggestion)
    return editing_suggestions


class EditingSuggestionsModel(kserve.Model):
    def __init__(self, name: str, model_path: str) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.editing_suggestions: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self.ready = False
        self.load()

    def load(self) -> None:
        logging.info("Loading suggestions from %s", self.model_path)
        self.editing_suggestions = load_from_csv(self.model_path)
        self.ready = True

    def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        inputs = validate_json_input(inputs)
        wiki_id = inputs.get("wiki_id")
        page_id = inputs.get("page_id")
        check_input_param(wiki_id=wiki_id, page_id=page_id)

        if not isinstance(wiki_id, str):
            error_message = "The input 'wiki_id' should be a string."
            logging.error(error_message)
            raise InvalidInput(error_message)

        page_id = parse_page_id(page_id)

        return {"wiki_id": wiki_id, "page_id": page_id}

    def predict(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, list[dict[str, Any]]]:
        wiki_id = inputs["wiki_id"]
        page_id = inputs["page_id"]
        suggestions = self.editing_suggestions.get(wiki_id, {}).get(page_id, [])
        return {"suggestions": suggestions}


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "editing-suggestions")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/suggestions.csv")
    model = EditingSuggestionsModel(model_name, model_path)
    ModelServer(workers=1).start([model])

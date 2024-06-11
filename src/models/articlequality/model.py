import logging
import os
from typing import Any, Dict

import kserve

from python.preprocess_utils import validate_json_input


logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ArticleQualityModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/model.pkl")
        self.load()

    def load(self) -> None:
        # TODO: Load the model from the model_path
        self.ready = True

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        # TODO: Preprocess features
        inputs = validate_json_input(inputs)
        return inputs

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        # TODO: Run model.predict() on the preprocessed inputs
        return request


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model = ArticleQuality(name=model_name)
    kserve.ModelServer(workers=1).start([model])

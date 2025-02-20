import logging
import os
from typing import Any, Dict

import kserve

from python.preprocess_utils import validate_json_input

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class EditCheckModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
        self.load()

    def load(self) -> None:
        # self.model = load_model(self.model_path)
        self.ready = True

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)

        # TODO: Pydantic object
        # check_type = inputs.get("check_type")
        # original_text = inputs.get("original_text")
        # modified_text = inputs.get("modified_text")
        # language = inputs.get("lang")

        # {
        #     "check_type": "string", # examples: "peacock", "npov", "weasel"
        #     "original_text": "string",
        #     "modified_text": "string",
        #     "lang": "string"
        # }
        return inputs

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:

        # We will probably need something like that
        # prediction, prob, details =  self.model.predict(request["original_text"], request["modified_text"])
        # request is output of preprocess()

        prediction: bool = True
        prob: float = 0.666
        details: dict = {
            "violations": ["string"]
        }  # list of words or phrases that are problematic according to the model

        out = {
            "model_name": self.name,
            "model_version": str(self.model.model_version),
            "check_type": request["check_type"],
            "language": request["lang"],
            "prediction": {bool(prediction)},
            "probability": prob,
            "details": details,
        }

        return out


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model = EditCheckModel(name=model_name)
    kserve.ModelServer(workers=1).start([model])

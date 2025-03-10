import json
import logging
import os
from typing import Any, Dict

import kserve

from python.preprocess_utils import validate_json_input
from request_model import RequestModel


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
        request_model = RequestModel(**inputs)
        request_model_dict = request_model.model_dump(mode="json")
        return request_model_dict

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
            # "model_version": str(self.model.model_version),
            "model_version": "v1",
            "check_type": request["check_type"],
            "language": request["lang"],
            "prediction": bool(prediction),
            "probability": prob,
            "details": details,
        }
        json_out = json.dumps(out, indent=4)
        return json_out


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model = EditCheckModel(name=model_name)
    kserve.ModelServer(workers=1).start([model])

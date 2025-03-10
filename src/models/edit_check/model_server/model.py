import json
import logging
import os
from typing import Any, Dict, List, Tuple

import kserve
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

from python.preprocess_utils import validate_json_input
from request_model import RequestModel


logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)
BATCH: int = 16  # 16
MAXLEN: int = 512  # 4096 #
OUTCOME_RULE: dict[str, bool] = {"00": False, "01": True, "10": False, "11": True}


class EditCheckModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
        self.load()

    def load(self) -> None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, truncation=True, max_length=MAXLEN, device=device
        )

        # Load the pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(
            device
        )

        # Build the pipeline
        self.model_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=BATCH,
        )
        self.ready = True

    def _postprocess(self, preds: List[Dict[str, Any]]) -> Tuple[Any]:
        # Extract results for original text
        original_txt_is_peacock = preds[0].get("label").split("_")[1]
        # original_text_score = preds[0].get("score")

        # Extract results for modified text
        modified_txt_is_peacock = preds[1].get("label").split("_")[1]
        modified_txt_score = preds[1].get("score")

        # Apply the defined peacock rules
        final_outcome = OUTCOME_RULE[
            f"{original_txt_is_peacock}{modified_txt_is_peacock}"
        ]

        # This will be needed when we can detect which are the peacock words in the text.
        details: dict = {
            "violations": ["string"]
        }  # list of words or phrases that are problematic according to the model
        return final_outcome, modified_txt_score, details

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

        original_text = request.get("original_text")
        modified_text = request.get("modified_text")

        # Predict both original and modified text
        tokenizer_kwargs = {"truncation": True, "max_length": 512}
        predictions = self.model_pipeline(
            [original_text, modified_text], **tokenizer_kwargs, batch_size=BATCH
        )

        # Post process the prediction results
        final_prediction, probability, details = self._postprocess(predictions)

        # Finilize the response schema
        out = {
            "model_name": self.name,
            # "model_version": str(self.model.model_version),
            "model_version": "v1",
            "check_type": request["check_type"],
            "language": request["lang"],
            "prediction": final_prediction,
            "probability": round(probability, 3),
            "details": details,
        }
        json_out = json.dumps(out, indent=4)
        return json_out


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model = EditCheckModel(name=model_name)
    kserve.ModelServer(workers=1).start([model])

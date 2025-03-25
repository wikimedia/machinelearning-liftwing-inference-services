import os
import json
import logging
from typing import Any, Dict, List, Tuple

import kserve
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from kserve.errors import InvalidInput

from python.preprocess_utils import validate_json_input
from request_model import RequestModel
from config import MAX_BATCH_SIZE, MODEL_NAME

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

BATCH = MAX_BATCH_SIZE * 2  # Batch size for the model pipeline which receives pairs.
MAXLEN: int = 512  # Maximum length for tokenization
OUTCOME_RULE: dict[str, bool] = {"00": False, "01": True, "10": False, "11": False}


class EditCheckModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
        self.model_pipeline = self.load()

    def load(self) -> None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            truncation=True,
            max_length=MAXLEN,
            padding=True,
            return_tensors="pt",
        )

        # Load the pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # Build the pipeline
        model_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=BATCH,
        )
        self.ready = True
        return model_pipeline

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Tuple[List[str], Dict[str, List]]:
        """Preprocess method that validates and passed input request
            into Pydantic RequestModel object.
            Extracts the original and modified text into a list,
            and stores the RequestModel instances into a list for further usage.
        Args:
            inputs (Dict[str, Any]): Input request comsidered as dict from Kserve.
            headers (Dict[str, str], optional): Request headers. Defaults to None.

        Raises:
            InvalidInput: Invalid input kserve error.

        Returns:
            Tuple[List[str], List[Dict[str, Any]]]: Tuple of the text list input for the ml model, and a list of RequestModel instances.
        """

        try:
            flattened_pair_texts = []
            validated_input = validate_json_input(inputs)
            request_model = RequestModel(**validated_input)
            processed_requests = request_model.process_instances()
            for request in processed_requests["Valid"]:
                flattened_pair_texts.append(request["instance"].original_text)
                flattened_pair_texts.append(request["instance"].modified_text)

        except (ValueError, AttributeError, json.decoder.JSONDecodeError) as e:
            raise InvalidInput(f"Wrong request! Message: {e}.")

        return flattened_pair_texts, processed_requests

    async def predict(
        self,
        request: Tuple[List[str], Dict[str, List]],
        headers: Dict[str, str] = None,
    ) -> Tuple[List[str], Dict[str, List]]:
        """Predict method using the model pipeline object for text classification.

        Args:
            request (Tuple[List[str], Dict[str, List]]): Texts to be fed into the ml model, and input requests dict.
            headers (Dict[str, str], optional): _description_. Request headers to None.

        Returns:
            Tuple[List[str], Dict[str, List]]: Predictions list and requests dict.
        """

        # Extract the two lists from preprocess output
        text_for_prediction, processed_requests = request

        # Predict both original and modified text
        tokenizer_kwargs = {"truncation": True, "max_length": MAXLEN}
        predictions = self.model_pipeline(
            text_for_prediction, **tokenizer_kwargs, batch_size=BATCH
        )

        return predictions, processed_requests

    async def postprocess(
        self,
        predictions: Tuple[List[str], Dict[str, List]],
        headers: Dict[str, str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Postprocess the predictions list.
        Extract the probability and the predicted class.
        Apply the rules for original and modified text predictions.
        """
        model_outputs, processed_requests = predictions

        if len(model_outputs) % 2:
            raise InvalidInput(
                f""" There is an error in the model output: length = {len(model_outputs)}.
                 Model output length needs to be an even number since it is based on input pairs:
                 `original_text` and `modified_text`. Check your input request again, or dig into the model pipeline."""
            )
        formatted_responses = []
        # Iterate over pairs of consecutive elements
        for valid_request, original_pred, modified_pred in zip(
            processed_requests["Valid"], model_outputs[::2], model_outputs[1::2]
        ):
            original_txt_label = original_pred.get("label").split("_")[1]
            modified_txt_label = modified_pred.get("label").split("_")[1]
            modified_txt_score = modified_pred.get("score")

            # Apply the defined peacock rules
            final_outcome = OUTCOME_RULE[f"{original_txt_label}{modified_txt_label}"]

            # This will be needed when we can detect which are the peacock words in the text.
            details: dict = {
                "violations": ["string"]
            }  # list of words or phrases that are problematic according to the model

            # Construct the final response payload
            formatted_responses.append(
                {
                    "index": valid_request["index"],
                    "status_code": valid_request["status_code"],
                    "model_name": self.name,
                    "model_version": "v1",
                    "check_type": valid_request["instance"].check_type,
                    "language": valid_request["instance"].lang,
                    "prediction": final_outcome,
                    "probability": round(modified_txt_score, 3),
                    "details": details,
                }
            )

        # Include malformed requests into response payload and sort it based on index.
        formatted_responses.extend(processed_requests["Malformed"])
        sorted_predictions = sorted(formatted_responses, key=lambda x: x["index"])
        for prediction in sorted_predictions:
            prediction.pop("index", None)
        response_payload = {"predictions": sorted_predictions}
        return response_payload


if __name__ == "__main__":
    model_name = MODEL_NAME
    model = EditCheckModel(name=model_name)
    kserve.ModelServer(workers=1).start([model])

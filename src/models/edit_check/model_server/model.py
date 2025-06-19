import json
import logging
import os
from distutils.util import strtobool
from typing import Any

import kserve
import shap
import torch
from fastapi.middleware.cors import CORSMiddleware
from kserve.errors import InvalidInput
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

# from shap.plots._text import unpack_shap_explanation_contents, process_shap_values
from python.preprocess_utils import validate_json_input
from src.models.edit_check.model_server.config import settings
from src.models.edit_check.model_server.request_model import RequestModel

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

BATCH = (
    settings.max_batch_size * 2
)  # Batch size for the model pipeline which receives pairs.
MAXLEN: int = 512  # Maximum length for tokenization
OUTCOME_RULE: dict[str, bool] = {"00": False, "01": True, "10": False, "11": False}


class EditCheckModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
        self.model_pipeline = self.load()
        self.explainer = shap.Explainer(self.model_pipeline)
        self.use_metadata = strtobool(
            os.environ.get("USE_METADATA", "False")
        )  # if using lang and page_title in model input

    def load(self) -> Pipeline:
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
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> tuple[list[str], list[str], dict[str, list]]:
        """Preprocess method that validates and passed input request
            into Pydantic RequestModel object.
            Extracts the original/modified text into a list to be fed into the ml-model,
            and stores the Valid/Malformed instances into a dict for further usage.
        Args:
            inputs (Dict[str, Any]): Input request comsidered as dict from Kserve.
            headers (Dict[str, str], optional): Request headers. Defaults to None.

        Raises:
            InvalidInput: Invalid input kserve error.

        Returns:
            Tuple[List[str], List[str], Dict[str, List]]:
                A tuple containing:
                - List[str]: The flattened_pair_texts list for the ml model.
                - List[str]: The modified text list for the shap explainer (if return_shap_values True).
                - Dict[str, List]: A dictionary of Valid/Malformed processed_requests instances.
        """

        try:
            flattened_pair_texts = []
            text_for_explanation = []
            validated_input = validate_json_input(inputs)
            request_model = RequestModel(**validated_input)
            processed_requests = request_model.process_instances()
            for request in processed_requests["Valid"]:
                inst = request["instance"]
                if not self.use_metadata:  # Model input: {text}
                    flattened_pair_texts.append(inst.original_text)
                    flattened_pair_texts.append(inst.modified_text)
                    # Extract the modified text when return_shap_values is True
                    if inst.return_shap_values:
                        text_for_explanation.append(inst.modified_text)
                else:  # Model input: {lang}[SEP]{page_title}[SEP]{text}
                    flattened_pair_texts.append(
                        "[SEP]".join((inst.lang, inst.page_title, inst.original_text))
                    )
                    flattened_pair_texts.append(
                        "[SEP]".join((inst.lang, inst.page_title, inst.modified_text))
                    )
                    if inst.return_shap_values:
                        text_for_explanation.append(
                            "[SEP]".join(
                                (inst.lang, inst.page_title, inst.modified_text)
                            )
                        )
        except (ValueError, AttributeError, json.decoder.JSONDecodeError) as e:
            raise InvalidInput(f"Wrong request! Message: {e}.")

        return flattened_pair_texts, text_for_explanation, processed_requests

    async def predict(
        self,
        request: tuple[list[str], list[str], dict[str, list]],
        headers: dict[str, str] = None,
    ) -> tuple[list[Any], list[Any], dict[str, list]]:
        """Predict method using the model pipeline object for text classification.

        Args:
            request (Tuple[List[str], List[str], Dict[str, List]]): Texts to be fed into the ml model, shap explainer, input requests dict.
            headers (Dict[str, str], optional): _description_. Request headers to None.

        Returns:
            Tuple[List[Any], List[Any], Dict[str, List]]: Predictions list, explaniner outputs and requests dict.
        """

        # Extract the three lists from preprocess output
        text_for_prediction, text_for_explanation, processed_requests = request

        # Predict both original and modified text
        tokenizer_kwargs = {"truncation": True, "max_length": MAXLEN}
        predictions = self.model_pipeline(
            text_for_prediction, **tokenizer_kwargs, batch_size=BATCH
        )
        # Pass the modified text to the explainer
        explainer_outputs = []
        if len(text_for_explanation) > 0:
            explainer_outputs = self.explainer(text_for_explanation)

        return predictions, explainer_outputs, processed_requests

    async def postprocess(
        self,
        predictions: tuple[list[Any], list[Any], dict[str, list]],
        headers: dict[str, str] = None,
        return_index: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Postprocess the predictions list.
        Extract the probability and the predicted class.
        Apply the rules for original and modified text predictions.
        """
        model_outputs, explainer_outputs, processed_requests = predictions
        explainer_outputs_idx = 0

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

            details: dict = {}
            # Return SHAP values and tokens if the request has return_shap_values set to True
            if valid_request["instance"].return_shap_values:
                shap_values = explainer_outputs[explainer_outputs_idx]
                (
                    unpacked_values,
                    clustering,
                ) = shap.plots._text.unpack_shap_explanation_contents(shap_values)
                # Reprocess the shap_values and tokens based on clustering data
                # https://github.com/shap/shap/blob/master/shap/plots/_text.py#L378
                tokens, values, _ = shap.plots._text.process_shap_values(
                    shap_values.data, unpacked_values[:, 1], 1, "", clustering, False
                )
                # Return top 3 tokens with highest SHAP values
                details["violations"] = sorted(
                    zip(tokens, values), key=lambda x: x[1], reverse=True
                )[:3]
                details["shap_values"] = values.tolist()
                details["tokens"] = tokens.tolist()
                explainer_outputs_idx += 1

            # Construct the final response payload
            formatted_responses.append(
                {
                    "index": valid_request["index"],
                    "status_code": valid_request["status_code"],
                    "model_name": self.name,
                    "model_version": "v1",
                    "check_type": valid_request["instance"].check_type,
                    "language": valid_request["instance"].lang,
                    "page_title": valid_request["instance"].page_title,
                    "prediction": final_outcome,
                    "probability": round(modified_txt_score, 3),
                    "details": details,
                }
            )

        # Include malformed requests into response payload and sort it based on index.
        formatted_responses.extend(processed_requests["Malformed"])
        sorted_predictions = sorted(formatted_responses, key=lambda x: x["index"])
        if not return_index:
            for prediction in sorted_predictions:
                prediction.pop("index", None)
        response_payload = {"predictions": sorted_predictions}
        return response_payload


if __name__ == "__main__":
    model = EditCheckModel(name=settings.model_name)
    model_server = kserve.ModelServer(workers=1)
    if settings.environment == "dev":
        model_server._rest_server.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"]
        )
    model_server.start([model])

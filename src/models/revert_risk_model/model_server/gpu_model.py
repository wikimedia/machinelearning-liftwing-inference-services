import logging
from typing import Any

import torch
from base_model import RevisionRevertRiskModel
from kserve.errors import InvalidInput

from python import events
from python.preprocess_utils import (
    check_input_param,
    get_lang,
    get_rev_id,
    is_domain_wikipedia,
    validate_json_input,
)


class RevertRiskMultilingualGPU(RevisionRevertRiskModel):
    def __init__(
        self,
        name: str,
        module_name: str,
        model_path: str,
        wiki_url: str,
        aiohttp_client_timeout: int,
        force_http: bool,
        allow_revision_json_input: bool,
        eventgate_url: str | None,
        eventgate_stream: str | None,
    ) -> None:
        super().__init__(
            name,
            module_name,
            model_path,
            wiki_url,
            aiohttp_client_timeout,
            force_http,
            allow_revision_json_input,
            eventgate_url=eventgate_url,
            eventgate_stream=eventgate_stream,
        )
        self.use_gpu()

    def use_gpu(self) -> None:
        """
        Loads the model on the appropriate device and updates its reference.
        In production this will typically be a GPU; for local development or
        environments without CUDA support we transparently fall back to CPU.
        This function needs to run after the webserver's initialization
        (that forks and creates new processes, see https://github.com/pytorch/pytorch/issues/83973).
        """
        if not self.device:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                logging.info(f"Using device: {self.device}")
            else:
                self.device = torch.device("cpu")
                logging.warning(
                    "CUDA is not available or PyTorch is CPU-only; using CPU instead."
                )
            # Move model components to the selected device
            self.model.title_model.model.to(self.device)
            self.model.insert_model.model.to(self.device)
            self.model.remove_model.model.to(self.device)
            self.model.change_model.model.to(self.device)
            # Update pipeline devices
            self.model.title_model.device = self.device
            self.model.insert_model.device = self.device
            self.model.remove_model.device = self.device
            self.model.change_model.device = self.device

    async def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Extend the base preprocess to support event-based inputs.

        For plain API hits (e.g. {\"lang\": ..., \"rev_id\": ...}) we defer
        entirely to the base implementation so behaviour is unchanged.

        For event-based inputs we:
        - ensure the event is for a Wikipedia project,
        - derive lang and rev_id from the event,
        - inject them into the request,
        - and then delegate to the base model's preprocess to fetch revision data.
        """
        if self.event_key in inputs:
            inputs = validate_json_input(inputs)
            source_event = inputs[self.event_key]
            if not is_domain_wikipedia(source_event):
                error_message = (
                    "This model is not recommended for use in projects outside of "
                    "Wikipedia (e.g. Wiktionary, Wikinews, etc)"
                )
                logging.error(error_message)
                raise InvalidInput(error_message)

            lang = get_lang(inputs, self.event_key)
            rev_id = get_rev_id(inputs, self.event_key)
            check_input_param(lang=lang, rev_id=rev_id)
            inputs["lang"] = lang
            inputs["rev_id"] = rev_id
            return await super().preprocess(inputs, headers)

        # No event wrapper: keep the original single-revision behaviour.
        return await super().preprocess(inputs, headers)

    async def predict(
        self, request: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Run prediction using the base model logic and, when an event payload
        is present, emit a prediction_classification event to EventGate so
        that downstream consumers (e.g. mediawiki.page_revert_risk_prediction_change)
        can incorporate multilingual revert risk scores.
        """
        prediction = super().predict(request, headers)

        if self.event_key in request:
            prediction_results = {
                "predictions": [str(prediction["output"]["prediction"]).lower()],
                "probabilities": prediction["output"]["probabilities"],
            }
            await self.send_event(
                request[self.event_key],
                prediction_results,
                prediction["model_version"],
            )

        return prediction

    async def send_event(
        self,
        page_change_event: dict[str, Any],
        prediction_results: dict[str, Any],
        model_version: str,
    ) -> None:
        """
        Send a multilingual revert risk prediction classification change event to EventGate
        generated from the page_change event and prediction_results passed as input.
        """
        if not self.eventgate_url or not self.eventgate_stream:
            logging.error(
                "EVENTGATE_URL or EVENTGATE_STREAM is not configured; "
                "skipping event emission for multilingual revert risk."
            )
            return

        revertrisk_multilingual_event = events.generate_prediction_classification_event(
            page_change_event,
            self.eventgate_stream,
            "revertrisk-multilingual",
            model_version,
            prediction_results,
        )
        await events.send_event(
            revertrisk_multilingual_event,
            self.eventgate_url,
            self.tls_cert_bundle_path,
            self.custom_user_agent,
            self.get_http_client_session("eventgate"),
        )

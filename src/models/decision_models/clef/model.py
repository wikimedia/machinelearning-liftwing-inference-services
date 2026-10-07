import asyncio
import base64
import binascii
import io
import logging
import os
import sys
import threading
import warnings

import kserve
import torch
from kserve.errors import InferenceError, InvalidInput
from PIL import Image

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

# Clef is not an AutoModelForCausalLM. The release is a Qwen backbone plus a
# separate joint schema head, loaded by `joint_schema_model.py`, which ships
# with the weights. That module also provides `systemone`, which takes a
# Jev/SystemOne request body and returns the answers, so this server is a thin
# wrapper around it rather than a prompt builder.
# See https://huggingface.co/Cloudflare/clef-flash#jev--systemone-api
QUESTION_TYPES = ("noul", "choice", "score")

# Downscaling to 640px wide costs nothing in answer quality and 1.73x less
# time. Measured on ML-Lab (MI210, gfx90a) with a 960x1279 Commons portrait:
# 960px 646ms, 640px 374ms, 480px 374ms, 320px 307ms, with the infobox verdict
# unchanged at every size and the largest drift 0.022 on a 0-1 probability.
# 480px and 640px cost the same because they land in the same vision tiling,
# so 640px is the most resolution available for the lower cost.
# LiftWing has MI300X, where these timings will differ; the ratios should hold.
DEFAULT_MAX_IMAGE_WIDTH = 640

# A base64 image arrives before anything can inspect it, so it is bounded by
# size first. 8 MiB encoded is far above a Commons thumbnail and far below
# what would trouble the pod's memory.
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024

# `encode_record` truncates at 16,384 tokens by default, and `systemone` does
# not expose that argument, so an over-long state would be silently cut rather
# than refused. This bounds the state before it gets there. Four characters per
# token is the usual rough ratio for English, and the point is to refuse the
# obviously oversized rather than to count precisely.
DEFAULT_MAX_STATE_CHARS = 60000


def parse_bool(value: str) -> bool:
    """
    Parse a boolean environment variable.

    `distutils.util.strtobool` would do this, but distutils left the standard
    library in Python 3.12 and now resolves only through a setuptools shim, so
    using it makes the server depend on setuptools staying installed.
    """
    return value.strip().lower() in ("1", "true", "yes", "y", "on", "t")


def positive_int(name: str, value: str) -> int:
    """
    Read a positive integer from the environment, failing at startup.

    A zero or negative bound would be accepted silently and then produce
    confusing failures per request, such as resizing an image to zero width.
    """
    try:
        parsed = int(value)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {value!r}") from e
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1, got {parsed}")
    return parsed


class ClefModel(kserve.Model):
    def __init__(
        self,
        name: str,
        model_path: str,
        device: str,
        max_images: int,
        max_questions: int,
        max_image_width: int,
        max_image_bytes: int,
        max_state_chars: int,
        warmup: bool,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.device = device
        self.max_images = max_images
        self.max_questions = max_questions
        self.max_image_width = max_image_width
        self.max_image_bytes = max_image_bytes
        self.max_state_chars = max_state_chars
        self.warmup = warmup
        self.model = None
        self.processor = None
        self.systemone = None
        # One GPU and one model instance, so forward passes are serialised.
        # `predict` runs the forward pass in a worker thread to keep the event
        # loop free (see the comment there), so two requests really can arrive
        # at once and this is what keeps them apart. Batching measured only
        # 1.20x at eight images, so little throughput is given up by holding
        # the lock for a whole request.
        self.inference_lock = threading.Lock()
        self.ready = False

    def load(self) -> None:
        """
        Load the Clef backbone and joint schema head.

        The loader lives with the weights rather than in transformers, so the
        model directory goes on sys.path and `joint_schema_model` is imported
        from there. This runs vendor code in the server process, which is why
        the module is imported from the model path rather than pip installed.
        """
        try:
            logging.info("Importing the release loader from %s...", self.model_path)
            if self.model_path not in sys.path:
                sys.path.insert(0, self.model_path)
            from joint_schema_model import load_release_model, systemone

            self.systemone = systemone

            logging.info("Loading model...")
            self.model, self.processor = load_release_model(
                self.model_path, device=self.device
            )

            if self.warmup:
                self._warm_up()

            self.ready = True
            logging.info("Model loaded successfully!")
        except Exception as e:
            error_message = f"Failed to load model. Reason: {e}"
            logging.critical(error_message)
            raise kserve.errors.ModelMissingError(error_message) from e

    def _warm_up(self) -> None:
        """
        Run one text request and one image request before reporting ready.

        A new input shape is compiled on first use, and the cost is large: on
        ML-Lab the first text request took 7.1s against 180ms warm, and the
        first image request took 22.3s against 649ms warm. Without this, the
        first real request after every deployment pays that.
        """
        warm_question = {"warm": {"type": "noul", "instructions": "Warm up."}}

        # The same context `predict` uses. torch.inference_mode() changes how
        # the forward pass executes, so warming up without it could cache a
        # different path from the one served and leave the first real request
        # paying the compile cost anyway. The inference lock is deliberately
        # not taken: this runs inside load(), before ModelServer().start(), so
        # nothing else can be in flight, and taking it would imply a
        # concurrency that does not exist yet.
        with torch.inference_mode():
            logging.info("Warming up the text path...")
            self.systemone(
                self.model,
                self.processor,
                {"model": self.name, "state": "warmup", "questions": warm_question},
            )

            logging.info("Warming up the image path...")
            self.systemone(
                self.model,
                self.processor,
                {
                    "model": self.name,
                    "state": "warmup",
                    "images": [Image.new("RGB", (self.max_image_width, 480), "white")],
                    "questions": warm_question,
                },
            )
        logging.info("Warmup complete.")

    def _decode_image(self, encoded: str, index: int) -> Image.Image:
        """
        Decode one base64 image and downscale it to the configured width.
        """
        if not isinstance(encoded, str):
            error_message = (
                f"Image at index {index} must be a base64 string, got "
                f"{type(encoded).__name__}."
            )
            logging.error(error_message)
            raise InvalidInput(error_message)

        if len(encoded) > self.max_image_bytes:
            error_message = (
                f"Image at index {index} is {len(encoded)} bytes encoded, "
                f"over the maximum of {self.max_image_bytes}."
            )
            logging.error(error_message)
            raise InvalidInput(error_message)

        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, TypeError, ValueError) as e:
            error_message = f"Image at index {index} is not valid base64: {e}"
            logging.error(error_message)
            raise InvalidInput(error_message) from e

        try:
            # Pillow only warns above MAX_IMAGE_PIXELS and raises above twice
            # that, so a small upload can still expand into hundreds of
            # megabytes: a 439 KiB PNG decodes to 12000x12000. Promoting the
            # warning to an error refuses those here.
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image = Image.open(io.BytesIO(raw))
                image.load()
            image = image.convert("RGB")
        except Image.DecompressionBombWarning as e:
            error_message = f"Image at index {index} is too large to decode: {e}"
            logging.error(error_message)
            raise InvalidInput(error_message) from e
        except Exception as e:
            error_message = f"Image at index {index} could not be read: {e}"
            logging.error(error_message)
            raise InvalidInput(error_message) from e

        if image.size[0] > self.max_image_width:
            height = max(1, round(image.size[1] * self.max_image_width / image.size[0]))
            image = image.resize((self.max_image_width, height), Image.LANCZOS)

        return image

    def _validate_questions(self, questions: dict) -> None:
        """
        Check the question schema before it reaches the vendor code.
        """
        if not isinstance(questions, dict) or not questions:
            error_message = "'questions' must be a non-empty mapping of id to question."
            logging.error(error_message)
            raise InvalidInput(error_message)

        if len(questions) > self.max_questions:
            error_message = (
                f"{len(questions)} questions exceeds the maximum of "
                f"{self.max_questions}."
            )
            logging.error(error_message)
            raise InvalidInput(error_message)

        for question_id, question in questions.items():
            if not isinstance(question, dict):
                error_message = f"Question '{question_id}' must be an object."
                logging.error(error_message)
                raise InvalidInput(error_message)

            question_type = question.get("type")
            if question_type not in QUESTION_TYPES:
                error_message = (
                    f"Question '{question_id}' has type {question_type!r}; "
                    f"must be one of {', '.join(QUESTION_TYPES)}."
                )
                logging.error(error_message)
                raise InvalidInput(error_message)

            # A choice needs named options and a score needs ordered ones.
            # Without criteria the model has nothing to score against, and the
            # failure would otherwise surface inside the vendor code.
            criteria = question.get("criteria")
            if question_type == "choice" and (
                not isinstance(criteria, dict) or len(criteria) < 2
            ):
                error_message = (
                    f"Question '{question_id}' is a choice and needs "
                    "'criteria' as a mapping of at least two options."
                )
                logging.error(error_message)
                raise InvalidInput(error_message)

            if question_type == "score" and (
                not isinstance(criteria, list) or len(criteria) < 2
            ):
                error_message = (
                    f"Question '{question_id}' is a score and needs "
                    "'criteria' as a list of at least two options."
                )
                logging.error(error_message)
                raise InvalidInput(error_message)

    def preprocess(self, payload: dict, headers: dict[str, str] = None) -> dict:
        """
        Validate the payload and build a Jev/SystemOne request body.

        The payload is the SystemOne shape with one change: `images` are
        base64 strings rather than PIL images, because this arrives over HTTP.
        """
        if not isinstance(payload, dict):
            error_message = "Invalid payload format. Must be a JSON object."
            logging.error(error_message)
            raise InvalidInput(error_message)

        if "state" not in payload:
            error_message = "Invalid payload format. Must contain a 'state' field."
            logging.error(error_message)
            raise InvalidInput(error_message)

        if "questions" not in payload:
            error_message = "Invalid payload format. Must contain a 'questions' field."
            logging.error(error_message)
            raise InvalidInput(error_message)

        # Video is part of the SystemOne API but is not served here, and
        # ignoring the field silently would look like it had been used.
        if payload.get("videos"):
            error_message = "'videos' is not supported by this server."
            logging.error(error_message)
            raise InvalidInput(error_message)

        state = payload["state"]
        state_size = len(state if isinstance(state, str) else str(state))
        if state_size > self.max_state_chars:
            error_message = (
                f"'state' is {state_size} characters, over the maximum of "
                f"{self.max_state_chars}."
            )
            logging.error(error_message)
            raise InvalidInput(error_message)

        questions = payload["questions"]
        self._validate_questions(questions)

        body = {"model": self.name, "state": state, "questions": questions}

        encoded_images = payload.get("images") or []
        if not isinstance(encoded_images, list):
            error_message = "'images' must be a list of base64 encoded images."
            logging.error(error_message)
            raise InvalidInput(error_message)

        if len(encoded_images) > self.max_images:
            error_message = (
                f"{len(encoded_images)} images exceeds the maximum of "
                f"{self.max_images}."
            )
            logging.error(error_message)
            raise InvalidInput(error_message)

        if encoded_images:
            body["images"] = [
                self._decode_image(encoded, index)
                for index, encoded in enumerate(encoded_images)
            ]

        return body

    async def predict(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """
        Score every option of every question in a single forward pass.

        Async, and the forward pass goes to a worker thread. KServe calls a
        sync `predict` directly from its async handler rather than through an
        executor, so a sync one would hold the event loop for the whole
        inference: about 650ms for an image on an MI210. Nothing else would
        run in that window, including the health and readiness probes, and
        under load that is enough to get a working pod restarted.
        """

        def run_inference() -> dict:
            # One GPU, so one forward pass at a time. The lock is held for the
            # whole call rather than around the model only, because the vendor
            # code encodes the record and the images on the way in.
            with self.inference_lock, torch.inference_mode():
                return self.systemone(self.model, self.processor, inputs)

        try:
            logging.info("Performing inference...")
            return await asyncio.to_thread(run_inference)
        except Exception as e:
            error_message = f"Error during inference: {e}"
            logging.error(error_message)
            raise InferenceError(error_message) from e

    def postprocess(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """
        Return the SystemOne response unchanged.

        There is no text to parse: `answers` already holds a probability for
        every allowed option of every question, so reshaping it here would
        only hide information the caller asked for.
        """
        try:
            return {
                "model": inputs.get("model", self.name),
                "answers": inputs["answers"],
                "usage": inputs.get("usage"),
            }
        except KeyError as e:
            error_message = f"Error during post-processing: missing {e}"
            logging.error(error_message)
            raise InferenceError(error_message) from e


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "clef-flash")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/snapshots/clef-flash")
    device = os.environ.get("DEVICE", "cuda")
    max_images = positive_int("MAX_IMAGES", os.environ.get("MAX_IMAGES", "4"))
    max_questions = positive_int("MAX_QUESTIONS", os.environ.get("MAX_QUESTIONS", "32"))
    max_image_width = positive_int(
        "MAX_IMAGE_WIDTH",
        os.environ.get("MAX_IMAGE_WIDTH", str(DEFAULT_MAX_IMAGE_WIDTH)),
    )
    max_image_bytes = positive_int(
        "MAX_IMAGE_BYTES",
        os.environ.get("MAX_IMAGE_BYTES", str(DEFAULT_MAX_IMAGE_BYTES)),
    )
    max_state_chars = positive_int(
        "MAX_STATE_CHARS",
        os.environ.get("MAX_STATE_CHARS", str(DEFAULT_MAX_STATE_CHARS)),
    )
    warmup = parse_bool(os.environ.get("WARMUP", "True"))

    model = ClefModel(
        name=model_name,
        model_path=model_path,
        device=device,
        max_images=max_images,
        max_questions=max_questions,
        max_image_width=max_image_width,
        max_image_bytes=max_image_bytes,
        max_state_chars=max_state_chars,
        warmup=warmup,
    )

    model.load()
    kserve.ModelServer().start([model])

"""Tests for the clef model-server.

The model is never loaded: `load()` imports vendor code and reads 15 GiB of
weights, so every test here builds the server directly and exercises the three
KServe stages around a stubbed vLLM engine, head and vendor module. What is
worth testing is the part this repo owns, which is the payload contract: what
is refused, what an image becomes before it reaches the model, what reaches
vLLM, and what comes back.

conftest.py stubs torch before this module is imported. Pillow is real.
"""

import asyncio
import base64
import contextlib
import inspect
import io
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest
from kserve.errors import InferenceError, InvalidInput
from PIL import Image

from src.models.decision_models.clef import model as clef_model
from src.models.decision_models.clef.model import (
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_IMAGE_WIDTH,
    DEFAULT_MAX_STATE_CHARS,
    ClefModel,
    parse_bool,
    positive_int,
)

NOUL = {"revert": {"type": "noul", "instructions": "Should this be reverted?"}}


def encoded_png(width: int, height: int) -> str:
    """A real PNG, base64 encoded, as a client would send one."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def model():
    """A server built without load(): no weights, no vendor import."""
    return ClefModel(
        name="clef-flash",
        model_path="/mnt/models",
        device="cuda",
        max_images=4,
        max_questions=32,
        max_image_width=DEFAULT_MAX_IMAGE_WIDTH,
        max_image_bytes=DEFAULT_MAX_IMAGE_BYTES,
        max_state_chars=DEFAULT_MAX_STATE_CHARS,
        warmup=False,
        gpu_memory_utilization=0.85,
    )


# What `encode_record` returns for a text-only record with no questions.
ENCODED = SimpleNamespace(input_ids=(1, 2, 3), questions=())


class FakeEngine:
    """Stands in for the vLLM engine.

    It returns the prompt tokens as vLLM does after it expands the images. By
    default these are the `encode_record` tokens, so the token check passes.
    """

    def __init__(self, prompt_token_ids=ENCODED.input_ids, delay=0.0):
        self.prompt_token_ids = prompt_token_ids
        self.delay = delay
        self.prompts = []
        self.now = 0
        self.most = 0

    async def encode(self, prompt, pooling_params, request_id):
        self.prompts.append(prompt)
        self.now += 1
        self.most = max(self.most, self.now)
        await asyncio.sleep(self.delay)
        self.now -= 1
        yield SimpleNamespace(
            prompt_token_ids=list(self.prompt_token_ids),
            outputs=SimpleNamespace(data=mock.MagicMock()),
        )


@pytest.fixture
def loaded(model):
    """A server with stand-ins for what load() sets: no GPU, no vendor code.

    torch is patched on the module, because other test directories replace
    `sys.modules["torch"]` with stubs that lack the names `_score` reads.
    """
    model.vendor = SimpleNamespace(
        encode_record=mock.MagicMock(return_value=ENCODED),
        systemone_answer=mock.MagicMock(),
    )
    model.vllm_engine = FakeEngine()
    model.head = mock.MagicMock(return_value=[[]])
    model.lm_head = mock.MagicMock()
    model.processor = SimpleNamespace(tokenizer=object())
    model.image_pad_id = 9
    fake_torch = mock.MagicMock(inference_mode=contextlib.nullcontext)
    with mock.patch.object(clef_model, "torch", fake_torch):
        yield model


class TestPayloadValidation:
    """Every rejection returns InvalidInput, which KServe turns into a 400
    rather than a 500."""

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ("not a dict", "JSON object"),
            ({"questions": NOUL}, "'state'"),
            ({"state": "x"}, "'questions'"),
            ({"state": "x", "questions": {}}, "non-empty"),
            ({"state": "x", "questions": []}, "non-empty"),
            ({"state": "x", "questions": {"a": "noul"}}, "must be an object"),
            ({"state": "x", "questions": {"a": {}}}, "must be one of"),
            ({"state": "x", "questions": {"a": {"type": "vibes"}}}, "must be one of"),
        ],
    )
    def test_malformed_payloads_are_refused(self, model, payload, expected):
        with pytest.raises(InvalidInput, match=expected):
            model.preprocess(payload)

    @pytest.mark.parametrize(
        "question",
        [
            {"type": "choice"},
            {"type": "choice", "criteria": {"only": "one option"}},
            {"type": "choice", "criteria": ["yes", "no"]},
        ],
    )
    def test_a_choice_needs_at_least_two_named_options(self, model, question):
        # Without criteria the model has nothing to score against, and the
        # failure would otherwise surface inside the vendor code.
        with pytest.raises(InvalidInput, match="at least two options"):
            model.preprocess({"state": "x", "questions": {"fit": question}})

    @pytest.mark.parametrize(
        "question",
        [
            {"type": "score"},
            {"type": "score", "criteria": ["only"]},
            {"type": "score", "criteria": {"0": "Minor"}},
        ],
    )
    def test_a_score_needs_at_least_two_ordered_options(self, model, question):
        with pytest.raises(InvalidInput, match="at least two options"):
            model.preprocess({"state": "x", "questions": {"severity": question}})

    def test_too_many_questions_are_refused(self, model):
        questions = {f"q{i}": {"type": "noul"} for i in range(33)}
        with pytest.raises(InvalidInput, match="exceeds the maximum"):
            model.preprocess({"state": "x", "questions": questions})

    def test_an_over_long_state_is_refused_rather_than_truncated(self, model):
        # encode_record truncates at 16,384 tokens and systemone does not
        # expose that argument, so without this the state would be silently
        # cut instead of refused.
        payload = {"state": "x" * (DEFAULT_MAX_STATE_CHARS + 1), "questions": NOUL}
        with pytest.raises(InvalidInput, match="over the maximum"):
            model.preprocess(payload)

    def test_video_is_refused_rather_than_ignored(self, model):
        # Video is part of the SystemOne API but is not served here, and
        # ignoring the field would look like it had been used.
        payload = {"state": "x", "questions": NOUL, "videos": ["anything"]}
        with pytest.raises(InvalidInput, match="not supported"):
            model.preprocess(payload)

    def test_a_valid_payload_becomes_a_systemone_body(self, model):
        body = model.preprocess(
            {
                "state": "Edit replaced the lead with promotional language.",
                "questions": {
                    "revert": {"type": "noul", "instructions": "Revert?"},
                    "fit": {"type": "choice", "criteria": {"y": "yes", "n": "no"}},
                    "severity": {"type": "score", "criteria": ["Minor", "Severe"]},
                },
            }
        )
        assert body["model"] == "clef-flash"
        assert body["state"].startswith("Edit replaced")
        assert set(body["questions"]) == {"revert", "fit", "severity"}
        # No images key at all rather than an empty list: the vendor code
        # treats a text-only record differently.
        assert "images" not in body

    def test_a_non_string_state_is_accepted(self, model):
        # The API takes JSON as a state, not only prose.
        body = model.preprocess({"state": {"edit": {"id": 1}}, "questions": NOUL})
        assert body["state"] == {"edit": {"id": 1}}


class TestImageHandling:
    @pytest.mark.parametrize(
        "images,expected",
        [
            ("not a list", "must be a list"),
            ([encoded_png(8, 8)] * 5, "exceeds the maximum"),
            ([123], "must be a base64 string"),
            (["not!valid!base64"], "not valid base64"),
            ([base64.b64encode(b"hello").decode()], "could not be read"),
        ],
    )
    def test_bad_images_are_refused(self, model, images, expected):
        payload = {"state": "x", "questions": NOUL, "images": images}
        with pytest.raises(InvalidInput, match=expected):
            model.preprocess(payload)

    def test_a_wide_image_is_downscaled(self, model):
        # 640px costs 374ms against 646ms at 960px on an MI210, with the
        # answers unchanged, so the server downscales rather than paying for
        # resolution the model does not use.
        body = model.preprocess(
            {"state": "x", "questions": NOUL, "images": [encoded_png(1200, 900)]}
        )
        assert body["images"][0].size == (DEFAULT_MAX_IMAGE_WIDTH, 480)

    def test_a_narrow_image_is_left_alone(self, model):
        body = model.preprocess(
            {"state": "x", "questions": NOUL, "images": [encoded_png(320, 240)]}
        )
        assert body["images"][0].size == (320, 240)

    def test_aspect_ratio_survives_the_downscale(self, model):
        body = model.preprocess(
            {"state": "x", "questions": NOUL, "images": [encoded_png(960, 1279)]}
        )
        width, height = body["images"][0].size
        assert width == DEFAULT_MAX_IMAGE_WIDTH
        assert abs(width / height - 960 / 1279) < 0.01

    def test_a_very_tall_image_keeps_at_least_one_row(self, model):
        # Rounding a 4000x1 image to 640 wide gives 0 rows without a floor,
        # and PIL raises on a zero-height resize.
        body = model.preprocess(
            {"state": "x", "questions": NOUL, "images": [encoded_png(4000, 1)]}
        )
        assert body["images"][0].size[1] >= 1

    def test_an_oversized_payload_is_refused_before_decoding(self, model):
        model.max_image_bytes = 1024
        with pytest.raises(InvalidInput, match="over the maximum"):
            model.preprocess(
                {"state": "x", "questions": NOUL, "images": [encoded_png(600, 600)]}
            )

    def test_a_decompression_bomb_is_refused(self, model):
        # Pillow only warns above MAX_IMAGE_PIXELS and raises above twice
        # that, so a 439 KiB PNG decodes to 12000x12000 and several hundred
        # megabytes. The server promotes the warning to an error.
        bomb = encoded_png(12000, 12000)
        with pytest.raises(InvalidInput, match="too large to decode"):
            model.preprocess({"state": "x", "questions": NOUL, "images": [bomb]})

    def test_several_images_are_all_decoded(self, model):
        body = model.preprocess(
            {
                "state": "x",
                "questions": NOUL,
                "images": [encoded_png(800, 600), encoded_png(100, 100)],
            }
        )
        assert [image.size for image in body["images"]] == [(640, 480), (100, 100)]

    def test_the_failing_image_is_named(self, model):
        payload = {
            "state": "x",
            "questions": NOUL,
            "images": [encoded_png(8, 8), "not!valid!base64"],
        }
        with pytest.raises(InvalidInput, match="index 1"):
            model.preprocess(payload)


class TestPredict:
    """`predict` is async and runs the forward pass in a worker thread.

    KServe calls a sync `predict` directly from its async handler rather than
    through an executor (see Model.__call__ in kserve 0.19), so a sync one
    would hold the event loop for the whole inference.
    """

    def test_predict_is_a_coroutine_function(self, model):
        # This is what KServe branches on: `inspect.iscoroutinefunction`
        # decides whether it awaits or calls directly. If this regresses to a
        # sync def, the forward pass silently moves back onto the event loop.
        assert inspect.iscoroutinefunction(model.predict)

    def test_the_body_reaches_encode_record_unchanged(self, loaded):
        body = {"model": "clef-flash", "state": "x", "questions": NOUL}
        asyncio.run(loaded.predict(body))
        assert loaded.vendor.encode_record.call_args.args[1] is body

    def test_the_response_follows_the_vendor_systemone(self, loaded):
        question = SimpleNamespace(question_id="revert", option_ids=("true", "false"))
        loaded.vendor.encode_record.return_value = SimpleNamespace(
            input_ids=(1, 2, 3), questions=(question,)
        )
        logits = mock.MagicMock()
        logits.float.return_value.softmax.return_value.tolist.return_value = [0.9, 0.1]
        loaded.head.return_value = [[logits]]
        loaded.vendor.systemone_answer.side_effect = lambda q, probabilities: (
            probabilities
        )
        body = {"model": "clef-flash", "state": "x", "questions": NOUL}
        response = asyncio.run(loaded.predict(body))
        assert response == {
            "model": "clef-flash",
            "answers": {"revert": {"true": 0.9, "false": 0.1}},
            "usage": {"input_tokens": 3, "output_tokens": 0},
        }
        loaded.vendor.systemone_answer.assert_called_once_with(
            NOUL["revert"], {"true": 0.9, "false": 0.1}
        )

    def test_a_failure_inside_the_model_becomes_an_inference_error(self, loaded):
        loaded.head.side_effect = RuntimeError("HIP out of memory")
        with pytest.raises(InferenceError, match="HIP out of memory"):
            asyncio.run(loaded.predict({"model": "clef-flash", "questions": {}}))

    def test_a_text_prompt_reaches_vllm_unchanged(self, loaded):
        asyncio.run(loaded.predict({"model": "clef-flash", "questions": {}}))
        assert loaded.vllm_engine.prompts == [{"prompt_token_ids": [1, 2, 3]}]

    def test_each_image_reaches_vllm_as_one_pad_token(self, loaded):
        """encode_record expands each image to a run of pad tokens, and vLLM
        expands each pad token again. Without the collapse, every position
        after the first image would move."""
        # 8 and 7 stand for <|vision_start|> and <|vision_end|>, 9 for the pad.
        expanded = (1, 8, 9, 9, 9, 7, 2, 8, 9, 9, 7, 3)
        loaded.vendor.encode_record.return_value = SimpleNamespace(
            input_ids=expanded, questions=()
        )
        loaded.vllm_engine = FakeEngine(prompt_token_ids=expanded)
        images = [object(), object()]
        asyncio.run(
            loaded.predict({"model": "clef-flash", "questions": {}, "images": images})
        )
        (prompt,) = loaded.vllm_engine.prompts
        assert prompt["prompt_token_ids"] == [1, 8, 9, 7, 2, 8, 9, 7, 3]
        assert prompt["multi_modal_data"] == {"image": images}

    def test_a_token_mismatch_with_vllm_is_an_error(self, loaded):
        """The head reads each question and option by position, so a shifted
        prompt would give wrong answers and no error."""
        loaded.vllm_engine = FakeEngine(prompt_token_ids=(1, 2, 3, 4))
        with pytest.raises(InferenceError, match="do not match"):
            asyncio.run(loaded.predict({"model": "clef-flash", "questions": {}}))

    def test_concurrent_requests_reach_vllm_together(self, loaded):
        """vLLM batches the backbone passes of concurrent requests. This is
        the gain over the transformers server, which ran one at a time."""
        loaded.vllm_engine = FakeEngine(delay=0.05)

        async def eight_at_once():
            await asyncio.gather(
                *(
                    loaded.predict({"model": "clef-flash", "questions": {}})
                    for _ in range(8)
                )
            )

        asyncio.run(eight_at_once())
        assert loaded.vllm_engine.most > 1

    def test_the_head_runs_one_request_at_a_time(self, loaded):
        counter = {"now": 0, "most": 0}
        guard = threading.Lock()

        def tracked(*args):
            with guard:
                counter["now"] += 1
                counter["most"] = max(counter["most"], counter["now"])
            time.sleep(0.02)
            with guard:
                counter["now"] -= 1
            return [[]]

        loaded.head.side_effect = tracked

        async def eight_at_once():
            await asyncio.gather(
                *(
                    loaded.predict({"model": "clef-flash", "questions": {}})
                    for _ in range(8)
                )
            )

        asyncio.run(eight_at_once())
        assert counter["most"] == 1

    def test_the_event_loop_keeps_running_during_inference(self, loaded):
        """The reason predict is async. A health or readiness probe arriving
        while the model is busy has to be answered, and under load a blocked
        loop is enough to get a working pod restarted."""
        ticks = []

        def slow(*args):
            time.sleep(0.3)
            return [[]]

        loaded.head.side_effect = slow

        async def probe_while_inferring():
            async def probe():
                for _ in range(5):
                    await asyncio.sleep(0.02)
                    ticks.append(1)

            await asyncio.gather(
                loaded.predict({"model": "clef-flash", "questions": {}}), probe()
            )

        asyncio.run(probe_while_inferring())
        assert len(ticks) == 5, "the event loop was blocked by the forward pass"


class TestPostprocess:
    def test_the_answers_are_returned_unchanged(self, model):
        # There is no text to parse: `answers` already holds a probability for
        # every allowed option, so reshaping would only hide information.
        answers = {
            "revert": {"type": "noul", "noul": 0.9381},
            "severity": {
                "type": "score",
                "score": 1.2813,
                "confidence": 0.5202,
                "legend": {"0": "Minor", "1": "Moderate", "2": "Severe"},
                "probabilities": {"0": 0.0993, "1": 0.5202, "2": 0.3806},
            },
        }
        response = model.postprocess(
            {"model": "clef-flash", "answers": answers, "usage": {"input_tokens": 228}}
        )
        assert response == {
            "model": "clef-flash",
            "answers": answers,
            "usage": {"input_tokens": 228},
        }

    def test_the_model_name_falls_back_to_the_server_name(self, model):
        response = model.postprocess({"answers": {}})
        assert response["model"] == "clef-flash"
        assert response["usage"] is None

    def test_a_response_without_answers_is_an_error(self, model):
        with pytest.raises(InferenceError, match="answers"):
            model.postprocess({"model": "clef-flash"})


class TestLifecycle:
    def test_not_ready_before_load(self, model):
        assert model.ready is False
        assert model.vllm_engine is None
        assert model.head is None

    def test_kserve_runs_start_engine_in_the_server_loop(self, model):
        """KServe calls `start_engine()` in the server event loop only when
        `engine` is True. The vLLM engine needs that loop for the warmup."""
        assert model.engine is True
        assert inspect.iscoroutinefunction(model.start_engine)

    @pytest.mark.parametrize("warmup", [True, False])
    def test_start_engine_runs_the_warmup_only_when_enabled(self, model, warmup):
        calls = []

        async def warm_up():
            calls.append(True)

        model.warmup = warmup
        model._warm_up = warm_up
        asyncio.run(model.start_engine())
        assert calls == ([True] if warmup else [])

    def test_warmup_exercises_both_paths(self, model):
        """The first request compiles the Triton kernels: 4.7s for text on an
        MI210, against 59ms warm. Without this the first real request after a
        deployment pays that."""
        calls = []

        async def predict(body):
            calls.append(body)
            return {"answers": {}}

        model.predict = predict
        asyncio.run(model._warm_up())
        assert len(calls) == 2
        assert "images" not in calls[0]
        assert len(calls[1]["images"]) == 1
        assert calls[1]["images"][0].size[0] == DEFAULT_MAX_IMAGE_WIDTH


class TestConfigHelpers:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("True", True),
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            (" T ", True),
            ("False", False),
            ("false", False),
            ("0", False),
            ("no", False),
            ("", False),
            ("anything else", False),
        ],
    )
    def test_parse_bool(self, value, expected):
        # distutils.util.strtobool would do this, but distutils left the
        # standard library in Python 3.12 and resolves only through a
        # setuptools shim.
        assert parse_bool(value) is expected

    def test_positive_int_accepts_a_positive_value(self):
        assert positive_int("MAX_IMAGES", "4") == 4

    @pytest.mark.parametrize("value", ["0", "-1", "abc", "", "1.5"])
    def test_positive_int_refuses_anything_else(self, value):
        # A zero or negative bound would be accepted silently and then fail
        # per request, such as resizing an image to zero width.
        with pytest.raises(ValueError, match="MAX_IMAGES"):
            positive_int("MAX_IMAGES", value)

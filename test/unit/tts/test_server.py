"""
Unit tests for the KServe server layer (preprocess / predict / postprocess).

Covers the contract boundary the orchestrator depends on:
  - input validation in ``preprocess`` (the InvalidInput cases)
  - PCM-to-base64 encoding and round-trip fidelity in ``postprocess``
  - the async ``predict`` threadpool offload (that it awaits and returns the
    pipeline result, and wraps failures as InferenceError)

Heavy deps are stubbed via the same mechanism as test_inference.py; kserve is
provided by the local stub module.
"""

import asyncio
import base64
import logging
import sys
import types

import numpy as np
import pytest


def _install_stubs() -> None:
    kokoro_module = types.ModuleType("kokoro_onnx")
    kokoro_module.Kokoro = type("Kokoro", (), {"__init__": lambda self, *a, **k: None})
    sys.modules.setdefault("kokoro_onnx", kokoro_module)

    onnxruntime = types.SimpleNamespace(
        InferenceSession=type(
            "InferenceSession", (), {"__init__": lambda self, *a, **k: None}
        ),
        SessionOptions=lambda: types.SimpleNamespace(),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=object()),
    )
    sys.modules.setdefault("onnxruntime", onnxruntime)

    scipy = types.SimpleNamespace(
        signal=types.SimpleNamespace(resample=lambda audio, n: audio)
    )
    sys.modules.setdefault("scipy", scipy)
    sys.modules.setdefault("scipy.signal", sys.modules["scipy"].signal)

    transformers = types.ModuleType("transformers")
    transformers.Wav2Vec2Processor = object
    sys.modules.setdefault("transformers", transformers)


_install_stubs()

from kserve.errors import InferenceError, InvalidInput  # noqa: E402

from src.models.tts.model_server.model import TTSModel  # noqa: E402

SAMPLE_RATE = 24000


def _model():
    """A model instance without a loaded pipeline (preprocess/postprocess need none)."""
    return TTSModel.__new__(TTSModel)


# ── preprocess validation ───────────────────────────────────────────────────


def test_preprocess_rejects_missing_segments():
    m = _model()
    with pytest.raises(InvalidInput):
        m.preprocess({})


def test_preprocess_rejects_empty_segments_list():
    m = _model()
    with pytest.raises(InvalidInput):
        m.preprocess({"segments": []})


def test_preprocess_rejects_segment_without_text():
    m = _model()
    with pytest.raises(InvalidInput):
        m.preprocess({"segments": [{"voice": "af_heart"}]})


def test_preprocess_rejects_empty_text():
    m = _model()
    with pytest.raises(InvalidInput):
        m.preprocess({"segments": [{"text": "   "}]})


def test_preprocess_rejects_non_string_text():
    m = _model()
    with pytest.raises(InvalidInput):
        m.preprocess({"segments": [{"text": 123}]})


def test_preprocess_applies_defaults():
    m = _model()
    out = m.preprocess({"segments": [{"text": "hello"}]})
    assert out["default_voice"] == "af_heart"
    assert out["default_speed"] == 1.0
    assert out["default_lang"] == "en-us"
    assert out["segments"] == [{"text": "hello"}]


def test_preprocess_passes_through_overrides():
    m = _model()
    out = m.preprocess(
        {
            "segments": [{"text": "hi"}],
            "default_voice": "af_bella",
            "default_speed": 1.5,
            "default_lang": "en-gb",
        }
    )
    assert out["default_voice"] == "af_bella"
    assert out["default_speed"] == 1.5
    assert out["default_lang"] == "en-gb"


def test_preprocess_oversized_segment_warns_but_accepts(caplog):
    """Segments over MAX_SEGMENT_CHARS are a warning, not a rejection."""
    m = _model()
    long_text = "word " * 200  # ~1000 chars, over the 800 threshold
    with caplog.at_level(logging.WARNING):
        out = m.preprocess({"segments": [{"text": long_text}]})
    assert out["segments"][0]["text"] == long_text  # accepted, not dropped
    assert "max recommended" in caplog.text  # the warning was actually emitted


# ── postprocess encoding ─────────────────────────────────────────────────────


def test_postprocess_roundtrip_preserves_audio_f32():
    """base64-decoding f32le response reconstructs the exact float32 PCM."""
    m = _model()
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    out = m.postprocess(
        {
            "audio": audio,
            "sample_rate": SAMPLE_RATE,
            "timestamps": [],
            "encoding": "pcm_f32le",
        }
    )

    decoded = np.frombuffer(base64.b64decode(out["audio_b64"]), dtype=np.float32)
    assert np.array_equal(decoded, audio)
    assert out["encoding"] == "pcm_f32le"


def test_postprocess_reports_correct_duration():
    m = _model()
    n = SAMPLE_RATE  # exactly 1 second
    audio = np.zeros(n, dtype=np.float32)
    out = m.postprocess({"audio": audio, "sample_rate": SAMPLE_RATE, "timestamps": []})
    assert out["duration_ms"] == pytest.approx(1000.0)


def test_postprocess_empty_audio():
    m = _model()
    audio = np.array([], dtype=np.float32)
    out = m.postprocess({"audio": audio, "sample_rate": SAMPLE_RATE, "timestamps": []})
    assert out["audio_b64"] == ""
    assert out["duration_ms"] == 0.0


def test_postprocess_passes_timestamps_through():
    m = _model()
    ts = [{"word": "Earth", "start_ms": 0.0, "end_ms": 380.0}]
    out = m.postprocess(
        {
            "audio": np.zeros(10, dtype=np.float32),
            "sample_rate": SAMPLE_RATE,
            "timestamps": ts,
        }
    )
    assert out["timestamps"] == ts
    assert out["sample_rate"] == SAMPLE_RATE


def test_postprocess_int16_is_the_default():
    """When no encoding is given, postprocess defaults to int16 and the
    base64-decoded roundtrip is within 16-bit quantization error."""
    m = _model()
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    out = m.postprocess({"audio": audio, "sample_rate": SAMPLE_RATE, "timestamps": []})

    assert out["encoding"] == "pcm_s16le"
    decoded_i16 = np.frombuffer(base64.b64decode(out["audio_b64"]), dtype=np.int16)
    decoded_f32 = decoded_i16.astype(np.float32) / 32767.0
    # 16-bit quantization error: max absolute error ≤ 1.53e-05 (1 / 2^15 / 2)
    max_err = np.max(np.abs(decoded_f32 - audio))
    assert max_err <= 1.53e-05


def test_postprocess_f32le_roundtrip():
    """Explicit pcm_f32le preserves float32 exactly."""
    m = _model()
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    out = m.postprocess(
        {
            "audio": audio,
            "sample_rate": SAMPLE_RATE,
            "timestamps": [],
            "encoding": "pcm_f32le",
        }
    )
    assert out["encoding"] == "pcm_f32le"
    decoded = np.frombuffer(base64.b64decode(out["audio_b64"]), dtype=np.float32)
    assert np.array_equal(decoded, audio)


def test_postprocess_clips_out_of_range_samples():
    """Samples outside [-1, 1] are clipped before int16 scaling to prevent
    integer-wrapping clicks."""
    m = _model()
    audio = np.array([-1.2, 0.0, 1.5], dtype=np.float32)
    out = m.postprocess({"audio": audio, "sample_rate": SAMPLE_RATE, "timestamps": []})
    decoded = np.frombuffer(base64.b64decode(out["audio_b64"]), dtype=np.int16)
    # -1.2 → -32767, 0.0 → 0, 1.5 → 32767
    assert decoded[0] == -32767
    assert decoded[1] == 0
    assert decoded[2] == 32767


# ── preprocess encoding validation ────────────────────────────────────────────


def test_preprocess_encoding_default():
    m = _model()
    out = m.preprocess({"segments": [{"text": "hi"}]})
    assert out["encoding"] == "pcm_s16le"


def test_preprocess_rejects_invalid_encoding():
    m = _model()
    with pytest.raises(InvalidInput, match="encoding"):
        m.preprocess({"segments": [{"text": "hi"}], "encoding": "pcm_u8"})


def test_preprocess_accepts_valid_encodings():
    m = _model()
    for enc in ("pcm_s16le", "pcm_f32le"):
        out = m.preprocess({"segments": [{"text": "hi"}], "encoding": enc})
        assert out["encoding"] == enc


# ── async predict offload ────────────────────────────────────────────────────


def test_predict_awaits_and_returns_pipeline_result():
    m = _model()

    class FakePipeline:
        def predict(self, segments, default_voice, default_speed, default_lang):
            return {
                "audio": np.zeros(3, dtype=np.float32),
                "sample_rate": SAMPLE_RATE,
                "timestamps": [],
            }

    m.pipeline = FakePipeline()
    inputs = {
        "segments": [{"text": "hi"}],
        "default_voice": "af_heart",
        "default_speed": 1.0,
        "default_lang": "en-us",
        "encoding": "pcm_s16le",
    }
    result = asyncio.run(m.predict(inputs))
    assert result["sample_rate"] == SAMPLE_RATE
    assert len(result["audio"]) == 3


def test_predict_wraps_pipeline_failure_as_inference_error():
    m = _model()

    class ExplodingPipeline:
        def predict(self, *a, **k):
            raise RuntimeError("kokoro choked on bad unicode")

    m.pipeline = ExplodingPipeline()
    inputs = {
        "segments": [{"text": "x"}],
        "default_voice": "af_heart",
        "default_speed": 1.0,
        "default_lang": "en-us",
    }
    with pytest.raises(InferenceError):
        asyncio.run(m.predict(inputs))


def test_predict_runs_off_the_event_loop_thread():
    """
    The offloaded pipeline.predict must execute on a *different* thread than
    the event loop, proving the run_in_executor offload is real (this is what
    keeps liveness probes responsive during synthesis).
    """
    import threading

    m = _model()
    loop_thread_ids = {}

    class ThreadCapturingPipeline:
        def predict(self, *a, **k):
            loop_thread_ids["worker"] = threading.get_ident()
            return {
                "audio": np.zeros(1, dtype=np.float32),
                "sample_rate": SAMPLE_RATE,
                "timestamps": [],
            }

    m.pipeline = ThreadCapturingPipeline()
    inputs = {
        "segments": [{"text": "x"}],
        "default_voice": "af_heart",
        "default_speed": 1.0,
        "default_lang": "en-us",
        "encoding": "pcm_s16le",
    }

    async def runner():
        loop_thread_ids["loop"] = threading.get_ident()
        return await m.predict(inputs)

    asyncio.run(runner())
    assert loop_thread_ids["worker"] != loop_thread_ids["loop"]

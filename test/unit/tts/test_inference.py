"""
Unit tests for the KServe TTS inference pipeline.

These tests exercise the crossfade concatenation and timestamp-accumulation
logic in ``inference.TTSInferencePipeline`` *without* loading any real models.
Following the v0 pattern (see ``tests/test_worker_split_text.py`` and
``tests/test_timestamps.py``), heavy dependencies are stubbed before import.

Kokoro and the Wav2Vec2 aligner are replaced with controllable fakes so we can
assert the exact sample math of the crossfade (how many samples each chunk
contributes to the output) and the timestamp clock advancement that depends on
it.
"""

import sys
import threading
import types

import numpy as np
import pytest


def _install_inference_dependency_stubs() -> None:
    """
    Stub the heavy native deps that ``inference``/``alignment`` import.

    We never load real models in these tests; Kokoro and the Aligner are
    injected per-test as fakes onto the pipeline instance.
    """
    # kokoro_onnx: provide a Kokoro class so `from kokoro_onnx import Kokoro`
    # works at construction. The instance is overwritten with a fake in tests.
    kokoro_module = types.ModuleType("kokoro_onnx")

    class _DummyKokoro:
        def __init__(self, *args, **kwargs):
            pass

        def create(self, *args, **kwargs):  # pragma: no cover - overwritten
            raise NotImplementedError

    kokoro_module.Kokoro = _DummyKokoro
    sys.modules.setdefault("kokoro_onnx", kokoro_module)

    # onnxruntime / scipy / transformers are imported by alignment.py at module
    # import time. inference.py imports alignment.py, so stub them too.
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


_install_inference_dependency_stubs()

from src.models.tts.model_server.inference import (  # noqa: E402
    FADE_LEN,
    MAX_SEGMENT_CHARS,
    TTSInferencePipeline,
)

SAMPLE_RATE = 24000


# ── Fakes ─────────────────────────────────────────────────────────────────


class FakeKokoro:
    """
    Returns audio of a deterministic length keyed by segment text.

    ``lengths`` maps text -> number of samples. Audio is filled with a constant
    value (1.0) so crossfade envelope effects are observable if we inspect
    samples, but the tests mainly assert lengths and timestamps.
    """

    def __init__(self, lengths: dict[str, int]):
        self.lengths = lengths
        self.calls: list[str] = []

    def create(self, text, voice=None, speed=None, lang=None):
        self.calls.append(text)
        n = self.lengths[text]
        return np.ones(n, dtype=np.float32), SAMPLE_RATE


class FakeAligner:
    """
    Returns caller-supplied per-chunk timestamps keyed by text.

    Timestamps are returned relative to the chunk (start at 0), exactly as the
    real aligner does since the pipeline is responsible for offsetting them by
    the accumulated clock.
    """

    def __init__(self, timestamps: dict[str, list[dict]] | None = None):
        self.timestamps = timestamps or {}

    def align(self, audio, sample_rate, text):
        # Return a deep-ish copy so the pipeline's in-place += offsetting does
        # not mutate our canned fixtures across calls.
        return [dict(t) for t in self.timestamps.get(text, [])]


def _make_pipeline(lengths, timestamps=None):
    """Construct a pipeline with fakes injected (no real model load)."""
    pipe = TTSInferencePipeline.__new__(TTSInferencePipeline)
    pipe.kokoro = FakeKokoro(lengths)
    pipe.aligner = FakeAligner(timestamps)
    pipe.sample_rate = SAMPLE_RATE
    pipe._synth_lock = threading.Lock()
    return pipe


def _ms(samples: int) -> float:
    return (samples / SAMPLE_RATE) * 1000


# ── Single-segment behaviour ────────────────────────────────────────────────


def test_single_segment_passes_through_untrimmed():
    """A lone segment is emitted whole, no crossfade trimming applied."""
    n = 1000
    pipe = _make_pipeline({"hello": n})
    result = pipe.predict([{"text": "hello"}])

    assert len(result["audio"]) == n
    assert result["sample_rate"] == SAMPLE_RATE


def test_single_segment_timestamps_not_offset():
    pipe = _make_pipeline(
        {"hello": 1000},
        {"hello": [{"word": "hello", "start_ms": 0.0, "end_ms": 40.0}]},
    )
    result = pipe.predict([{"text": "hello"}])

    assert result["timestamps"] == [{"word": "hello", "start_ms": 0.0, "end_ms": 40.0}]


def test_empty_segment_list_returns_empty_audio():
    pipe = _make_pipeline({})
    result = pipe.predict([])

    assert len(result["audio"]) == 0
    assert result["timestamps"] == []


# ── Crossfade sample accounting (two segments) ──────────────────────────────


def test_two_segments_overlap_by_fade_len():
    """
    Two long chunks crossfade: output length = n1 + n2 - FADE_LEN.

    First chunk contributes n1 - FADE_LEN (tail trimmed, stored as prev_tail);
    last chunk contributes its full n2 (head overlaps onto the stored tail).
    """
    n1, n2 = 1000, 1200
    pipe = _make_pipeline({"a": n1, "b": n2})
    result = pipe.predict([{"text": "a"}, {"text": "b"}])

    assert len(result["audio"]) == n1 + n2 - FADE_LEN


def test_two_segment_second_chunk_timestamp_offset_matches_contribution():
    """
    The second chunk's timestamps must be offset by the first chunk's
    *contributed* samples (n1 - FADE_LEN), not its full length.
    """
    n1, n2 = 1000, 800
    pipe = _make_pipeline(
        {"a": n1, "b": n2},
        {
            "a": [{"word": "a", "start_ms": 0.0, "end_ms": _ms(n1)}],
            "b": [{"word": "b", "start_ms": 0.0, "end_ms": _ms(n2)}],
        },
    )
    result = pipe.predict([{"text": "a"}, {"text": "b"}])

    expected_offset = _ms(n1 - FADE_LEN)
    a_ts, b_ts = result["timestamps"]
    assert a_ts["start_ms"] == 0.0
    assert b_ts["start_ms"] == pytest.approx(expected_offset)
    assert b_ts["end_ms"] == pytest.approx(expected_offset + _ms(n2))


# ── Crossfade sample accounting (three+ segments) ───────────────────────────


def test_three_segments_total_length():
    """Middle chunk also trims its tail: total = n1 + n2 + n3 - 2*FADE_LEN."""
    n1, n2, n3 = 1000, 1100, 900
    pipe = _make_pipeline({"a": n1, "b": n2, "c": n3})
    result = pipe.predict([{"text": "a"}, {"text": "b"}, {"text": "c"}])

    assert len(result["audio"]) == n1 + n2 + n3 - 2 * FADE_LEN


def test_three_segment_clock_advances_by_contribution_not_raw_length():
    """
    Each boundary advances the clock by (chunk_len - FADE_LEN) for non-final
    chunks. Verify the third chunk's offset is the sum of the first two
    contributions.
    """
    n1, n2, n3 = 1000, 1100, 900
    pipe = _make_pipeline(
        {"a": n1, "b": n2, "c": n3},
        {
            "a": [{"word": "a", "start_ms": 0.0, "end_ms": 10.0}],
            "b": [{"word": "b", "start_ms": 0.0, "end_ms": 10.0}],
            "c": [{"word": "c", "start_ms": 0.0, "end_ms": 10.0}],
        },
    )
    result = pipe.predict([{"text": "a"}, {"text": "b"}, {"text": "c"}])

    a_ts, b_ts, c_ts = result["timestamps"]
    assert a_ts["start_ms"] == 0.0
    assert b_ts["start_ms"] == pytest.approx(_ms(n1 - FADE_LEN))
    assert c_ts["start_ms"] == pytest.approx(_ms((n1 - FADE_LEN) + (n2 - FADE_LEN)))


def test_timestamps_never_exceed_total_duration():
    """
    No accumulated end_ms should run past the actual audio duration
    as this is the regression guard for the original drift bug.
    """
    n1, n2, n3 = 1000, 1100, 900
    pipe = _make_pipeline(
        {"a": n1, "b": n2, "c": n3},
        {
            "a": [{"word": "a", "start_ms": 0.0, "end_ms": _ms(n1)}],
            "b": [{"word": "b", "start_ms": 0.0, "end_ms": _ms(n2)}],
            "c": [{"word": "c", "start_ms": 0.0, "end_ms": _ms(n3)}],
        },
    )
    result = pipe.predict([{"text": "a"}, {"text": "b"}, {"text": "c"}])

    total_duration_ms = _ms(len(result["audio"]))
    last_end = result["timestamps"][-1]["end_ms"]
    # The last chunk is emitted whole, so its end aligns with total duration.
    assert last_end == pytest.approx(total_duration_ms)
    for t in result["timestamps"]:
        assert t["end_ms"] <= total_duration_ms + 1e-6


# ── Short-chunk handling ─────────────────────────────────────────────────────


def test_short_chunk_emitted_whole_and_breaks_chain():
    """
    A chunk shorter than FADE_LEN is emitted untrimmed and resets prev_tail.

    Sequence: long, short, long. The short chunk contributes its full length;
    the chain is broken so the third chunk does NOT fade its head in (prev_tail
    is None), and is emitted whole as the final chunk.
    """
    n1, short, n3 = 1000, FADE_LEN - 1, 1000
    pipe = _make_pipeline({"a": n1, "s": short, "c": n3})
    result = pipe.predict([{"text": "a"}, {"text": "s"}, {"text": "c"}])

    # a: n1 - FADE_LEN (tail trimmed) ; s: short (whole) ; c: n3 (whole, final)
    expected = (n1 - FADE_LEN) + short + n3
    assert len(result["audio"]) == expected


def test_short_first_chunk_does_not_crash():
    """A short *leading* chunk must not break the slicing logic."""
    short, n2 = FADE_LEN - 5, 1000
    pipe = _make_pipeline({"s": short, "b": n2})
    result = pipe.predict([{"text": "s"}, {"text": "b"}])

    # short emitted whole (chain broken) ; second chunk has no prev_tail to
    # fade against, and as the final chunk is emitted whole.
    assert len(result["audio"]) == short + n2


def test_short_chunk_clock_uses_full_short_length():
    """
    When a short chunk is emitted whole, the clock advances by its full
    length (not length - FADE_LEN).
    """
    n1, short = 1000, FADE_LEN - 1
    pipe = _make_pipeline(
        {"a": n1, "s": short},
        {
            "a": [{"word": "a", "start_ms": 0.0, "end_ms": 10.0}],
            "s": [{"word": "s", "start_ms": 0.0, "end_ms": 5.0}],
        },
    )
    result = pipe.predict([{"text": "a"}, {"text": "s"}])

    # First chunk is non-final long -> contributes n1 - FADE_LEN.
    _, s_ts = result["timestamps"]
    assert s_ts["start_ms"] == pytest.approx(_ms(n1 - FADE_LEN))


# ── Defensive copy ───────────────────────────────────────────────────────────


def test_kokoro_output_is_not_mutated_in_place():
    """
    The pipeline must copy Kokoro's output before applying in-place fades.

    We hand back a read-only array from create(); if the pipeline failed to
    copy, the in-place *= would raise. Success means the copy is working.
    """
    n1, n2 = 1000, 1000

    class ReadOnlyKokoro(FakeKokoro):
        def create(self, text, voice=None, speed=None, lang=None):
            audio, sr = super().create(text, voice, speed, lang)
            audio.setflags(write=False)  # simulate a read-only / cached buffer
            return audio, sr

    pipe = TTSInferencePipeline.__new__(TTSInferencePipeline)
    pipe.kokoro = ReadOnlyKokoro({"a": n1, "b": n2})
    pipe.aligner = FakeAligner()
    pipe.sample_rate = SAMPLE_RATE
    pipe._synth_lock = threading.Lock()

    # Should not raise despite read-only source arrays.
    result = pipe.predict([{"text": "a"}, {"text": "b"}])
    assert len(result["audio"]) == n1 + n2 - FADE_LEN


# ── Per-segment overrides ────────────────────────────────────────────────────


def test_segment_level_voice_passed_through():
    captured = {}

    class CapturingKokoro(FakeKokoro):
        def create(self, text, voice=None, speed=None, lang=None):
            captured[text] = {"voice": voice, "speed": speed, "lang": lang}
            return super().create(text, voice, speed, lang)

    pipe = TTSInferencePipeline.__new__(TTSInferencePipeline)
    pipe.kokoro = CapturingKokoro({"x": 500, "y": 500})
    pipe.aligner = FakeAligner()
    pipe.sample_rate = SAMPLE_RATE
    pipe._synth_lock = threading.Lock()

    pipe.predict(
        [{"text": "x", "voice": "af_bella"}, {"text": "y"}],
        default_voice="af_heart",
        default_speed=1.25,
        default_lang="en-gb",
    )

    # Segment-level voice overrides default; unspecified fields fall back.
    assert captured["x"]["voice"] == "af_bella"
    assert captured["x"]["speed"] == 1.25
    assert captured["x"]["lang"] == "en-gb"
    # Second segment uses all defaults.
    assert captured["y"]["voice"] == "af_heart"


# ── Overlap correctness ──────────────────────────────────────────────────────


def test_overlap_preserves_timestamp_offsets():
    """Overlapped alignment must reproduce serial-accumulation offsets
    exactly: each chunk's timestamps are offset by the audio contributed
    before it (post-crossfade), regardless of when alignment completes."""
    L = 1000
    n_segs = 4
    segs = [{"text": f"seg{i}"} for i in range(n_segs)]
    lengths = {f"seg{i}": L for i in range(n_segs)}
    ts = {
        f"seg{i}": [{"word": f"w{i}", "start_ms": 0.0, "end_ms": 10.0}]
        for i in range(n_segs)
    }

    pipe = _make_pipeline(lengths, ts)
    result = pipe.predict(segs)

    # chunk 0..n-2 each contribute L - FADE_LEN; last chunk contributes L
    contribution = _ms(L - FADE_LEN)
    for i, t in enumerate(result["timestamps"]):
        expected_offset = i * contribution
        assert t["start_ms"] == pytest.approx(expected_offset)
        assert t["end_ms"] == pytest.approx(expected_offset + 10.0)

    # Total audio length matches serial expectation: n*L - (n-1)*FADE_LEN
    assert len(result["audio"]) == n_segs * L - (n_segs - 1) * FADE_LEN


# ── Constants sanity ─────────────────────────────────────────────────────────


def test_max_segment_chars_constant_exposed():
    """model.py imports this constant; guard against accidental removal."""
    assert isinstance(MAX_SEGMENT_CHARS, int)
    assert MAX_SEGMENT_CHARS > 0

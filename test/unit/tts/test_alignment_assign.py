"""Tests for CTC frame-to-word assignment (T436758).

No ONNX model or audio needed: the unit under test takes CTC segments and
a character alignment, both of which are cheap to construct. Scenario
helper ``_case`` builds a synthetic recognition where chosen words are
missed entirely, which is what the recogniser does on spelled-out numbers
beside titles and on quoted phrases.
"""

import sys
import types

import pytest

# The module imports onnxruntime/transformers/scipy at import time; stub
# them so these tests stay dependency-free and fast.
for name, stub in (
    (
        "onnxruntime",
        types.SimpleNamespace(
            InferenceSession=object,
            SessionOptions=lambda: types.SimpleNamespace(),
            GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=0),
        ),
    ),
    ("transformers", types.SimpleNamespace(Wav2Vec2Processor=object)),
    (
        "scipy",
        types.SimpleNamespace(signal=types.SimpleNamespace(resample=lambda a, n: a)),
    ),
):
    sys.modules.setdefault(name, stub)
sys.modules.setdefault("scipy.signal", sys.modules["scipy"].signal)

from src.models.tts.model_server import alignment as al  # noqa: E402

FRAMES_PER_CHAR = 5


def _case(text: str, missed: tuple[str, ...] = ()) -> list[dict]:
    """Align ``text`` where the words in ``missed`` produced no CTC output."""
    words = text.split()
    clean = ["".join(c for c in w if c.isalnum()).upper() for w in words]
    reference = "".join(clean)
    recognised_chars: list[str] = []
    for clean_word in clean:
        if clean_word in {m.upper() for m in missed}:
            continue
        recognised_chars.extend(clean_word)
    segments = [
        (0, i * FRAMES_PER_CHAR, i * FRAMES_PER_CHAR + FRAMES_PER_CHAR - 1)
        for i in range(len(recognised_chars))
    ]
    alignment = al._character_align("".join(recognised_chars), reference)
    return al._assign_frames_to_words(
        segments, alignment, words, clean, len(segments) * FRAMES_PER_CHAR
    )


def _assert_contract(result: list[dict], text: str) -> None:
    """Invariants the VTT writer and cue-stepping players depend on."""
    words = text.split()
    assert [t["word"] for t in result] == words, "one cue per word, in order"
    for t in result:
        assert t["end_ms"] > t["start_ms"], f"degenerate cue: {t}"
        assert t["start_ms"] >= 0
    starts = [t["start_ms"] for t in result]
    assert starts == sorted(starts), "cue starts must not go backwards"
    for earlier, later in zip(result, result[1:]):
        assert later["start_ms"] >= earlier["end_ms"], (
            f"cues overlap: {earlier} then {later}"
        )


EARTH = (
    "The Earth is the third planet from the Sun and the only place known to host life"
)


# ── The bug this change fixes ────────────────────────────────────────────


def test_missed_run_midway_does_not_shift_later_words():
    """Words after a missed run keep their own audio positions.

    Pre-fix, the lockstep walk handed later words frames belonging to
    other audio and collapsed the tail onto one frame: captions ran up to
    a second behind and carried zero-length cues.
    """
    text = "and Highway sixty one Revisited. Critics often rank it highly"
    result = _case(text, missed=("sixty", "one", "Revisited"))
    _assert_contract(result, text)
    by_word = {t["word"]: t for t in result}
    # "Critics" onwards are matched, so they must sit after the missed run
    # and before the end of the audio, not bunched at the final frame.
    assert by_word["Critics"]["start_ms"] > by_word["Highway"]["end_ms"]
    assert (
        by_word["highly"]["end_ms"]
        <= len("".join(c for c in text if c.isalnum()))
        * FRAMES_PER_CHAR
        * al.FRAME_DURATION_MS
    )
    # The later words keep distinct, non-trivial durations.
    assert by_word["Critics"]["end_ms"] - by_word["Critics"]["start_ms"] > 100


def test_missed_run_produces_no_zero_length_cues():
    for missed in (("The",), ("third", "planet"), ("host", "life")):
        result = _case(EARTH, missed=missed)
        _assert_contract(result, EARTH)


def test_non_alphanumeric_token_gets_a_real_duration():
    """Em-dashes and bare quotes clean to empty strings; they still need a
    non-degenerate cue or players may skip the surrounding words."""
    text = "Dylan recorded it in Nashville — a first for him"
    result = _case(text)
    _assert_contract(result, text)
    dash = next(t for t in result if t["word"] == "—")
    assert dash["end_ms"] - dash["start_ms"] >= al.FRAME_DURATION_MS


# ── No change when alignment is complete ─────────────────────────────────


def test_full_match_timings_come_straight_from_the_frames():
    result = _case(EARTH)
    _assert_contract(result, EARTH)
    clean = ["".join(c for c in w if c.isalnum()).upper() for w in EARTH.split()]
    cursor = 0
    for word_clean, t in zip(clean, result):
        expected_start = cursor * FRAMES_PER_CHAR * al.FRAME_DURATION_MS
        cursor += len(word_clean)
        expected_end = (cursor * FRAMES_PER_CHAR - 1) * al.FRAME_DURATION_MS
        assert t["start_ms"] == pytest.approx(expected_start, abs=1)
        assert t["end_ms"] == pytest.approx(expected_end, abs=al.FRAME_DURATION_MS)


# ── Degenerate inputs ────────────────────────────────────────────────────


def test_no_words_returns_empty():
    assert al._assign_frames_to_words([], [], [], [], 0) == []


def test_single_word():
    result = _case("Earth")
    _assert_contract(result, "Earth")


def test_every_word_missed_still_returns_one_cue_per_word():
    """Nothing recognised: all cues are interpolated, contract still holds."""
    words = EARTH.split()
    clean = ["".join(c for c in w if c.isalnum()).upper() for w in words]
    result = al._assign_frames_to_words([], [], words, clean, 500)
    _assert_contract(result, EARTH)
    assert result[-1]["end_ms"] <= 500 * al.FRAME_DURATION_MS


def test_no_room_to_interpolate_still_yields_separate_cues():
    """Window with zero width (the last anchor already reaches the end of
    the audio): words must still get distinct, non-overlapping cues."""
    words = ["Hi", "There", "Friend"]
    clean = ["HI", "THERE", "FRIEND"]
    segments = [(1, 0, 5), (2, 5, 10)]
    result = al._assign_frames_to_words(segments, [0, 1], words, clean, 10)
    _assert_contract(result, "Hi There Friend")


def test_alignment_index_past_reference_is_ignored():
    """A defensive case: alignment entries must never index outside the
    reference text, but if they do we skip rather than raise."""
    words = ["Earth", "is", "round"]
    clean = ["EARTH", "IS", "ROUND"]
    segments = [(0, 0, 4), (0, 5, 9)]
    result = al._assign_frames_to_words(segments, [999, None], words, clean, 20)
    _assert_contract(result, "Earth is round")

"""
Unit tests for the pure-function parts of the forced-alignment module.

These cover the logic that does NOT require the ONNX model:
  - ``_character_align`` (Needleman-Wunsch char alignment, monotonicity)
  - ``_assign_frames_to_words`` (segment -> word timestamp mapping, flush paths)
  - ``_proportional_timestamps`` (the fallback used when CTC quality is low)

The Aligner class itself (ONNX inference) is integration-tested separately
against real models per the README; here we guard the algorithmic core.
"""

import sys
import types

import pytest


def _install_stubs() -> None:
    scipy = types.SimpleNamespace(
        signal=types.SimpleNamespace(resample=lambda audio, n: audio)
    )
    sys.modules.setdefault("scipy", scipy)
    sys.modules.setdefault("scipy.signal", sys.modules["scipy"].signal)

    transformers = types.ModuleType("transformers")
    transformers.Wav2Vec2Processor = object
    sys.modules.setdefault("transformers", transformers)


_install_stubs()

from src.models.tts.model_server.alignment import (  # noqa: E402
    FRAME_DURATION_MS,
    _assign_frames_to_words,
    _character_align,
    _proportional_timestamps,
)

# ── _character_align ─────────────────────────────────────────────────────────


def test_character_align_identical_strings_maps_one_to_one():
    alignment = _character_align("ABC", "ABC")
    assert alignment == [0, 1, 2]


def test_character_align_empty_returns_empty():
    assert _character_align("", "ABC") == []
    assert _character_align("ABC", "") == []


def test_character_align_insertion_marked_none():
    """An extra recognised char (insertion) maps to None and the rest still
    aligns to reference indices."""
    alignment = _character_align("AXBC", "ABC")
    # Exactly one None (the inserted X); reference indices appear in order.
    assert alignment.count(None) == 1
    ref_indices = [a for a in alignment if a is not None]
    assert ref_indices == sorted(ref_indices)  # monotonic non-decreasing


def test_character_align_reference_index_is_monotonic():
    alignment = _character_align("HELLOWORLD", "HELOWORLD")
    ref_indices = [a for a in alignment if a is not None]
    assert ref_indices == sorted(ref_indices)


# ── _assign_frames_to_words ──────────────────────────────────────────────────


def _seg(char_id, start, end):
    return (char_id, start, end)


def test_assign_frames_single_word():
    """Two segments forming the 2-char word 'HI' -> one timestamp spanning
    from the first segment's start frame to the second's end frame."""
    segments = [_seg(1, 0, 5), _seg(2, 5, 10)]
    alignment = [0, 1]  # both chars align to reference
    words = ["Hi"]
    clean_words = ["HI"]

    result = _assign_frames_to_words(
        segments, alignment, words, clean_words, total_frames=10
    )

    assert result == [{"word": "Hi", "start_ms": 0, "end_ms": 10 * FRAME_DURATION_MS}]


def test_assign_frames_two_words():
    """'HI' + 'YO', 4 segments, each 5 frames wide."""
    segments = [_seg(1, 0, 5), _seg(2, 5, 10), _seg(3, 10, 15), _seg(4, 15, 20)]
    alignment = [0, 1, 2, 3]
    words = ["Hi", "Yo"]
    clean_words = ["HI", "YO"]

    result = _assign_frames_to_words(
        segments, alignment, words, clean_words, total_frames=20
    )

    assert result[0] == {"word": "Hi", "start_ms": 0, "end_ms": 10 * FRAME_DURATION_MS}
    assert result[1] == {
        "word": "Yo",
        "start_ms": 10 * FRAME_DURATION_MS,
        "end_ms": 20 * FRAME_DURATION_MS,
    }


def test_assign_frames_flushes_unassigned_words_at_end():
    """
    If alignment runs out before all words are placed, remaining words are
    appended pinned to the final timestamp, no word is dropped.
    """
    segments = [_seg(1, 0, 5), _seg(2, 5, 10)]
    alignment = [0, 1]
    words = ["Hi", "There", "Friend"]  # more words than segments cover
    clean_words = ["HI", "THERE", "FRIEND"]

    result = _assign_frames_to_words(
        segments, alignment, words, clean_words, total_frames=10
    )

    # Every input word must appear in the output.
    assert [r["word"] for r in result] == ["Hi", "There", "Friend"]


def test_assign_frames_preserves_original_word_casing_and_punctuation():
    """Output words use the original tokens (with punctuation), not the
    cleaned/uppercased reference forms."""
    segments = [_seg(1, 0, 5), _seg(2, 5, 10), _seg(3, 10, 15)]
    alignment = [0, 1, 2]
    words = ["Earth,"]
    clean_words = ["EARTH"]  # 5 chars

    # Only 3 segments for a 5-char word -> word flushed via the partial path.
    result = _assign_frames_to_words(
        segments, alignment, words, clean_words, total_frames=15
    )
    assert result[0]["word"] == "Earth,"  # original token preserved


def test_assign_frames_non_alnum_word_gets_zero_duration_entry():
    """
    A word with no alphanumeric chars (e.g an em-dash) receives a zero-duration
    timestamp so it stays index-aligned with the recognised words.
    """
    words = ["Hi", "—", "there"]
    clean_words = ["HI", "", "THERE"]

    # 7 segments for the 7 chars of "HITHERE", 5 frames each.
    segments = [
        _seg(1, 0, 5),  # H
        _seg(2, 5, 10),  # I
        _seg(3, 10, 15),  # T
        _seg(4, 15, 20),  # H
        _seg(5, 20, 25),  # E
        _seg(6, 25, 30),  # R
        _seg(7, 30, 35),  # E
    ]
    alignment = [0, 1, 2, 3, 4, 5, 6]

    result = _assign_frames_to_words(
        segments, alignment, words, clean_words, total_frames=35
    )

    assert len(result) == 3

    ms_per_frame = FRAME_DURATION_MS

    # "Hi" spans frames 0–10.
    assert result[0] == {
        "word": "Hi",
        "start_ms": 0,
        "end_ms": 10 * ms_per_frame,
    }

    # "—" is a zero-duration stub pinned to the end of "Hi".
    assert result[1] == {
        "word": "—",
        "start_ms": 10 * ms_per_frame,
        "end_ms": 10 * ms_per_frame,
    }

    # "there" spans frames 10–35.
    assert result[2] == {
        "word": "there",
        "start_ms": 10 * ms_per_frame,
        "end_ms": 35 * ms_per_frame,
    }


# ── _proportional_timestamps ─────────────────────────────────────────────────


def test_proportional_distributes_by_character_length():
    result = _proportional_timestamps("a bbb", 800)
    assert result == [
        {"word": "a", "start_ms": 0.0, "end_ms": 200.0},
        {"word": "bbb", "start_ms": 200.0, "end_ms": 800.0},
    ]


def test_proportional_handles_blank_input():
    assert _proportional_timestamps("", 1000) == []
    assert _proportional_timestamps("   ", 1000) == []


def test_proportional_is_contiguous_and_covers_full_duration():
    """Successive words must be gap-free and the last word must end exactly at
    the total duration."""
    total = 1234.5
    result = _proportional_timestamps("alpha beta gamma", total)
    for a, b in zip(result, result[1:]):
        assert a["end_ms"] == pytest.approx(b["start_ms"])
    assert result[-1]["end_ms"] == pytest.approx(total)
    assert result[0]["start_ms"] == 0.0

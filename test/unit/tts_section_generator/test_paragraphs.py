"""Tests for paragraph boundary recovery and entry assembly.

The interesting cases are the three recovery tiers (identity, alignment,
proportional) and the dropped-word handling: the isvc aligner drops words
whose cue has zero duration, so the timestamp list is a subsequence of the
section's words, not an equal-length list. paragraph_entries must walk the
two in lockstep or every paragraph after a dropped word is mis-timed.
"""

import logging

import pytest

from src.models.tts_section_generator.tts_generator.paragraphs import (
    ALIGNMENT,
    IDENTITY,
    PROPORTIONAL,
    paragraph_entries,
    paragraph_word_ranges,
)


def _identity(text):
    return text


def _words(count):
    return [f"w{i}" for i in range(count)]


def _stamps(count, step=100.0):
    return [
        {"word": f"w{i}", "start_ms": i * step, "end_ms": (i + 1) * step}
        for i in range(count)
    ]


# ── paragraph_word_ranges ───────────────────────────────────────────────────


def test_identity_tier_returns_cumulative_ranges():
    blocks = ["w0 w1", "w2 w3"]
    cleaned = "w0 w1 w2 w3"

    ranges, method = paragraph_word_ranges(cleaned, blocks, _identity)

    assert ranges == [(0, 2), (2, 4)]
    assert method == IDENTITY


def test_alignment_tier_maps_boundaries_when_streams_diverge():
    blocks = ["w0 w1", "w2 w3"]
    cleaned = "w0 w1 w2 w3"

    def normalize(text):
        # The second block normalizes to an extra trailing token: a word
        # the normalizer would add across a join that the section text
        # does not contain. Within budget, so boundaries are aligned.
        return text + " w4" if text == "w2 w3" else text

    ranges, method = paragraph_word_ranges(cleaned, blocks, normalize)

    assert ranges == [(0, 2), (2, 4)]
    assert method == ALIGNMENT


def test_proportional_tier_when_streams_genuinely_diverge():
    blocks = ["alpha beta", "gamma delta"]
    cleaned = "one two three four five six seven eight"

    def normalize(text):
        return "x"

    ranges, method = paragraph_word_ranges(cleaned, blocks, normalize)

    assert method == PROPORTIONAL
    assert ranges == [(0, 4), (4, 8)]


def test_empty_blocks_or_words_return_empty_identity():
    assert paragraph_word_ranges("", [], _identity) == ([], IDENTITY)
    assert paragraph_word_ranges("some words", [], _identity) == ([], IDENTITY)
    assert paragraph_word_ranges("", ["block"], _identity) == ([], IDENTITY)


# ── paragraph_entries ───────────────────────────────────────────────────────


def test_entries_take_their_times_from_the_word_timestamps():
    words = _words(4)
    ts = _stamps(4)
    ranges = [(0, 2), (2, 4)]
    blocks = ["w0 w1", "w2 w3"]

    entries = paragraph_entries(ranges, blocks, words, ts, 400.0, IDENTITY)

    assert [e["start_ms"] for e in entries] == [0.0, 200.0]
    assert [e["end_ms"] for e in entries] == [200.0, 400.0]
    assert [e["excerpt"] for e in entries] == ["w0 w1", "w2 w3"]
    assert [e["word_count"] for e in entries] == [2, 2]
    assert [e["index"] for e in entries] == [0, 1]
    assert all(e["method"] == IDENTITY for e in entries)


def test_a_dropped_word_does_not_shift_later_paragraphs():
    """Regression: the aligner drops a zero-duration word, so timestamps is
    a subsequence of words, not an equal-length list. Indexing timestamps by
    word position would read w3's cue as w2's and shift every paragraph
    after the drop; the lockstep walk reads each paragraph's own words."""
    words = ["w0", "w1", "w2", "w3", "w4", "w5"]
    ts = [
        {"word": "w0", "start_ms": 0.0, "end_ms": 100.0},
        {"word": "w1", "start_ms": 100.0, "end_ms": 200.0},
        # "w2" dropped by the aligner (zero-width cue)
        {"word": "w3", "start_ms": 300.0, "end_ms": 400.0},
        {"word": "w4", "start_ms": 400.0, "end_ms": 500.0},
        {"word": "w5", "start_ms": 500.0, "end_ms": 600.0},
    ]
    ranges = [(0, 3), (3, 6)]
    blocks = ["w0 w1 w2", "w3 w4 w5"]

    entries = paragraph_entries(ranges, blocks, words, ts, 600.0, ALIGNMENT)

    assert entries[0]["end_ms"] == 200.0  # w1's cue, not w3's
    assert entries[1]["start_ms"] == 300.0  # w3's cue, not w4's
    assert entries[1]["end_ms"] == 600.0


def test_block_with_all_dropped_words_gets_zero_length_at_neighbour():
    words = _words(4)
    ts = _stamps(1)  # only w0 kept
    ranges = [(0, 2), (2, 4)]
    blocks = ["w0 w1", "w2 w3"]

    entries = paragraph_entries(ranges, blocks, words, ts, 400.0, ALIGNMENT)

    assert entries[0]["end_ms"] == 100.0
    assert entries[1]["start_ms"] == 100.0
    assert entries[1]["end_ms"] == 100.0


def test_proportional_entries_spread_duration_by_char_share():
    blocks = ["aa", "bbbb"]  # char shares 2/6 and 4/6
    words = _words(4)
    ranges = [(0, 2), (2, 4)]

    entries = paragraph_entries(ranges, blocks, words, [], 600.0, PROPORTIONAL)

    assert entries[0]["start_ms"] == 0.0
    assert entries[0]["end_ms"] == pytest.approx(200.0)
    assert entries[1]["start_ms"] == pytest.approx(200.0)
    assert entries[1]["end_ms"] == 600.0


def test_proportional_entries_keep_real_times_when_timestamps_present():
    """Proportional word RANGES are char-share estimates, but the timestamps
    are real (paragraph artifacts force alignment), so entries must keep
    their timestamp-derived times. Spreading by char share would make them
    disagree with captions_vtt in the same request."""
    words = _words(4)
    ts = _stamps(4)
    ranges = [(0, 2), (2, 4)]
    blocks = ["aa", "bbbb"]

    entries = paragraph_entries(ranges, blocks, words, ts, 600.0, PROPORTIONAL)

    assert [e["start_ms"] for e in entries] == [0.0, 200.0]
    assert [e["end_ms"] for e in entries] == [200.0, 400.0]
    assert [e["method"] for e in entries] == [PROPORTIONAL, PROPORTIONAL]


def test_desynced_tokenization_warns_and_collapses_to_zero(caplog):
    """A different-word desync (not a dropped word) makes the lockstep walk
    return all None, collapsing every paragraph to zero length at 0.0. The
    warning is the only signal that the boundaries are unusable rather than
    genuinely empty."""
    words = ["fifty", "six", "point", "seven"]
    ts = [
        {"word": "fifty-six", "start_ms": 0.0, "end_ms": 100.0},
        {"word": "point", "start_ms": 100.0, "end_ms": 200.0},
        {"word": "seven", "start_ms": 200.0, "end_ms": 300.0},
    ]
    ranges = [(0, 2), (2, 4)]
    blocks = ["fifty six", "point seven"]

    with caplog.at_level(logging.WARNING):
        entries = paragraph_entries(ranges, blocks, words, ts, 300.0, ALIGNMENT)

    assert "may be unusable" in caplog.text
    assert all(e["start_ms"] == 0.0 and e["end_ms"] == 0.0 for e in entries)

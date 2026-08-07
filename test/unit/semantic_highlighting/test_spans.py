"""Tests for the highlight-span rules in spans.py.

These encode the span constraints OpenSearch enforces on a semantic-highlighter
response, including the UTF-16 offset caveat. A span that breaks them fails the
whole `_search` rather than just one hit, so the rules are pinned here instead of
being left to the model's good behaviour.

spans.py imports nothing beyond the standard library, so no stubbing is needed.
"""

from src.models.semantic_highlighting.model_server.spans import (
    harden,
    spans_from_result,
    to_utf16,
    utf16_len,
)

# --- to_utf16 / utf16_len ----------------------------------------------------


def test_to_utf16_bmp_matches_code_points():
    text = "Gustave Eiffel"
    assert to_utf16(text, 0) == 0
    assert to_utf16(text, 7) == 7
    assert utf16_len(text) == len(text)


def test_to_utf16_drifts_after_non_bmp():
    # "😀" is one Python code point but two UTF-16 code units.
    text = "😀 Gustave"
    assert to_utf16(text, 2) == 3  # code point 2 -> utf-16 unit 3
    assert to_utf16(text, 9) == 10  # end of "Gustave"
    assert utf16_len(text) == 10


# --- harden ------------------------------------------------------------------


def test_harden_clamps_to_bounds():
    assert harden([{"start": -5, "end": 1000}], 10) == [{"start": 0, "end": 10}]


def test_harden_drops_inverted_and_zero_length():
    assert harden([{"start": 5, "end": 5}, {"start": 8, "end": 3}], 10) == []


def test_harden_drops_overlap_and_duplicate_start():
    assert harden([{"start": 0, "end": 5}, {"start": 3, "end": 9}], 10) == [
        {"start": 0, "end": 5}
    ]
    assert harden([{"start": 2, "end": 10}, {"start": 2, "end": 4}], 20) == [
        {"start": 2, "end": 10}
    ]


def test_harden_sorts_and_allows_touching():
    assert harden([{"start": 8, "end": 10}, {"start": 0, "end": 3}], 20) == [
        {"start": 0, "end": 3},
        {"start": 8, "end": 10},
    ]
    assert harden([{"start": 0, "end": 5}, {"start": 5, "end": 9}], 20) == [
        {"start": 0, "end": 5},
        {"start": 5, "end": 9},
    ]


# --- spans_from_result -------------------------------------------------------


def test_spans_from_result_converts_offsets_to_utf16():
    # emoji before the answer -> code-point offsets must shift to UTF-16 units.
    result = {"answer": "Gustave", "score": 0.9, "start": 2, "end": 9}
    assert spans_from_result(result, "😀 Gustave Eiffel") == [{"start": 3, "end": 10}]


def test_spans_from_result_empty_on_abstain():
    result = {"answer": "", "score": 0.99, "start": 0, "end": 0}
    assert spans_from_result(result, "no relevant content here") == []


def test_spans_from_result_honours_min_score():
    result = {"answer": "Gustave", "score": 0.4, "start": 0, "end": 7}
    assert spans_from_result(result, "Gustave Eiffel", min_score=0.5) == []
    assert spans_from_result(result, "Gustave Eiffel", min_score=0.3) == [
        {"start": 0, "end": 7}
    ]

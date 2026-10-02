"""Bare metric unit abbreviations NeMo leaves unexpanded (T426756).

NeMo's grammar expands km, cm, kg, mi, lb, °C and many others, but leaves
a bare "m" and "mm" alone: "m" is ambiguous (metres, minutes, million)
and NeMo declines to guess. The leftover letter reaches espeak, which
reads it as a letter name: "56.7 m long" was spoken as "fifty six point
seven M long", reported against the Beta app by a volunteer.

The rules live in _COMPOUND_UNIT_SUBS, which runs BEFORE NeMo, and not in
_UNIT_SUBS, which runs only on the NeMo-unavailable fallback path that
production never takes. That placement is why the original T426756 rule
never took effect.
"""

import pytest
from tts_generator.text import (
    _COMPOUND_UNIT_SUBS,
    _norm_compound_units,
    clean_spoken_text,
    init_nemo,
    nemo_available,
)

# ── The reported defect ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The bridge is 56.7 m long.", "56.7 meters"),
        ("The tower is 300 m tall.", "300 meters"),
        ("a 56.7 m span", "56.7 meters"),
        ("It is 3 mm wide.", "3 millimeters"),
        ("It is 0.5 mm thick.", "0.5 millimeters"),
    ],
)
def test_bare_metre_and_millimetre_are_expanded(text, expected):
    assert expected in _norm_compound_units(text)


# ── Ordering: the compound units must consume their tokens first ─────────


@pytest.mark.parametrize(
    "text,expected,not_expected",
    [
        ("It travels at 56.7 m/s.", "56.7 meters per second", "meters/s"),
        ("Gravity is 9.8 m/s² here.", "9.8 meters per second squared", "meters/s"),
        ("The area is 56.7 m² total.", "56.7 square meters", "meters²"),
        ("It is 12 km² wide.", "12 square kilometers", "square meters"),
    ],
)
def test_compound_units_win_over_the_bare_metre(text, expected, not_expected):
    """The bare-metre rule is LAST in the table for this reason: it matches
    "56.7 m" inside "56.7 m/s", so m/s and m/s² must substitute first."""
    out = _norm_compound_units(text)
    assert expected in out
    assert not_expected not in out


def test_the_bare_metre_rule_is_last_in_the_table():
    """Pins the ordering structurally, not just by its effects: inserting a
    rule after it would silently reintroduce "meters/s"."""
    assert _COMPOUND_UNIT_SUBS[-1][1] == r"\1 meters"
    assert _COMPOUND_UNIT_SUBS[-2][1] == r"\1 millimeters"


@pytest.mark.parametrize("text", ["It is 56.7 km away.", "It is 5 cm thick."])
def test_prefixed_metres_are_untouched_by_the_bare_rule(text):
    """km and cm cannot match: the digit must sit immediately before the
    "m", and the k/c intervenes. NeMo expands these itself."""
    assert _norm_compound_units(text) == text


# ── What must NOT change ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "the m in theorem is silent",
        "Mr. M arrived late",
        "section m of the act",
    ],
)
def test_a_bare_m_with_no_number_before_it_is_untouched(text):
    assert _norm_compound_units(text) == text


def test_financial_m_reads_as_metres_and_that_is_accepted():
    """Documented trade, pinned so nobody "fixes" it and breaks metres:
    "m" after a number is read as metres. Financial "m" is rare in article
    prose and usually written out; bare metric measurements are common."""
    assert "50 meters" in _norm_compound_units("It cost £50 m in total.")


# ── End to end, through the real normalizer ──────────────────────────────


def test_end_to_end_through_nemo():
    init_nemo()
    if not nemo_available():
        pytest.skip("NeMo unavailable; the fallback path is covered above")
    out = clean_spoken_text("The bridge is 56.7 m (186 ft) long.")
    assert "fifty six point seven meters" in out
    assert " M " not in out and not out.endswith(" M")

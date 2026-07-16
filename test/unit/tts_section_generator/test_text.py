"""Port of v0's text normalization tests (regex-path subset runs everywhere;
NeMo cases skip when the package is absent, matching v0's importorskip)."""

import pytest

from src.models.tts_section_generator.tts_generator.text import (
    clean_spoken_text,
    init_nemo,
)


def test_removes_citation_brackets():
    assert (
        clean_spoken_text("Earth is the third planet[1].")
        == "Earth is the third planet."
    )


def test_removes_phonetic_guides():
    assert clean_spoken_text("Earth (/ˈɜːrθ/) is a planet.") == "Earth is a planet."


def test_normalizes_percent():
    assert "seventy point eight percent" in clean_spoken_text(
        "Covering 70.8% of Earth."
    )


def test_normalizes_integer():
    assert "forty-two" in clean_spoken_text("42 is the answer.")


def test_unit_expansion_metric():
    assert "meters" in clean_spoken_text("The mountain is 2060 m tall.")


def test_unit_expansion_speed():
    assert "kilometers per hour" in clean_spoken_text("Winds reached 120 km/h.")


def test_unit_expansion_temperature():
    assert "degrees Celsius" in clean_spoken_text("Water boils at 100 °C.")


def test_unit_expansion_no_false_positive():
    assert "meters" not in clean_spoken_text("The m in theorem is silent.")


def test_unit_expansion_mm_before_m():
    assert "millimeters" in clean_spoken_text("The pipe is 50 mm wide.")


def test_strips_html_tags():
    result = clean_spoken_text("CO<sub>2</sub> concentration.")
    assert "CO" in result and "concentration" in result
    assert "<sub>" not in result


def test_en_dash_to_word():
    assert "to" in clean_spoken_text("100–900 million years.")


def test_compound_unit_m_s2_with_superscript():
    result = clean_spoken_text("Acceleration is 9.8 m/s².")
    assert "meters per second" in result or "metres per second" in result
    assert "squared" in result


def test_handles_empty():
    assert clean_spoken_text("") == ""
    assert clean_spoken_text(None) == ""


def test_strips_extra_whitespace():
    assert (
        clean_spoken_text("Hello    world.\n\nNext paragraph.")
        == "Hello world. Next paragraph."
    )


def test_nemo_currency_normalization():
    pytest.importorskip("nemo_text_processing")
    init_nemo()
    assert "dollars" in clean_spoken_text("It costs $99.99.")


def test_int_to_words_handles_huge_numbers_without_crashing():
    """Regression: v0's fallback crashed (IndexError) on numbers beyond
    'billion'; found by the Phase 3 corpus scan on real FA text."""
    from src.models.tts_section_generator.tts_generator.text import _int_to_words

    assert "trillion" in _int_to_words(1_400_000_000_000)
    assert "quadrillion" in _int_to_words(2 * 10**15)
    huge = _int_to_words(10**18)  # beyond named scales: digit-by-digit
    assert huge.startswith("one zero zero")
    out = clean_spoken_text("The estimate was 1400000000000 units.")
    assert len(out) > 0  # did not crash on number beyond named scales

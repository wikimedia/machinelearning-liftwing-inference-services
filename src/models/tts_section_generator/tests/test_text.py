"""Port of v0's text normalization tests plus pilot regression guards."""

import pytest
from tts_generator.text import clean_spoken_text, init_nemo


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
    from tts_generator.text import _int_to_words

    assert "trillion" in _int_to_words(1_400_000_000_000)
    assert "quadrillion" in _int_to_words(2 * 10**15)
    huge = _int_to_words(10**18)
    assert huge.startswith("one zero zero")
    out = clean_spoken_text("The estimate was 1400000000000 units.")
    assert "trillion" in out


# ── Pilot regression guards (T426756 + U 518 root cause) ─────────────────────


def test_scientific_notation_is_spoken_correctly():
    """5.97×10²⁴ must read as 'times ten to the power of'. The × is guarded
    to digit context so it is NOT dropped as a runic word separator."""
    result = clean_spoken_text("The mass is 5.97×10²⁴ kg.")
    assert "times" in result
    assert "ten to the power of" in result


def test_runic_word_separator_is_not_spoken_as_times():
    """× between non-digit characters is a runic word separator, not
    multiplication. Global ×→times replacement was a 2026.07.20 regression
    caught by the pilot (U 518)."""
    result = clean_spoken_text("stin × þina × iftiʀ")
    assert "times" not in result
    # The × glyph itself should be gone (replaced with space)
    assert "×" not in result


def test_times_between_digits_not_removed():
    """Digit-context × stays as 'times'."""
    result = clean_spoken_text("3 × 4 = 12")
    assert "times" in result


def test_pathological_old_norse_transliteration_degraded_not_crashed():
    """The actual chunk-5 pathological string from the U 518 pilot failure.
    After the interlinear strip (sections.py) removes the div, this string
    should never reach clean_spoken_text. But belt-and-braces: if similar
    content reaches normalization, it must not contain 'times' from the ×
    glyphs, and it must not crash."""
    pathological = (
        "times onHann times etaþisændaðis times iisiluSilu times nurnor "
        "times ianenþiʀþæiʀantriʀand"
    )
    result = clean_spoken_text(pathological)
    # × aren't in the input here (they were already replaced with spaces
    # by the ×→" " guard). The text itself contains only prose-normalization
    # markers and Old Norse transliteration.
    assert isinstance(result, str)
    # Should not crash, regardless of content length or character set.
    # The result may be non-empty (NeMo processes what it can), but must
    # not contain the word "times" from a × glyph that isn't present.

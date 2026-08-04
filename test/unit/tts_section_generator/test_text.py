"""Port of v0's text normalization tests plus pilot regression guards."""

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
    from src.models.tts_section_generator.tts_generator.text import _int_to_words

    assert "trillion" in _int_to_words(1_400_000_000_000)
    assert "quadrillion" in _int_to_words(2 * 10**15)
    huge = _int_to_words(10**18)
    assert huge.startswith("one zero zero")
    out = clean_spoken_text("The estimate was 1400000000000 units.")
    assert len(out) > 0  # did not crash on number beyond named scales


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


# ── Listening-pass regression guards (ruleset 2026.07.23) ────────────────────


def test_km_s_spoken_as_kilometers_per_second():
    result = clean_spoken_text("The star moves at 90 km/s.")
    assert "kilometers per second" in result


def test_ms_spoken_as_meters_per_second():
    result = clean_spoken_text("Velocity of 15 m/s was recorded.")
    assert "meters per second" in result


def test_unicode_minus_is_normalised():
    """U+2212 minus sign breaks NeMo number tokenization; normalise to
    ASCII 'minus' so the number is classifiable."""
    result = clean_spoken_text("The value is −110 km/s.")
    assert "minus" in result
    assert "kilometers per second" in result


def test_plus_minus_is_detached():
    """± glues numbers into one unclassifiable token (142.6±0.2).
    Detaching it lets NeMo classify both halves."""
    result = clean_spoken_text("of 142.6±0.2 km/s")
    assert "plus or minus" in result


def test_plus_minus_in_sentence_preserves_surrounding_text():
    """± normalization must not corrupt the sentence around it.
    Engine-agnostic: passes with or without NeMo."""
    result = clean_spoken_text("The measurement was 142.6±0.2 km/s.")
    assert "plus or minus" in result
    assert "kilometers per second" in result


def test_plus_minus_both_numbers_spoken_as_words():
    """With NeMo, both numbers around ± must be spoken, not digit-by-digit.
    NeMo-only strong form: without the ± detach, NeMo sees '142.6±0.2' as
    one unclassifiable token and reads it symbol-by-symbol."""
    pytest.importorskip("nemo_text_processing")
    init_nemo()
    result = clean_spoken_text("of 142.6±0.2 km/s")
    assert "one hundred" in result
    assert "point" in result
    assert "plus or minus" in result


def test_mg_per_l_spoken_correctly():
    result = clean_spoken_text("fluoride level of 1.5 mg/L is recommended.")
    assert "milligrams per liter" in result


def test_mg_per_day_spoken_correctly():
    result = clean_spoken_text("The dose is 6 mg/day.")
    assert "milligrams per day" in result


def test_mg_per_kg_still_works():
    """mg/kg is in NeMo's measure lexicon; must not be broken by the
    slash-unit whitelist. NeMo-only: the fallback path doesn't expand
    mg/kg, so skip when NeMo is absent."""
    pytest.importorskip("nemo_text_processing")
    init_nemo()
    result = clean_spoken_text("0.05 mg/kg is safe.")
    assert "milligrams per kilogram" in result


def test_aud_currency_prefix_expanded():
    """Post-normalization, '1.1' becomes 'one point one'; assert the
    currency expansion and that the $ symbol is gone."""
    result = clean_spoken_text("The project cost A$1.1 million to build.")
    assert "Australian dollars" in result
    assert "$" not in result


def test_trailing_comma_number_classified_by_nemo():
    """NeMo 1.2.0 fixes the decimal+comma verbatim bug (1.1.0: 'eight dot
    five comma'; 1.2.0: 'eight point five,'). NeMo-only: the fallback
    path does its best but can't match NeMo's grammar coverage."""
    pytest.importorskip("nemo_text_processing")
    init_nemo()
    result = clean_spoken_text("about 8.5, roughly.")
    assert "point five" in result
    assert "comma" not in result


def test_non_latin_cjk_is_stripped_romanization_kept():
    result = clean_spoken_text("Lightning (Japanese: ライトニング, Raitoningu)")
    assert "Raitoningu" in result
    assert "ライトニング" not in result
    # Must not leave dangling ": ," from the script strip
    assert ": ," not in result


def test_year_alternative_slash_read_as_or():
    """{{circa|1352/1362}} means either year; NeMo read the slash pair as
    a fraction ("...sixty-seconds"). Engine-agnostic assertions."""
    result = clean_spoken_text("born c. 1352/1362 in Vilnius")
    assert " or " in result
    assert "seconds" not in result
    assert "circa" in result and "c." not in result


def test_circa_guard_leaves_initialisms():
    result = clean_spoken_text("dated 25 B.C. 1350 years ago")
    assert "circa" not in result
    # Lowercase initialisms are the real work of the lookbehind
    assert "circa" not in clean_spoken_text("in 44 b.c. 1350 artifacts were found")


def test_year_slash_nemo_reads_years():
    """NeMo-only strong form: with the slash rewritten to 'or', NeMo
    classifies both halves as years, not a fraction."""
    pytest.importorskip("nemo_text_processing")
    init_nemo()
    result = clean_spoken_text("born c. 1352/1362 in Vilnius")
    assert "thirteen fifty two" in result
    assert "thirteen sixty two" in result


# ── Ruleset 2026.08.03 regression guards (T433594 follow-up) ──────────────


def test_regnal_roman_reads_ordinal():
    out = clean_spoken_text("Henry VIII married; Louis XIV built Versailles.")
    assert "Henry the Eighth" in out and "Louis the Fourteenth" in out


def test_regnal_roman_unicode_names():
    """[A-Za-z] is ASCII-only; royal names are not. Name shape is checked
    with str.isupper()/islower(). Władysław is whitelist-respelled when
    NeMo is live (Vladiswav, T433923), so assert the roman conversion via
    'the Second' rather than pinning the surname's spelling; Æthelred has
    no whitelist entry and pins the name-preserving path."""
    out = clean_spoken_text("Władysław II ruled.")
    assert "the Second ruled" in out and " II " not in out
    assert "Æthelred the Second" in clean_spoken_text("Æthelred II fled.")


def test_name_I_converts_with_guards():
    assert "Elizabeth the First was" in clean_spoken_text("Elizabeth I was crowned.")
    assert "Charles the First of England" in clean_spoken_text(
        "Charles I of England was tried."
    )


def test_structural_roman_reads_cardinal():
    out = clean_spoken_text(
        "He served in World War II and World War I; see Volume XIV, Part IV."
    )
    assert "World War Two" in out and "World War One" in out
    assert "Volume Fourteen" in out and "Part Four" in out


def test_roman_guards():
    # Middle initials: single-char V/X/I followed by "." never convert.
    assert "John V. Smith" in clean_spoken_text("John V. Smith wrote.")
    assert "John I. Smith" in clean_spoken_text("John I. Smith wrote.")
    # The famous exception.
    out = clean_spoken_text("Malcolm X spoke in Harlem.")
    assert "Malcolm X" in out and "Tenth" not in out
    # I followed by a Capitalized word is left alone (Mary I Tudor: logged).
    assert "Mary I Tudor" in clean_spoken_text("Mary I Tudor reigned.")
    # Pronoun I and ALLCAPS predecessors never convert.
    assert "It is I who" in clean_spoken_text("It is I who decides.")
    assert "NASA II" in clean_spoken_text("The NASA II mission.")


# ── No. / r. / fl. / dagger ──────────────────────────────────────────────


def test_numero_before_digit():
    out = clean_spoken_text("The song reached No. 1 on the chart.")
    assert "number " in out and "No." not in out


def test_numero_capital_only():
    assert "number" not in clean_spoken_text("The answer was no. 5 people.")


def test_reigned_floruit_expand_with_initialism_guard():
    assert "reigned " in clean_spoken_text("Władysław (r. 1386) ruled.")
    assert "flourished " in clean_spoken_text("The poet (fl. 1200) wrote.")
    out = clean_spoken_text("The B.R. 1904 model.")
    assert "B.R." in out and "reigned" not in out


def test_dagger_before_year_reads_died():
    out = clean_spoken_text("John Smith († 1434) was a bishop.")
    assert "died " in out and "†" not in out


# ── Tilde / DMS / micro / latin abbreviations ────────────────────────────


def test_tilde_before_digit():
    out = clean_spoken_text("The city lies ~50 km east.")
    assert "approximately " in out and "~" not in out


def test_dms_primes_and_compass():
    out = clean_spoken_text("Located at 51°28′40″N 0°5′W.")
    assert "minutes" in out and "seconds" in out
    assert "north" in out and "west" in out
    assert "′" not in out and "″" not in out


def test_micro_units_both_glyphs():
    out = clean_spoken_text("The cell is 10 µm long; a dose of 5 μg.")
    assert "micrometers" in out and "micrograms" in out


def test_latin_abbreviations_expand():
    out = clean_spoken_text("Birds, e.g., crows, adapt, i.e., quickly.")
    assert "for example, crows" in out and "that is, quickly" in out


def test_st_is_never_touched():
    out = clean_spoken_text("St. Peter's stands on Main St. today.")
    assert "St. Peter" in out and "Main St." in out


# ── NeMo-strong composition (the probe sentence) ─────────────────────────


def test_monarch_lead_composition_nemo():
    pytest.importorskip("nemo_text_processing")
    init_nemo()
    from tts_generator.text import nemo_available

    if not nemo_available():
        pytest.skip("NeMo init failed")
    out = clean_spoken_text("Henry VIII (r. 1509–1547) succeeded Henry VII.")
    assert "Henry the Eighth" in out
    assert "reigned fifteen oh nine to fifteen forty seven" in out
    assert "Henry the Seventh" in out


def test_pronunciation_whitelist_names_nemo():
    """ASR round-trip findings (T433923): non-English names espeak
    mangles (Wladyslaw was heard with a spelled letter W). Respellings
    are phoneme-verified; this pins them through the whitelist.
    NOTE for local runs: the grammar cache is keyed by whitelist
    FILENAME, not content; clear TTS_GEN_NEMO_CACHE after editing the
    tsv or the old FST is silently reused."""
    pytest.importorskip("nemo_text_processing")
    from tts_generator.text import nemo_available

    init_nemo()
    if not nemo_available():
        pytest.skip("NeMo init failed")
    out = clean_spoken_text(
        "Władysław II Jagiełło, born Jogaila, crossed the Wkra with "
        "Vytautas; the złoty and Białowieża endure."
    )
    assert "Vladiswav the Second Yahg yehwo" in out
    assert "Yo guyla" in out
    assert "Vukra" in out and "Veetowtas" in out
    assert "zwoty" in out and "Byahwovyezha" in out

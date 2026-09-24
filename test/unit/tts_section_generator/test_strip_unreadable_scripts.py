"""Tests for the unreadable-script strip (T438647 follow-up).

The cleaned string is both what the voice speaks and what listeners read:
service.py sends it to the isvc, whose word timestamps become
captions_vtt. So these assert on the RETURNED STRING, not on whether
espeak stays silent: a surviving CJK full stop or Tibetan tsheg is
caption residue even when nothing is spoken.
"""

import re
import unicodedata

import pytest

from src.models.tts_section_generator.tts_generator.text import (
    _KEEP_NON_ASCII,
    _strip_unreadable_scripts,
    clean_spoken_text,
)

# ── What must survive ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "£3 million and €4.2 million",  # currency: NeMo reads these later
        "25 °C at 51°28′40″N",  # degrees and DMS primes
        "10 µm and 5 μg",  # both micro signs
        "142.6±0.2 km",  # plus-minus
        "5.97×10²⁴ kg",  # times and superscripts
        "Add ½ cup and ¾ of the rest",  # vulgar fractions
        "Khafre → Menkaure ⟶ Shepseskaf",  # arrow rule runs before this
        "α particle and β decay",  # single Greek letters
        "Kamaʻehuakanaloa and ʻOumuamua",  # Hawaiian okina (modifier letter)
        "naïve café Señor Brontë Müller",  # espeak reads these correctly
        "Æthelwulf and Władysław and Dvořák",  # whitelist keys, folded later
        "an em—dash, an en–dash, ’quotes’",  # typography: caption fidelity
        "2+1⁄2 cups and 10+2⁄3 litres",  # FRACTION SLASH: see below
        "a ∗ operation where x ≠ y",  # math symbols espeak ignores
        "(Spanish: ¡Átame!) and ¿Qué?",  # Spanish punctuation
        "√2 ≈ 1.41 and ∞ is not a number",
    ],
)
def test_readable_text_is_returned_unchanged(text):
    assert _strip_unreadable_scripts(text) == text


def test_symbols_are_kept_by_default_not_stripped():
    """The failure modes are asymmetric, which is why letters and symbols
    get opposite defaults. An unreadable LETTER makes espeak recite its
    codepoint (unintelligible, and 45 phonemes for a six-letter name); an
    unreadable SYMBOL is merely silent. So symbols are kept unless they
    belong to a script being stripped.

    Dropping one is not harmless: removing the fraction slash from
    "2+1⁄2" leaves "2+12", which NeMo verbalizes as "two plus twelve".
    """
    for text in ("2+1⁄2", "10+2⁄3", "a ∗ b", "x ≠ y", "¡Átame!", "√2", "∞"):
        assert _strip_unreadable_scripts(text) == text


def test_script_owned_punctuation_still_goes():
    """Kept-by-default applies to symbols in general, not to punctuation
    belonging to a stripped script: those would be caption residue
    sitting beside the words just removed."""
    for text in ("。、「」・〜", "،؛", "｡｢｣", "Ａ２", "हिन्दी।"):
        assert _strip_unreadable_scripts(text).strip() == ""


def test_iteration_marks_go_with_their_script():
    """Modifier letters are kept so the Hawaiian okina survives, but CJK
    and Thai iteration marks are modifier letters too, and they appear
    inside the very glosses whose letters are stripped."""
    for text in ("々", "ゝゞ", "ヽヾ", "ๆ"):
        assert _strip_unreadable_scripts(text).strip() == ""
    for text in ("Kamaʻehuakanaloa", "ʻOumuamua"):
        assert _strip_unreadable_scripts(text) == text


def test_every_symbol_the_rules_depend_on_survives():
    """The strip runs before NeMo and before some symbol rules. Dropping
    any of these would break a rule silently rather than loudly."""
    for char in _KEEP_NON_ASCII:
        assert _strip_unreadable_scripts(f"x{char}x") == f"x{char}x", char


# ── What must go, letters AND punctuation ────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Здание мэрии Москвы",  # Cyrillic
        "Իջևան",  # Armenian
        "𒀭𒊹𒉽𒀸",  # Cuneiform
        "𓂋𓂝𓇳",  # Egyptian hieroglyphs
        "ⲣⲏ",  # Coptic
        "𐤓𐤏",  # Phoenician
        "ᬦᬲ",  # Balinese
        "ⴰⵎⴰⵣⵉⵖ",  # Tifinagh
        "ལྷ་ས",  # Tibetan, including the tsheg
        "ງ",  # Lao
        "မြန်မာ",  # Burmese, including its marks
        "幞头",  # CJK
        "ライトニング",  # Katakana
        "한국어",  # Hangul
        "العربية",  # Arabic
        "ܣܘܪܝܝܐ",  # Syriac
        "ไทย",  # Thai
        "हिन्दी",  # Devanagari
        "ქართული",  # Georgian
        "。、「」・〜",  # CJK punctuation: caption residue
        "،؛",  # Arabic punctuation
        "Ａ２",  # fullwidth forms
    ],
)
def test_unreadable_scripts_leave_nothing_behind(text):
    """Only whitespace may remain: clean_spoken_text collapses that at
    the end, so the words vanish without a trace in the captions."""
    assert _strip_unreadable_scripts(text).strip() == ""


def test_scripts_are_removed_from_around_the_english(text=None):
    assert (
        _strip_unreadable_scripts("Futou (simplified Chinese: 幞头) was headwear.")
        == "Futou (simplified Chinese: ) was headwear."
    )
    assert (
        re.sub(
            r"\s+",
            " ",
            _strip_unreadable_scripts(
                "Moscow City Hall (Russian: Здание мэрии) is here."
            ),
        )
        == "Moscow City Hall (Russian: ) is here."
    )


def test_parity_with_the_blocklist_this_replaced():
    """Every character the enumerated blocklist removed must still be
    removed. Written after a regression the other tests missed: the
    Japanese prolonged-sound mark "ー" survived, because its name is
    "KATAKANA-HIRAGANA PROLONGED SOUND MARK" and the script check reads
    the first word. The same hole let through 400+ enclosed CJK forms
    whose names begin CIRCLED, PARENTHESIZED or SQUARE.

    Combining marks are excluded: tested alone they have no base to
    inherit from, while in real text they follow one. The gloss tests
    above cover them in context.
    """
    old_blocklist = re.compile(
        "[֐-׿؀-ۿऀ-ॿ฀-๿ᄀ-ᇿ　-〿぀-ゟ゠-ヿ㄰-㆏ㇰ-ㇿ㈀-㏿"
        "㐀-䶿一-鿿가-힯豈-﫿＀-￯\U00020000-\U0002ffff]"
    )
    ranges = [
        (0x0590, 0x0700),
        (0x0900, 0x0E80),
        (0x1100, 0x1200),
        (0x3000, 0x3400),
        (0x4E00, 0x4E20),
        (0xAC00, 0xAC20),
        (0xFF00, 0xFFF0),
    ]
    leaks = []
    for low, high in ranges:
        for codepoint in range(low, high):
            char = chr(codepoint)
            if unicodedata.category(char) in ("Mn", "Mc", "Me"):
                continue
            if old_blocklist.match(char) and _strip_unreadable_scripts(char) != "":
                leaks.append(char)
    assert leaks == [], f"{len(leaks)} characters leak: {''.join(leaks[:20])!r}"


def test_prolonged_sound_mark_goes():
    """The regression that prompted the parity test above."""
    assert _strip_unreadable_scripts("ー") == ""
    assert _strip_unreadable_scripts("ポケモン ー").strip() == ""


# ── Marks follow their base (the stateful case) ──────────────────────────


def test_mark_on_a_kept_base_survives():
    """Decomposed Latin: the acute belongs to an ASCII "e" and stays."""
    decomposed = "cafe\u0301"
    assert _strip_unreadable_scripts(decomposed) == decomposed


def test_mark_on_a_stripped_base_goes_with_it():
    """Cyrillic "а" + combining acute: both go, no orphan mark."""
    assert _strip_unreadable_scripts("\u0430\u0301") == ""


def test_marks_do_not_leak_between_words():
    assert _strip_unreadable_scripts("Lhasa \u0f63\u0fb7 city") == "Lhasa  city"


# ── Greek: the deliberate split ──────────────────────────────────────────


def test_single_greek_letters_are_kept_by_the_strip():
    assert _strip_unreadable_scripts("α β μ Δ") == "α β μ Δ"


def test_greek_words_are_removed_by_the_pipeline():
    out = clean_spoken_text("Alexios Apokaukos (Greek: Ἀλέξιος Ἀπόκαυκος) ruled.")
    assert "Alexios Apokaukos" in out
    assert "Ἀλέξιος" not in out and "Ἀπόκαυκος" not in out


# ── End to end: nothing unreadable reaches the captions ──────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Futou (simplified Chinese: 幞头; traditional Chinese: 襆頭) was headwear.",
        "Ra (Ancient Egyptian: 𓂋𓂝𓇳; cuneiform: 𒊑𒀀; Coptic: ⲣⲏ) was a deity.",
        "China–Myanmar relations (Burmese: တရုတ်–မြန်မာ ဆက်ဆံရေး) are old.",
        "Ijevan (Armenian: Իջևան) is a town.",
        "Muang Sing (Lao: ງ) is a district.",
    ],
)
def test_no_unreadable_character_reaches_the_caption_text(text):
    out = clean_spoken_text(text)
    residue = [
        ch
        for ch in out
        if not ch.isascii() and ch not in _KEEP_NON_ASCII and not ch.isalpha()
    ]
    assert residue == [], f"caption residue: {residue!r} in {out!r}"
    for ch in out:
        if ch.isascii() or ch in _KEEP_NON_ASCII:
            continue
        import unicodedata

        script = unicodedata.name(ch, "?").split()[0]
        assert script in ("LATIN", "GREEK"), f"{ch!r} ({script}) survived in {out!r}"

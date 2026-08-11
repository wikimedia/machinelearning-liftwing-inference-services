"""Text normalization for TTS: port of v0 ``wiki_tts/text.py``.

Behavior-preserving port; the v0 test suite carries over verbatim. The only
changes are config imports and module logging. Normalization is part of
``generation_version`` (see version.py): any rule change here that alters
output text for identical input must bump NORMALIZATION_RULESET.
"""

import logging
import re

from tts_generator.config import NEMO_GRAMMAR_CACHE, NEMO_WHITELIST

logger = logging.getLogger(__name__)

_WORDS = (
    "zero one two three four five six seven eight nine ten "
    "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()

_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()

_SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]

# ── Unit abbreviation expansion ────────────────────────────────────────────

# Full list used as fallback when NeMo is unavailable (no singular/plural
# distinction).
_UNIT_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*km/h\b"), r"\1 kilometers per hour"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*km²\b"), r"\1 square kilometers"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*km\b"), r"\1 kilometers"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*m²\b"), r"\1 square meters"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*mm\b"), r"\1 millimeters"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*cm\b"), r"\1 centimeters"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*m\b"), r"\1 meters"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*mph\b"), r"\1 miles per hour"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*ft\b"), r"\1 feet"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*mi\b"), r"\1 miles"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*in\b"), r"\1 inches"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*kg\b"), r"\1 kilograms"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*mg\b"), r"\1 milligrams"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*g\b"), r"\1 grams"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*lb\b"), r"\1 pounds"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*oz\b"), r"\1 ounces"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*ml\b"), r"\1 milliliters"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*L\b"), r"\1 liters"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*m/s²\b"), r"\1 meters per second squared"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*m/s\b"), r"\1 meters per second"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*°C\b"), r"\1 degrees Celsius"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*°F\b"), r"\1 degrees Fahrenheit"),
]

# Compound / special units that NeMo's MEASURE grammar doesn't handle
# natively. Expanded before NeMo so the number is still in digit form.
_COMPOUND_UNIT_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*km/h\b"), r"\1 kilometers per hour"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*km²\b"), r"\1 square kilometers"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*m²\b"), r"\1 square meters"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*m/s²\b"), r"\1 meters per second squared"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*m/s\b"), r"\1 meters per second"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*mph\b"), r"\1 miles per hour"),
]

_SUP_TO_DIGIT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUB_TO_DIGIT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# ── Pre-NeMo token normalization tables (listening-pass ruleset) ──────────────
# Hoisted to module level so they are not rebuilt every call.

# Slash units NeMo's measure lexicon misses (mg/kg works; these don't).
# Whitelist, not generic: unit expansion must be deliberate.
_SLASH_UNITS = {
    "km/s": "kilometers per second",
    "m/s": "meters per second",
    "km/h": "kilometers per hour",
    "mg/L": "milligrams per liter",
    "g/L": "grams per liter",
    "mg/day": "milligrams per day",
    "g/day": "grams per day",
    "mL/day": "milliliters per day",
}

# Currency prefixes NeMo's money grammar can't parse.
_CURRENCY_PREFIX = {
    "A$": "Australian dollars",
    "NZ$": "New Zealand dollars",
    "US$": "US dollars",
    "C$": "Canadian dollars",
    "HK$": "Hong Kong dollars",
}

# Non-Latin script ranges stripped from English audio. Deliberately NOT
# Greek (α/β in astronomy names are directly pronounceable). This is v1's
# explicit stance: non-Latin content is omitted pending multilingual
# model-servers, per the README architecture separation.
_NON_LATIN_RE = re.compile(
    "["
    "֐-׿"  # Hebrew
    "؀-ۿ"  # Arabic
    "ऀ-ॿ"  # Devanagari
    "฀-๿"  # Thai
    "ᄀ-ᇿ"  # Hangul jamo
    "\u3000-\u303f"  # CJK symbols & punctuation (、。「」 + ideographic space)
    "぀-ゟ゠-ヿ"  # hiragana, katakana
    "㄰-㆏ㇰ-ㇿ"  # Hangul compat jamo, katakana phonetic ext
    "㈀-㏿"  # CJK enclosed / compatibility
    "㐀-䶿一-鿿"  # CJK ext A, CJK unified ideographs
    "가-힯"  # Hangul syllables
    "豈-﫿"  # CJK compatibility ideographs
    "＀-￯"  # fullwidth / halfwidth forms (includes fullwidth
    #   Latin letters and digits: deliberate, they
    #   appear almost exclusively inside CJK glosses)
    "\U00020000-\U0002ffff"  # CJK ext B+
    "]+"
)


# ── Roman numerals (listening-pass ruleset 2026.08) ───────────────────────
# espeak (Kokoro's G2P) reads roman numerals by literally saying "roman":
# "Henry VIII" was heard as "Henry roman eight", "Elizabeth I" as
# "Elizabeth eye". Two contexts, two readings (the industry-standard split):
#   * structural words take CARDINALS:  World War II -> World War Two
#   * regnal names take ORDINALS:       Henry VIII -> Henry the Eighth
# Guards, in order of the damage they prevent:
#   * multi-char romans first in the alternation (XVIII before XVII before X)
#   * single-char V and X are skipped when followed by "." (middle
#     initials: "John V. Smith"); multi-char romans keep sentence-final "."
#   * "Malcolm X" is a literal exception (the famous counterexample)
#   * bare I after a name converts only when NOT followed by "." (middle
#     initial "John I. Smith") and NOT followed by another Capitalized word
#     ("Mary I Tudor" is left alone and logged). Encyclopedic prose has no
#     first-person I outside quotations, which bounds the pronoun risk.
_ROMANS_1_30 = (
    "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX "
    "XXI XXII XXIII XXIV XXV XXVI XXVII XXVIII XXIX XXX"
).split()
_ORDINAL_WORDS = (
    "First Second Third Fourth Fifth Sixth Seventh Eighth Ninth Tenth "
    "Eleventh Twelfth Thirteenth Fourteenth Fifteenth Sixteenth Seventeenth "
    "Eighteenth Nineteenth Twentieth Twenty-first Twenty-second Twenty-third "
    "Twenty-fourth Twenty-fifth Twenty-sixth Twenty-seventh Twenty-eighth "
    "Twenty-ninth Thirtieth"
).split()
_CARDINAL_WORDS = (
    "One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen "
    "Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen Twenty Twenty-one "
    "Twenty-two Twenty-three Twenty-four Twenty-five Twenty-six Twenty-seven "
    "Twenty-eight Twenty-nine Thirty"
).split()
_ROMAN_TO_N = {r: i + 1 for i, r in enumerate(_ROMANS_1_30)}
# Longest-first alternation: regex alternation is ordered, and X must not
# shadow XVIII.
_ROMAN_ALT = "|".join(sorted(_ROMANS_1_30, key=len, reverse=True))
_ROMAN_MULTI_ALT = "|".join(
    sorted((r for r in _ROMANS_1_30 if len(r) > 1), key=len, reverse=True)
)

# Structural words whose roman suffix reads as a CARDINAL. Generous on
# purpose: every entry is a word after which "the Nth" would sound wrong.
_ROMAN_STRUCT_RE = re.compile(
    r"\b(World\s+War|Act|Part|Chapter|Volume|Book|Section|Phase|Class|Type"
    r"|Mark|Grade|Stage|Level|Camp|Appendix|Article|Table|Figure|Title)"
    rf"\s+({_ROMAN_ALT})\b"
)
# Regnal/name context. The name pattern matches any Unicode word
# ([^\W\d_] is "\w minus digits/underscore" = letters, Unicode-aware);
# the capitalized-name SHAPE (upper first, lower last) is enforced in the
# callable with str.isupper()/islower(), because [A-Za-z] is ASCII-only
# and royal names are not: "Władysław II" (ł), "Æthelred II" (Æ) must
# match. Multi-char romans (II..XXX): sentence-final "." is fine since
# "VIII." can never be an initial.
_ROMAN_NAME_MULTI_RE = re.compile(rf"\b([^\W\d_]+)\s+({_ROMAN_MULTI_ALT})\b")
# Single-char V and X after a name: the trailing-dot guard protects middle
# initials ("John V. Smith"); Malcolm X is excepted in the callable.
_ROMAN_NAME_VX_RE = re.compile(r"\b([^\W\d_]+)\s+([VX])\b(?!\.)")
# Bare I after a name: dot guard (initial), next-Capital guard (Mary I Tudor).
_ROMAN_NAME_I_RE = re.compile(r"\b([^\W\d_]+)\s+I\b(?!\.)(?!\s+[A-Z])")


def _is_name_shaped(word: str) -> bool:
    """Capitalized-name shape, Unicode-correct: upper first, lower last
    (filters lowercase words, ALLCAPS acronyms, and single letters)."""
    return len(word) >= 2 and word[0].isupper() and word[-1].islower()


def _roman_struct(m: re.Match) -> str:
    return f"{m.group(1)} {_CARDINAL_WORDS[_ROMAN_TO_N[m.group(2)] - 1]}"


def _roman_name(m: re.Match) -> str:
    name, roman = m.group(1), m.group(2)
    if not _is_name_shaped(name):
        return m.group(0)
    if name == "Malcolm" and roman == "X":
        return m.group(0)
    return f"{name} the {_ORDINAL_WORDS[_ROMAN_TO_N[roman] - 1]}"


def _norm_roman_numerals(text: str) -> str:
    """Structural first, so "World War I" reads cardinal before the
    name-ordinal rules could ever see it."""
    text = _ROMAN_STRUCT_RE.sub(_roman_struct, text)
    text = _ROMAN_NAME_MULTI_RE.sub(_roman_name, text)
    text = _ROMAN_NAME_VX_RE.sub(_roman_name, text)
    text = _ROMAN_NAME_I_RE.sub(
        lambda m: f"{m.group(1)} the First"
        if _is_name_shaped(m.group(1))
        else m.group(0),
        text,
    )
    return text


def _norm_units(text: str) -> str:
    """Expand measurement unit abbreviations following numeric values."""
    for pattern, replacement in _UNIT_SUBS:
        text = pattern.sub(replacement, text)
    return text


def _norm_compound_units(text: str) -> str:
    """Expand compound or special units that NeMo's grammar doesn't handle."""
    for pattern, replacement in _COMPOUND_UNIT_SUBS:
        text = pattern.sub(replacement, text)
    return text


# ── NeMo Text Processing ────────────────────────────────────────────────────

_nemo_normalizer = None


def init_nemo() -> None:
    """Initialize the NeMo text normalizer (called once at service startup)."""
    global _nemo_normalizer
    if _nemo_normalizer is not None:
        return

    try:
        logger.info("Initialising NeMo text normalizer...")
        from nemo_text_processing.text_normalization.normalize import Normalizer

        _nemo_normalizer = Normalizer(
            input_case="cased",
            lang="en",
            whitelist=NEMO_WHITELIST,
            cache_dir=NEMO_GRAMMAR_CACHE,
            overwrite_cache=False,
        )
        _nemo_normalizer.normalize("Warm up.")  # trigger grammar compilation
        logger.info("NeMo text normalizer ready.")
    except Exception:
        logger.warning(
            "NeMo text normalizer unavailable; falling back to regex.", exc_info=True
        )


def nemo_available() -> bool:
    return _nemo_normalizer is not None


def _norm_nemo(text: str) -> str:
    if _nemo_normalizer is not None:
        return _nemo_normalizer.normalize(text)
    return text


def _int_to_words(n: int) -> str:
    """Convert a non-negative integer to English words.

    Numbers beyond the named scales (>= 10**18) are read digit by digit:
    a FALLBACK normalizer must degrade, never crash. (The original v0
    port raised IndexError past "billion"; found by the Phase 3 corpus
    scan on real Featured Article text.)
    """
    if n == 0:
        return "zero"
    if n >= 10 ** (3 * len(_SCALES)):
        return " ".join(_WORDS[int(d)] for d in str(n))

    def _hundreds(n: int) -> str:
        if n == 0:
            return ""
        parts = []
        if n >= 100:
            parts.append(_WORDS[n // 100] + " hundred")
            n %= 100
        if n >= 20:
            t, o = divmod(n, 10)
            chunk = _TENS[t - 2]
            if o:
                chunk += "-" + _WORDS[o]
            parts.append(chunk)
        elif n > 0:
            parts.append(_WORDS[n])
        return " ".join(parts)

    result = []
    scale_idx = 0
    while n > 0:
        chunk = n % 1000
        if chunk:
            label = _hundreds(chunk)
            if scale := _SCALES[scale_idx]:
                label += " " + scale
            result.append(label)
        n //= 1000
        scale_idx += 1
    return " ".join(reversed(result))


def _norm_numbers(text: str) -> str:
    """Convert numeric tokens to their spoken form (fallback path)."""

    def _replace_decimal(m: re.Match) -> str:
        integer_word = _int_to_words(int(m.group(1)))
        decimal_digits = " ".join(_WORDS[int(d)] for d in m.group(2))
        suffix = " percent" if m.group(3) else ""
        return f"{integer_word} point {decimal_digits}{suffix}"

    def _replace_int_percent(m: re.Match) -> str:
        return f"{_int_to_words(int(m.group(1)))} percent"

    def _replace_int(m: re.Match) -> str:
        return _int_to_words(int(m.group(0)))

    text = re.sub(r"(\d+)\.(\d+)(%)?", _replace_decimal, text)
    text = re.sub(r"(?<!\d)(\d+)%", _replace_int_percent, text)
    text = re.sub(r"(?<!\d)(\d+)(?!\.\d)", _replace_int, text)
    return text


def clean_spoken_text(text: str) -> str:
    """Normalize Wikipedia text for TTS: removes citations, HTML, phonetic
    guides, expands units, normalizes numbers, dates, currency, and
    abbreviations."""
    if not text:
        return ""

    # ── 1. Strip markup ─────────────────────────────────────────────────────
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(/.*?/\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags (<sub>, <sup>, etc.)

    # ── 2. Normalize special characters ─────────────────────────────────────
    # En-dash / em-dash between numbers -> "to"
    text = re.sub(r"(\d+)\s*[–—]\s*(\d+)", r"\1 to \2", text)

    # ── 3. Compound unit expansion (units NeMo doesn't handle natively) ────
    text = _norm_compound_units(text)

    # Scientific notation and superscript runs (must precede the single
    # ²/³ replacements, or 10²⁴ would read as "10 squared 4").
    # × means "times" only between numbers (5.97×10²⁴); elsewhere (runic
    # word separators, dimension glyphs in odd contexts) it is dropped.
    # Global replacement was a 2026.07.20 regression caught by the pilot.
    text = re.sub(r"(?<=[\d⁰¹²³⁴⁵⁶⁷⁸⁹])\s*×\s*(?=[\d⁰¹²³⁴⁵⁶⁷⁸⁹])", " times ", text)
    text = text.replace("×", " ")
    text = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]{2,}",
        lambda m: " to the power of " + m.group(0).translate(_SUP_TO_DIGIT),
        text,
    )
    # Subscript digits read correctly as plain digits (H₂O -> "H2O",
    # the v0-validated behavior), now applied consistently.
    text = text.translate(_SUB_TO_DIGIT)

    # Remaining single superscripts (after unit expansion so km²/m²/m/s²
    # match first)
    text = text.replace("²", " squared")
    text = text.replace("³", " cubed")

    # ── 3.5. Pre-NeMo token normalisation (listening-pass ruleset) ─────────
    # NeMo's deterministic mode classifies whole tokens; any token its
    # grammars can't fully classify degrades to symbol-by-symbol reading.
    # These rules reshape tokens so NeMo's number/measure/money grammars
    # recognise them.

    # Unicode minus and ± break NeMo's number tokenization
    text = text.replace("−", "minus ")
    text = re.sub(r"\s*±\s*", " plus or minus ", text)

    # Slash units NeMo's measure lexicon misses
    for u, spoken in _SLASH_UNITS.items():
        text = re.sub(rf"(?<=\d)\s*{re.escape(u)}\b", f" {spoken}", text)

    # Currency prefixes NeMo's money grammar can't parse
    for sym, words in _CURRENCY_PREFIX.items():
        text = re.sub(
            rf"{re.escape(sym)}([\d][\d,]*(?:\.\d+)?)"
            rf"(\s*(?:thousand|million|billion|trillion))?",
            rf"\1\2 {words}",
            text,
        )

    # Year-alternative slash ({{circa|1352/1362}}: uncertain year, meaning
    # "either"). NeMo reads YYYY/YYYY as a fraction, denominator as plural
    # ordinal ("...sixty-seconds"). Rewritten with "or", NeMo classifies
    # both as years (verified 1.2.0: "thirteen fifty two or thirteen
    # sixty two").
    text = re.sub(r"\b(1\d{3}|20\d{2})/(1\d{3}|20\d{2})\b", r"\1 or \2", text)

    # "c." before a digit -> "circa" (bio leads; the voice reads bare
    # "c." as "see"). Lookbehind guards initialisms: "B.C. 1350" intact.
    text = re.sub(r"(?<![A-Za-z]\.)\bc\.\s*(?=\d)", "circa ", text)

    # "r." / "fl." before a digit -> "reigned" / "flourished" (monarch and
    # medieval-figure leads: "(r. 1386-1434)" was heard as "ar thirteen
    # eighty six"; "(fl. 1200)" as "ef-el"). Same initialism lookbehind as
    # the circa rule above.
    text = re.sub(r"(?<![A-Za-z]\.)\br\.\s*(?=\d)", "reigned ", text)
    text = re.sub(r"(?<![A-Za-z]\.)\bfl\.\s*(?=\d)", "flourished ", text)

    # Roman numerals: espeak says the word "roman" for them ("Henry roman
    # eight"). Structural contexts read cardinal, names read ordinal; see
    # the guard notes on the module-level tables.
    text = _norm_roman_numerals(text)

    # "No." before a digit -> "number" ("reached No. 1" was heard as
    # "reached no one": actively misleading). Capital-only on purpose:
    # lowercase "no." before a digit is almost always a sentence ending
    # ("The answer was no. 5 people agreed"), and MOS writes numero as
    # "No.".
    text = re.sub(r"\bNo\.\s*(?=\d)", "number ", text)

    # Dagger before a year -> "died" (bio convention "(† 1434)"; the glyph
    # is silent in espeak, leaving an orphaned parenthetical year).
    text = re.sub(r"†\s*(?=\d)", "died ", text)

    # Arrow glyphs -> "to": succession lists and reactions ("Khafre ->
    # Menkaure -> Shepseskaf") are otherwise spoken as "right arrow"
    # between every item (espeak verbalizes the glyph; T433923 standing
    # benchmark, relative-chronology worst-10).
    text = re.sub(r"\s*[→⟶]\s*", " to ", text)

    # "~" before a digit -> "approximately". Without this NeMo classifies
    # "~50" as one verbatim token: the output glues ("approximatelyfifty")
    # AND the following unit escapes the measure grammar ("km" unread).
    text = re.sub(r"~\s*(?=\d)", "approximately ", text)

    # Coordinate/DMS notation: compass letters first (while the prime
    # glyphs still mark the context), then the primes themselves, which
    # espeak renders as silence ("28' 40\" N" was heard as "twenty eight
    # forty en").
    text = re.sub(
        r"([\u00b0\u2032\u2033])\s*([NSEW])\b",
        lambda m: m.group(1)
        + " "
        + {"N": "north", "S": "south", "E": "east", "W": "west"}[m.group(2)],
        text,
    )
    text = re.sub(r"(?<=\d)\s*\u2032", " minutes ", text)
    text = re.sub(r"(?<=\d)\s*\u2033", " seconds ", text)

    # Micro-sign units (both U+00B5 MICRO SIGN and U+03BC GREEK MU appear
    # in wiki text): "10 um" was heard as "ten micro-em". Digit-guarded,
    # slash-units style: unit expansion must be deliberate.
    for _mu_unit, _mu_spoken in (
        ("m", "micrometers"),
        ("g", "micrograms"),
        ("s", "microseconds"),
        ("L", "microliters"),
    ):
        text = re.sub(rf"(?<=\d)\s*[\u00b5\u03bc]{_mu_unit}\b", f" {_mu_spoken}", text)

    # Latin abbreviations espeak spells as letters ("ee-jee", "eye-ee").
    # Lowercase-only (the written convention); "etc." and "et al." are
    # already spoken correctly by espeak and stay untouched.
    text = re.sub(r"\be\.g\.,?\s*", "for example, ", text)
    text = re.sub(r"\bi\.e\.,?\s*", "that is, ", text)

    # Non-Latin script runs: strip, keep romanization
    text = _NON_LATIN_RE.sub("", text)
    # Collapse damage left by script removal: ": ," -> ": ";
    # a gloss with nothing left ("(Japanese: )") -> removed.
    text = re.sub(r":\s*,\s*", ": ", text)
    text = re.sub(r"\(\s*[A-Za-z][A-Za-z ]*:\s*\)", "", text)
    text = re.sub(r"\s+,\s*", ", ", text)

    # ── 4. NeMo full normalisation ──────────────────────────────────────────
    text = _norm_nemo(text)

    # ── 5. Fallback when NeMo is unavailable ────────────────────────────────
    if _nemo_normalizer is None:
        text = _norm_units(text)  # full unit list (always plural)
        text = _norm_numbers(text)

    # ── 6. Remove orphaned punctuation from stripped Wikipedia symbols ────
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r",+", ",", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

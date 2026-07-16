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

    # Remaining superscripts (after unit expansion so km²/m²/m/s² match first)
    text = text.replace("²", " squared")
    text = text.replace("³", " cubed")

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

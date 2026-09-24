"""Text normalization for TTS: port of v0 ``wiki_tts/text.py``.

Behavior-preserving port; the v0 test suite carries over verbatim. The only
changes are config imports and module logging. Normalization is part of
``generation_version`` (see version.py): any rule change here that alters
output text for identical input must bump NORMALIZATION_RULESET.
"""

import functools
import logging
import re
import unicodedata

from anyascii import anyascii
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

# Non-Latin script ranges are stripped from English audio, per the README
# architecture separation (non-Latin content is omitted pending
# multilingual model-servers). Greek is the deliberate exception: single
# letters are kept — α/β in astronomy names are directly pronounceable,
# and the scientific use ("the alpha particle", "beta decay") reads
# correctly — but Greek WORDS are removed. espeak has no Greek
# pronunciation in en-us, so it reads a word as its codepoint and then
# names each letter: "Ἀλέξιος" is 67 phonemes for 7 characters, and a
# Byzantine gloss can consume half a segment's phoneme budget before any
# English is spoken (T438647).
_GREEK_WORD_RE = re.compile(
    r"[\u0370-\u03ff\u1f00-\u1fff][\u0370-\u03ff\u1f00-\u1fff\u0300-\u036f'\u2019]+"
)

# Characters that are not ASCII but MUST survive the script strip below:
# the rules further down (and NeMo, which runs after) consume them, so
# dropping one would silently break coordinates, units, fractions,
# currency or the arrow rule. Punctuation is here for a second reason:
# the cleaned string is also the caption text (service.py sends it to the
# isvc, whose word timestamps become captions_vtt), so anything that
# survives is read by a person, not just by espeak.
_KEEP_NON_ASCII = frozenset(
    "°′″µμ×÷±−–—‘’“”„…†‡§¶€£¥¢₹½¼¾⅓⅔⅛⅜⅝⅞⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉→⟶≈≤≥©®™«»"
)
_MARK_CATEGORIES = frozenset({"Mn", "Mc", "Me"})
_LETTER_CATEGORIES = frozenset({"Lo", "Lu", "Ll", "Lt"})
# espeak's en-us voice can pronounce Latin and (single) Greek letters.
# Everything else it reads as a codepoint followed by letter names, which
# is both unintelligible and enormous: "Ἀλέξιος" is 67 phonemes for 7
# characters, "Здание" 26 for 6, one Burmese gloss 587 (T438647).
_READABLE_SCRIPTS = frozenset({"LATIN", "GREEK"})
# Punctuation blocks belonging to scripts we strip, for the characters
# whose Unicode name does not start with the script: 「 is "LEFT CORNER
# BRACKET", 〜 is "WAVE DASH". Small and stable, unlike the letter
# blocklist this design replaced.
_SCRIPT_PUNCT_RANGES = (
    (0x3000, 0x303F),  # CJK symbols and punctuation 。、「」〜
    (0xFE10, 0xFE1F),  # vertical forms
    (0xFE30, 0xFE4F),  # CJK compatibility forms
    (0xFF00, 0xFFEF),  # fullwidth and halfwidth forms
    (0x0964, 0x0965),  # Devanagari danda
)

# Punctuation and symbols are KEPT by default: espeak is silent on the
# ones it does not know, so the cost of keeping one is a glyph in the
# caption, while the cost of dropping one is a wrong reading. Stripping
# "1⁄2"'s fraction slash turned "2+1⁄2" into "two plus twelve".
# Only punctuation BELONGING to a stripped script goes, identified by
# the first word of its Unicode name.
_UNREADABLE_SCRIPT_NAMES = frozenset(
    {
        "ARABIC",
        "ARMENIAN",
        "BALINESE",
        "BENGALI",
        "COPTIC",
        "CJK",
        "CUNEIFORM",
        "DEVANAGARI",
        "ETHIOPIC",
        "EGYPTIAN",
        "FULLWIDTH",
        "GEORGIAN",
        "GUJARATI",
        "GURMUKHI",
        "HALFWIDTH",
        "HANGUL",
        "HEBREW",
        "HIRAGANA",
        "IDEOGRAPHIC",
        "KANNADA",
        "KATAKANA",
        "KHMER",
        "LAO",
        "MALAYALAM",
        "MONGOLIAN",
        "MYANMAR",
        "NKO",
        "ORIYA",
        "PHOENICIAN",
        "SINHALA",
        "SYRIAC",
        "TAMIL",
        "TELUGU",
        "THAI",
        "TIBETAN",
        "TIFINAGH",
    }
)


@functools.lru_cache(maxsize=4096)
def _readable(ch: str) -> bool | None:
    """True = keep, False = strip, None = inherit the preceding character.

    Marks inherit: a combining acute belongs to the letter it sits on, so
    it survives on "cafe\u0301" and is removed with a stripped Cyrillic
    base.

    Modifier letters (category Lm) are kept-by-default LIKE SYMBOLS, not
    judged by the letter rule. The distinction matters: the Hawaiian
    okina is "MODIFIER LETTER TURNED COMMA", so the letter rule (keep
    only LATIN and GREEK) would strip it and break
    "Kama\u02bbehuakanaloa". Kept-by-default keeps it, while CJK and Thai
    iteration marks (\u3005 \u309d \u30fe \u0e46, also Lm) still go, because their
    names carry the script they belong to.
    """
    if ch.isascii() or ch in _KEEP_NON_ASCII:
        return True
    category = unicodedata.category(ch)
    if category in _MARK_CATEGORIES:
        return None
    if category in _LETTER_CATEGORIES:
        try:
            script = unicodedata.name(ch).split()[0]
        except ValueError:  # unnamed codepoint: not something we can read
            return False
        return script in _READABLE_SCRIPTS
    # Modifier letters, punctuation, symbols and digits: KEPT unless the
    # character belongs to a script we strip. Lm lands here deliberately
    # (see the docstring): judging it by the letter rule would strip the
    # okina, whose name begins MODIFIER rather than LATIN.
    codepoint = ord(ch)
    if any(low <= codepoint <= high for low, high in _SCRIPT_PUNCT_RANGES):
        return False
    try:
        script = unicodedata.name(ch).split()[0]
    except ValueError:
        return False
    return script not in _UNREADABLE_SCRIPT_NAMES


def _strip_unreadable_scripts(text: str) -> str:
    """Remove characters espeak cannot read, and their attached marks.

    Replaces an enumerated block list. Enumerating scripts does not
    converge: the first production run turned up Cyrillic, Coptic,
    Cuneiform, Phoenician, Egyptian hieroglyphs, Syriac, Tibetan, Lao,
    Burmese, Armenian, Balinese and Tifinagh, and the next corpus would
    turn up more. Keeping what is readable and dropping the rest handles
    a script nobody has seen yet, on arrival.

    Greek is kept here and handled by _GREEK_WORD_RE above: single
    letters are pronounceable and meaningful ("the alpha particle"),
    whole Greek words are not.
    """
    if text.isascii():
        return text
    out = []
    keeping = True
    for ch in text:
        verdict = _readable(ch)
        if verdict is None:
            if keeping:
                out.append(ch)
            continue
        keeping = verdict
        if verdict:
            out.append(ch)
    return "".join(out)


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


# Latin characters espeak cannot pronounce. Its language data has no
# pronunciation and no accent name for these, so it falls back to spelling
# out the codepoint one digit at a time (the ``$accent`` fallback in
# espeak-ng's dictionary documentation): "Nguyễn" becomes 45 phonemes
# reading "N G U Y letter one E C five N", where "Nguyen" is 7 (T438647).
#
# DERIVED, not hand-written: which characters these are depends on the
# espeak behind kokoro-onnx, so scripts/derive_fold_set.py regenerates and
# re-verifies this set inside the tts image (the only place with espeak).
# The set is listed exactly rather than as a range: Latin Extended
# Additional is 255 of 256 characters, and the exception (U+1E9E capital
# sharp s) reads correctly, so folding the whole block would regress it.
#
# Characters espeak DOES read are deliberately absent: folding é ñ ç ü ö
# would make pronunciation worse ("Señor" sɛnjˈɔːɹ -> sˈɛnɚ, "Brontë"
# bɹˈɔntɛ -> bɹˈɔnt) across far more of the corpus than it fixes, and
# macrons (ā ē ī ō ū) read correctly too.
_UNREADABLE_LATIN = frozenset(
    "ĐđŉſƀƂƃƄƅƇƈƋƌƍƑƒƔƕƖƘƙƚƛƞƤƥƧƨƪƫƬƭƱƵƶƸƹƺƻƼƽƾƿǀǁǂǃǄ"
    "ǅǆǇǈǉǊǋǌǤǥǦǧǨǩǮǯǰǱǲǳǴǵǶǷǸǹȐȑȒȓȘșȚțȞȟȠȡȤȥȴȵȶȷȸȹȺȻ"
    "ȼȽȾȿɀɁɂɃɈɉɊɋɌɍḀḁḂḃḄḅḆḇḈḉḊḋḌḍḎḏḐḑḒḓḔḕḖḗḘḙḚḛḜḝḞḟḠḡ"
    "ḢḣḤḥḦḧḨḩḪḫḬḭḮḯḰḱḲḳḴḵḶḷḸḹḺḻḼḽḾḿṀṁṂṃṄṅṆṇṈṉṊṋṌṍṎṏṐṑ"
    "ṒṓṔṕṖṗṘṙṚṛṜṝṞṟṠṡṢṣṤṥṦṧṨṩṪṫṬṭṮṯṰṱṲṳṴṵṶṷṸṹṺṻṼṽṾṿẀẁ"
    "ẂẃẄẅẆẇẈẉẊẋẌẍẎẏẐẑẒẓẔẕẖẗẘẙẚẛẜẝẟẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲ"
    "ẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢ"
    "ợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹỺỻỼỽỾỿɞɡɣɩɰɷɸʊʗʘʚʩʪʫʬʭʮʯⱠ"
    "ⱡⱣⱥⱦⱧⱨⱩⱪⱫⱬⱱⱲⱳⱴⱵⱶⱷⱸⱹⱺⱻⱼⱽⱾⱿꝽꞬ"
)


def _fold_unreadable_latin(text: str) -> str:
    """Transliterate the Latin characters espeak cannot pronounce.

    Only characters in ``_UNREADABLE_LATIN`` are touched; everything else
    is returned byte-for-byte. The mapping itself comes from anyascii
    (ISC-licensed, no dependencies) rather than a hand-maintained table.

    MUST run after NeMo normalisation. Eleven pronunciation whitelist
    entries are keyed on diacritic spellings (Władysław, Jagiełło,
    Białowieża, Æthelwulf and others); folding first would stop every one
    of them matching and silently undo those fixes.

    anyascii's table feeds this function, so its output is part of
    ``content_sha256`` and of the normalizer identity: the pin in
    requirements.txt and the version component in version.py must move
    together with any upgrade.
    """
    if text.isascii():
        return text
    # ``or ch``: anyascii can return "" for a character it does not know,
    # which would silently delete a letter mid-word — a missing-token G2P
    # error worse than espeak's codepoint spelling. No current gate
    # character hits this, but the guard is cheap insurance against a
    # future anyascii table change. Preserve the character when there is no
    # transliteration.
    return "".join(
        (anyascii(ch) or ch) if ch in _UNREADABLE_LATIN else ch for ch in text
    )


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

    # Scripts espeak cannot read: strip, keep the romanization beside them
    text = _GREEK_WORD_RE.sub("", text)
    text = _strip_unreadable_scripts(text)
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

    # ── 6. Fold the Latin characters espeak cannot pronounce ──────────────
    # After NeMo: the pronunciation whitelist is keyed on the original
    # spellings, diacritics included.
    text = _fold_unreadable_latin(text)

    # ── 7. Remove orphaned punctuation from stripped Wikipedia symbols ────
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r",+", ",", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

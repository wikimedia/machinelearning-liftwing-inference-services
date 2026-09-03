"""Tests for boundaries.py, the answer-edge corrections.

A raw answer edge is a token edge, and a token is not a unit a reader sees.
`answer_span` moves it onto one, in three steps that are each tested here:

- whitespace the SentencePiece span carries is dropped;
- each edge widens to the tokenizer's own word (`_align_to_words`), the way the
  removed `question-answering` pipeline did with `align_to_words`;
- each edge is pushed off the inside of a grapheme cluster
  (`_extend_to_graphemes`), which holds in every script.

The service ships in every language, so the multilingual cases are the point
rather than an afterthought: an earlier version classified characters with
`str.isalnum`, which reads False for every combining mark, and so treated a Thai
tone mark or an Indic matra as the end of a word.

boundaries.py reads no torch, transformers or kserve, so nothing here needs the
stubs conftest.py installs for the test modules that reach qa.py. `regex` is a
real dependency of boundaries.py and is imported here for the same reason.
"""

import unicodedata
from bisect import bisect_left, bisect_right

import pytest
import regex

from src.models.semantic_highlighting.model_server.boundaries import (
    _align_to_words,
    _extend_to_graphemes,
    _has_no_space_script,
    _is_separator,
    answer_span,
    word_spans,
)

# The passage from the code review that made the correction necessary.
EIFFEL = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
    "Paris, France. It is named after the engineer Gustave Eiffel, whose company "
    "designed and built the tower from 1887 to 1889 as the centrepiece of the "
    "World Exposition."
)


def _whitespace_word(context, index):
    """The whitespace-delimited run holding ``index``.

    The tokenizer splits on whitespace before it splits into subwords, so this
    is the word span a fast tokenizer reports for a token at ``index`` -- which
    lets these tests exercise the real alignment without loading a tokenizer.
    """
    start = index
    while start > 0 and not context[start - 1].isspace():
        start -= 1
    end = index
    while end < len(context) and not context[end].isspace():
        end += 1
    return (start, end)


def _highlight(context, start, end):
    """`answer_span` over a raw span, with the tokenizer's words filled in."""
    char_start, char_end = answer_span(
        context,
        start,
        end,
        _whitespace_word(context, start),
        _whitespace_word(context, max(start, end - 1)),
    )
    return context[char_start:char_end]


def _raw(context, start, end):
    """`answer_span` with no word information, as on a slow tokenizer."""
    char_start, char_end = answer_span(context, start, end)
    return context[char_start:char_end]


# --- word alignment ----------------------------------------------------------


def test_token_edge_inside_a_word_widens_to_the_word():
    # "centrepiece" tokenizes as "centre" + "piece", so the model can answer
    # from the second half of it.
    start = EIFFEL.index("piece of the World")
    assert _highlight(EIFFEL, start, start + len("piece of the World")) == (
        "centrepiece of the World"
    )


def test_whole_word_answer_is_unchanged():
    start = EIFFEL.index("Gustave Eiffel")
    end = start + len("Gustave Eiffel")
    assert _highlight(EIFFEL, start, end) == "Gustave Eiffel"


def test_leading_whitespace_is_dropped():
    start = EIFFEL.index(" Gustave Eiffel")
    end = start + len(" Gustave Eiffel")
    assert _highlight(EIFFEL, start, end) == "Gustave Eiffel"


def test_digits_are_word_characters():
    start = EIFFEL.index("889 as")
    assert _highlight(EIFFEL, start, start + 3) == "1889"


def test_trailing_punctuation_is_not_swallowed():
    # The tokenizer's word here is "Eiffel," -- the comma and all -- so this
    # tests the separator stop, not a missing word.
    end = EIFFEL.index("Gustave Eiffel,") + len("Gustave Eiffel")
    assert _whitespace_word(EIFFEL, end - 1) == (
        EIFFEL.index("Eiffel,"),
        EIFFEL.index("Eiffel,") + len("Eiffel,"),
    )
    start = EIFFEL.index("Gustave Eiffel")
    assert _highlight(EIFFEL, start, end) == "Gustave Eiffel"


def test_hyphen_bounds_the_widening():
    # "wrought-iron" is one tokenizer word, but the hyphen stops the walk, so a
    # cut inside "wrought" takes that half only.
    start = EIFFEL.index("rought-iron")
    assert _highlight(EIFFEL, start, start + len("rought")) == "wrought"


def test_no_cap_on_how_far_an_edge_moves():
    # A compound is one word however long it is. The earlier version capped the
    # move at 12 characters, so a cut this deep into the word was left alone --
    # the same word snapping or not depending on where the token edge fell.
    for word in ("Weltausstellung", "Rechtsschutzversicherung", "kirjastonhoitaja"):
        context = "Die " + word + " von 1889"
        cut = 4 + 13
        assert _highlight(context, cut, 4 + len(word)) == word


def test_widening_stops_at_the_word_and_not_beyond():
    context = "aaa bbbbbbbb ccc"
    assert _highlight(context, 6, 9) == "bbbbbbbb"


def test_without_word_information_the_token_edge_stands():
    start = EIFFEL.index("piece of the World")
    assert _raw(EIFFEL, start, start + len("piece of the World")) == (
        "piece of the World"
    )


def test_align_to_words_leaves_a_none_word_alone():
    assert _align_to_words(EIFFEL, 10, 20, None, None) == (10, 20)


# --- grapheme clusters: the case str.isalnum got wrong ------------------------

THAI = "หอไอเฟลตั้งอยู่ในปารีส"  # "the Eiffel Tower is in Paris"
HINDI = "एफिल टॉवर पेरिस में है"  # "the Eiffel Tower is in Paris"


def test_thai_marks_are_not_orphaned_by_the_end_edge():
    # Index 7 is the base consonant and 8-9 are its vowel and tone marks. An
    # edge at 8 leaves them outside the highlight, where they render broken.
    assert unicodedata.category(THAI[7]) == "Lo"
    assert [unicodedata.category(ch) for ch in THAI[8:10]] == ["Mn", "Mn"]
    assert _extend_to_graphemes(THAI, 0, 8) == (0, 10)


def test_thai_highlight_never_opens_on_a_bare_mark():
    assert _extend_to_graphemes(THAI, 8, len(THAI)) == (7, len(THAI))


def test_devanagari_matra_does_not_end_a_word():
    # Index 6 is a matra, so widening the start of "ॉवर" has to walk over it to
    # reach the start of the word. Classifying with str.isalnum stopped there.
    assert unicodedata.category(HINDI[6]) == "Mc"
    start = HINDI.index("टॉवर") + 1
    assert _highlight(HINDI, start, start + 3) == "टॉवर"


def test_nfd_accent_stays_with_its_letter():
    context = unicodedata.normalize("NFD", "Tiếng Việt")
    assert [unicodedata.category(ch) for ch in context[3:5]] == ["Mn", "Mn"]
    assert _extend_to_graphemes(context, 0, 3) == (0, 5)


def test_arabic_haraka_stays_with_its_letter():
    context = "مَدرسة"  # مَدرسة
    assert unicodedata.category(context[1]) == "Mn"
    assert _extend_to_graphemes(context, 0, 1) == (0, 2)


def test_persian_zwnj_does_not_end_a_word():
    # ZWNJ is category Cf, so it is not alphanumeric, but it sits inside a
    # Persian word. Widening has to cross it.
    context = "می‌رود"  # می‌رود
    assert not context[2].isalnum()
    assert _highlight(context, 3, len(context)) == context


def test_emoji_zwj_sequence_is_one_cluster():
    context = "\U0001f468‍\U0001f469‍\U0001f467"  # family
    assert _extend_to_graphemes(context, 0, 1) == (0, len(context))


def test_plain_latin_edges_are_left_alone():
    assert _extend_to_graphemes(EIFFEL, 4, 10) == (4, 10)


# --- scripts with no space between words -------------------------------------

JAPANESE = "エッフェル塔はパリにある鉄の塔です"


def test_japanese_clause_does_not_widen():
    # The whole run is one tokenizer word, so widening to it would highlight the
    # clause. The model's token edge is the best boundary available.
    assert _highlight(JAPANESE, 6, 12) == JAPANESE[6:12]


def test_japanese_edge_still_leaves_a_cluster_whole():
    # Alignment is off here, but the grapheme correction is not: a decomposed
    # dakuten must not be cut off its kana.
    context = unicodedata.normalize("NFD", "パリにある")
    assert unicodedata.category(context[1]) == "Mn"
    assert _extend_to_graphemes(context, 0, 1) == (0, 2)


def test_latin_glued_to_japanese_does_not_widen():
    # No space between them, so the tokenizer calls the lot one word.
    context = "塔はParisにある"
    start = context.index("Paris")
    assert _highlight(context, start, start + 5) == "Paris"


@pytest.mark.parametrize(
    "text",
    [
        "エッフェル塔",  # Japanese
        "巴黎",  # Chinese
        "หอไอเฟล",  # Thai
        "ຫໍໄອເຟ",  # Lao
        "ប៉ារីស",  # Khmer
        "မြန်မာ",  # Myanmar
        "བོད་ཡིག",  # Tibetan
    ],
)
def test_scripts_without_word_gaps(text):
    assert _has_no_space_script(text)


@pytest.mark.parametrize(
    "text",
    [
        "centrepiece",  # Latin
        "Weltausstellung",
        "Эйфелева",  # Cyrillic
        "Πύργος",  # Greek
        "برج ايفل",  # Arabic
        "מגדל",  # Hebrew
        "एफिल टॉवर",  # Devanagari -- Hindi does use spaces
        "에펠탑",  # Hangul -- so does Korean
        "1889",
    ],
)
def test_scripts_with_word_gaps(text):
    assert not _has_no_space_script(text)


# --- what ends a word --------------------------------------------------------


@pytest.mark.parametrize(
    "ch",
    [
        "ั",  # Thai vowel sign, Mn
        "่",  # Thai tone mark, Mn
        "ा",  # Devanagari matra, Mc
        "्",  # Devanagari virama, Mn
        "َ",  # Arabic fatha, Mn
        "ִ",  # Hebrew niqqud, Mn
        "́",  # combining acute, Mn
        "‌",  # ZWNJ, Cf
        "‍",  # ZWJ, Cf
        "e",
        "9",
        "$",  # symbol, not punctuation: "$100" widens whole
        "+",
    ],
)
def test_word_internal_characters_are_not_separators(ch):
    assert not _is_separator(ch)


@pytest.mark.parametrize(
    "ch",
    [
        " ",
        "\t",
        "\n",
        " ",
        ",",
        ".",
        "-",
        "'",
        "’",
        "%",  # Unicode files the percent sign under punctuation, not symbols
    ],
)
def test_whitespace_and_punctuation_are_separators(ch):
    assert _is_separator(ch)


# --- the tokenizer's words ---------------------------------------------------


def test_word_spans_unions_the_offsets_of_a_word():
    # CLS, two question tokens, SEP, three context tokens, SEP, PAD.
    sequence_ids = [None, 0, 0, None, 1, 1, 1, None, None]
    word_ids = [None, 0, 1, None, 0, 0, 1, None, None]
    offsets = [
        (0, 0),
        (0, 3),
        (4, 6),
        (0, 0),
        (0, 6),
        (6, 11),
        (12, 17),
        (0, 0),
        (0, 0),
    ]
    assert word_spans(word_ids, sequence_ids, offsets) == [
        None,
        None,
        None,
        None,
        (0, 11),  # "centre" and "piece" are one word
        (0, 11),
        (12, 17),
        None,
        None,
    ]


def test_word_spans_skips_a_token_that_covers_no_characters():
    sequence_ids = [1, 1]
    word_ids = [0, 0]
    offsets = [(0, 0), (0, 4)]
    assert word_spans(word_ids, sequence_ids, offsets) == [(0, 4), (0, 4)]


def test_word_spans_without_word_ids_is_all_none():
    assert word_spans(None, [None, 1, 1], [(0, 0), (0, 3), (3, 6)]) == [
        None,
        None,
        None,
    ]


# --- degenerate spans --------------------------------------------------------


def test_whitespace_only_span_collapses():
    assert _highlight("a   b", 1, 4) == ""


def test_span_at_the_edges_of_the_context():
    assert _highlight("Eiffel", 2, 4) == "Eiffel"
    assert _highlight("Eiffel", 0, 6) == "Eiffel"


def test_empty_context():
    assert answer_span("", 0, 0) == (0, 0)


def test_empty_span_at_the_end_of_the_text_does_not_widen():
    # The end of the text is a cluster boundary, so there is nothing to widen.
    for context in ("Eiffel", THAI, JAPANESE):
        assert _extend_to_graphemes(context, len(context), len(context)) == (
            len(context),
            len(context),
        )


@pytest.mark.parametrize(
    "context",
    [
        EIFFEL[:80],
        THAI,
        HINDI,
        JAPANESE,
        unicodedata.normalize("NFD", "Tiếng Việt"),
        "a\r\nb\tc  d",  # CRLF is one cluster; \n alone is not a safe anchor
        "\U0001f468\u200d\U0001f469\u200d\U0001f467 family",
        "no-space-anywhere-in-this-string",
    ],
)
def test_bounded_scan_matches_a_scan_of_the_whole_text(context):
    """`_extend_to_graphemes` counts clusters from the last space, not from 0.

    That rests on a claim -- a space always begins a cluster -- so this checks
    the shortcut against the boundaries of the whole text, for every span.
    """
    bounds = [match.start() for match in regex.finditer(r"\X", context)]
    bounds.append(len(context))
    for start in range(len(context) + 1):
        for end in range(start, len(context) + 1):
            assert _extend_to_graphemes(context, start, end) == (
                bounds[bisect_right(bounds, start) - 1],
                bounds[bisect_left(bounds, end)],
            )

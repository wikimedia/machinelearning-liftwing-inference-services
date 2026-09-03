"""Move an answer edge onto a boundary a reader sees.

The QA model in ``qa.py`` answers with whole tokens, and a token edge is not a
boundary a reader sees. This tokenizer splits "centrepiece" into "centre" and
"piece", so a raw token edge highlights "piece of the World".

:func:`answer_span` corrects such an edge in three steps: it drops the
whitespace a SentencePiece span carries, it widens each edge to the tokenizer's
own word, and it pushes each edge off the inside of a grapheme cluster.

:func:`word_spans` reads the tokenizer's word boundaries out of one encoding, so
``qa.py`` hands this module plain lists and a string and nothing else.
"""

import unicodedata
from typing import Optional

import regex

# Codepoint ranges of the scripts that write with no space between words.
#
# The tokenizer splits on whitespace before it splits into subwords, so its
# notion of a "word" is a whitespace-delimited run. In these scripts that run is
# a whole clause, and widening an answer edge to it grows the highlight over the
# clause -- worse than an edge inside a word. `_align_to_words` therefore leaves
# an edge alone when its word holds one of these characters.
#
# Hangul is deliberately absent: Korean puts spaces between its words. So do
# the Indic scripts, so they align like Latin.
_NO_SPACE_SCRIPTS = (
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x0F00, 0x0FFF),  # Tibetan -- ends a syllable with a tsheg, not a space
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x19E0, 0x19FF),  # Khmer symbols
    (0x3005, 0x3007),  # ideographic iteration and number marks
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xA9E0, 0xA9FF),  # Myanmar extended-B
    (0xAA60, 0xAA7F),  # Myanmar extended-A
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF66, 0xFF9F),  # half-width Katakana
    (0x20000, 0x3FFFF),  # CJK unified ideographs, extension B and later
)


def _has_no_space_script(text: str) -> bool:
    """True when ``text`` holds a character from a script with no word gaps."""
    return any(
        any(low <= point <= high for low, high in _NO_SPACE_SCRIPTS)
        for point in map(ord, text)
    )


def _is_separator(ch: str) -> bool:
    """True for a character that ends a word: whitespace or punctuation.

    A combining mark such as a thai vowel or tone mark is not alphanumeric,
    but it is word-internal in every script that has one. Testing for
    alphanumeric would stop a walk in the middle of a Thai word. Instead we
    check for either a space or unicode punctuation as the boundary.

    Symbol categories are left out to allow widening across "$100" or "C++". A
    percent sign does stop the walk, unicode files it under punctuation, not
    symbols."""
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def word_spans(word_ids, sequence_ids, offsets) -> list:
    """Character span of the tokenizer word each token of one item belongs to.

    Offsets are relative to the context string.

    ``word_ids`` reports which pre-tokenizer word each token came from. A
    word's span is the union of the offsets of its tokens.

    The union is taken here rather than asking the encoding for
    ``word_to_chars`` so that this module keeps to plain lists and a string.
    This allows testing without transformers.

    Returns all ``None`` when the tokenizer reports no words, which leaves
    :func:`_align_to_words` nothing to do and the raw token offsets in place.
    Words come from a pre-tokenizer, and not every tokenizer has one.
    """
    # spans[token] holds (char_start, char_end) or None
    spans = [None] * len(sequence_ids)
    if word_ids is None:
        return spans
    bounds = {}
    for token, (sequence, word) in enumerate(zip(sequence_ids, word_ids)):
        if sequence != 1 or word is None:
            continue
        start, end = int(offsets[token][0]), int(offsets[token][1])
        if end <= start:
            # A token covering no characters says nothing about its word.
            continue
        low, high = bounds.get(word, (start, end))
        bounds[word] = (min(low, start), max(high, end))
    for token, (sequence, word) in enumerate(zip(sequence_ids, word_ids)):
        if sequence == 1 and word in bounds:
            spans[token] = bounds[word]
    return spans


def _align_to_words(
    context: str,
    char_start: int,
    char_end: int,
    start_word: Optional[tuple],
    end_word: Optional[tuple],
) -> tuple:
    """Widen each answer edge to the edge of the tokenizer word holding it.

    The model result tokens can cut a word: this tokenizer splits "centrepiece"
    into "centre" + "piece", so a raw token edge highlights "piece of the
    World.". This widens it to "centrepiece of the World".

    ``start_word`` and ``end_word`` come from :func:`word_spans`.

    bounds for the widening:
    * It stops at a separator, so it does not swallow punctuation the tokenizer
      counts as part of the same word: ``Gustave Eiffel`` rather than the
      pipeline's ``Gustave Eiffel,``, and ``iron`` rather than ``wrought-iron``.
    * It does not run at all when the word is in a script from _NO_SPACE_SCRIPTS.
    """
    if start_word is not None and not _has_no_space_script(
        context[start_word[0] : start_word[1]]
    ):
        while char_start > start_word[0] and not _is_separator(context[char_start - 1]):
            char_start -= 1
    if end_word is not None and not _has_no_space_script(
        context[end_word[0] : end_word[1]]
    ):
        while char_end < end_word[1] and not _is_separator(context[char_end]):
            char_end += 1
    return char_start, char_end


def _extend_to_graphemes(context: str, char_start: int, char_end: int) -> tuple:
    r"""Widen the edges off the inside of any grapheme cluster they cut.

    A token edge can land between a base character and its combining marks: a
    Thai vowel or tone mark, an Indic matra, Arabic harakat, an NFD accent, a
    ZWJ inside an emoji sequence. A highlight that opens on a bare mark, or
    that drops the marks off its last character, renders as a broken glyph. It
    moves an edge by a character or two at most.

    ``regex``'s ``\X`` matches one cluster under the UAX #29 rules, so the
    cluster table is Unicode's rather than one kept here.

    Clusters are counted from the last space before the span up to its end,
    rather than over the whole passage: this runs per answer and a passage is
    up to 2000 characters. The anchor is exact, not an estimate. A space always
    begins a cluster -- no rule joins a cluster to a space that comes after it
    -- so counting from one gives the same boundaries as counting from the
    start of the text. A passage with no space in it anchors at 0.
    """
    if char_start >= len(context):
        # An empty span at the end of the text. The end of the text is itself a
        # boundary, so there is nothing to widen, and the loop below would take
        # the last character in.
        return len(context), len(context)
    anchor = max(context.rfind(" ", 0, char_start), 0)
    start = end = anchor
    for match in regex.finditer(r"\X", context, pos=anchor):
        if match.start() <= char_start:
            start = match.start()
        if match.start() >= char_end:
            end = match.start()
            break
    else:
        end = len(context)
    return start, end


def answer_span(
    context: str,
    char_start: int,
    char_end: int,
    start_word: Optional[tuple] = None,
    end_word: Optional[tuple] = None,
) -> tuple:
    """Turn one raw token span into the character span to highlight.

    Three corrections, in this order:

    1. Drop whitespace the span carries. SentencePiece tokens hold the space in
       front of a word, so a raw span can start or end on one.
    2. Widen each edge to its tokenizer word (:func:`_align_to_words`), which
       cannot re-add whitespace because it stops at a separator.
    3. Push each edge off the inside of a grapheme cluster
       (:func:`_extend_to_graphemes`). Last, so nothing after it reopens one.
    """
    while char_start < char_end and context[char_start].isspace():
        char_start += 1
    while char_end > char_start and context[char_end - 1].isspace():
        char_end -= 1
    if char_start >= char_end:
        return char_start, char_end
    char_start, char_end = _align_to_words(
        context, char_start, char_end, start_word, end_word
    )
    return _extend_to_graphemes(context, char_start, char_end)

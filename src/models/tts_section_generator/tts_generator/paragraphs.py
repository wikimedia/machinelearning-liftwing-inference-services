"""Paragraph boundaries in the spoken text, for paragraph-level highlighting.

A client highlighting an article as it is read needs to answer one
question: at audio time T, which paragraph is playing? The service
already produces everything needed to answer it -- the word timestamps,
in order -- but nothing records where the paragraphs are, because
extraction flattens them.

This module recovers them. Given a section's normalized text and the
blocks it came from, it returns each paragraph's half-open range of word
indices, which the caller turns into a time range by looking up the
first and last word's timestamps.

Why not simply normalize each block and count words
---------------------------------------------------
Because normalization is not paragraph-local. NeMo reads across the
join, so ``clean(a + " " + b)`` can differ from
``clean(a) + " " + clean(b)`` -- "one hundred forty five" against "one
hundred and forty five", or a blockquote attribution gaining a space
before its dash. Measured across a 50-article sample (394 sections),
per-block normalization matched whole-section normalization for the
large majority of sections; most of the remainder were recoverable by
token alignment, leaving a small proportional tail.

The three tiers, in order:

1. **identity** -- the per-block tokens concatenate to exactly the
   section's tokens. Ranges are exact.
2. **alignment** -- they differ, but difflib matches them well enough
   to locate each boundary in the real token stream.
3. **proportional** -- alignment is poor (the blocks and the text have
   genuinely diverged). Time is split by character count and the
   paragraph is flagged, so a consumer can tell a measured boundary
   from an estimated one.
"""

import difflib
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# How far the two token streams may diverge before their boundaries stop
# meaning anything. Expressed as a share of unmatched tokens rather than
# difflib's ratio, because that ratio is length-dependent: one inserted
# word in a 9-word section scores 0.89, the same insertion in a 200-word
# section scores 0.998, so a fixed ratio judges short sections far more
# harshly than long ones for the same drift.
#
# The floor keeps very short sections workable; the share keeps long ones
# honest. Normalization drift is a word or two per join, which passes
# either way; a block list that genuinely does not describe the text --
# the case this guards against -- fails both.
MAX_UNMATCHED_SHARE = 0.10
MIN_UNMATCHED_TOKENS = 4

Method = Literal["identity", "alignment", "proportional"]

IDENTITY: Method = "identity"
ALIGNMENT: Method = "alignment"
PROPORTIONAL: Method = "proportional"


def paragraph_word_ranges(
    cleaned: str,
    blocks: list[str],
    normalize,
) -> tuple[list[tuple[int, int]], Method]:
    """Locate each block in the section's normalized word stream.

    Args:
        cleaned: the section's normalized text, exactly as synthesized.
        blocks: the raw block texts the section was extracted from.
        normalize: the normalizer, passed in rather than imported so
            this module stays testable without NeMo.

    Returns:
        ``(ranges, method)`` where ranges are half-open ``(start, end)``
        word indices into ``cleaned.split()``, one per block, and method
        is the tier that produced them. Ranges are non-decreasing and
        non-overlapping; a block whose prose normalizes away entirely
        (an all-Greek parenthetical, say) gets an empty range at its
        neighbour's boundary rather than being dropped, so indices stay
        aligned with the blocks the caller holds.
    """
    words = cleaned.split()
    if not blocks or not words:
        return [], IDENTITY

    per_block = [normalize(b).split() for b in blocks]
    flat = [t for block in per_block for t in block]

    # Tier 1: the streams agree, so block lengths are the boundaries.
    if flat == words:
        return _cumulative(per_block), IDENTITY

    # Tier 2: they differ. Match them and carry each block's boundary
    # through to the real stream, if the match is good enough to trust.
    matcher = difflib.SequenceMatcher(None, flat, words, autojunk=False)
    unmatched = _unmatched_tokens(matcher)
    budget = max(MIN_UNMATCHED_TOKENS, int(len(words) * MAX_UNMATCHED_SHARE))
    if unmatched <= budget:
        return _aligned(per_block, matcher, len(words)), ALIGNMENT

    logger.info(
        "paragraph boundaries fell back to proportional timing "
        "(%d unmatched tokens over a budget of %d, %d blocks)",
        unmatched,
        budget,
        len(blocks),
    )
    return _proportional(blocks, len(words)), PROPORTIONAL


def _unmatched_tokens(matcher: difflib.SequenceMatcher) -> int:
    """Tokens on either side that the match could not pair up."""
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def _cumulative(per_block: list[list[str]]) -> list[tuple[int, int]]:
    """Ranges straight from the block lengths."""
    ranges, pos = [], 0
    for tokens in per_block:
        ranges.append((pos, pos + len(tokens)))
        pos += len(tokens)
    return ranges


def _aligned(
    per_block: list[list[str]],
    matcher: difflib.SequenceMatcher,
    total: int,
) -> list[tuple[int, int]]:
    """Carry block boundaries through a difflib match.

    The opcodes map positions in the concatenated per-block stream onto
    positions in the real one. Each block's boundary is translated
    through the nearest equal run; boundaries inside an inserted or
    deleted run land at that run's edge, which is the closest the two
    streams can be said to correspond.
    """
    # Position in the per-block stream -> position in the real stream.
    translate: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                translate[i1 + offset] = j1 + offset
        translate.setdefault(i1, j1)
        translate[i2] = j2

    boundaries, pos = [0], 0
    for tokens in per_block:
        pos += len(tokens)
        boundaries.append(pos)

    mapped = [_nearest(translate, b, total) for b in boundaries]
    # Alignment can map two boundaries onto the same point, or (in a
    # transposition) out of order. Clamp to monotonic so the ranges a
    # caller gets are always usable as time ranges.
    for i in range(1, len(mapped)):
        mapped[i] = max(mapped[i], mapped[i - 1])
    return list(zip(mapped, mapped[1:]))


def _nearest(translate: dict[int, int], index: int, total: int) -> int:
    """Translate one boundary, falling back to the closest known point."""
    if index in translate:
        return min(translate[index], total)
    known = [i for i in translate if i <= index]
    if not known:
        return 0
    return min(translate[max(known)], total)


def _proportional(blocks: list[str], total: int) -> list[tuple[int, int]]:
    """Split the word stream by each block's share of the characters.

    A last resort: the boundaries are estimates, and the caller marks
    them as such so nobody mistakes them for measured ones.
    """
    chars = [len(b) for b in blocks]
    total_chars = sum(chars) or 1
    ranges, pos = [], 0
    for i, n in enumerate(chars):
        if i == len(chars) - 1:
            end = total  # the last block absorbs any rounding
        else:
            end = min(total, pos + round(n / total_chars * total))
        end = max(end, pos)
        ranges.append((pos, end))
        pos = end
    return ranges


def paragraph_entries(
    ranges: list[tuple[int, int]],
    blocks: list[str],
    words: list[str],
    timestamps: list[dict],
    duration_ms: float,
    method: Method,
    excerpt_chars: int = 120,
) -> list[dict]:
    """Turn word ranges into the published paragraph list.

    ``words`` is the section's full normalized word stream -- what
    ``paragraph_word_ranges`` indexed into. ``timestamps`` is the word
    timestamps the isvc returned for it. The two usually line up one to
    one, but the aligner drops a word whose cue has no duration
    (zero-width cues are unrenderable), so ``timestamps`` is a subsequence
    of ``words``, not an equal-length list. Indexing timestamps by word
    position would therefore mis-read every paragraph after the first
    dropped word; the lookup walks the two streams in lockstep instead.

    ``excerpt`` is the block's pruned prose, not its normalized form:
    citation markers and other non-spoken noise are already stripped (the
    same pruning as the section text), but "1000" stays "1000" rather
    than "one thousand". It matches what the voice speaks, modulo that
    normalization -- it is not the fully-rendered page text, which still
    shows citation markers.

    A paragraph with no words of its own -- or one whose words all lost
    their timestamps -- takes a zero-length range at the preceding
    paragraph's end, keeping the list one-to-one with the blocks.
    """
    ts_index = _timestamp_index_map(words, timestamps)
    if timestamps:
        matched = sum(1 for i in ts_index if i is not None)
        if matched < len(timestamps) * 0.9:
            logger.warning(
                "paragraph timing matched %d of %d timestamps to words; "
                "paragraph boundaries may be unusable",
                matched,
                len(timestamps),
            )
    entries: list[dict] = []
    last_end = 0.0
    for index, ((start, end), block) in enumerate(zip(ranges, blocks)):
        stamps = [i for i in ts_index[start:end] if i is not None]
        if stamps:
            start_ms = float(timestamps[stamps[0]]["start_ms"])
            end_ms = float(timestamps[stamps[-1]]["end_ms"])
            last_end = end_ms
        else:
            start_ms = end_ms = last_end
        entries.append(
            {
                "index": index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "word_count": end - start,
                "char_count": len(block),
                "excerpt": block[:excerpt_chars],
                "method": method,
            }
        )

    if entries and not timestamps:
        # No word timestamps to read (a paragraph artifact was requested
        # with timestamps="none"): every range collapsed to zero length,
        # so spread the section's duration over them by character share.
        _spread(entries, duration_ms)
    return entries


def _timestamp_index_map(words: list[str], timestamps: list[dict]) -> list[int | None]:
    """Map each full word index to its timestamp index, None where dropped.

    ``timestamps`` is a subsequence of ``words`` in the same order: the
    aligner drops a word whose cue has no duration. Walking the two in
    lockstep recovers which words kept a cue without trusting the two
    lists to be equal length.
    """
    index: list[int | None] = []
    p = 0
    for word in words:
        if p < len(timestamps) and timestamps[p].get("word") == word:
            index.append(p)
            p += 1
        else:
            index.append(None)
    return index


def _spread(entries: list[dict], duration_ms: float) -> None:
    """Assign times by character share, in place."""
    total = sum(e["char_count"] for e in entries) or 1
    pos = 0.0
    for i, entry in enumerate(entries):
        share = entry["char_count"] / total * duration_ms
        entry["start_ms"] = pos
        pos = duration_ms if i == len(entries) - 1 else min(duration_ms, pos + share)
        entry["end_ms"] = pos

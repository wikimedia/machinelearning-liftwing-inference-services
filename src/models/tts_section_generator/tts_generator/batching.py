"""Group a section's text chunks into isvc-sized batches.

A section is synthesized in one isvc request per batch. Splitting exists
because a single request must finish inside the 600 s chain
(``spec.predictor.timeout`` -> Knative ``timeoutSeconds`` ->
``TTS_ISVC_TIMEOUT_S`` -> ``mesh.upstream_timeout``), and one
19,877-character section exceeded it during the v2 trial (T441983).

A batch closes only at a chunk that ends a sentence, so the silence gap
the caller inserts between batches always lands at a sentence end, where
the voice already pauses. Measured on the corpus: 97.3% of the chunk
boundaries ``split_text`` produces end a sentence, and the longest run
without one is 1,344 characters -- so waiting for the next sentence end
overshoots the budget by a bounded, small amount.

Nothing here does I/O: ``partition`` is a pure function of its inputs,
which is what makes the batch layout reproducible for a given
configuration.
"""

import logging
import re

logger = logging.getLogger(__name__)

# A chunk ends a sentence if its last character is sentence-final
# punctuation, allowing for a closing quote or bracket after it.
_SENTENCE_END = re.compile(r"""[.!?]["')\]]*$""")

# Close a batch even without a sentence end once it reaches this multiple
# of the budget. Never reached by the measured corpus (worst case is
# budget + 1,344 chars), but a section of unpunctuated text -- a flattened
# table, a list of names -- would otherwise grow one batch without limit
# and reproduce the timeout this module exists to prevent.
#
# 1.5 keeps the worst-case mid-sentence batch (~6,400 chars at the 4,000
# budget) at ~485 s of wall at containerConcurrency 4, under the 600 s
# chain. A factor of 2 would close at 8,000+ chars and reproduce the
# ~613 s cc:4 failure the split exists to prevent.
_OVERSHOOT_FACTOR = 1.5


def ends_sentence(chunk: str) -> bool:
    """Whether ``chunk`` ends at a sentence boundary."""
    return bool(_SENTENCE_END.search(chunk.rstrip()))


def partition(chunks: list[str], max_batch_chars: int) -> list[list[str]]:
    """Group ``chunks`` into batches of roughly ``max_batch_chars``.

    The budget is a target, not a cap: a batch closes at the first
    sentence-ending chunk at or past the budget, so it is usually a
    little over. Returns one batch per isvc request, in order; a chunk
    list at or under the budget returns a single batch, which is the
    unchanged single-request path for the ~93% of sections below it.
    """
    if max_batch_chars <= 0:
        raise ValueError(f"max_batch_chars must be positive, got {max_batch_chars}")

    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    hard_limit = int(max_batch_chars * _OVERSHOOT_FACTOR)

    for chunk in chunks:
        current.append(chunk)
        size += len(chunk)

        if size < max_batch_chars:
            continue
        if ends_sentence(chunk):
            batches.append(current)
            current, size = [], 0
        elif size >= hard_limit:
            # No sentence end in sight: split here rather than let one
            # request grow past the timeout. The join is mid-sentence and
            # will be audible, so say so.
            logger.warning(
                "closing a batch mid-sentence: %d chars with no sentence end "
                "(budget %d); the join will be audible",
                size,
                max_batch_chars,
            )
            batches.append(current)
            current, size = [], 0

    if current:
        batches.append(current)
    return batches

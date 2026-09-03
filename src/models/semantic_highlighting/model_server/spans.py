"""Highlight-span arithmetic for the OpenSearch semantic highlighter.

Turns a QA result (``qa.answer_question_batch``, with Python code-point offsets)
into the span list OpenSearch expects, honouring the constraints in
``opensearch_semantic_highlighting.md`` section 5. Those constraints are
load-bearing: a span that violates them fails the *entire* ``_search``, not just
the one hit, so every span leaving this module goes through :func:`harden`.

Each span also carries the two things offsets alone cannot say: ``answer``, the
text it selects, and ``score``, the model's confidence in it. Both are additive
-- OpenSearch reads ``start`` and ``end`` -- and ``answer`` is derived from the
final offsets rather than copied from the decoder, so the two cannot disagree.

Deliberately free of torch, transformers and kserve so the rules can be unit
tested on any Python without installing the serving stack.
"""

from typing import Optional


def to_utf16(text: str, index: int) -> int:
    """Convert a Python code-point index into a UTF-16 code-unit index.

    OpenSearch measures offsets in Java ``String`` units (UTF-16 code units).
    For BMP-only text this equals the code-point index; non-BMP characters
    (emoji, rare CJK, math symbols) count as two UTF-16 units, so a naive
    code-point offset would drift. See spec section 5.
    """
    return len(text[:index].encode("utf-16-le")) // 2


def utf16_len(text: str) -> int:
    """Length of ``text`` in UTF-16 code units (how OpenSearch sees it)."""
    return len(text.encode("utf-16-le")) // 2


def slice_utf16(text: str, start: int, end: int) -> str:
    """The substring that UTF-16 offsets ``start``/``end`` select.

    The inverse of :func:`to_utf16`, and the only correct way to read a span back
    out: slicing ``text`` with those offsets directly would drift on any passage
    carrying a non-BMP character, exactly as producing them naively would.
    """
    buffer = text.encode("utf-16-le")
    return buffer[start * 2 : end * 2].decode("utf-16-le")


def harden(spans: list[dict], length: int) -> list[dict]:
    """Force OpenSearch's span constraints (spec section 5).

    Guarantees in-bounds integer offsets, sorted by start, unique starts, and
    no overlaps. Offsets and ``length`` must both be in UTF-16 code units.
    """
    cleaned = []
    for s in spans:
        start = max(0, min(int(s["start"]), length))
        end = max(0, min(int(s["end"]), length))
        if start < end:
            cleaned.append((start, end))
    cleaned.sort(key=lambda t: t[0])

    out: list[dict] = []
    last_end = -1
    for start, end in cleaned:
        if start < last_end:  # overlaps previous (or duplicate start) -> drop
            continue
        out.append({"start": start, "end": end})
        last_end = end
    return out


def spans_from_result(
    result: dict, context: str, min_score: Optional[float] = None
) -> list[dict]:
    """Convert one ``qa`` result into hardened UTF-16 highlight spans.

    The model's best answer span is the highlight; an abstention (or a score
    below ``min_score``, when set) yields no highlight.

    Every span returned carries a non-empty ``answer``: the text it selects, read
    back out of ``context`` with the *hardened* offsets. Reading it back rather
    than copying ``result["answer"]`` is what makes "``answer`` is exactly what
    ``start``/``end`` select" true by construction -- after any clamping
    :func:`harden` applied, and on non-BMP text, where the decoder's code-point
    offsets and these UTF-16 ones part ways.

    ``score`` is unambiguous in a span, unlike in the result it comes from: a
    span exists only when the model did not abstain, so it is always the answer's
    own probability and never the ``[CLS]`` null probability that
    ``decode_answer`` reports in its place. It is also the exact number
    ``min_score`` was compared against, unrounded, so a recorded response can be
    replayed against a candidate ``SH_MIN_SCORE`` -- which is what the README's
    open calibration question needs.
    """
    if not result["answer"].strip():
        return []
    if min_score is not None and result["score"] < min_score:
        return []
    spans = [
        {
            "start": to_utf16(context, result["start"]),
            "end": to_utf16(context, result["end"]),
        }
    ]
    # `harden` rebuilds each span from its offsets alone, so these are attached
    # after it rather than threaded through and silently dropped.
    return [
        {
            **span,
            "answer": slice_utf16(context, span["start"], span["end"]),
            "score": result["score"],
        }
        for span in harden(spans, utf16_len(context))
    ]

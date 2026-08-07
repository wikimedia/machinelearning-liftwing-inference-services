"""Highlight-span arithmetic for the OpenSearch semantic highlighter.

Turns a QA result (``qa.answer_question_batch``, with Python code-point offsets)
into the span list OpenSearch expects, honouring the constraints in
``opensearch_semantic_highlighting.md`` section 5. Those constraints are
load-bearing: a span that violates them fails the *entire* ``_search``, not just
the one hit, so every span leaving this module goes through :func:`harden`.

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
    return harden(spans, utf16_len(context))

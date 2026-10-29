"""Batching a section's chunks into isvc requests (T441983).

A long section is synthesized in several isvc requests so no single one
outlives the 600 s chain. These pin the two properties the audio depends
on: batches close at sentence ends (so the caller's silence gap lands
where the voice already pauses), and a section under the budget still
produces exactly one request -- the unchanged path for ~93% of sections.
"""

import pytest
from tts_generator.batching import ends_sentence, partition

SENTENCE = "x" * 348 + "."  # 349 chars, ends a sentence
FRAGMENT = "y" * 400  # 400 chars, no sentence end


def sizes(batches: list[list[str]]) -> list[int]:
    return [sum(len(c) for c in b) for b in batches]


# ── The no-op path: most sections must be unaffected ─────────────────────


def test_a_section_under_the_budget_is_one_batch():
    """~93% of sections take this path and must be byte-identical to
    their pre-split output: one request, no gap, nothing new."""
    assert len(partition([SENTENCE] * 5, 4000)) == 1


def test_a_section_exactly_at_the_budget_is_one_batch():
    chunks = ["x" * 399 + "."] * 10  # 4,000 chars exactly
    assert sizes(partition(chunks, 4000)) == [4000]


def test_one_chunk_over_the_budget_splits():
    """The discriminating case: this fails if partition never splits."""
    chunks = ["x" * 399 + "."] * 11  # 4,400 chars
    assert len(partition(chunks, 4000)) == 2


def test_empty_input_produces_no_batches():
    assert partition([], 4000) == []


# ── Where the joins land ─────────────────────────────────────────────────


def test_every_join_is_at_a_sentence_end():
    """The silence gap goes between batches, so each batch but the last
    must end a sentence or the gap lands mid-sentence."""
    batches = partition([SENTENCE] * 30, 4000)
    assert len(batches) > 1
    for batch in batches[:-1]:
        assert ends_sentence(batch[-1])


def test_a_run_without_sentence_ends_defers_the_split():
    """split_text emits mid-sentence chunks when one sentence exceeds the
    chunk size. The batcher must keep accumulating to the next sentence
    end rather than splitting mid-sentence."""
    chunks = [SENTENCE] * 11 + [FRAGMENT] * 3 + ["y" * 144 + "."] + [SENTENCE] * 5
    batches = partition(chunks, 4000)
    for batch in batches[:-1]:
        assert ends_sentence(batch[-1])


def test_an_unpunctuated_run_is_still_bounded():
    """Defence in depth: text with no sentence-ending punctuation at all
    (a flattened table, a list of names) would otherwise grow one batch
    without limit and reproduce the timeout this module prevents. The
    measured corpus never reaches this, but the v2 corpus is a different
    set of articles.

    The cap is budget * _OVERSHOOT_FACTOR (4000 * 1.5 = 6000): sized so a
    mid-sentence close stays under the 600 s chain at containerConcurrency
    4, unlike a 2x cap which would reproduce the ~613 s failure."""
    batches = partition([FRAGMENT] * 30, 4000)
    assert max(sizes(batches)) <= 6000


# ── Nothing is lost or reordered ─────────────────────────────────────────


@pytest.mark.parametrize("budget", [20, 400, 4000, 100000])
def test_chunks_are_preserved_in_order(budget):
    chunks = [f"sentence number {i}." for i in range(50)]
    assert [c for batch in partition(chunks, budget) for c in batch] == chunks


def test_a_non_positive_budget_is_rejected():
    """A zero budget would close a batch on every chunk; silently doing
    so would turn one request into hundreds."""
    with pytest.raises(ValueError):
        partition([SENTENCE], 0)


# ── The sentence-end predicate ───────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A sentence.", True),
        ("A question?", True),
        ("An exclamation!", True),
        ('He said "go."', True),  # closing quote after the stop
        ("Ends in a bracket.)", True),
        ("A sentence. ", True),  # trailing space is stripped
        ("No terminator", False),
        ("A clause,", False),
        ("An abbreviation such as Dr", False),
        ("", False),
    ],
)
def test_ends_sentence(text, expected):
    assert ends_sentence(text) is expected

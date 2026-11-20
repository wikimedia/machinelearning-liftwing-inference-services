"""Checks the vectorized decode against the loop it replaces.

`qa._best_spans` scores the start and end candidates of a whole batch with
tensor operations. This module compares it with `_reference`, the per-item loop
that `qa.py` used before, on random logits.

It needs the real torch, so it is skipped unless ``SH_REAL_MODEL=1``, which also
turns off the stubs in conftest.py. Run it inside the service image, as
test_real_tokenizer.py shows; it needs no tokenizer and no weights:

    docker run --rm --user "$(id -u):$(id -g)" \\
      -v "$PWD":/srv/app:ro -w /srv/app \\
      -e USER="$USER" -e HOME=/tmp -e SH_REAL_MODEL=1 \\
      -e PYTHONPATH=/srv/overlay/lib/python3.12/site-packages:/srv/venv/lib/python3.12/site-packages:/srv/app \\
      --entrypoint python semantic-highlighting:prod \\
      -m pytest test/unit/semantic_highlighting/test_best_spans.py -q
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SH_REAL_MODEL") != "1",
    reason="needs the real torch; set SH_REAL_MODEL=1 to run",
)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def qa():
    from src.models.semantic_highlighting.model_server import qa

    return qa


def _reference(
    torch, start_logits, end_logits, attention_mask, context_mask, max_len, top_k
):
    """The per-item loop that `qa.py` used before `_best_spans`."""
    neg_inf = torch.finfo(start_logits.dtype).min
    pad = ~attention_mask.bool()
    start_probs = start_logits.masked_fill(pad, neg_inf).softmax(dim=-1)
    end_probs = end_logits.masked_fill(pad, neg_inf).softmax(dim=-1)
    null_prob = (start_probs[0] * end_probs[0]).item()

    context_tokens = [i for i, inside in enumerate(context_mask.tolist()) if inside]
    top_starts = sorted(context_tokens, key=lambda i: start_probs[i], reverse=True)
    top_ends = sorted(context_tokens, key=lambda i: end_probs[i], reverse=True)

    best_prob = 0.0
    best_tokens = None
    for s in top_starts[:top_k]:
        for e in top_ends[:top_k]:
            if e < s or (e - s + 1) > max_len:
                continue
            prob = (start_probs[s] * end_probs[e]).item()
            if prob > best_prob:
                best_prob = prob
                best_tokens = (s, e)
    return null_prob, best_prob, best_tokens


def _batch(torch, generator, batch, length, scale):
    """Random logits, with a question, a context and padding in each item."""
    start_logits = torch.randn(batch, length, generator=generator) * scale
    end_logits = torch.randn(batch, length, generator=generator) * scale
    attention_mask = torch.zeros(batch, length, dtype=torch.long)
    context_mask = torch.zeros(batch, length, dtype=torch.bool)
    for i in range(batch):
        # [CLS] question [SEP] context [SEP] padding
        real = int(torch.randint(4, length + 1, (1,), generator=generator))
        question = int(torch.randint(1, max(2, real - 2), (1,), generator=generator))
        attention_mask[i, :real] = 1
        context_mask[i, question + 1 : real - 1] = True
    return start_logits, end_logits, attention_mask, context_mask


@pytest.mark.parametrize(
    "batch,length,scale,max_len,top_k",
    [
        (8, 64, 1.0, 30, 20),
        (8, 64, 8.0, 30, 20),  # peaked logits, as a trained model gives
        (16, 384, 4.0, 30, 20),
        (4, 12, 2.0, 30, 20),  # fewer context tokens than top_k
        (8, 64, 4.0, 1, 20),  # one-token answers only
        (8, 64, 4.0, 30, 1),
    ],
)
def test_best_spans_matches_the_loop(torch, qa, batch, length, scale, max_len, top_k):
    generator = torch.Generator().manual_seed(batch * length + max_len + top_k)
    for _ in range(20):
        args = _batch(torch, generator, batch, length, scale)
        null_probs, best_probs, starts, ends = qa._best_spans(*args, max_len, top_k)
        for i in range(batch):
            item = [tensor[i] for tensor in args]
            null_prob, best_prob, best_tokens = _reference(torch, *item, max_len, top_k)
            # The batched softmax may differ from the per-item one in the last
            # bit, thus compare the probabilities with a tolerance.
            assert null_probs[i] == pytest.approx(null_prob, rel=1e-6)
            assert best_probs[i] == pytest.approx(best_prob, rel=1e-6, abs=0)
            if best_tokens is not None:
                assert (starts[i], ends[i]) == best_tokens


def test_an_item_with_no_context_has_no_span(torch, qa):
    logits = torch.zeros(1, 6)
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    context_mask = torch.zeros(1, 6, dtype=torch.bool)
    null_probs, best_probs, _, _ = qa._best_spans(
        logits, logits, attention_mask, context_mask, 30, 20
    )
    assert best_probs == [0.0]
    assert null_probs[0] > 0.0

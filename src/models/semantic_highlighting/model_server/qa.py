"""Extractive question answering with timpal0l/mdeberta-v3-base-squad2.

Inference code behind ``highlighter.py``. The model reads a *context* passage and
a *question* and extracts the span of the context that answers it. Because it was
fine-tuned on SQuAD 2.0, it can also decide a question is unanswerable and return
an empty answer. The checkpoint is multilingual, so one deployment serves every
wiki.

Nothing here is specific to that checkpoint: any extractive-QA model exposing
start/end logits and the SQuAD 2.0 ``[CLS]`` abstention works, so swapping
``MODEL_PATH`` is all a different one takes.

We call the model directly (tokenizer + AutoModelForQuestionAnswering) rather
than via the high-level ``pipeline("question-answering")``, because that
pipeline task was removed in transformers 5.x. The classes used here need
transformers 4.56 or later: before that release ``from_pretrained`` takes the
weight precision as ``torch_dtype`` only, and a ``dtype`` keyword reaches the
model constructor and raises ``TypeError``.

``answer_question_batch`` runs a single padded forward pass over many
(question, context) pairs. What it does with the winning token edges lives in
``boundaries.py``, which reads no torch and is therefore tested in CI. That
module keeps a pipeline behaviour the answers depend on: moving an answer edge
out of the middle of a word, which the pipeline called ``align_to_words``.
"""

from typing import Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.models.semantic_highlighting.model_server import boundaries

MODEL_NAME = "timpal0l/mdeberta-v3-base-squad2"

# Cap on the tokenized (question + context) length; contexts are truncated.
# This is the only bound in play -- mDeBERTa's tokenizer reports no
# model_max_length, so nothing else stops a long context reaching the model's
# 512 position buckets.
MAX_LENGTH = 384

# Weight precision, by the name `SH_DTYPE` takes.
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_model(
    model_name: str = MODEL_NAME,
    device: Optional[str] = None,
    dtype: str = "float32",
):
    """Load the tokenizer and model once, on ``device``, and set eval mode.

    Downloads the model on first call (~1.1 GB) and caches it under
    ~/.cache/huggingface, unless ``model_name`` is a local directory. Callers
    should hold on to the returned pair and reuse it for every request rather
    than reloading per call.

    ``device`` defaults to the first GPU when one is visible, else CPU. ROCm
    builds of torch report AMD GPUs through the ``cuda`` API, so this covers
    both CUDA and ROCm without a separate code path.

    ``dtype`` is one of :data:`DTYPES`. Half precision halves the weights and is
    much faster on a GPU with half-precision matrix units; the decoder is
    unaffected either way, because ``answer_question_batch`` casts the logits
    back to float32 before decoding.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        # Offset mapping and sequence_ids() come from the Rust tokenizer; the
        # decoder cannot map answers back to characters without them. This
        # checkpoint ships tokenizer.json and no spm.model, so a directory
        # published without it silently loses the fast path.
        raise RuntimeError(
            f"{model_name} loaded a slow tokenizer, but the highlighter needs "
            "the fast one for offset mapping (is tokenizer.json missing?)"
        )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name, dtype=DTYPES[dtype]
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def highlight(context: str, start: int, end: int) -> str:
    """Wrap the answer span in the context with markers, for a quick visual."""
    return f"{context[:start]}[[ {context[start:end]} ]]{context[end:]}"


def _best_spans(
    start_logits,
    end_logits,
    attention_mask,
    context_mask,
    max_answer_len: int,
    top_k: int,
) -> tuple:
    """Find the best answer token span of each item of a batch.

    All arguments are ``(batch, length)`` tensors. ``attention_mask`` masks
    padding tokens out of the softmax so a padded (batched) item scores
    identically to the same item run on its own. ``context_mask`` is true on the
    tokens of the context, the only valid answer positions.

    Returns four lists with one entry per item: the null probability, the best
    span probability, and the start and end tokens of that span. The best
    probability is 0.0 when no span is valid.

    This is one set of tensor operations for the whole batch. It computes the
    same thing as a loop over the ``top_k`` starts and the ``top_k`` ends of each
    item, without ``top_k * top_k`` Python reads per item.
    """
    neg_inf = torch.finfo(start_logits.dtype).min
    pad = ~attention_mask.bool()
    start_probs = start_logits.masked_fill(pad, neg_inf).softmax(dim=-1)
    end_probs = end_logits.masked_fill(pad, neg_inf).softmax(dim=-1)

    # SQuAD 2.0 abstention: probability on the [CLS] token (index 0) is the
    # model's "no answer in this context" option.
    null_probs = start_probs[:, 0] * end_probs[:, 0]

    # A token outside the context gets -1, which sorts it after every context
    # token. When the context has fewer than k tokens, the top k thus include
    # such tokens, and `valid` below removes them.
    k = min(top_k, start_probs.shape[1])
    outside = ~context_mask
    top_start_probs, top_starts = start_probs.masked_fill(outside, -1.0).topk(k)
    top_end_probs, top_ends = end_probs.masked_fill(outside, -1.0).topk(k)

    # (batch, k, k): the rows are the start candidates, the columns are the end
    # candidates, both in decreasing probability.
    probs = top_start_probs[:, :, None] * top_end_probs[:, None, :]
    lengths = top_ends[:, None, :] - top_starts[:, :, None] + 1
    valid = (
        (lengths >= 1)
        & (lengths <= max_answer_len)
        & (top_start_probs[:, :, None] >= 0)
        & (top_end_probs[:, None, :] >= 0)
    )
    # An invalid pair gets 0.0, and a span with probability 0.0 never wins.
    # `max` returns the first maximum in (start, end) order, which is the pair a
    # loop over the starts and then the ends keeps.
    best_probs, best = probs.masked_fill(~valid, 0.0).flatten(1).max(dim=-1)
    best_starts = top_starts.gather(1, (best // k)[:, None]).squeeze(1)
    best_ends = top_ends.gather(1, (best % k)[:, None]).squeeze(1)
    return (
        null_probs.tolist(),
        best_probs.tolist(),
        best_starts.tolist(),
        best_ends.tolist(),
    )


def _word_ids(enc, i):
    """The word ids of item ``i``, or None from a slow tokenizer."""
    try:
        return enc.word_ids(i)
    except ValueError:
        # Raised by a slow tokenizer, which reports no words. `load_model`
        # rejects one, so this covers a caller that brought its own.
        return None


def answer_question_batch(
    tokenizer,
    model,
    pairs,
    max_answer_len: int = 30,
    top_k: int = 20,
) -> list:
    """Extract the best answer span for each (question, context) pair.

    Runs a *single* padded forward pass over the whole ``pairs`` sequence, so N
    pairs cost one batched inference instead of N. Returns a list of result
    dicts (``answer``/``score``/``start``/``end``) aligned to ``pairs``.
    """
    if not pairs:
        return []
    questions = [q for q, _ in pairs]
    contexts = [c for _, c in pairs]
    enc = tokenizer(
        questions,
        contexts,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation="only_second",  # never truncate the question
        max_length=MAX_LENGTH,
        padding=True,  # pad to the longest item in the batch
    )
    offset_mapping = enc["offset_mapping"]
    attention_mask = enc["attention_mask"]
    inputs = {k: v.to(model.device) for k, v in enc.items() if k != "offset_mapping"}

    with torch.inference_mode():
        out = model(**inputs)

    # Decode on the CPU. The logits are small, and the answer edges must come
    # back to the CPU anyway. Two transfers per batch.
    start_logits = out.start_logits.float().cpu()
    end_logits = out.end_logits.float().cpu()

    sequence_ids = [enc.sequence_ids(i) for i in range(len(pairs))]
    # Only context tokens (sequence id 1) are valid answer positions; this also
    # excludes padding, whose sequence id is None.
    context_mask = torch.tensor(
        [[sid == 1 for sid in ids] for ids in sequence_ids], dtype=torch.bool
    )
    null_probs, best_probs, best_starts, best_ends = _best_spans(
        start_logits,
        end_logits,
        attention_mask,
        context_mask,
        max_answer_len,
        top_k,
    )

    results = []
    for i, context in enumerate(contexts):
        # The null probability is never negative, so this also abstains when no
        # span is valid and the best probability is 0.0.
        if null_probs[i] >= best_probs[i]:
            results.append({"answer": "", "score": null_probs[i], "start": 0, "end": 0})
            continue
        # Read the word boundaries only for an item that answers. Most items
        # abstain.
        offsets = offset_mapping[i]
        token_words = boundaries.word_spans(_word_ids(enc, i), sequence_ids[i], offsets)
        token_start, token_end = best_starts[i], best_ends[i]
        char_start, char_end = boundaries.answer_span(
            context,
            int(offsets[token_start][0]),
            int(offsets[token_end][1]),
            token_words[token_start],
            token_words[token_end],
        )
        results.append(
            {
                "answer": context[char_start:char_end],
                "score": best_probs[i],
                "start": char_start,
                "end": char_end,
            }
        )
    return results

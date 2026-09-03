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


def _decode_answer(
    start_logits,
    end_logits,
    attention_mask,
    sequence_ids,
    offsets,
    token_words,
    context: str,
    max_answer_len: int,
    top_k: int,
) -> dict:
    """Turn one item's start/end logits into an answer dict.

    ``attention_mask`` masks padding tokens out of the softmax so a padded
    (batched) item scores identically to the same item run on its own.

    ``token_words`` holds one entry per token -- the span of the word that token
    belongs to, from :func:`boundaries.word_spans`. It carries the tokenizer's
    word boundaries through to :func:`boundaries.answer_span`, which turns the
    winning token span into the characters a reader sees.
    """
    neg_inf = torch.finfo(start_logits.dtype).min
    pad = ~attention_mask.bool()
    start_probs = start_logits.masked_fill(pad, neg_inf).softmax(dim=-1)
    end_probs = end_logits.masked_fill(pad, neg_inf).softmax(dim=-1)

    # SQuAD 2.0 abstention: probability on the [CLS] token (index 0) is the
    # model's "no answer in this context" option.
    null_prob = (start_probs[0] * end_probs[0]).item()

    # Only context tokens (sequence id 1) are valid answer positions; this also
    # excludes padding, whose sequence id is None.
    context_tokens = [i for i, sid in enumerate(sequence_ids) if sid == 1]
    top_starts = sorted(context_tokens, key=lambda i: start_probs[i], reverse=True)[
        :top_k
    ]
    top_ends = sorted(context_tokens, key=lambda i: end_probs[i], reverse=True)[:top_k]

    best_prob = 0.0
    best_tokens = None  # (start token, end token)
    for s in top_starts:
        for e in top_ends:
            if e < s or (e - s + 1) > max_answer_len:
                continue
            prob = (start_probs[s] * end_probs[e]).item()
            if prob > best_prob:
                best_prob = prob
                best_tokens = (s, e)

    if best_tokens is None or null_prob >= best_prob:
        return {"answer": "", "score": null_prob, "start": 0, "end": 0}

    token_start, token_end = best_tokens
    char_start, char_end = boundaries.answer_span(
        context,
        int(offsets[token_start][0]),
        int(offsets[token_end][1]),
        token_words[token_start],
        token_words[token_end],
    )
    return {
        "answer": context[char_start:char_end],
        "score": best_prob,
        "start": char_start,
        "end": char_end,
    }


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
    try:
        # One call per item, and the only tokenizer feature `word_spans` needs.
        word_ids = [enc.word_ids(i) for i in range(len(pairs))]
    except ValueError:
        # Raised by a slow tokenizer, which reports no words. `load_model`
        # rejects one, so this covers a caller that brought its own.
        word_ids = [None] * len(pairs)
    inputs = {k: v.to(model.device) for k, v in enc.items() if k != "offset_mapping"}

    with torch.inference_mode():
        out = model(**inputs)

    # Decode on the CPU. `_decode_answer` reads individual probabilities as
    # Python scalars (top_k * top_k of them per item), and on a GPU each read is
    # a host-device sync -- far more expensive than the forward pass itself. Two
    # transfers per batch instead.
    start_logits = out.start_logits.float().cpu()
    end_logits = out.end_logits.float().cpu()

    results = []
    for i, context in enumerate(contexts):
        sequence_ids = enc.sequence_ids(i)
        results.append(
            _decode_answer(
                start_logits[i],
                end_logits[i],
                attention_mask[i],
                sequence_ids,
                offset_mapping[i],
                boundaries.word_spans(word_ids[i], sequence_ids, offset_mapping[i]),
                context,
                max_answer_len,
                top_k,
            )
        )
    return results

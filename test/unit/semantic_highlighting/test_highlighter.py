"""Contract tests for the semantic highlighter.

These assert the wire contract the OpenSearch neural-search plugin depends on:
which request shapes are accepted, that batch responses keep input order with
empty lists preserved for abstentions, and that the response body is an object
with a single `highlights` key. An array body, or any wrapper key, silently
leaves every hit unhighlighted instead of failing, so the envelope is pinned
here.

conftest.py stubs torch and transformers before this module is imported: the
model itself is monkeypatched out, so CI does not need a multi-hundred-MB wheel
to test span arithmetic.
"""

import pytest

from src.models.semantic_highlighting.model_server import (
    highlighter as hl_module,
)
from src.models.semantic_highlighting.model_server.config import Settings
from src.models.semantic_highlighting.model_server.highlighter import (
    Highlighter,
)


@pytest.fixture
def hl():
    # Constructed without load(): no model, no download. batch_size is
    # deliberately small so the chunking loop runs more than once below.
    # min_context_tokens is switched off because the contexts here are a few
    # words long and would otherwise all be skipped before reaching the stubbed
    # model; the skip has its own tests, which set it explicitly.
    return Highlighter(Settings(batch_size=2, min_context_tokens=0))


class _FakeTokenizer:
    """Whitespace tokenizer standing in for the SentencePiece one.

    ``run_batch`` only needs a token *count* per context, so the skip can be
    driven deterministically without loading the real 16 MB tokenizer.json --
    and one word here stands for one token, which keeps the tests readable.
    """

    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [text.split() for text in texts]}


class _PerCharTokenizer:
    """One token per character: the floor for CJK, where SentencePiece merges at
    most a couple of characters into a token."""

    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [list(text) for text in texts]}


def _tokens(text, n):
    """Extend `text` to exactly `n` whitespace tokens, leaving offsets alone."""
    return text + " word" * (n - len(text.split()))


def _skipping(**kwargs):
    """A Highlighter with the skip enabled and a tokenizer it can count with."""
    hl = Highlighter(Settings(**kwargs))
    hl.tokenizer = _FakeTokenizer()
    return hl


def _fake_answer_question_batch(tok, model, pairs, max_answer_len=30, top_k=20):
    """Answer "Gustave" wherever the context contains it, else abstain."""
    out = []
    for _question, context in pairs:
        i = context.find("Gustave")
        if i == -1:
            out.append({"answer": "", "score": 0.99, "start": 0, "end": 0})
        else:
            out.append({"answer": "Gustave", "score": 0.9, "start": i, "end": i + 7})
    return out


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    monkeypatch.setattr(hl_module, "answer_question_batch", _fake_answer_question_batch)


# --- extract_pairs -----------------------------------------------------------


def test_extract_pairs_batch_shape(hl):
    pairs = hl.extract_pairs(
        {
            "inputs": [
                {"question": "q1", "context": "c1"},
                {"question": "q2", "context": "c2"},
            ]
        }
    )
    assert pairs == [("q1", "c1"), ("q2", "c2")]


def test_extract_pairs_empty_body_highlights_nothing(hl):
    # Not an error: a 4xx would fail the caller's entire `_search`.
    assert hl.extract_pairs({}) == []


def test_extract_pairs_single_document_shape_highlights_nothing(hl):
    # The shape OpenSearch 3.0-3.2 sent. No longer supported, and deliberately
    # not an error either: such a caller gets unhighlighted hits, not a failed
    # `_search`.
    assert hl.extract_pairs({"question": "q", "context": "c"}) == []


def test_extract_pairs_ignores_unknown_fields(hl):
    pairs = hl.extract_pairs(
        {"inputs": [{"question": "q", "context": "c"}], "future_option": 1}
    )
    assert pairs == [("q", "c")]


def test_extract_pairs_decodes_a_bytes_body(hl):
    # KServe passes the raw bytes through whenever the request's Content-Type is
    # not one of its JSON types (e.g. the header was omitted altogether).
    pairs = hl.extract_pairs(b'{"inputs": [{"question": "q", "context": "c"}]}')
    assert pairs == [("q", "c")]


@pytest.mark.parametrize(
    "body",
    [
        {"inputs": [{"question": "q"}]},  # context missing
        {"inputs": "not a list"},
        [{"question": "q", "context": "c"}],  # array body, not an object
        b'[{"question": "q", "context": "c"}]',  # valid JSON, wrong shape
    ],
)
def test_extract_pairs_rejects_malformed_bodies(hl, body):
    with pytest.raises(ValueError):
        hl.extract_pairs(body)


# --- run_batch ---------------------------------------------------------------


def test_run_batch_preserves_order_across_chunks(hl):
    # 3 pairs with batch_size=2 -> two forward passes; order must survive.
    spans = hl.run_batch(
        [
            ("who?", "see Gustave here"),  # "Gustave" at index 4
            ("who?", "no name here"),  # abstain -> []
            ("who?", "Gustave first"),  # index 0
        ]
    )
    assert spans == [
        [{"start": 4, "end": 11, "answer": "Gustave", "score": 0.9}],
        [],
        [{"start": 0, "end": 7, "answer": "Gustave", "score": 0.9}],
    ]


def test_run_batch_empty_input(hl):
    assert hl.run_batch([]) == []


def test_run_batch_converts_offsets_to_utf16(hl):
    # The emoji shifts the UTF-16 offset one unit past the code-point offset.
    assert hl.run_batch([("who?", "😀 Gustave Eiffel")]) == [
        [{"start": 3, "end": 10, "answer": "Gustave", "score": 0.9}]
    ]


def test_run_batch_applies_min_score():
    # Skip disabled: this is about the score filter, not the passage length.
    hl = Highlighter(Settings(min_score=0.95, min_context_tokens=0))
    assert hl.run_batch([("who?", "see Gustave here")]) == [[]]


# --- short-passage skip -------------------------------------------------------


def test_short_passage_is_not_highlighted():
    hl = _skipping(min_context_tokens=5)
    assert hl.run_batch([("who?", "aa Gustave bb")]) == [[]]


def test_passage_at_the_threshold_is_highlighted():
    # The bound is inclusive: >= min_context_tokens runs, one token less does
    # not. Both contexts hold "Gustave" at index 3.
    hl = _skipping(min_context_tokens=5)
    assert hl.run_batch([("who?", _tokens("aa Gustave bb", 5))]) == [
        [{"start": 3, "end": 10, "answer": "Gustave", "score": 0.9}]
    ]
    assert hl.run_batch([("who?", _tokens("aa Gustave bb", 4))]) == [[]]


def test_skip_preserves_order_in_a_mixed_batch():
    # Skipped slots must stay in place, or every hit after one shifts.
    hl = _skipping(min_context_tokens=5, batch_size=2)
    spans = hl.run_batch(
        [
            ("who?", "aa Gustave bb"),  # 3 tokens -> []
            ("who?", _tokens("aa Gustave bb", 8)),  # index 3
            ("who?", "Gustave short"),  # 2 tokens -> []
            ("who?", _tokens("Gustave first", 8)),  # index 0
            ("who?", _tokens("no name here", 8)),  # long, but abstains -> []
        ]
    )
    assert spans == [
        [],
        [{"start": 3, "end": 10, "answer": "Gustave", "score": 0.9}],
        [],
        [{"start": 0, "end": 7, "answer": "Gustave", "score": 0.9}],
        [],
    ]


def test_skipped_passages_never_reach_the_model(monkeypatch):
    """The forward pass avoided is the whole point, so pin it."""
    seen = []

    def recording(tok, model, pairs, max_answer_len=30, top_k=20):
        seen.extend(ctx for _q, ctx in pairs)
        return _fake_answer_question_batch(tok, model, pairs, max_answer_len, top_k)

    monkeypatch.setattr(hl_module, "answer_question_batch", recording)
    long_ctx = _tokens("aa Gustave bb", 8)
    hl = _skipping(min_context_tokens=5)
    hl.run_batch([("who?", "aa Gustave bb"), ("who?", long_ctx)])
    assert seen == [long_ctx]


def test_all_short_batch_runs_no_inference_at_all(monkeypatch):
    calls = []

    def recording(tok, model, pairs, max_answer_len=30, top_k=20):
        calls.append(pairs)
        return []

    monkeypatch.setattr(hl_module, "answer_question_batch", recording)
    hl = _skipping(min_context_tokens=5)
    assert hl.run_batch([("who?", "tiny"), ("who?", "also tiny")]) == [[], []]
    assert calls == []


def test_a_dense_script_passage_is_not_skipped(monkeypatch):
    """The whole reason for counting tokens rather than characters.

    This paragraph says what a 235-character English one says, in 77 Chinese
    characters, and tokenizes to 55 real tokens. A 150-character cut would drop
    it while highlighting its English twin; a 40-token cut asks the model.
    """
    seen = []

    def recording(tok, model, pairs, max_answer_len=30, top_k=20):
        seen.extend(ctx for _q, ctx in pairs)
        return _fake_answer_question_batch(tok, model, pairs, max_answer_len, top_k)

    monkeypatch.setattr(hl_module, "answer_question_batch", recording)
    zh = (
        "埃菲尔铁塔是位于法国巴黎战神广场的一座锻铁格子塔。它以工程师古斯塔夫·埃菲尔的"
        "名字命名，他的公司在1887年至1889年间为世界博览会设计并建造了这座塔。"
    )
    assert len(zh) < 150  # would have fallen under a character-based cut
    hl = Highlighter(Settings(min_context_tokens=40))
    hl.tokenizer = _PerCharTokenizer()
    hl.run_batch([("who?", zh)])
    assert seen == [zh]  # reached the model instead of being skipped


def test_zero_disables_the_skip_without_a_tokenizer():
    # Highlighter.load() has not run, so self.tokenizer is None: disabling the
    # skip must not go anywhere near it.
    hl = Highlighter(Settings(min_context_tokens=0))
    assert hl.tokenizer is None
    assert hl.run_batch([("who?", "aa Gustave bb")]) == [
        [{"start": 3, "end": 10, "answer": "Gustave", "score": 0.9}]
    ]


def test_default_threshold_is_40_tokens():
    # Pinned: changing the default changes what every deployment highlights.
    assert Settings().min_context_tokens == 40


# --- load and warm-up ---------------------------------------------------------


class _FakeModel:
    device = "cpu"


def test_load_warms_up_at_the_configured_batch_size(monkeypatch):
    """One full-size forward pass has to happen before the model reports ready.

    It must not go through run_batch: the filler would have to clear
    min_context_tokens or the skip would drop it and warm nothing at all.
    """
    seen = []

    def recording(tok, model, pairs, max_answer_len=30, top_k=20):
        seen.append(pairs)
        return [{"answer": "", "score": 0.0, "start": 0, "end": 0} for _ in pairs]

    monkeypatch.setattr(hl_module, "answer_question_batch", recording)
    monkeypatch.setattr(
        hl_module, "load_model", lambda p, dtype=None: ("tokenizer", _FakeModel())
    )

    hl = Highlighter(Settings(batch_size=4, min_context_tokens=40))
    hl.load()

    assert hl.ready
    assert len(seen) == 1, "exactly one warm-up pass"
    assert len(seen[0]) == 4, "sized by SH_BATCH_SIZE"
    # Comfortably past any plausible min_context_tokens, so the skip could not
    # have swallowed it even if the warm-up went through run_batch.
    assert len(seen[0][0][1]) > 1000


def test_load_passes_the_configured_dtype(monkeypatch):
    # fp32 stays the default: half precision shifts scores, and the abstention
    # threshold is sensitive to that, so switching must be deliberate.
    assert Settings().dtype == "float32"

    seen = {}

    def fake_load_model(path, dtype="float32"):
        seen.update(path=path, dtype=dtype)
        return "tokenizer", _FakeModel()

    monkeypatch.setattr(hl_module, "load_model", fake_load_model)
    monkeypatch.setattr(hl_module, "answer_question_batch", lambda *a, **k: [])

    Highlighter(Settings(model_path="/mnt/models/", dtype="bfloat16")).load()
    assert seen == {"path": "/mnt/models/", "dtype": "bfloat16"}


# --- envelope ----------------------------------------------------------------


def test_envelope_batch_is_object_with_list_of_lists(hl):
    body = hl.envelope([[{"start": 3, "end": 8}], []])
    assert isinstance(body, dict)
    assert set(body) == {"highlights"}
    assert body["highlights"] == [[{"start": 3, "end": 8}], []]


def test_envelope_empty_batch(hl):
    assert hl.envelope([]) == {"highlights": []}


# --- end to end through all three stages -------------------------------------


def test_full_pipeline_batch(hl):
    pairs = hl.extract_pairs(
        {
            "inputs": [
                {"question": "who?", "context": "aa Gustave bb"},
                {"question": "who?", "context": "no name here"},
                {"question": "who?", "context": "Gustave first"},
            ]
        }
    )
    assert hl.envelope(hl.run_batch(pairs)) == {
        "highlights": [
            [{"start": 3, "end": 10, "answer": "Gustave", "score": 0.9}],
            [],
            [{"start": 0, "end": 7, "answer": "Gustave", "score": 0.9}],
        ]
    }


def test_full_pipeline_empty_body(hl):
    pairs = hl.extract_pairs({})
    assert hl.envelope(hl.run_batch(pairs)) == {"highlights": []}

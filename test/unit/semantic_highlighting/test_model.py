"""Tests for the KServe shell in model.py.

Only the delegation and the async plumbing live there; the contract itself is
covered by test_highlighter.py. The three stages are driven the way KServe's
dataplane drives them, so a signature drift in a future kserve bump shows up
here rather than in production.

conftest.py stubs torch and transformers before this module is imported.
"""

import pytest
from kserve.errors import InvalidInput

from src.models.semantic_highlighting.model_server import (
    highlighter as hl_module,
)
from src.models.semantic_highlighting.model_server.config import Settings
from src.models.semantic_highlighting.model_server.highlighter import (
    Highlighter,
)
from src.models.semantic_highlighting.model_server.model import (
    SemanticHighlighterModel,
)


def _fake_answer_question_batch(tok, model, pairs, max_answer_len=30, top_k=20):
    out = []
    for _question, context in pairs:
        i = context.find("Gustave")
        if i == -1:
            out.append({"answer": "", "score": 0.99, "start": 0, "end": 0})
        else:
            out.append({"answer": "Gustave", "score": 0.9, "start": i, "end": i + 7})
    return out


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(hl_module, "answer_question_batch", _fake_answer_question_batch)
    # Constructed without load(): no model, no download. min_context_tokens is
    # switched off so these short contexts reach the stubbed model -- the skip
    # is the highlighter's concern and is tested in test_highlighter.py.
    return SemanticHighlighterModel(
        "semantic-highlighter", highlighter=Highlighter(Settings(min_context_tokens=0))
    )


async def _infer(model, body):
    """Drive the three stages the way KServe's dataplane does."""
    return await model.postprocess(await model.predict(await model.preprocess(body)))


@pytest.mark.asyncio
async def test_batch_response_shape_order_and_empty(model):
    body = await _infer(
        model,
        {
            "inputs": [
                {"question": "who?", "context": "aa Gustave bb"},
                {"question": "who?", "context": "no name here"},
                {"question": "who?", "context": "Gustave first"},
            ]
        },
    )
    assert isinstance(body, dict)
    assert set(body) == {"highlights"}
    assert body["highlights"] == [
        [{"start": 3, "end": 10}],
        [],
        [{"start": 0, "end": 7}],
    ]


@pytest.mark.asyncio
async def test_empty_body_returns_empty_highlights(model):
    assert await _infer(model, {}) == {"highlights": []}


@pytest.mark.asyncio
async def test_bytes_body_is_decoded(model):
    body = await _infer(
        model, b'{"inputs": [{"question": "who?", "context": "Gustave"}]}'
    )
    assert body == {"highlights": [[{"start": 0, "end": 7}]]}


@pytest.mark.asyncio
async def test_malformed_body_raises_invalid_input(model):
    # InvalidInput is what KServe turns into a 400 rather than a 500.
    with pytest.raises(InvalidInput):
        await model.preprocess({"inputs": [{"question": "q"}]})


def test_not_ready_before_load(model):
    assert model.ready is False
    assert model.highlighter.ready is False

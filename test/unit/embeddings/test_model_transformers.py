import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from kserve.errors import InferenceError, InvalidInput


class _FakeRow:
    """Minimal row stub so embedding.tolist() works without numpy."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


# Mock GPU-only deps before importing the model.
_torch = types.ModuleType("torch")
_torch.cuda = MagicMock()
_torch.cuda.is_available.return_value = False
_torch.dtype = type("dtype", (), {})  # for type hints only
_torch.bfloat16 = "bfloat16"
_torch.float16 = "float16"
sys.modules["torch"] = _torch

_st = types.ModuleType("sentence_transformers")
_st.SentenceTransformer = MagicMock()
sys.modules["sentence_transformers"] = _st

from src.models.embeddings.model_server.model_transformers import (  # noqa: E402
    EmbeddingModel,
)


def _default_kwargs(**overrides):
    kwargs = dict(
        name="jina-embedding",
        model_path="/mnt/models",
        model_version="",
        dtype="bfloat16",
        attn_implementation="eager",
        default_prompt_name="query",
        truncate_dim=None,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def model():
    m = EmbeddingModel(**_default_kwargs())
    m.model = MagicMock()
    m.ready = True
    return m


class TestPreprocess:
    def test_list_input_unchanged(self, model):
        result = model.preprocess({"input": ["a", "b"]}, None)
        assert result == {"sentences": ["a", "b"], "prompt_name": "query"}

    def test_string_input_wrapped_in_list(self, model):
        result = model.preprocess({"input": "single"}, None)
        assert result == {"sentences": ["single"], "prompt_name": "query"}

    def test_missing_input_raises_invalid_input(self, model):
        with pytest.raises(InvalidInput):
            model.preprocess({}, None)

    def test_prompt_name_override(self, model):
        result = model.preprocess(
            {"input": ["doc text"], "prompt_name": "document"}, None
        )
        assert result == {"sentences": ["doc text"], "prompt_name": "document"}

    def test_empty_prompt_name_becomes_none(self, model):
        result = model.preprocess({"input": ["x"], "prompt_name": ""}, None)
        assert result == {"sentences": ["x"], "prompt_name": None}

    def test_default_prompt_name_from_constructor(self):
        m = EmbeddingModel(**_default_kwargs(default_prompt_name="document"))
        result = m.preprocess({"input": ["x"]}, None)
        assert result["prompt_name"] == "document"


class TestLoad:
    def _make_model(self, **overrides):
        return EmbeddingModel(**_default_kwargs(**overrides))

    def test_default_kwargs_passed_to_sentence_transformer(self):
        with patch(
            "src.models.embeddings.model_server.model_transformers.SentenceTransformer"
        ) as mock_st:
            mock_instance = MagicMock()
            mock_st.return_value = mock_instance

            m = self._make_model()
            m.load()

            kwargs = mock_st.call_args.kwargs
            assert mock_st.call_args.args[0] == "/mnt/models"
            assert kwargs["device"] == "cpu"
            assert kwargs["trust_remote_code"] is True
            assert kwargs["model_kwargs"] == {"dtype": "bfloat16"}
            assert kwargs["config_kwargs"] == {"_attn_implementation": "eager"}
            assert kwargs["truncate_dim"] is None
            mock_instance.eval.assert_called_once()
            assert m.ready is True

    def test_attn_and_truncate_dim_passed_through(self):
        with patch(
            "src.models.embeddings.model_server.model_transformers.SentenceTransformer"
        ) as mock_st:
            m = self._make_model(
                attn_implementation="flash_attention_2",
                truncate_dim=512,
            )
            m.load()

            kwargs = mock_st.call_args.kwargs
            assert kwargs["config_kwargs"] == {
                "_attn_implementation": "flash_attention_2"
            }
            assert kwargs["truncate_dim"] == 512

    def test_sentence_transformer_failure_raises_inference_error(self):
        with patch(
            "src.models.embeddings.model_server.model_transformers.SentenceTransformer",
            side_effect=RuntimeError("boom"),
        ):
            m = self._make_model()
            with pytest.raises(InferenceError):
                m.load()
            assert m.ready is False


class TestPredict:
    def test_returns_openai_format(self, model):
        model.model.encode.return_value = [
            _FakeRow([0.6, 0.8]),
            _FakeRow([0.0, 1.0]),
        ]

        result = model.predict(
            {"sentences": ["text1", "text2"], "prompt_name": "query"}
        )

        model.model.encode.assert_called_once_with(
            sentences=["text1", "text2"],
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        assert result["object"] == "list"
        assert result["model"] == "jina-embedding"
        assert len(result["data"]) == 2

        first = result["data"][0]
        assert first["object"] == "embedding"
        assert first["index"] == 0
        assert first["embedding"] == [0.6, 0.8]

        second = result["data"][1]
        assert second["index"] == 1
        assert second["embedding"] == [0.0, 1.0]

    def test_model_version_used_when_set(self, model):
        model.model_version = "v1.2.3"
        model.model.encode.return_value = [_FakeRow([1.0, 0.0])]

        result = model.predict({"sentences": ["text"], "prompt_name": "query"})

        assert result["model"] == "v1.2.3"

    def test_model_falls_back_to_name_when_version_empty(self, model):
        model.model_version = ""
        model.model.encode.return_value = [_FakeRow([1.0, 0.0])]

        result = model.predict({"sentences": ["text"], "prompt_name": "document"})

        assert result["model"] == "jina-embedding"

    def test_inference_exception_raises_inference_error(self, model):
        model.model.encode.side_effect = RuntimeError("gpu failed")

        with pytest.raises(InferenceError):
            model.predict({"sentences": ["text"], "prompt_name": "query"})

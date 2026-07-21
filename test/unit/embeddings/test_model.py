import math
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from kserve.errors import InferenceError, InvalidInput


def _make_mock_package(name):
    """Create a mock module that acts as a package with __path__."""
    mod = MagicMock()
    mod.__path__ = []
    mod.__name__ = name
    return mod


class _FakeTensor:
    """Minimal tensor stub for L2-normalization checks without real torch."""

    def __init__(self, data):
        self._data = data._data if isinstance(data, _FakeTensor) else data

    def float(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_args):
        flat = self._data
        while isinstance(flat, list) and flat and isinstance(flat[0], list):
            flat = [x for row in flat for x in row]
        return _FakeTensor(flat)

    def tolist(self):
        return self._data

    def __iter__(self):
        for item in self._data:
            yield item if isinstance(item, _FakeTensor) else _FakeTensor(item)


def _fake_tensor(data):
    return data if isinstance(data, _FakeTensor) else _FakeTensor(data)


def _fake_stack(tensors):
    return _FakeTensor(
        [t.tolist() if isinstance(t, _FakeTensor) else list(t) for t in tensors]
    )


def _fake_normalize(tensor, p=2, dim=1):
    rows = tensor.tolist()
    if not rows or not isinstance(rows[0], (list, tuple)):
        rows = [rows]
    out = []
    for row in rows:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0.0 for x in row])
    return _FakeTensor(out)


# Mock GPU-only deps before importing the model.
# Use ModuleType (not MagicMock) so `import torch.nn.functional` resolves correctly.
_torch = types.ModuleType("torch")
_torch.tensor = _fake_tensor
_torch.stack = _fake_stack
_torch.Tensor = _FakeTensor
_nn = types.ModuleType("torch.nn")
_functional = types.ModuleType("torch.nn.functional")
_functional.normalize = _fake_normalize
_nn.functional = _functional
_torch.nn = _nn
sys.modules["torch"] = _torch
sys.modules["torch.nn"] = _nn
sys.modules["torch.nn.functional"] = _functional

sys.modules["vllm"] = _make_mock_package("vllm")
sys.modules["vllm.config"] = _make_mock_package("vllm.config")
sys.modules["vllm.config.pooler"] = MagicMock()

from src.models.embeddings.model_server.model import EmbeddingModel  # noqa: E402


def _default_kwargs(**overrides):
    kwargs = dict(
        name="qwen3-embedding",
        model_path="/mnt/models",
        model_version="",
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        disable_log_stats=False,
        vllm_runner="",
        pooling_type="",
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
        assert result == ["a", "b"]

    def test_string_input_wrapped_in_list(self, model):
        result = model.preprocess({"input": "single"}, None)
        assert result == ["single"]

    def test_missing_input_raises_invalid_input(self, model):
        with pytest.raises(InvalidInput):
            model.preprocess({}, None)


class TestLoad:
    def _make_model(self, **overrides):
        return EmbeddingModel(**_default_kwargs(**overrides))

    def test_default_kwargs_passed_to_llm(self):
        with patch("src.models.embeddings.model_server.model.LLM") as mock_llm:
            m = self._make_model()
            m.load()

            kwargs = mock_llm.call_args.kwargs
            assert kwargs["model"] == "/mnt/models"
            assert kwargs["dtype"] == "float16"
            assert kwargs["trust_remote_code"] is True
            assert kwargs["gpu_memory_utilization"] == 0.9
            assert kwargs["max_model_len"] == 8192
            assert kwargs["max_num_batched_tokens"] == 8192
            assert kwargs["enforce_eager"] is False
            assert kwargs["enable_prefix_caching"] is False
            assert kwargs["served_model_name"] == "qwen3-embedding"
            assert kwargs["disable_log_stats"] is False
            assert "runner" not in kwargs
            assert "pooler_config" not in kwargs
            assert m.ready is True

    def test_vllm_runner_passed_to_llm(self):
        with patch("src.models.embeddings.model_server.model.LLM") as mock_llm:
            m = self._make_model(vllm_runner="pooling")
            m.load()

            assert mock_llm.call_args.kwargs["runner"] == "pooling"

    def test_pooling_type_sets_pooler_config(self):
        # PoolerConfig is imported inside load(); ensure the mock is used.
        pooler_mod = sys.modules["vllm.config.pooler"]
        pooler_mod.PoolerConfig = MagicMock(return_value="fake-pooler-config")

        with patch("src.models.embeddings.model_server.model.LLM") as mock_llm:
            m = self._make_model(pooling_type="LAST")
            m.load()

            pooler_mod.PoolerConfig.assert_called_once_with(seq_pooling_type="LAST")
            assert mock_llm.call_args.kwargs["pooler_config"] == "fake-pooler-config"

    def test_llm_failure_raises_inference_error(self):
        with patch(
            "src.models.embeddings.model_server.model.LLM",
            side_effect=RuntimeError("boom"),
        ):
            m = self._make_model()
            with pytest.raises(InferenceError):
                m.load()
            assert m.ready is False


class TestPredict:
    def _embed_output(self, embedding):
        output = MagicMock()
        output.outputs.embedding = embedding
        return output

    def _encode_output(self, data):
        output = MagicMock()
        output.outputs.data = _FakeTensor(data)
        return output

    def test_embed_path_returns_openai_format_and_normalizes(self, model):
        model.model.embed.return_value = [
            self._embed_output([3.0, 4.0]),
            self._embed_output([0.0, 5.0]),
        ]

        result = model.predict(["text1", "text2"])

        model.model.embed.assert_called_once_with(["text1", "text2"])
        assert result["object"] == "list"
        assert result["model"] == "qwen3-embedding"
        assert len(result["data"]) == 2

        first = result["data"][0]
        assert first["object"] == "embedding"
        assert first["index"] == 0
        assert first["embedding"] == pytest.approx([0.6, 0.8])

        second = result["data"][1]
        assert second["index"] == 1
        assert second["embedding"] == pytest.approx([0.0, 1.0])

    def test_model_version_used_when_set(self, model):
        model.model_version = "v1.2.3"
        model.model.embed.return_value = [self._embed_output([1.0, 0.0])]

        result = model.predict(["text"])

        assert result["model"] == "v1.2.3"

    def test_model_falls_back_to_name_when_version_empty(self, model):
        model.model_version = ""
        model.model.embed.return_value = [self._embed_output([1.0, 0.0])]

        result = model.predict(["text"])

        assert result["model"] == "qwen3-embedding"

    def test_pooling_path_uses_encode(self, model):
        model.vllm_runner = "pooling"
        model.model.encode.return_value = [
            self._encode_output([3.0, 4.0]),
        ]
        # Ensure hasattr(self.model, "encode") is True (MagicMock has it).
        model.model.embed = MagicMock()

        result = model.predict(["text"])

        model.model.encode.assert_called_once_with(["text"], pooling_task="embed")
        model.model.embed.assert_not_called()
        assert result["object"] == "list"
        assert result["data"][0]["object"] == "embedding"
        assert result["data"][0]["index"] == 0
        assert result["data"][0]["embedding"] == pytest.approx([0.6, 0.8])

    def test_inference_exception_raises_inference_error(self, model):
        model.model.embed.side_effect = RuntimeError("gpu failed")

        with pytest.raises(InferenceError):
            model.predict(["text"])

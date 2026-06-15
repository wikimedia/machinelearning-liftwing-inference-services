import sys
from unittest.mock import MagicMock, patch

import pytest
from kserve.errors import InvalidInput


def _make_mock_package(name):
    """Create a mock module that acts as a package with __path__."""
    mod = MagicMock()
    mod.__path__ = []
    mod.__name__ = name
    return mod


# Mock GPU-only deps before importing the model.
# Package-like modules need __path__ so Python's import system can traverse them.
sys.modules["vllm"] = _make_mock_package("vllm")
sys.modules["vllm.engine"] = MagicMock()
sys.modules["vllm.engine.arg_utils"] = MagicMock()
sys.modules["vllm.engine.async_llm_engine"] = MagicMock()
sys.modules["vllm.entrypoints"] = _make_mock_package("vllm.entrypoints")
sys.modules["vllm.entrypoints.openai"] = _make_mock_package("vllm.entrypoints.openai")
sys.modules["vllm.entrypoints.openai.chat_completion"] = _make_mock_package(
    "vllm.entrypoints.openai.chat_completion"
)
sys.modules["vllm.entrypoints.openai.chat_completion.protocol"] = MagicMock()
sys.modules["vllm.entrypoints.openai.completion"] = _make_mock_package(
    "vllm.entrypoints.openai.completion"
)
sys.modules["vllm.entrypoints.openai.completion.protocol"] = MagicMock()
sys.modules["vllm.entrypoints.openai.engine"] = _make_mock_package(
    "vllm.entrypoints.openai.engine"
)
sys.modules["vllm.entrypoints.openai.engine.protocol"] = MagicMock()
sys.modules["vllm.entrypoints.pooling"] = _make_mock_package("vllm.entrypoints.pooling")
sys.modules["vllm.entrypoints.pooling.embed"] = _make_mock_package(
    "vllm.entrypoints.pooling.embed"
)
sys.modules["vllm.entrypoints.pooling.embed.protocol"] = MagicMock()
sys.modules["vllm.entrypoints.pooling.scoring"] = _make_mock_package(
    "vllm.entrypoints.pooling.scoring"
)
sys.modules["vllm.entrypoints.pooling.scoring.protocol"] = MagicMock()
sys.modules["vllm.entrypoints.chat_utils"] = MagicMock()
sys.modules["vllm.outputs"] = MagicMock()

from src.models.qwen36.model_server.model import Qwen36Model  # noqa: E402


@pytest.fixture
def model():
    with patch.object(Qwen36Model, "__init__", return_value=None):
        m = Qwen36Model()
        m.name = "qwen36-27b"
        m.tokenizer = MagicMock()
        m.tokenizer.encode.return_value = [100, 200, 300, 400, 500]
        m.tokenizer.apply_chat_template.return_value = "mocked template output"
        yield m


class TestBuildMessages:
    def test_builds_messages_with_system_and_user(self, model):
        messages = model._build_messages("Hello", system="You are helpful.")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_no_system_message_when_not_provided(self, model):
        messages = model._build_messages("Hi")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi"


class TestPreprocessInputValidation:
    def test_missing_prompt_raises_error(self, model):
        with pytest.raises(InvalidInput):
            model.preprocess({}, None)

    def test_non_string_prompt_raises_error(self, model):
        with pytest.raises(InvalidInput):
            model.preprocess({"prompt": 123}, None)

    def test_empty_prompt_raises_error(self, model):
        with pytest.raises(InvalidInput):
            model.preprocess({"prompt": ""}, None)


class TestPreprocessReasoning:
    def test_reasoning_defaults_to_false(self, model):
        model.preprocess({"prompt": "Hello"}, None)
        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["enable_thinking"] == 0

    def test_reasoning_true_enables_thinking(self, model):
        model.preprocess({"prompt": "Hello", "reasoning": True}, None)
        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["enable_thinking"] == 1

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, 1),
            (False, 0),
            ("true", 1),
            ("false", 0),
            ("yes", 1),
            ("no", 0),
            ("1", 1),
            ("0", 0),
        ],
    )
    def test_reasoning_values(self, model, value, expected):
        model.preprocess({"prompt": "Hello", "reasoning": value}, None)
        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["enable_thinking"] == expected

    def test_invalid_reasoning_raises_value_error(self, model):
        with pytest.raises(ValueError):
            model.preprocess({"prompt": "Hello", "reasoning": "invalid"}, None)


class TestPreprocessSamplingParams:
    """Tests for default sampling parameters in preprocess."""

    def test_defaults_are_instruct_mode(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        model.preprocess({"prompt": "Hello"}, None)
        call_kwargs = SamplingParams.call_args.kwargs
        assert call_kwargs["max_tokens"] == 32768
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["top_k"] == 20
        assert call_kwargs["presence_penalty"] == 1.5
        assert call_kwargs["repetition_penalty"] == 1.0

    def test_thinking_mode_switches_defaults(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        model.preprocess({"prompt": "Hello", "reasoning": True}, None)
        call_kwargs = SamplingParams.call_args.kwargs
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["presence_penalty"] == 0.0

    def test_reasoning_can_override_defaults(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        model.preprocess(
            {"prompt": "Hello", "reasoning": True, "temperature": 0.5},
            None,
        )
        call_kwargs = SamplingParams.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_custom_sampling_params(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        model.preprocess(
            {
                "prompt": "Hello",
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 50,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.2,
            },
            None,
        )
        call_kwargs = SamplingParams.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["presence_penalty"] == 1.5
        assert call_kwargs["repetition_penalty"] == 1.2


class TestApplyChatTemplate:
    def test_returns_chat_prompt(self, model):
        request = MagicMock()
        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Hello"
        request.messages = [user_msg]

        result = model.apply_chat_template(request)

        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert not kwargs["tokenize"]
        assert kwargs["add_generation_prompt"]
        assert not kwargs["enable_thinking"]
        assert result.prompt == "mocked template output"
        assert result.response_role == "assistant"

    def test_falls_back_without_enable_thinking(self, model):
        def _apply_side_effect(*args, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword")
            return "fallback template"

        model.tokenizer.apply_chat_template.side_effect = _apply_side_effect
        request = MagicMock()
        request.messages = [MagicMock(role="user", content="Hi")]

        result = model.apply_chat_template(request)

        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert "enable_thinking" not in kwargs
        assert result.prompt == "fallback template"

    def test_converts_messages_to_dicts(self, model):
        # vLLM ChatCompletion messages support dict() conversion natively
        msg = {"role": "system", "content": "You are helpful."}
        request = MagicMock()
        request.messages = [msg]

        model.apply_chat_template(request)

        call_args = model.tokenizer.apply_chat_template.call_args.args
        messages_arg = call_args[0]
        assert isinstance(messages_arg[0], dict)
        assert messages_arg[0]["role"] == "system"
        assert messages_arg[0]["content"] == "You are helpful."


class TestCreateCompletion:
    @pytest.fixture
    def mock_output(self):
        """Create a mock RequestOutput for the async generator."""
        output = MagicMock()
        output.index = 0
        output.text = "Hello, how are you?"
        output.token_ids = [100, 200, 300, 400, 500]
        output.finish_reason = "stop"
        request_output = MagicMock()
        request_output.prompt_token_ids = [1, 2, 3]
        request_output.outputs = [output]
        return request_output

    async def _async_gen(self, items):
        """Helper to create an async generator from a list."""
        for item in items:
            yield item

    def test_non_streaming_returns_completion(self, model, mock_output):
        # Use real class to capture args for Completion and friends
        class _FakeType:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        import src.models.qwen36.model_server.model as model_module

        with patch.multiple(
            model_module,
            Completion=_FakeType,
            CompletionChoice=_FakeType,
            UsageInfo=_FakeType,
        ):
            model.model = MagicMock()
            gen = self._async_gen([mock_output])
            model.model.generate.return_value = gen

            request = MagicMock()
            request.prompt = "Hello"
            request.max_tokens = None
            request.temperature = None
            request.top_p = None
            request.top_k = None
            request.presence_penalty = None
            request.repetition_penalty = None
            request.stream = False
            request.request_id = None
            request.model = "test-model"

            import asyncio

            result = asyncio.run(model.create_completion(request))

            assert result.model == "test-model"
            assert result.choices[0].text == "Hello, how are you?"
            assert result.choices[0].finish_reason == "stop"
            assert result.choices[0].index == 0
            assert result.usage.prompt_tokens == 3
            assert result.usage.completion_tokens == 5
            assert result.usage.total_tokens == 8

    def test_defaults_are_thinking_mode(self, model, mock_output):
        from src.models.qwen36.model_server.model import SamplingParams

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([mock_output])

        request = MagicMock()
        request.prompt = "Hello"
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None
        request.stream = False
        request.request_id = None
        request.model = "test-model"

        import asyncio

        asyncio.run(model.create_completion(request))

        kwargs = SamplingParams.call_args.kwargs
        assert kwargs["max_tokens"] == 32768
        assert kwargs["temperature"] == 1.0
        assert kwargs["top_p"] == 0.95
        assert kwargs["top_k"] == 20
        assert kwargs["presence_penalty"] == 0.0
        assert kwargs["repetition_penalty"] == 1.0

    def test_custom_params_override_defaults(self, model, mock_output):
        from src.models.qwen36.model_server.model import SamplingParams

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([mock_output])

        request = MagicMock()
        request.prompt = "Hello"
        request.max_tokens = 100
        request.temperature = 0.5
        request.top_p = 0.9
        request.top_k = 50
        request.presence_penalty = 0.5
        request.repetition_penalty = 1.2
        request.stream = False
        request.request_id = None
        request.model = "test-model"

        import asyncio

        asyncio.run(model.create_completion(request))

        kwargs = SamplingParams.call_args.kwargs
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 50
        assert kwargs["presence_penalty"] == 0.5
        assert kwargs["repetition_penalty"] == 1.2

    def test_streaming_returns_async_generator(self, model, mock_output):
        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([mock_output])

        request = MagicMock()
        request.prompt = "Hello"
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None
        request.stream = True
        request.request_id = None
        request.model = "test-model"

        import asyncio

        result = asyncio.run(model.create_completion(request))

        assert hasattr(result, "__aiter__")

    def test_handles_prompt_list(self, model, mock_output):
        from src.models.qwen36.model_server.model import SamplingParams

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([mock_output])
        model.tokenizer.decode.return_value = "decoded prompt"

        request = MagicMock()
        request.prompt = [100, 200, 300]
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None
        request.stream = False
        request.request_id = None
        request.model = "test-model"

        import asyncio

        asyncio.run(model.create_completion(request))

        model.tokenizer.decode.assert_called_once_with([100, 200, 300])
        assert SamplingParams.call_args is not None


class TestStreamCompletion:
    async def _request_outputs(self):
        """Simulate a streaming sequence: two intermediate chunks + final."""
        out1 = MagicMock()
        out1.index = 0
        out1.text = "Hello"
        out1.token_ids = [100, 200]
        out1.finish_reason = None

        out2 = MagicMock()
        out2.index = 0
        out2.text = "Hello world"
        out2.token_ids = [100, 200, 300, 400]
        out2.finish_reason = None

        out3 = MagicMock()
        out3.index = 0
        out3.text = "Hello world!"
        out3.token_ids = [100, 200, 300, 400, 500]
        out3.finish_reason = "stop"

        req1 = MagicMock()
        req1.prompt_token_ids = [1, 2, 3]
        req1.outputs = [out1]

        req2 = MagicMock()
        req2.prompt_token_ids = [1, 2, 3]
        req2.outputs = [out2]

        req3 = MagicMock()
        req3.prompt_token_ids = [1, 2, 3]
        req3.outputs = [out3]

        for req in [req1, req2, req3]:
            yield req

    def test_yields_delta_text(self, model):
        # Use real classes so model_dump_json produces valid JSON
        class _FakeChoice:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class _FakeCompletionChunk:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def model_dump_json(self):
                import json as _json

                def _convert(obj):
                    if isinstance(obj, (_FakeCompletionChunk, _FakeChoice)):
                        d = {}
                        for k, v in obj.__dict__.items():
                            d[k] = _convert(v)
                        return d
                    if isinstance(obj, list):
                        return [_convert(i) for i in obj]
                    return obj

                return _json.dumps(_convert(self))

        import src.models.qwen36.model_server.model as model_module

        with patch.multiple(
            model_module,
            CompletionChunk=_FakeCompletionChunk,
            CompletionChunkChoice=_FakeChoice,
            UsageInfo=_FakeChoice,
        ):

            async def _collect():
                chunks = []
                async for chunk in model._stream_completion(
                    self._request_outputs(), "req-1", 1000, "test-model"
                ):
                    chunks.append(chunk)
                return chunks

            import asyncio
            import json

            chunks = asyncio.run(_collect())

            # 3 SSE chunks + 1 [DONE]
            assert len(chunks) == 4
            assert chunks[3] == "data: [DONE]\n\n"

            # Parse SSE data to verify delta text computation
            def parse_sse(chunk):
                assert chunk.startswith("data: ")
                return json.loads(chunk[len("data: ") :])

            c1 = parse_sse(chunks[0])
            assert c1["choices"][0]["text"] == "Hello"
            assert c1["choices"][0]["finish_reason"] is None
            assert c1["usage"] is None

            c2 = parse_sse(chunks[1])
            assert c2["choices"][0]["text"] == " world"

            c3 = parse_sse(chunks[2])
            assert c3["choices"][0]["text"] == "!"
            assert c3["choices"][0]["finish_reason"] == "stop"
            assert c3["usage"]["prompt_tokens"] == 3
            assert c3["usage"]["completion_tokens"] == 5
            assert c3["usage"]["total_tokens"] == 8

    def test_stream_error_raises_inference_error(self, model):
        from kserve.errors import InferenceError

        async def _error_gen():
            out = MagicMock()
            out.index = 0
            out.text = "Hello"
            out.token_ids = [100]
            req = MagicMock()
            req.prompt_token_ids = [1]
            req.outputs = [out]
            yield req
            raise RuntimeError("GPU error")

        async def _collect():
            chunks = []
            async for chunk in model._stream_completion(
                _error_gen(), "req-1", 1000, "test-model"
            ):
                chunks.append(chunk)
            return chunks

        import asyncio

        with pytest.raises(InferenceError):
            asyncio.run(_collect())

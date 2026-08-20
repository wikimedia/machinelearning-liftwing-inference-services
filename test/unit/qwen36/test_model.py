import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

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
sys.modules["vllm.reasoning"] = _make_mock_package("vllm.reasoning")
sys.modules["vllm.sampling_params"] = MagicMock()
sys.modules["vllm.tool_parsers"] = _make_mock_package("vllm.tool_parsers")

from src.models.qwen36.model_server.model import (  # noqa: E402
    RAW_COMPLETIONS_DEFAULTS,
    PerRequestOptions,
    Qwen36Model,  # noqa: E402
)


class _FakeType:
    """Minimal type that captures constructor kwargs as attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump_json(self):
        """Serialize like a pydantic model, keeping None-valued fields."""
        import json as _json

        def _convert(obj):
            if isinstance(obj, _FakeType):
                return {k: _convert(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj

        return _json.dumps(_convert(self))

    def model_dump(self, exclude_none=False):
        """Serialize like pydantic's model_dump; exclude None when asked."""

        def _convert(obj):
            if isinstance(obj, _FakeType):
                result = {}
                for k, v in obj.__dict__.items():
                    if exclude_none and v is None:
                        continue
                    result[k] = _convert(v)
                return result
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj

        return _convert(self)


def _completion_request_mock(**overrides):
    """A completion-request mock with the attributes create_completion branches on.

    MagicMock defaults every attribute to truthy, so an unpinned mock would
    silently take the ``request.stream`` branch and return a generator where
    a Completion is expected.
    """
    return MagicMock(stream=False, request_id=None, prompt="test prompt", **overrides)


@contextmanager
def _patched_chat_types(model_module):
    """Patch all OpenAI protocol types used by the non-streaming chat path."""
    with patch.multiple(
        model_module,
        Completion=_FakeType,
        CompletionChoice=_FakeType,
        UsageInfo=_FakeType,
        CompletionRequest=_FakeType,
        ChatPrompt=_FakeType,
        ChatCompletion=_FakeType,
        ChatCompletionChoice=_FakeType,
        ChatMessage=_FakeType,
        ChoiceDelta=_FakeType,
        ChunkChoice=_FakeType,
    ):
        yield


@contextmanager
def _patched_stream_types(model_module):
    """Patch all OpenAI protocol types used by the streaming chat path."""
    with patch.multiple(
        model_module,
        ChatCompletionChunk=_FakeType,
        ChunkChoice=_FakeType,
        ChoiceDelta=_FakeType,
        UsageInfo=_FakeType,
        CompletionRequest=_FakeType,
    ):
        yield


@pytest.fixture
def model():
    with patch.object(Qwen36Model, "__init__", return_value=None):
        m = Qwen36Model()
        m.name = "qwen36-27b"
        m.tool_calling_enabled = False
        m.tokenizer = MagicMock()
        m.tokenizer.encode.return_value = [100, 200, 300, 400, 500]
        m.tokenizer.apply_chat_template.return_value = "mocked template output"
        yield m


class TestLoad:
    def _make_model(self, **overrides):
        kwargs = dict(
            name="qwen36-27b",
            model_path="/mnt/models",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=32768,
            tensor_parallel_size=1,
            dtype="auto",
            language_model_only_flag=True,
            skip_mm_profiling_flag=True,
        )
        kwargs.update(overrides)
        return Qwen36Model(**kwargs)

    def test_kv_cache_dtype_defaults_to_auto(self):
        from src.models.qwen36.model_server.model import AsyncEngineArgs

        self._make_model().load()
        assert AsyncEngineArgs.call_args.kwargs["kv_cache_dtype"] == "auto"

    def test_kv_cache_dtype_passed_to_engine_args(self):
        from src.models.qwen36.model_server.model import AsyncEngineArgs

        self._make_model(kv_cache_dtype="fp8").load()
        assert AsyncEngineArgs.call_args.kwargs["kv_cache_dtype"] == "fp8"

    def test_reasoning_parser_initialized_with_tokenizer(self):
        import src.models.qwen36.model_server.model as model_module

        parser_cls = MagicMock()
        parser_instance = MagicMock()
        parser_cls.return_value = parser_instance

        with patch.object(
            model_module.ReasoningParserManager,
            "get_reasoning_parser",
            return_value=parser_cls,
        ) as mock_get:
            m = self._make_model()
            m.load()

            mock_get.assert_called_once_with("qwen3")
            parser_cls.assert_called_once_with(m.tokenizer)
            assert m.reasoning_parser is parser_instance

    def test_tool_parser_smoke_constructed_at_load(self):
        import src.models.qwen36.model_server.model as model_module

        parser_cls = MagicMock()

        with patch.object(
            model_module.ToolParserManager,
            "get_tool_parser",
            return_value=parser_cls,
        ) as mock_get:
            m = self._make_model()
            m.load()

            mock_get.assert_called_once_with("hermes")
            assert m.tool_parser_cls is parser_cls
            parser_cls.assert_called_once_with(m.tokenizer)


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


class TestPreprocessThinkingTokenBudget:
    """Tests for thinking_token_budget in preprocess."""

    def test_budget_present(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        model.preprocess({"prompt": "Hello", "thinking_token_budget": 100}, None)
        assert SamplingParams.call_args.kwargs["thinking_token_budget"] == 100

    def test_budget_absent_is_none(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        model.preprocess({"prompt": "Hello"}, None)
        assert SamplingParams.call_args.kwargs["thinking_token_budget"] is None

    def test_budget_string_coerced(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        model.preprocess({"prompt": "Hello", "thinking_token_budget": "100"}, None)
        assert SamplingParams.call_args.kwargs["thinking_token_budget"] == 100

    def test_budget_garbage_raises_invalid_input(self, model):
        with pytest.raises(InvalidInput, match="must be an integer"):
            model.preprocess(
                {"prompt": "Hello", "thinking_token_budget": "not_a_number"},
                None,
            )

    def test_budget_zero_raises_invalid_input(self, model):
        with pytest.raises(InvalidInput, match="must be >= 1"):
            model.preprocess({"prompt": "Hello", "thinking_token_budget": 0}, None)

    def test_budget_negative_raises_invalid_input(self, model):
        with pytest.raises(InvalidInput, match="must be >= 1"):
            model.preprocess({"prompt": "Hello", "thinking_token_budget": -5}, None)


class TestResolveEnableThinking:
    """Tests for _resolve_enable_thinking."""

    def test_no_chat_template_kwargs_defaults_false(self, model):
        request = MagicMock(spec=[])
        result = model._resolve_enable_thinking(request)
        assert result is False

    def test_empty_kwargs_defaults_false(self, model):
        request = MagicMock(chat_template_kwargs={})
        result = model._resolve_enable_thinking(request)
        assert result is False

    def test_enable_thinking_true(self, model):
        request = MagicMock(chat_template_kwargs={"enable_thinking": True})
        result = model._resolve_enable_thinking(request)
        assert result is True

    def test_enable_thinking_false(self, model):
        request = MagicMock(chat_template_kwargs={"enable_thinking": False})
        result = model._resolve_enable_thinking(request)
        assert result is False

    def test_kwargs_without_enable_thinking_key(self, model):
        request = MagicMock(chat_template_kwargs={"other_key": "value"})
        result = model._resolve_enable_thinking(request)
        assert result is False

    def test_kwargs_is_none(self, model):
        request = MagicMock(chat_template_kwargs=None)
        result = model._resolve_enable_thinking(request)
        assert result is False

    def test_kwargs_is_string_does_not_crash(self, model):
        request = MagicMock(chat_template_kwargs="garbage")
        result = model._resolve_enable_thinking(request)
        assert result is False

    def test_kwargs_is_list_does_not_crash(self, model):
        request = MagicMock(chat_template_kwargs=["garbage"])
        result = model._resolve_enable_thinking(request)
        assert result is False

    def test_string_true_coerces_to_true(self, model):
        request = MagicMock(chat_template_kwargs={"enable_thinking": "true"})
        result = model._resolve_enable_thinking(request)
        assert result is True

    def test_string_false_coerces_to_false(self, model):
        request = MagicMock(chat_template_kwargs={"enable_thinking": "false"})
        result = model._resolve_enable_thinking(request)
        assert result is False

    def test_string_yes_coerces_to_true(self, model):
        request = MagicMock(chat_template_kwargs={"enable_thinking": "yes"})
        result = model._resolve_enable_thinking(request)
        assert result is True

    def test_string_no_coerces_to_false(self, model):
        request = MagicMock(chat_template_kwargs={"enable_thinking": "no"})
        result = model._resolve_enable_thinking(request)
        assert result is False

    def test_garbage_string_raises_invalid_input(self, model):
        with pytest.raises(InvalidInput, match="enable_thinking must be a boolean"):
            model._resolve_enable_thinking(
                MagicMock(chat_template_kwargs={"enable_thinking": "banana"})
            )


class TestBuildSamplingParamsDefaultsByMode:
    """Tests for _build_sampling_params_from_request thinking parameter."""

    def test_instruct_defaults(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        request = MagicMock()
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None

        model._build_sampling_params_from_request(
            request, options=PerRequestOptions(enable_thinking=False)
        )
        kwargs = SamplingParams.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.8
        assert kwargs["presence_penalty"] == 1.5

    def test_thinking_defaults(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        request = MagicMock()
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None

        model._build_sampling_params_from_request(
            request, options=PerRequestOptions(enable_thinking=True)
        )
        kwargs = SamplingParams.call_args.kwargs
        assert kwargs["temperature"] == 1.0
        assert kwargs["top_p"] == 0.95
        assert kwargs["presence_penalty"] == 0.0

    def test_user_values_override_defaults(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        request = MagicMock()
        request.max_tokens = None
        request.temperature = 0.3
        request.top_p = 0.5
        request.top_k = None
        request.presence_penalty = 0.2
        request.repetition_penalty = None

        model._build_sampling_params_from_request(
            request, options=PerRequestOptions(enable_thinking=False)
        )
        kwargs = SamplingParams.call_args.kwargs
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.5
        assert kwargs["presence_penalty"] == 0.2

    def test_default_is_thinking(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        request = MagicMock()
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None

        model._build_sampling_params_from_request(
            request, options=RAW_COMPLETIONS_DEFAULTS
        )
        kwargs = SamplingParams.call_args.kwargs
        assert kwargs["temperature"] == 1.0

    def test_budget_passed_through(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        request = MagicMock()
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None

        model._build_sampling_params_from_request(
            request,
            options=PerRequestOptions(enable_thinking=True, thinking_token_budget=200),
        )
        assert SamplingParams.call_args.kwargs["thinking_token_budget"] == 200

    def test_budget_none_by_default(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        request = MagicMock()
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None

        model._build_sampling_params_from_request(request, options=PerRequestOptions())
        assert SamplingParams.call_args.kwargs["thinking_token_budget"] is None

    def test_skip_special_tokens_passed_through_when_set(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        request = MagicMock()
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None

        model._build_sampling_params_from_request(
            request, options=PerRequestOptions(skip_special_tokens=False)
        )
        assert SamplingParams.call_args.kwargs["skip_special_tokens"] is False

    def test_skip_special_tokens_absent_by_default(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        request = MagicMock()
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None

        model._build_sampling_params_from_request(request, options=PerRequestOptions())
        assert "skip_special_tokens" not in SamplingParams.call_args.kwargs


class TestPerRequestOptions:
    """Tests for the PerRequestOptions dataclass and its default constant."""

    def test_dataclass_defaults(self):
        options = PerRequestOptions()
        assert options.enable_thinking is False
        assert options.thinking_token_budget is None
        assert options.structured_outputs is None
        assert options.skip_special_tokens is None

    def test_raw_completions_defaults_keep_thinking(self):
        assert RAW_COMPLETIONS_DEFAULTS.enable_thinking is True
        assert RAW_COMPLETIONS_DEFAULTS.thinking_token_budget is None
        assert RAW_COMPLETIONS_DEFAULTS.structured_outputs is None
        assert RAW_COMPLETIONS_DEFAULTS.skip_special_tokens is None


class TestApplyChatTemplate:
    def test_returns_chat_prompt(self, model):
        request = MagicMock()
        request.tools = None
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
        request.tools = None
        request.messages = [MagicMock(role="user", content="Hi")]

        result = model.apply_chat_template(request)

        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert "enable_thinking" not in kwargs
        assert result.prompt == "fallback template"

    def test_converts_messages_to_dicts(self, model):
        # vLLM ChatCompletion messages support dict() conversion natively
        msg = {"role": "system", "content": "You are helpful."}
        request = MagicMock()
        request.tools = None
        request.messages = [msg]

        model.apply_chat_template(request)

        call_args = model.tokenizer.apply_chat_template.call_args.args
        messages_arg = call_args[0]
        assert isinstance(messages_arg[0], dict)
        assert messages_arg[0]["role"] == "system"
        assert messages_arg[0]["content"] == "You are helpful."

    def test_enable_thinking_true_passed_to_template(self, model):
        request = MagicMock()
        request.tools = None
        request.messages = [MagicMock(role="user", content="Hi")]

        model.apply_chat_template(request, enable_thinking=True)

        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["enable_thinking"] is True

    def test_enable_thinking_defaults_false(self, model):
        request = MagicMock()
        request.tools = None
        request.messages = [MagicMock(role="user", content="Hi")]

        model.apply_chat_template(request)

        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["enable_thinking"] is False

    def test_tool_choice_auto_renders_tools(self, model):
        model.tool_calling_enabled = True
        request = MagicMock()
        request.tools = [{"type": "function", "function": {}}]
        request.messages = [MagicMock(role="user", content="Hi")]

        model.apply_chat_template(request, tool_choice="auto")

        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["tools"] == [{"type": "function", "function": {}}]

    def test_tool_choice_none_does_not_render_tools(self, model):
        model.tool_calling_enabled = True
        request = MagicMock()
        request.tools = [{"type": "function", "function": {}}]
        request.messages = [MagicMock(role="user", content="Hi")]

        model.apply_chat_template(request, tool_choice="none")

        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert "tools" not in kwargs

    def test_tool_choice_none_without_tools_is_noop(self, model):
        model.tool_calling_enabled = False
        request = MagicMock()
        request.tools = None
        request.messages = [MagicMock(role="user", content="Hi")]

        model.apply_chat_template(request, tool_choice="none")

        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        assert "tools" not in kwargs


class TestToolCallsToOpenai:
    def test_maps_function_name_and_arguments(self, model):
        tool_call = _FakeType(
            id="call_abc12345",
            type="function",
            function=_FakeType(name="get_weather", arguments='{"city": "Kampala"}'),
        )
        result = model._tool_calls_to_openai([tool_call])
        assert result == [
            {
                "id": "call_abc12345",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Kampala"}',
                },
            }
        ]

    def test_generates_id_when_absent(self, model):
        tool_call = _FakeType(
            function=_FakeType(name="get_time", arguments='{"timezone": "UTC"}')
        )
        result = model._tool_calls_to_openai([tool_call])
        assert len(result) == 1
        assert result[0]["id"].startswith("call_")
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_time"


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

    def test_stream_error_yields_sse_error_and_done(self, model):
        import json

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

        chunks = asyncio.run(_collect())
        # First chunk is the successful token, then error SSE, then [DONE]
        assert len(chunks) >= 2
        # Last chunk should be [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"
        # Second-to-last should be the error payload
        error_chunk = json.loads(chunks[-2][len("data: ") :])
        assert error_chunk["error"]["message"] == "GPU error"
        assert error_chunk["error"]["type"] == "server_error"


class TestReasoningExtractionNonStreaming:
    """Tests for reasoning trace split in the non-streaming chat path."""

    @pytest.fixture
    def model_with_parser(self, model):
        """Attach a stubbed reasoning parser to the model."""
        mock_parser = MagicMock()
        mock_parser.extract_reasoning.return_value = (
            "think trace",
            "final answer",
        )
        model.reasoning_parser = mock_parser
        return model

    @staticmethod
    async def _async_gen(items):
        for item in items:
            yield item

    @staticmethod
    def _make_request(thinking=True, tools=None):
        request = MagicMock()
        request.n = 1
        request.stream = False
        request.tools = tools
        request.tool_choice = None
        request.chat_template_kwargs = {"enable_thinking": True} if thinking else {}
        return request

    def test_reasoning_extracted_when_thinking_enabled(self, model_with_parser):
        model = model_with_parser
        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "<think>\nthink trace\n</think>\n\nfinal answer"
        mock_output.token_ids = list(range(10))
        mock_output.finish_reason = "stop"
        request_output = MagicMock()
        request_output.prompt_token_ids = [1, 2, 3]
        request_output.outputs = [mock_output]

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([request_output])

        import src.models.qwen36.model_server.model as model_module

        with _patched_chat_types(model_module):
            with patch.object(
                model_module.OpenAIChatAdapterModel,
                "chat_completion_params_to_completion_params",
                return_value=_completion_request_mock(),
            ):
                with patch.object(
                    model,
                    "apply_chat_template",
                    return_value=MagicMock(prompt="template output"),
                ):
                    with patch.object(
                        model,
                        "completion_to_chat_completion",
                    ) as mock_c2c:
                        mock_c2c.return_value = _FakeType(
                            id="test-id",
                            created=1000,
                            model="test-model",
                            object="chat.completion",
                            choices=[
                                _FakeType(
                                    index=0,
                                    message=_FakeType(
                                        role="assistant",
                                        content="old content",
                                        reasoning=None,
                                    ),
                                    finish_reason="stop",
                                )
                            ],
                            usage=_FakeType(
                                prompt_tokens=3,
                                completion_tokens=10,
                                total_tokens=13,
                            ),
                        )

                        import asyncio

                        result = asyncio.run(
                            model.create_chat_completion(
                                self._make_request(thinking=True)
                            )
                        )

        msg = result.choices[0].message
        assert msg.reasoning == "think trace"
        assert msg.content == "final answer"

    def test_no_parser_called_when_thinking_disabled(self, model):
        model.reasoning_parser = MagicMock()
        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "plain answer"
        mock_output.token_ids = list(range(5))
        mock_output.finish_reason = "stop"
        request_output = MagicMock()
        request_output.prompt_token_ids = [1, 2, 3]
        request_output.outputs = [mock_output]

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([request_output])

        import src.models.qwen36.model_server.model as model_module

        with _patched_chat_types(model_module):
            with patch.object(
                model_module.OpenAIChatAdapterModel,
                "chat_completion_params_to_completion_params",
                return_value=_completion_request_mock(),
            ):
                with patch.object(
                    model,
                    "apply_chat_template",
                    return_value=MagicMock(prompt="template output"),
                ):
                    with patch.object(
                        model,
                        "completion_to_chat_completion",
                    ) as mock_c2c:
                        mock_c2c.return_value = _FakeType(
                            id="test-id",
                            created=1000,
                            model="test-model",
                            object="chat.completion",
                            choices=[
                                _FakeType(
                                    index=0,
                                    message=_FakeType(
                                        role="assistant",
                                        content="old content",
                                        reasoning=None,
                                    ),
                                    finish_reason="stop",
                                )
                            ],
                            usage=_FakeType(
                                prompt_tokens=3,
                                completion_tokens=5,
                                total_tokens=8,
                            ),
                        )

                        import asyncio

                        result = asyncio.run(
                            model.create_chat_completion(
                                self._make_request(thinking=False)
                            )
                        )

        model.reasoning_parser.extract_reasoning.assert_not_called()
        msg = result.choices[0].message
        assert msg.reasoning is None
        assert msg.content == "plain answer"

    def test_tool_calls_with_reasoning(self, model_with_parser):
        model = model_with_parser
        model.tool_calling_enabled = True

        tool_call = _FakeType(
            id="call_abc12345",
            type="function",
            function=_FakeType(name="get_weather", arguments='{"city": "Kampala"}'),
        )
        tool_parser = MagicMock()
        model.tool_parser_cls = MagicMock(return_value=tool_parser)
        tool_parser.extract_tool_calls.return_value = _FakeType(
            tools_called=True,
            tool_calls=[tool_call],
            content="\n\n",
        )

        text_with_tools = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Kampala"}}\n</tool_call>'
        # The real parser keeps tool-call text on the content side (probe
        # case 5); the fixture's fixed tuple would discard it, so override.
        model.reasoning_parser.extract_reasoning.return_value = (
            "think trace",
            f"\n\n{text_with_tools}",
        )
        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = f"<think>\nthink trace\n</think>\n\n{text_with_tools}"
        mock_output.token_ids = list(range(20))
        mock_output.finish_reason = "stop"
        request_output = MagicMock()
        request_output.prompt_token_ids = [1, 2, 3]
        request_output.outputs = [mock_output]

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([request_output])

        import src.models.qwen36.model_server.model as model_module

        with _patched_chat_types(model_module):
            with patch.object(
                model_module.OpenAIChatAdapterModel,
                "chat_completion_params_to_completion_params",
                return_value=_completion_request_mock(),
            ):
                with patch.object(
                    model,
                    "apply_chat_template",
                    return_value=MagicMock(prompt="template output"),
                ):
                    import asyncio

                    result = asyncio.run(
                        model.create_chat_completion(
                            self._make_request(thinking=True, tools=[MagicMock()])
                        )
                    )

        msg = result.choices[0].message
        assert msg.reasoning == "think trace"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_truncated_thinking_with_tools_no_error(self, model_with_parser):
        """Truncated mid-think output yields text=None and must not crash.

        Thinking + tools + a generation cut off before </think> makes the
        reasoning parser return (reasoning, None). The tool parser is then
        skipped because there is no content text to parse.
        """
        model = model_with_parser
        model.tool_calling_enabled = True

        model.reasoning_parser.extract_reasoning.return_value = (
            "think trace",
            None,
        )
        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "<think>\nthink trace"  # no closing </think>
        mock_output.token_ids = list(range(10))
        mock_output.finish_reason = "length"
        request_output = MagicMock()
        request_output.prompt_token_ids = [1, 2, 3]
        request_output.outputs = [mock_output]

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([request_output])

        import src.models.qwen36.model_server.model as model_module

        with _patched_chat_types(model_module):
            with patch.object(
                model_module.OpenAIChatAdapterModel,
                "chat_completion_params_to_completion_params",
                return_value=_completion_request_mock(),
            ):
                with patch.object(
                    model,
                    "apply_chat_template",
                    return_value=MagicMock(prompt="template output"),
                ):
                    with patch.object(
                        model,
                        "completion_to_chat_completion",
                    ) as mock_c2c:
                        mock_c2c.return_value = _FakeType(
                            id="test-id",
                            created=1000,
                            model="test-model",
                            object="chat.completion",
                            choices=[
                                _FakeType(
                                    index=0,
                                    message=_FakeType(
                                        role="assistant",
                                        content="old content",
                                        reasoning=None,
                                    ),
                                    finish_reason="length",
                                )
                            ],
                            usage=_FakeType(
                                prompt_tokens=3,
                                completion_tokens=10,
                                total_tokens=13,
                            ),
                        )

                        import asyncio

                        result = asyncio.run(
                            model.create_chat_completion(
                                self._make_request(thinking=True, tools=[MagicMock()])
                            )
                        )

        msg = result.choices[0].message
        assert msg.reasoning == "think trace"
        assert msg.content is None
        assert getattr(msg, "tool_calls", None) is None
        assert result.choices[0].finish_reason == "length"


class TestReasoningExtractionStreaming:
    """Tests for reasoning trace split in the streaming chat path."""

    @pytest.fixture
    def model_with_parser(self, model):
        """Attach a stubbed reasoning parser to the model."""
        model.reasoning_parser = MagicMock()
        return model

    @staticmethod
    def _make_delta(reasoning=None, content=None):
        """Create a mock DeltaMessage with reasoning and content."""
        d = MagicMock()
        d.reasoning = reasoning
        d.content = content
        return d

    @staticmethod
    async def _async_gen(items):
        for item in items:
            yield item

    async def _make_request_outputs(self, parser_deltas):
        """Build an async generator of RequestOutputs from parser deltas."""
        full_text = ""
        full_ids = []
        for i, d in enumerate(parser_deltas):
            if d.reasoning:
                full_text += d.reasoning
            if d.content:
                full_text += d.content
            new_tokens = [100 + i]
            full_ids.extend(new_tokens)

            out = MagicMock()
            out.index = 0
            out.text = full_text
            out.token_ids = list(full_ids)
            out.finish_reason = "stop" if i == len(parser_deltas) - 1 else None

            req = MagicMock()
            req.prompt_token_ids = [1, 2, 3]
            req.outputs = [out]
            yield req

    @staticmethod
    def _make_final_outputs(steps):
        """Build outputs from a list of (text, token_ids, finished) tuples."""
        outputs = []
        for text, tokens, finished in steps:
            out = MagicMock()
            out.index = 0
            out.text = text
            out.token_ids = tokens
            out.finish_reason = "stop" if finished else None
            req = MagicMock()
            req.prompt_token_ids = [1, 2, 3]
            req.outputs = [out]
            outputs.append(req)
        return outputs

    def _collect_stream(self, model, thinking=True):
        """Run _stream_chat_completion and return parsed data chunks."""
        import asyncio
        import json

        import src.models.qwen36.model_server.model as model_module

        with _patched_stream_types(model_module):
            with patch.object(
                model,
                "_build_sampling_params_from_request",
                return_value=MagicMock(),
            ):
                request = MagicMock()
                request.model = "test-model"
                chat_prompt = MagicMock()
                chat_prompt.response_role = "assistant"
                completion_request = MagicMock()
                completion_request.request_id = None
                completion_request.prompt = "test prompt"

                async def _collect():
                    chunks = []
                    async for chunk in model._stream_chat_completion(
                        request,
                        chat_prompt,
                        completion_request,
                        options=PerRequestOptions(enable_thinking=thinking),
                    ):
                        chunks.append(chunk)
                    return chunks

                chunks = asyncio.run(_collect())

        return [
            json.loads(c[len("data: ") :])
            for c in chunks
            if c.startswith("data: ") and "DONE" not in c
        ]

    def test_streaming_reasoning_routed_to_delta(self, model_with_parser):
        model = model_with_parser

        mock_parser = MagicMock()
        deltas = [
            self._make_delta(reasoning="think "),
            self._make_delta(reasoning="trace"),
            self._make_delta(content="answer"),
        ]
        mock_parser.extract_reasoning_streaming.side_effect = deltas
        model.reasoning_parser = mock_parser

        model.model = MagicMock()
        model.model.generate.return_value = self._make_request_outputs(deltas)

        data_chunks = self._collect_stream(model, thinking=True)

        assert len(data_chunks) >= 3
        assert data_chunks[0]["choices"][0]["delta"]["reasoning"] == "think "
        assert data_chunks[0]["choices"][0]["delta"]["content"] is None
        assert data_chunks[1]["choices"][0]["delta"]["reasoning"] == "trace"
        assert data_chunks[2]["choices"][0]["delta"]["content"] == "answer"

    def test_streaming_no_parser_when_thinking_disabled(self, model):
        model.reasoning_parser = MagicMock()

        out = MagicMock()
        out.index = 0
        out.text = "plain answer"
        out.token_ids = [100]
        out.finish_reason = "stop"
        req = MagicMock()
        req.prompt_token_ids = [1, 2, 3]
        req.outputs = [out]

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([req])

        data_chunks = self._collect_stream(model, thinking=False)

        model.reasoning_parser.extract_reasoning_streaming.assert_not_called()
        assert len(data_chunks) >= 1
        assert data_chunks[0]["choices"][0]["delta"]["content"] == "plain answer"
        assert data_chunks[0]["choices"][0]["delta"]["reasoning"] is None

    def test_streaming_final_chunk_carries_usage(self, model_with_parser):
        model = model_with_parser

        mock_parser = MagicMock()
        mock_parser.extract_reasoning_streaming.return_value = self._make_delta(
            content="answer"
        )
        model.reasoning_parser = mock_parser

        out = MagicMock()
        out.index = 0
        out.text = "answer"
        out.token_ids = [100, 200]
        out.finish_reason = "stop"
        req = MagicMock()
        req.prompt_token_ids = [1, 2, 3]
        req.outputs = [out]

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([req])

        data_chunks = self._collect_stream(model, thinking=True)

        final = data_chunks[-1]
        assert final["usage"] is not None
        assert final["choices"][0]["finish_reason"] == "stop"

    def test_streaming_parser_returns_none_skips_chunk(self, model_with_parser):
        model = model_with_parser

        mock_parser = MagicMock()
        mock_parser.extract_reasoning_streaming.side_effect = [
            None,
            None,
            self._make_delta(content="visible answer"),
        ]
        model.reasoning_parser = mock_parser

        outputs = self._make_final_outputs(
            [
                ("<think>", [100], False),
                ("<think>\nthink", [100, 200], False),
                ("<think>\nthink\n</think>\n\nvisible answer", [100, 200, 300], True),
            ]
        )

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen(outputs)

        data_chunks = self._collect_stream(model, thinking=True)

        assert len(data_chunks) == 2
        assert data_chunks[0]["choices"][0]["delta"]["content"] == "visible answer"

    def test_final_delta_none_preserves_finish_reason(self, model_with_parser):
        """Must-fix 1: parser returns None on final delta — finish_reason
        and usage must still be captured for the closing chunk."""
        model = model_with_parser

        mock_parser = MagicMock()
        mock_parser.extract_reasoning_streaming.side_effect = [
            self._make_delta(content="some text"),
            None,  # tag-boundary at end of generation
        ]
        model.reasoning_parser = mock_parser

        outputs = self._make_final_outputs(
            [
                ("some text", [100, 200], False),
                ("some text</think>", [100, 200, 300], True),
            ]
        )

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen(outputs)

        data_chunks = self._collect_stream(model, thinking=True)

        final = data_chunks[-1]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"]["prompt_tokens"] == 3
        assert final["usage"]["completion_tokens"] == 3
        assert final["usage"]["total_tokens"] == 6


class TestToolCallParsingStreaming:
    """Tests for streamed tool calls routed through vLLM's streaming parser."""

    @pytest.fixture
    def model_with_tool_parser(self, model):
        """Attach a stubbed tool-parser class to the model."""
        model.tool_parser_cls = MagicMock()
        model.reasoning_parser = None
        return model

    @staticmethod
    def _make_tool_delta(content=None, tool_calls=None):
        d = MagicMock()
        d.content = content
        d.tool_calls = tool_calls
        return d

    @staticmethod
    def _make_reasoning_delta(reasoning=None, content=None):
        d = MagicMock()
        d.reasoning = reasoning
        d.content = content
        return d

    @staticmethod
    async def _async_gen(items):
        for item in items:
            yield item

    @staticmethod
    def _make_request_outputs(steps):
        """Build outputs from a list of (text, token_ids, finished) tuples."""
        outputs = []
        for text, tokens, finished in steps:
            out = MagicMock()
            out.index = 0
            out.text = text
            out.token_ids = tokens
            out.finish_reason = "stop" if finished else None
            req = MagicMock()
            req.prompt_token_ids = [1, 2, 3]
            req.outputs = [out]
            outputs.append(req)
        return outputs

    def _collect_stream(self, model, thinking=False, parse_tool_calls=True):
        """Run _stream_chat_completion and return parsed data chunks."""
        import asyncio
        import json

        import src.models.qwen36.model_server.model as model_module

        with _patched_stream_types(model_module):
            with patch.object(
                model,
                "_build_sampling_params_from_request",
                return_value=MagicMock(),
            ):
                request = MagicMock()
                request.model = "test-model"
                chat_prompt = MagicMock()
                chat_prompt.response_role = "assistant"
                completion_request = MagicMock()
                completion_request.request_id = None
                completion_request.prompt = "test prompt"

                async def _collect():
                    chunks = []
                    async for chunk in model._stream_chat_completion(
                        request,
                        chat_prompt,
                        completion_request,
                        parse_tool_calls,
                        options=PerRequestOptions(enable_thinking=thinking),
                    ):
                        chunks.append(chunk)
                    return chunks

                chunks = asyncio.run(_collect())

        return [
            json.loads(c[len("data: ") :])
            for c in chunks
            if c.startswith("data: ") and "DONE" not in c
        ]

    def test_tool_call_streamed_as_delta_not_content(self, model_with_tool_parser):
        model = model_with_tool_parser
        tool_parser = MagicMock()
        model.tool_parser_cls.return_value = tool_parser

        tool_delta = _FakeType(
            index=0,
            id="call_abc12345",
            type="function",
            function=_FakeType(name="get_weather", arguments='{"city": "Kampala"}'),
        )
        tool_parser.extract_tool_calls_streaming.side_effect = [
            None,  # <tool_call> tag boundary, suppressed
            None,  # accumulating arguments, suppressed
            self._make_tool_delta(content=None, tool_calls=[tool_delta]),
        ]

        steps = [
            ("<tool_call>", [100], False),
            ('<tool_call>{"name": "get_weather"', [100, 200], False),
            (
                '<tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Kampala"}}</tool_call>',
                [100, 200, 300],
                True,
            ),
        ]
        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen(
            self._make_request_outputs(steps)
        )

        data_chunks = self._collect_stream(model)

        # Suppressed deltas collapse to one tool-call chunk plus the final
        # chunk. No content chunk ever carried the raw tag text.
        assert len(data_chunks) == 2
        tool_chunk = data_chunks[0]
        assert tool_chunk["choices"][0]["finish_reason"] is None
        assert tool_chunk["choices"][0]["delta"]["tool_calls"] == [
            {
                "index": 0,
                "id": "call_abc12345",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Kampala"}',
                },
            }
        ]
        for chunk in data_chunks:
            content = chunk["choices"][0]["delta"].get("content")
            assert content is None or "<tool_call>" not in content

        final = data_chunks[-1]
        assert final["choices"][0]["finish_reason"] == "tool_calls"
        assert final["usage"] is not None

    def test_content_precedes_tool_call(self, model_with_tool_parser):
        model = model_with_tool_parser
        tool_parser = MagicMock()
        model.tool_parser_cls.return_value = tool_parser

        tool_delta = _FakeType(
            index=0,
            id="call_x",
            type="function",
            function=_FakeType(name="get_time", arguments='{"tz": "UTC"}'),
        )
        tool_parser.extract_tool_calls_streaming.side_effect = [
            self._make_tool_delta(content="Let me check.\n"),
            self._make_tool_delta(content=None, tool_calls=[tool_delta]),
        ]

        steps = [
            ("Let me check.\n", [100], False),
            (
                'Let me check.\n<tool_call>{"name": "get_time", '
                '"arguments": {"tz": "UTC"}}</tool_call>',
                [100, 200],
                True,
            ),
        ]
        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen(
            self._make_request_outputs(steps)
        )

        data_chunks = self._collect_stream(model)

        assert data_chunks[0]["choices"][0]["delta"]["content"] == "Let me check.\n"
        assert data_chunks[0]["choices"][0]["delta"].get("tool_calls") is None
        tool_call = data_chunks[1]["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "get_time"
        assert data_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_parser_none_on_finish_preserves_finish_reason(
        self, model_with_tool_parser
    ):
        model = model_with_tool_parser
        tool_parser = MagicMock()
        model.tool_parser_cls.return_value = tool_parser

        tool_parser.extract_tool_calls_streaming.side_effect = [
            self._make_tool_delta(content="some text"),
            None,  # trailing tag boundary on the final delta
        ]

        steps = [
            ("some text", [100, 200], False),
            ("some text</tool_call>", [100, 200, 300], True),
        ]
        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen(
            self._make_request_outputs(steps)
        )

        data_chunks = self._collect_stream(model)

        assert data_chunks[0]["choices"][0]["delta"]["content"] == "some text"
        final = data_chunks[-1]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"]["prompt_tokens"] == 3
        assert final["usage"]["completion_tokens"] == 3
        assert final["usage"]["total_tokens"] == 6

    def test_thinking_and_tools_routed_separately(self, model):
        model.reasoning_parser = MagicMock()
        model.tool_parser_cls = MagicMock()
        tool_parser = MagicMock()
        model.tool_parser_cls.return_value = tool_parser

        model.reasoning_parser.extract_reasoning_streaming.side_effect = [
            self._make_reasoning_delta(reasoning="think trace", content=None),
            self._make_reasoning_delta(
                reasoning=None,
                content=(
                    '<tool_call>{"name": "get_weather", '
                    '"arguments": {"city": "Kampala"}}</tool_call>'
                ),
            ),
        ]
        tool_delta = _FakeType(
            index=0,
            id="call_x",
            type="function",
            function=_FakeType(name="get_weather", arguments='{"city": "Kampala"}'),
        )
        tool_parser.extract_tool_calls_streaming.side_effect = [
            self._make_tool_delta(content=None, tool_calls=[tool_delta]),
        ]

        steps = [
            ("<think>\nthink trace", [100], False),
            (
                "<think>\nthink trace</think>\n\n"
                '<tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Kampala"}}</tool_call>',
                [100, 200],
                True,
            ),
        ]
        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen(
            self._make_request_outputs(steps)
        )

        data_chunks = self._collect_stream(model, thinking=True)

        assert data_chunks[0]["choices"][0]["delta"]["reasoning"] == "think trace"
        assert data_chunks[0]["choices"][0]["delta"]["content"] is None
        tool_call = data_chunks[1]["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "get_weather"
        assert data_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_suppressed_tool_delta_keeps_reasoning(self, model):
        """A suppressed content head must not swallow a reasoning fragment.

        When a delta spans </think>, the reasoning parser can return both a
        reasoning tail and a content head. If the tool parser suppresses that
        content head (a <tool_call> tag boundary), the reasoning fragment must
        still be emitted as a reasoning-only chunk.
        """
        model.reasoning_parser = MagicMock()
        model.tool_parser_cls = MagicMock()
        tool_parser = MagicMock()
        model.tool_parser_cls.return_value = tool_parser

        model.reasoning_parser.extract_reasoning_streaming.side_effect = [
            self._make_reasoning_delta(reasoning="tail", content="<tool_call>"),
        ]
        tool_parser.extract_tool_calls_streaming.side_effect = [None]

        steps = [("tail<tool_call>", [100], True)]
        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen(
            self._make_request_outputs(steps)
        )

        data_chunks = self._collect_stream(model, thinking=True)

        assert data_chunks[0]["choices"][0]["delta"]["reasoning"] == "tail"
        assert data_chunks[0]["choices"][0]["delta"]["content"] is None
        assert data_chunks[0]["choices"][0]["delta"].get("tool_calls") is None
        assert data_chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_no_tool_emitted_streams_content_normally(self, model_with_tool_parser):
        model = model_with_tool_parser
        tool_parser = MagicMock()
        model.tool_parser_cls.return_value = tool_parser

        tool_parser.extract_tool_calls_streaming.side_effect = [
            self._make_tool_delta(content="just a normal answer"),
        ]

        steps = [("just a normal answer", [100, 200], True)]
        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen(
            self._make_request_outputs(steps)
        )

        data_chunks = self._collect_stream(model)

        assert (
            data_chunks[0]["choices"][0]["delta"]["content"] == "just a normal answer"
        )
        assert data_chunks[0]["choices"][0]["delta"].get("tool_calls") is None
        assert data_chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_fresh_parser_constructed_per_stream(self, model):
        model.reasoning_parser = None
        model.tool_parser_cls = MagicMock()
        parser_instance = MagicMock()
        parser_instance.extract_tool_calls_streaming.return_value = (
            self._make_tool_delta(content="answer")
        )
        model.tool_parser_cls.return_value = parser_instance

        def _run_once():
            model.model = MagicMock()
            out = MagicMock()
            out.index = 0
            out.text = "answer"
            out.token_ids = [100]
            out.finish_reason = "stop"
            req = MagicMock()
            req.prompt_token_ids = [1, 2, 3]
            req.outputs = [out]
            model.model.generate.return_value = self._async_gen([req])
            return self._collect_stream(model)

        _run_once()
        _run_once()

        assert model.tool_parser_cls.call_count == 2
        model.tool_parser_cls.assert_called_with(model.tokenizer)


class TestResolveStructuredOutputs:
    """Tests for _resolve_structured_outputs."""

    def test_absent_returns_none(self, model):
        request = MagicMock(response_format=None)
        assert model._resolve_structured_outputs(request) is None

    def test_bare_magicmock_returns_none(self, model):
        # A bare MagicMock auto-creates a truthy response_format; it must not
        # be mistaken for a structured-output request.
        assert model._resolve_structured_outputs(MagicMock()) is None

    def test_text_returns_none(self, model):
        request = MagicMock(response_format={"type": "text"})
        assert model._resolve_structured_outputs(request) is None

    def test_json_object(self, model):
        from src.models.qwen36.model_server.model import StructuredOutputsParams

        result = model._resolve_structured_outputs(
            MagicMock(response_format={"type": "json_object"})
        )
        assert StructuredOutputsParams.call_args.kwargs == {"json_object": True}
        assert result is StructuredOutputsParams.return_value

    def test_json_schema_inline(self, model):
        from src.models.qwen36.model_server.model import StructuredOutputsParams

        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
        model._resolve_structured_outputs(
            MagicMock(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "s", "schema": schema},
                }
            )
        )
        assert StructuredOutputsParams.call_args.kwargs == {"json": schema}

    def test_json_schema_strict_is_ignored(self, model):
        from src.models.qwen36.model_server.model import StructuredOutputsParams

        schema = {"type": "object"}
        model._resolve_structured_outputs(
            MagicMock(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "s", "schema": schema, "strict": True},
                }
            )
        )
        assert StructuredOutputsParams.call_args.kwargs == {"json": schema}

    def test_json_schema_missing_json_schema_raises(self, model):
        with pytest.raises(InvalidInput, match="json_schema must be an object"):
            model._resolve_structured_outputs(
                MagicMock(response_format={"type": "json_schema"})
            )

    def test_json_schema_missing_schema_raises(self, model):
        with pytest.raises(InvalidInput, match="schema is required"):
            model._resolve_structured_outputs(
                MagicMock(
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "s"},
                    }
                )
            )

    def test_json_schema_schema_not_dict_raises(self, model):
        with pytest.raises(InvalidInput, match="must be a JSON Schema object"):
            model._resolve_structured_outputs(
                MagicMock(
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "s", "schema": "not-a-dict"},
                    }
                )
            )

    def test_unknown_type_raises(self, model):
        with pytest.raises(InvalidInput, match="Unsupported response_format type"):
            model._resolve_structured_outputs(
                MagicMock(response_format={"type": "bogus"})
            )

    def test_response_format_pydantic_model_uses_alias(self, model):
        from src.models.qwen36.model_server.model import StructuredOutputsParams

        schema = {"type": "object"}
        outer = MagicMock(spec=["model_dump"])
        outer.model_dump.side_effect = lambda **kw: (
            {
                "type": "json_schema",
                "json_schema": {"name": "s", "schema": schema},
            }
            if kw.get("by_alias")
            else {
                "type": "json_schema",
                "json_schema": {"name": "s", "json_schema": schema},
            }
        )
        model._resolve_structured_outputs(MagicMock(response_format=outer))
        outer.model_dump.assert_called_once_with(by_alias=True)
        assert StructuredOutputsParams.call_args.kwargs == {"json": schema}


class TestResolveToolChoice:
    """Tests for _resolve_tool_choice."""

    def test_absent_returns_auto(self, model):
        assert model._resolve_tool_choice(MagicMock(tool_choice=None)) == "auto"

    def test_auto_returns_auto(self, model):
        assert model._resolve_tool_choice(MagicMock(tool_choice="auto")) == "auto"

    def test_none_returns_none(self, model):
        assert model._resolve_tool_choice(MagicMock(tool_choice="none")) == "none"

    def test_required_raises(self, model):
        with pytest.raises(InvalidInput, match='tool_choice must be "auto" or "none"'):
            model._resolve_tool_choice(MagicMock(tool_choice="required"))

    def test_named_function_raises(self, model):
        named = _FakeType(type="function", function=_FakeType(name="get_weather"))
        with pytest.raises(InvalidInput, match="named functions are not supported"):
            model._resolve_tool_choice(MagicMock(tool_choice=named))


class TestBuildSamplingParamsStructuredOutputs:
    """Tests for structured_outputs threading in _build_sampling_params_from_request."""

    @staticmethod
    def _request():
        request = MagicMock()
        request.max_tokens = None
        request.temperature = None
        request.top_p = None
        request.top_k = None
        request.presence_penalty = None
        request.repetition_penalty = None
        return request

    def test_structured_outputs_passed_through(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        sentinel = MagicMock()
        model._build_sampling_params_from_request(
            self._request(), options=PerRequestOptions(structured_outputs=sentinel)
        )
        assert SamplingParams.call_args.kwargs["structured_outputs"] is sentinel

    def test_structured_outputs_none_by_default(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        model._build_sampling_params_from_request(
            self._request(), options=PerRequestOptions()
        )
        assert SamplingParams.call_args.kwargs["structured_outputs"] is None


class TestCreateCompletionStructuredOutputs:
    """Tests for structured_outputs threading in create_completion."""

    @staticmethod
    async def _async_gen(items):
        for item in items:
            yield item

    @staticmethod
    def _request():
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
        return request

    def _mock_output(self):
        output = MagicMock()
        output.index = 0
        output.text = "Hello"
        output.token_ids = [100, 200]
        output.finish_reason = "stop"
        request_output = MagicMock()
        request_output.prompt_token_ids = [1, 2, 3]
        request_output.outputs = [output]
        return request_output

    def test_threads_structured_outputs_to_sampling_params(self, model):
        from src.models.qwen36.model_server.model import SamplingParams

        model.model = MagicMock()
        model.model.generate.return_value = self._async_gen([self._mock_output()])

        sentinel = MagicMock()
        import asyncio

        asyncio.run(
            model.create_completion(
                self._request(), options=PerRequestOptions(structured_outputs=sentinel)
            )
        )
        assert SamplingParams.call_args.kwargs["structured_outputs"] is sentinel

    def test_valueerror_mapped_to_invalid_input(self, model):
        async def _error_gen():
            raise ValueError("unsupported schema feature")
            yield  # pragma: no cover

        model.model = MagicMock()
        model.model.generate.return_value = _error_gen()

        import asyncio

        with pytest.raises(InvalidInput, match="Invalid structured output request"):
            asyncio.run(
                model.create_completion(
                    self._request(),
                    options=PerRequestOptions(structured_outputs=MagicMock()),
                )
            )


class TestCreateChatCompletionStructuredOutputs:
    """Tests that create_chat_completion resolves and threads structured outputs."""

    @staticmethod
    def _chat_request(stream):
        request = MagicMock()
        request.n = 1
        request.stream = stream
        request.tools = None
        request.tool_choice = None
        request.chat_template_kwargs = {}
        request.thinking_token_budget = None
        return request

    def test_streaming_threads_structured_outputs(self, model):
        import asyncio

        import src.models.qwen36.model_server.model as model_module

        sentinel = MagicMock()
        request = self._chat_request(stream=True)

        with patch.object(
            model,
            "_resolve_request_options",
            return_value=PerRequestOptions(structured_outputs=sentinel),
        ):
            with patch.object(
                model, "apply_chat_template", return_value=MagicMock(prompt="t")
            ):
                with patch.object(
                    model_module.OpenAIChatAdapterModel,
                    "chat_completion_params_to_completion_params",
                    return_value=_completion_request_mock(),
                ):
                    with patch.object(model, "_stream_chat_completion") as mock_stream:
                        asyncio.run(model.create_chat_completion(request))

        mock_stream.assert_called_once()
        assert mock_stream.call_args.kwargs["options"].structured_outputs is sentinel

    def test_non_streaming_threads_structured_outputs(self, model):
        import asyncio

        import src.models.qwen36.model_server.model as model_module

        sentinel = MagicMock()
        request = self._chat_request(stream=False)

        with patch.object(
            model,
            "_resolve_request_options",
            return_value=PerRequestOptions(structured_outputs=sentinel),
        ):
            with patch.object(
                model, "apply_chat_template", return_value=MagicMock(prompt="t")
            ):
                with patch.object(
                    model_module.OpenAIChatAdapterModel,
                    "chat_completion_params_to_completion_params",
                    return_value=_completion_request_mock(),
                ):
                    with _patched_chat_types(model_module):
                        with patch.object(
                            model, "create_completion", new=AsyncMock()
                        ) as mock_cc:
                            mock_cc.return_value = _FakeType(
                                id="x",
                                created=1,
                                model="m",
                                object="text_completion",
                                choices=[
                                    _FakeType(index=0, text="", finish_reason="stop")
                                ],
                                usage=_FakeType(),
                            )
                            with patch.object(
                                model, "completion_to_chat_completion"
                            ) as mock_c2c:
                                mock_c2c.return_value = _FakeType(
                                    choices=[
                                        _FakeType(
                                            message=_FakeType(
                                                content=None, reasoning=None
                                            )
                                        )
                                    ]
                                )
                                asyncio.run(model.create_chat_completion(request))

        mock_cc.assert_awaited_once()
        assert mock_cc.await_args.kwargs["options"].structured_outputs is sentinel

    def test_tools_with_structured_outputs_skips_tool_call_parsing(self, model):
        import asyncio

        import src.models.qwen36.model_server.model as model_module

        model.tool_calling_enabled = True
        tool_parser = MagicMock()
        model.tool_parser_cls = MagicMock(return_value=tool_parser)
        request = self._chat_request(stream=False)
        request.tools = [{"type": "function", "function": {}}]

        # A tool call that WOULD parse if parse_tool_calls stayed True.
        tool_text = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'

        completion = _FakeType(
            id="x",
            created=1,
            model="m",
            object="text_completion",
            choices=[_FakeType(index=0, text=tool_text, finish_reason="stop")],
            usage=_FakeType(),
        )

        with patch.object(
            model,
            "_resolve_request_options",
            return_value=PerRequestOptions(structured_outputs=MagicMock()),
        ):
            with patch.object(
                model, "apply_chat_template", return_value=MagicMock(prompt="t")
            ):
                with patch.object(
                    model_module.OpenAIChatAdapterModel,
                    "chat_completion_params_to_completion_params",
                    return_value=_completion_request_mock(),
                ):
                    with _patched_chat_types(model_module):
                        with patch.object(
                            model, "create_completion", new=AsyncMock()
                        ) as mock_cc:
                            mock_cc.return_value = completion
                            with patch.object(
                                model, "completion_to_chat_completion"
                            ) as mock_c2c:
                                mock_c2c.return_value = _FakeType(
                                    choices=[
                                        _FakeType(
                                            message=_FakeType(
                                                content=None, reasoning=None
                                            )
                                        )
                                    ]
                                )
                                result = asyncio.run(
                                    model.create_chat_completion(request)
                                )

        tool_parser.extract_tool_calls.assert_not_called()
        assert result.choices[0].message.content == tool_text


class TestChatCompletionToolCallParsing:
    """Tests for non-streaming tool-call parsing via vLLM's tool parser."""

    @staticmethod
    def _chat_request():
        request = MagicMock()
        request.n = 1
        request.stream = False
        request.tools = [{"type": "function", "function": {}}]
        request.tool_choice = None
        request.chat_template_kwargs = {}
        request.thinking_token_budget = None
        request.response_format = None
        return request

    @staticmethod
    def _completion(text):
        return _FakeType(
            id="x",
            created=1,
            model="m",
            object="text_completion",
            choices=[_FakeType(index=0, text=text, finish_reason="stop")],
            usage=_FakeType(),
        )

    def test_tool_calls_routed_to_openai_message(self, model):
        import asyncio

        import src.models.qwen36.model_server.model as model_module

        model.tool_calling_enabled = True
        tool_call = _FakeType(
            id="call_abc12345",
            type="function",
            function=_FakeType(name="get_weather", arguments='{"city": "Kampala"}'),
        )
        tool_parser = MagicMock()
        model.tool_parser_cls = MagicMock(return_value=tool_parser)
        tool_parser.extract_tool_calls.return_value = _FakeType(
            tools_called=True,
            tool_calls=[tool_call],
            content="Let me check.\n",
        )

        request = self._chat_request()
        completion = self._completion(
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Kampala"}}\n</tool_call>'
        )

        with patch.object(
            model, "_resolve_request_options", return_value=PerRequestOptions()
        ):
            with patch.object(
                model, "apply_chat_template", return_value=MagicMock(prompt="t")
            ):
                with patch.object(
                    model_module.OpenAIChatAdapterModel,
                    "chat_completion_params_to_completion_params",
                    return_value=_completion_request_mock(),
                ):
                    with _patched_chat_types(model_module):
                        with patch.object(
                            model, "create_completion", new=AsyncMock()
                        ) as mock_cc:
                            mock_cc.return_value = completion
                            result = asyncio.run(model.create_chat_completion(request))

        tool_parser.extract_tool_calls.assert_called_once()
        model.tool_parser_cls.assert_called_once_with(model.tokenizer)
        call_args = tool_parser.extract_tool_calls.call_args.args
        assert call_args[0] == completion.choices[0].text
        assert call_args[1] is request
        assert mock_cc.await_args.kwargs["options"].skip_special_tokens is False

        msg = result.choices[0].message
        assert msg.tool_calls == [
            {
                "id": "call_abc12345",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Kampala"}',
                },
            }
        ]
        assert msg.content == "Let me check.\n"
        assert result.choices[0].finish_reason == "tool_calls"

    def test_no_tool_calls_falls_back_to_normal_completion(self, model):
        import asyncio

        import src.models.qwen36.model_server.model as model_module

        model.tool_calling_enabled = True
        tool_parser = MagicMock()
        model.tool_parser_cls = MagicMock(return_value=tool_parser)
        tool_parser.extract_tool_calls.return_value = _FakeType(
            tools_called=False,
            tool_calls=[],
            content=None,
        )

        request = self._chat_request()
        completion = self._completion("just a normal answer")

        with patch.object(
            model, "_resolve_request_options", return_value=PerRequestOptions()
        ):
            with patch.object(
                model, "apply_chat_template", return_value=MagicMock(prompt="t")
            ):
                with patch.object(
                    model_module.OpenAIChatAdapterModel,
                    "chat_completion_params_to_completion_params",
                    return_value=_completion_request_mock(),
                ):
                    with _patched_chat_types(model_module):
                        with patch.object(
                            model, "create_completion", new=AsyncMock()
                        ) as mock_cc:
                            mock_cc.return_value = completion
                            with patch.object(
                                model, "completion_to_chat_completion"
                            ) as mock_c2c:
                                mock_c2c.return_value = _FakeType(
                                    choices=[
                                        _FakeType(
                                            message=_FakeType(
                                                content=None, reasoning=None
                                            )
                                        )
                                    ]
                                )
                                result = asyncio.run(
                                    model.create_chat_completion(request)
                                )

        assert getattr(result.choices[0].message, "tool_calls", None) is None
        assert result.choices[0].message.content == "just a normal answer"
        assert mock_cc.await_args.kwargs["options"].skip_special_tokens is False

    def test_no_tools_leaves_skip_special_tokens_unset(self, model):
        import asyncio

        import src.models.qwen36.model_server.model as model_module

        model.tool_calling_enabled = True
        request = self._chat_request()
        request.tools = None
        completion = self._completion("plain answer")

        with patch.object(
            model, "_resolve_request_options", return_value=PerRequestOptions()
        ):
            with patch.object(
                model, "apply_chat_template", return_value=MagicMock(prompt="t")
            ):
                with patch.object(
                    model_module.OpenAIChatAdapterModel,
                    "chat_completion_params_to_completion_params",
                    return_value=_completion_request_mock(),
                ):
                    with _patched_chat_types(model_module):
                        with patch.object(
                            model, "create_completion", new=AsyncMock()
                        ) as mock_cc:
                            mock_cc.return_value = completion
                            with patch.object(
                                model, "completion_to_chat_completion"
                            ) as mock_c2c:
                                mock_c2c.return_value = _FakeType(
                                    choices=[
                                        _FakeType(
                                            message=_FakeType(
                                                content=None, reasoning=None
                                            )
                                        )
                                    ]
                                )
                                asyncio.run(model.create_chat_completion(request))

        assert mock_cc.await_args.kwargs["options"].skip_special_tokens is None

    def test_tool_choice_none_skips_tool_parsing(self, model):
        import asyncio

        import src.models.qwen36.model_server.model as model_module

        model.tool_calling_enabled = True
        model.tool_parser_cls = MagicMock()
        request = self._chat_request()
        request.tool_choice = "none"
        completion = self._completion("just an answer")

        with patch.object(
            model, "_resolve_request_options", return_value=PerRequestOptions()
        ):
            with patch.object(
                model, "apply_chat_template", return_value=MagicMock(prompt="t")
            ):
                with patch.object(
                    model_module.OpenAIChatAdapterModel,
                    "chat_completion_params_to_completion_params",
                    return_value=_completion_request_mock(),
                ):
                    with _patched_chat_types(model_module):
                        with patch.object(
                            model, "create_completion", new=AsyncMock()
                        ) as mock_cc:
                            mock_cc.return_value = completion
                            with patch.object(
                                model, "completion_to_chat_completion"
                            ) as mock_c2c:
                                mock_c2c.return_value = _FakeType(
                                    choices=[
                                        _FakeType(
                                            message=_FakeType(
                                                content=None, reasoning=None
                                            )
                                        )
                                    ]
                                )
                                result = asyncio.run(
                                    model.create_chat_completion(request)
                                )

        model.tool_parser_cls.assert_not_called()
        assert mock_cc.await_args.kwargs["options"].skip_special_tokens is None
        assert result.choices[0].message.content == "just an answer"

    def test_tool_choice_auto_parses_tools(self, model):
        import asyncio

        import src.models.qwen36.model_server.model as model_module

        model.tool_calling_enabled = True
        tool_parser = MagicMock()
        model.tool_parser_cls = MagicMock(return_value=tool_parser)
        tool_call = _FakeType(
            id="call_abc12345",
            type="function",
            function=_FakeType(name="get_weather", arguments='{"city": "Kampala"}'),
        )
        tool_parser.extract_tool_calls.return_value = _FakeType(
            tools_called=True,
            tool_calls=[tool_call],
            content="Let me check.\n",
        )

        request = self._chat_request()
        request.tool_choice = "auto"
        completion = self._completion(
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Kampala"}}\n</tool_call>'
        )

        with patch.object(
            model, "_resolve_request_options", return_value=PerRequestOptions()
        ):
            with patch.object(
                model, "apply_chat_template", return_value=MagicMock(prompt="t")
            ):
                with patch.object(
                    model_module.OpenAIChatAdapterModel,
                    "chat_completion_params_to_completion_params",
                    return_value=_completion_request_mock(),
                ):
                    with _patched_chat_types(model_module):
                        with patch.object(
                            model, "create_completion", new=AsyncMock()
                        ) as mock_cc:
                            mock_cc.return_value = completion
                            result = asyncio.run(model.create_chat_completion(request))

        tool_parser.extract_tool_calls.assert_called_once()
        assert mock_cc.await_args.kwargs["options"].skip_special_tokens is False
        assert result.choices[0].finish_reason == "tool_calls"

    def test_tool_choice_none_with_schema_no_warning(self, model):
        import asyncio

        import src.models.qwen36.model_server.model as model_module

        model.tool_calling_enabled = True
        model.tool_parser_cls = MagicMock()
        request = self._chat_request()
        request.tool_choice = "none"
        completion = self._completion("just an answer")

        with patch.object(
            model,
            "_resolve_request_options",
            return_value=PerRequestOptions(structured_outputs=MagicMock()),
        ):
            with patch.object(
                model, "apply_chat_template", return_value=MagicMock(prompt="t")
            ):
                with patch.object(
                    model_module.OpenAIChatAdapterModel,
                    "chat_completion_params_to_completion_params",
                    return_value=_completion_request_mock(),
                ):
                    with _patched_chat_types(model_module):
                        with patch.object(
                            model, "create_completion", new=AsyncMock()
                        ) as mock_cc:
                            mock_cc.return_value = completion
                            with patch.object(
                                model, "completion_to_chat_completion"
                            ) as mock_c2c:
                                mock_c2c.return_value = _FakeType(
                                    choices=[
                                        _FakeType(
                                            message=_FakeType(
                                                content=None, reasoning=None
                                            )
                                        )
                                    ]
                                )
                                with patch.object(
                                    model_module.logging, "warning"
                                ) as mock_warning:
                                    result = asyncio.run(
                                        model.create_chat_completion(request)
                                    )

        mock_warning.assert_not_called()
        model.tool_parser_cls.assert_not_called()
        assert result.choices[0].message.content == "just an answer"

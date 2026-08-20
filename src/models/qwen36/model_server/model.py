import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from typing import Any, Union

import kserve
from kserve.errors import InferenceError, InvalidInput
from kserve.protocol.rest.openai import ChatPrompt, OpenAIChatAdapterModel
from kserve.protocol.rest.openai.types import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    ChoiceDelta,
    ChunkChoice,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    ErrorResponse,
    UsageInfo,
)
from vllm import RequestOutput, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers import ToolParserManager

from python.type_utils import strtobool

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

# Sampling defaults keyed by enable_thinking (True = thinking, False = instruct).
# See https://huggingface.co/Qwen/Qwen3.6-27B-FP8#recommended-sampling-parameters
SAMPLING_DEFAULTS = {
    True: {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
    False: {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5},
}


@dataclass(frozen=True)
class PerRequestOptions:
    """Per-request options resolved once from a chat completions request.

    Groups the threaded thinking / structured-output options so they travel
    together instead of being passed as a growing list of positional args.
    """

    enable_thinking: bool = False
    thinking_token_budget: int | None = None
    structured_outputs: StructuredOutputsParams | None = None
    # Derived (not client-sent): set to False when tool calls are active so
    # the tool-call special tokens survive decoding for the parser.
    skip_special_tokens: bool | None = None


# The raw /openai/v1/completions endpoint keeps its historical thinking
# default; chat completions resolves its own options per request.
RAW_COMPLETIONS_DEFAULTS = PerRequestOptions(enable_thinking=True)


class Qwen36Model(kserve.Model, OpenAIChatAdapterModel):
    def __init__(
        self,
        name: str,
        model_path: str,
        trust_remote_code: bool,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
        dtype: str,
        language_model_only_flag: bool,
        skip_mm_profiling_flag: bool,
        max_num_seqs: int = 128,
        max_num_batched_tokens: int = 32768,
        block_size: int = 64,
        attention_backend: str = "TRITON_ATTN",
        kv_cache_dtype: str = "auto",
        disable_custom_all_reduce: bool = False,
        enforce_eager: bool = False,
        disable_log_stats: bool = False,
        tool_calling_enabled: bool = False,
        tool_call_parser: str = "hermes",
    ) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.language_model_only_flag = language_model_only_flag
        self.skip_mm_profiling_flag = skip_mm_profiling_flag
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.attention_backend = attention_backend
        self.kv_cache_dtype = kv_cache_dtype
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.enforce_eager = enforce_eager
        self.disable_log_stats = disable_log_stats
        self.tool_calling_enabled = tool_calling_enabled
        self.tool_call_parser = tool_call_parser
        self.model = None
        self.tokenizer = None
        self.reasoning_parser = None
        self.tool_parser_cls = None
        self.ready = False

    def load(self) -> None:
        try:
            logging.info(f"Loading Qwen 3.6 model from {self.model_path}...")
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                trust_remote_code=self.trust_remote_code,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_num_seqs=self.max_num_seqs,
                block_size=self.block_size,
                attention_backend=self.attention_backend,
                # KV cache dtype for the 16 gated-attention layers ("auto" = model
                # dtype, i.e. bf16; "fp8" = e4m3 with static scale 1.0). Do NOT add
                # calculate_kv_scales: on GDN+attention hybrids the warmup pass runs
                # with uninitialized recurrent state and produces corrupted scales
                # (https://github.com/vllm-project/vllm/issues/37554).
                kv_cache_dtype=self.kv_cache_dtype,
                tensor_parallel_size=self.tensor_parallel_size,
                disable_custom_all_reduce=self.disable_custom_all_reduce,
                enforce_eager=self.enforce_eager,
                dtype=self.dtype,
                language_model_only=self.language_model_only_flag,
                skip_mm_profiling=self.skip_mm_profiling_flag,
                enable_prefix_caching=True,
                reasoning_parser="qwen3",
                served_model_name=self.name,
                disable_log_stats=self.disable_log_stats,
            )
            self.model = AsyncLLMEngine.from_engine_args(engine_args)
            self.tokenizer = self.model.tokenizer
            self.reasoning_parser = ReasoningParserManager.get_reasoning_parser(
                "qwen3"
            )(self.tokenizer)
            self.tool_parser_cls = ToolParserManager.get_tool_parser(
                self.tool_call_parser
            )
            # Construct one instance and discard it: fail fast at startup if
            # the parser is incompatible with this tokenizer. Serving paths
            # build fresh instances per request.
            self.tool_parser_cls(self.tokenizer)
            self.ready = True
            logging.info("Model loaded successfully!")
        except Exception as e:
            error_message = f"Failed to load model. Reason: {e}"
            logging.critical(error_message)
            raise kserve.errors.ModelMissingError(error_message)

    def _build_messages(self, prompt: str, system: str | None = None) -> list:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return messages

    @staticmethod
    def _validate_thinking_token_budget(value: Any) -> int | None:
        """Validate and coerce a thinking_token_budget value.

        Returns an int >= 1, or None if the input is None.
        Raises InvalidInput for non-integer or out-of-range values.
        """
        if value is None:
            return None
        try:
            budget = int(value)
        except (TypeError, ValueError):
            raise InvalidInput("thinking_token_budget must be an integer")
        if budget < 1:
            raise InvalidInput("thinking_token_budget must be >= 1")
        return budget

    @staticmethod
    def _resolve_enable_thinking(request: ChatCompletionRequest) -> bool:
        """Extract enable_thinking from chat_template_kwargs; default off."""
        kwargs = getattr(request, "chat_template_kwargs", None)
        if not isinstance(kwargs, dict):
            return False
        value = kwargs.get("enable_thinking", False)
        if isinstance(value, bool):
            return value
        try:
            return bool(strtobool(str(value)))
        except ValueError:
            raise InvalidInput("enable_thinking must be a boolean")

    @staticmethod
    def _resolve_structured_outputs(
        request: ChatCompletionRequest,
    ) -> StructuredOutputsParams | None:
        """Resolve OpenAI ``response_format`` into vLLM structured outputs.

        Returns None when no constraint is requested (field absent, or
        ``{"type": "text"}``).  Maps ``json_schema`` to a JSON-schema
        constraint and ``json_object`` to vLLM's valid-JSON flag.  ``strict``
        is accepted and ignored: vLLM's json constraint always enforces the
        schema exactly.  Raises InvalidInput for any other mode or a malformed
        schema so callers get a 400 instead of a mid-generation failure.
        """
        response_format = getattr(request, "response_format", None)
        if response_format is None:
            return None
        if not isinstance(response_format, dict) and hasattr(
            response_format, "model_dump"
        ):
            # OpenAI's inner field is "schema"; pydantic reserves that name, so
            # vLLM stores it under an alias. Dump by_alias to recover the wire
            # key ("schema") instead of the internal field name ("json_schema").
            response_format = response_format.model_dump(by_alias=True)
        if not isinstance(response_format, dict):
            return None

        fmt_type = response_format.get("type", "text")
        if fmt_type == "text":
            return None
        if fmt_type == "json_object":
            return StructuredOutputsParams(json_object=True)
        if fmt_type == "json_schema":
            json_schema = response_format.get("json_schema")
            if not isinstance(json_schema, dict) and hasattr(json_schema, "model_dump"):
                json_schema = json_schema.model_dump(by_alias=True)
            if not isinstance(json_schema, dict):
                raise InvalidInput("response_format.json_schema must be an object")
            schema = json_schema.get("schema", json_schema.get("json_schema"))
            if schema is None:
                raise InvalidInput("response_format.json_schema.schema is required")
            if not isinstance(schema, dict):
                raise InvalidInput(
                    "response_format.json_schema.schema must be a JSON Schema object"
                )
            return StructuredOutputsParams(json=schema)
        raise InvalidInput(
            f"Unsupported response_format type: {fmt_type!r} "
            "(supported: text, json_object, json_schema)"
        )

    @staticmethod
    def _resolve_tool_choice(request: ChatCompletionRequest) -> str:
        """Resolve tool_choice; only "auto" and "none" are supported.

        "required" and named-function forcing are rejected until they can
        be implemented properly (named forcing maps to structured outputs).
        """
        tool_choice = getattr(request, "tool_choice", None)
        if tool_choice is None or tool_choice == "auto":
            return "auto"
        if tool_choice == "none":
            return "none"
        raise InvalidInput(
            'tool_choice must be "auto" or "none"; "required" and named '
            "functions are not supported"
        )

    def _resolve_request_options(
        self, request: ChatCompletionRequest
    ) -> PerRequestOptions:
        """Resolve the per-request thinking / structured-output options once."""
        enable_thinking = self._resolve_enable_thinking(request)
        thinking_token_budget = self._validate_thinking_token_budget(
            getattr(request, "thinking_token_budget", None)
        )
        structured_outputs = self._resolve_structured_outputs(request)
        return PerRequestOptions(
            enable_thinking=enable_thinking,
            thinking_token_budget=thinking_token_budget,
            structured_outputs=structured_outputs,
        )

    def _apply_chat_template(
        self, messages: list, enable_thinking: bool = True, tools: list | None = None
    ) -> str:
        """Apply the tokenizer chat template with optional thinking mode and tools.

        Falls back to calling without enable_thinking if the tokenizer doesn't
        support that parameter (e.g. older Qwen models).
        """
        kwargs: dict[str, Any] = {}
        if tools is not None:
            kwargs["tools"] = tools
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )

    def apply_chat_template(
        self,
        request: ChatCompletionRequest,
        enable_thinking: bool = False,
        tool_choice: str = "auto",
    ) -> ChatPrompt:
        messages = [dict(msg) for msg in request.messages]
        tools = None
        if request.tools and tool_choice != "none":
            if self.tool_calling_enabled:
                tools = [
                    tool.model_dump() if hasattr(tool, "model_dump") else tool
                    for tool in request.tools
                ]
            else:
                logging.warning(
                    "Request included tools but tool calling is disabled "
                    "(TOOL_CALLING_ENABLED); ignoring them."
                )
        # TODO: pass through arbitrary chat_template_kwargs instead of only
        # enable_thinking, so that preserve_thinking (27B) and future
        # template kwargs are honoured without per-field plumbing.
        text = self._apply_chat_template(
            messages, enable_thinking=enable_thinking, tools=tools
        )
        return ChatPrompt(prompt=text, response_role="assistant")

    def _build_sampling_params_from_request(
        self,
        request: CompletionRequest,
        options: PerRequestOptions,
    ) -> SamplingParams:
        """Extract sampling parameters from a CompletionRequest with defaults.

        ``options.enable_thinking`` selects the default temperature / top_p /
        presence_penalty from SAMPLING_DEFAULTS.  The chat completions
        endpoint passes ``False`` when thinking is off (the common case); the
        raw /openai/v1/completions endpoint keeps its historical ``True``
        default because it has no chat-template concept.

        ``options.skip_special_tokens`` is passed through only when it is not
        None; tool-calling chat requests set it to False so the model's
        tool-call special tokens are preserved in the decoded text for the
        parser.
        """
        defaults = SAMPLING_DEFAULTS[options.enable_thinking]
        params: dict[str, Any] = dict(
            max_tokens=request.max_tokens or 32768,
            temperature=(
                request.temperature
                if request.temperature is not None
                else defaults["temperature"]
            ),
            top_p=request.top_p if request.top_p is not None else defaults["top_p"],
            top_k=request.top_k or 20,
            presence_penalty=(
                request.presence_penalty
                if request.presence_penalty is not None
                else defaults["presence_penalty"]
            ),
            repetition_penalty=request.repetition_penalty or 1.0,
            thinking_token_budget=options.thinking_token_budget,
            structured_outputs=options.structured_outputs,
        )
        if options.skip_special_tokens is not None:
            params["skip_special_tokens"] = options.skip_special_tokens
        return SamplingParams(**params)

    async def _collect_generator(self, results_generator) -> RequestOutput:
        """Consume the async generator and return the final RequestOutput."""
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output

    def _build_completion(
        self,
        final_output: RequestOutput,
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> Completion:
        """Build a non-streaming Completion from the final RequestOutput."""
        completion = final_output.outputs[0]
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(completion.token_ids)
        return Completion(
            id=request_id,
            created=created_time,
            model=model_name,
            object="text_completion",
            choices=[
                CompletionChoice(
                    index=0,
                    finish_reason=completion.finish_reason,
                    text=completion.text,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            system_fingerprint=None,
        )

    @staticmethod
    def _tool_calls_to_openai(tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Map vLLM ToolCall objects to OpenAI tool_calls dicts.

        vLLM's parsers return ToolCall objects whose ``function`` carries the
        name and a JSON-string ``arguments``. The ``id`` may already be minted
        by the parser; fall back to generating one when absent.
        """
        converted: list[dict[str, Any]] = []
        for tc in tool_calls:
            fn = tc.function
            converted.append(
                {
                    "id": getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}",
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": fn.name,
                        "arguments": fn.arguments,
                    },
                }
            )
        return converted

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request=None,
        context: dict[str, Any] | None = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletion, ErrorResponse]:
        if request.n != 1:
            raise InvalidInput("n != 1 is not supported")

        tool_choice = self._resolve_tool_choice(request)

        options = self._resolve_request_options(request)
        if options.thinking_token_budget is not None and not options.enable_thinking:
            logging.warning(
                "thinking_token_budget=%s received with thinking disabled; "
                "budget has no effect",
                options.thinking_token_budget,
            )

        chat_prompt = self.apply_chat_template(
            request,
            enable_thinking=options.enable_thinking,
            tool_choice=tool_choice,
        )
        completion_request = (
            OpenAIChatAdapterModel.chat_completion_params_to_completion_params(
                request, chat_prompt.prompt
            )
        )

        parse_tool_calls = (
            self.tool_calling_enabled and bool(request.tools) and tool_choice != "none"
        )

        if options.structured_outputs is not None and parse_tool_calls:
            logging.warning(
                "response_format and tools both set; constrained output "
                "cannot emit tool calls, tools will not be used"
            )
            parse_tool_calls = False

        if parse_tool_calls:
            # Tool-call tags are special tokens in some tokenizers and would
            # be stripped from the decoded output before the parser sees them.
            options = replace(options, skip_special_tokens=False)

        if request.stream:
            return self._stream_chat_completion(
                request,
                chat_prompt,
                completion_request,
                parse_tool_calls,
                options=options,
            )

        completion = await self.create_completion(
            completion_request,
            raw_request,
            context,
            options=options,
        )
        assert isinstance(completion, Completion)

        text = completion.choices[0].text if completion.choices else ""
        reasoning_text = None
        if options.enable_thinking:
            reasoning_text, text = self.reasoning_parser.extract_reasoning(
                text, request
            )

        if parse_tool_calls and text:
            tool_parser = self.tool_parser_cls(self.tokenizer)
            tool_call_info = tool_parser.extract_tool_calls(text, request)
            if tool_call_info.tools_called:
                message = ChatMessage(
                    role="assistant",
                    content=tool_call_info.content,
                    tool_calls=self._tool_calls_to_openai(tool_call_info.tool_calls),
                )
                message.reasoning = reasoning_text
                return ChatCompletion(
                    id=completion.id,
                    created=completion.created,
                    model=completion.model,
                    object="chat.completion",
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=message,
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=completion.usage,
                )

        chat_completion = self.completion_to_chat_completion(
            completion, chat_prompt.response_role
        )
        chat_completion.choices[0].message.content = text
        chat_completion.choices[0].message.reasoning = reasoning_text
        return chat_completion

    async def _stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        chat_prompt: ChatPrompt,
        completion_request: CompletionRequest,
        parse_tool_calls: bool = False,
        *,
        options: PerRequestOptions,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completions as ``object:"chat.completion.chunk"`` chunks.

        Per-token chunks carry ``delta.content`` (and ``delta.reasoning`` when
        thinking is enabled) with ``finish_reason=None``.  With tool calling
        active, streamed tool calls arrive as incremental ``delta.tool_calls``
        deltas and the raw tool-call tag text is never emitted as content.
        Exactly one final chunk follows the stream, carrying usage and either
        finish_reason "tool_calls" (when a tool call was streamed) or the
        engine's finish_reason.
        """
        request_id = completion_request.request_id or uuid.uuid4().hex
        sampling_params = self._build_sampling_params_from_request(
            completion_request,
            options,
        )

        try:
            results_generator = self.model.generate(
                prompt=completion_request.prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )
        except Exception as e:
            logging.error("Error during streaming inference: %s", e)
            raise InferenceError(f"Error during streaming inference: {e}")

        previous_texts: list[str] = [""]
        previous_num_tokens: list[int] = [0]
        previous_token_ids: list[list[int]] = [[]]
        previous_contents: list[str] = [""]
        created_time = int(time.time())
        prompt_tokens = 0
        final_finish_reason: str | None = None
        final_completion_tokens = 0
        tools_streamed = False

        parser = self.reasoning_parser if options.enable_thinking else None
        tool_parser = self.tool_parser_cls(self.tokenizer) if parse_tool_calls else None

        try:
            async for request_output in results_generator:
                for output in request_output.outputs:
                    i = output.index
                    track_tokens = parser is not None or tool_parser is not None
                    self._ensure_output_capacity(
                        previous_texts,
                        previous_num_tokens,
                        i,
                        previous_token_ids=(
                            previous_token_ids if track_tokens else None
                        ),
                        previous_contents=(
                            previous_contents if tool_parser is not None else None
                        ),
                    )

                    prev_text = previous_texts[i]
                    curr_text = output.text
                    delta_text = curr_text[len(prev_text) :]
                    curr_tokens = output.token_ids

                    previous_texts[i] = curr_text
                    previous_num_tokens[i] = len(curr_tokens)

                    if output.finish_reason is not None:
                        prompt_tokens = len(request_output.prompt_token_ids)
                        final_finish_reason = output.finish_reason
                        final_completion_tokens = len(curr_tokens)

                    if track_tokens:
                        prev_tokens = previous_token_ids[i]
                        delta_tokens = curr_tokens[len(prev_tokens) :]
                        previous_token_ids[i] = list(curr_tokens)
                    else:
                        prev_tokens = None
                        delta_tokens = None

                    if parser is not None:
                        d = parser.extract_reasoning_streaming(
                            prev_text,
                            curr_text,
                            delta_text,
                            prev_tokens,
                            curr_tokens,
                            delta_tokens,
                        )
                        if d is None:
                            continue
                        delta_reasoning = d.reasoning
                        delta_content = d.content
                    else:
                        delta_reasoning = None
                        delta_content = delta_text

                    tool_calls = None
                    if tool_parser is not None and delta_content:
                        prev_content = previous_contents[i]
                        curr_content = prev_content + delta_content
                        previous_contents[i] = curr_content
                        d = tool_parser.extract_tool_calls_streaming(
                            prev_content,
                            curr_content,
                            delta_content,
                            prev_tokens,
                            curr_tokens,
                            delta_tokens,
                            request,
                        )
                        if d is None:
                            # The tool parser suppresses only the content side;
                            # a delta that still carries a reasoning fragment
                            # (e.g. spanning the </think> boundary) must emit
                            # as a reasoning-only chunk rather than be dropped.
                            if not delta_reasoning:
                                continue
                            delta_content = None
                        else:
                            delta_content = d.content
                            raw_tool_calls = d.tool_calls
                            if raw_tool_calls:
                                tools_streamed = True
                                tool_calls = [
                                    tc.model_dump(exclude_none=True)
                                    for tc in raw_tool_calls
                                ]

                    delta_kwargs: dict[str, Any] = {
                        "role": chat_prompt.response_role,
                        "content": delta_content,
                        "reasoning": delta_reasoning,
                    }
                    if tool_calls:
                        delta_kwargs["tool_calls"] = tool_calls

                    chunk = ChatCompletionChunk(
                        id=request_id,
                        created=created_time,
                        model=request.model,
                        object="chat.completion.chunk",
                        choices=[
                            ChunkChoice(
                                index=i,
                                delta=ChoiceDelta(**delta_kwargs),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as e:
            logging.error("Error during streaming inference: %s", e)
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=final_completion_tokens,
            total_tokens=prompt_tokens + final_completion_tokens,
        )
        final_chunk = ChatCompletionChunk(
            id=request_id,
            created=created_time,
            model=request.model,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason=(
                        "tool_calls" if tools_streamed else final_finish_reason
                    ),
                )
            ],
            usage=usage,
        )

        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request=None,
        context: dict | None = None,
        options: PerRequestOptions | None = None,
    ) -> Union[AsyncGenerator[str, None], Completion]:
        if options is None:
            options = RAW_COMPLETIONS_DEFAULTS
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = self.tokenizer.decode(prompt)

        sampling_params = self._build_sampling_params_from_request(request, options)
        request_id = request.request_id or uuid.uuid4().hex

        try:
            results_generator = self.model.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )
        except Exception as e:
            logging.error("Error during inference: %s", e)
            raise InferenceError(f"Error during inference: {e}")

        if request.stream:
            return self._stream_completion(
                results_generator, request_id, int(time.time()), request.model
            )

        try:
            final_output = await self._collect_generator(results_generator)
        except ValueError as e:
            if options.structured_outputs is not None:
                raise InvalidInput(f"Invalid structured output request: {e}") from e
            raise
        return self._build_completion(
            final_output, request_id, int(time.time()), request.model
        )

    @staticmethod
    def _ensure_output_capacity(
        previous_texts: list,
        previous_num_tokens: list,
        index: int,
        previous_token_ids: list | None = None,
        previous_contents: list | None = None,
    ) -> None:
        """Grow tracking lists to accommodate output at the given index.

        vLLM can return results from multiple parallel outputs (e.g. n > 1).
        We lazily expand the tracking lists so we can compute per-output deltas.
        """
        if index >= len(previous_texts):
            gap = index - len(previous_texts) + 1
            previous_texts.extend([""] * gap)
            previous_num_tokens.extend([0] * gap)
            if previous_token_ids is not None:
                previous_token_ids.extend([[]] * gap)
            if previous_contents is not None:
                previous_contents.extend([""] * gap)

    @staticmethod
    def _build_stream_chunk(
        request_id: str,
        created_time: int,
        model_name: str,
        index: int,
        delta_text: str,
        finish_reason: str | None,
        usage: UsageInfo | None,
    ) -> CompletionChunk:
        """Build a single SSE chunk for the streaming response."""
        return CompletionChunk(
            id=request_id,
            created=created_time,
            model=model_name,
            object="text_completion",
            choices=[
                CompletionChunkChoice(
                    index=index,
                    finish_reason=finish_reason,
                    text=delta_text,
                )
            ],
            usage=usage,
            system_fingerprint=None,
        )

    async def _stream_completion(
        self,
        results_generator,
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> AsyncGenerator[str, None]:
        previous_texts = [""]
        previous_num_tokens = [0]

        try:
            async for request_output in results_generator:
                for output in request_output.outputs:
                    i = output.index
                    self._ensure_output_capacity(previous_texts, previous_num_tokens, i)

                    delta_text = output.text[len(previous_texts[i]) :]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)

                    usage = None
                    if output.finish_reason is not None:
                        prompt_tokens = len(request_output.prompt_token_ids)
                        completion_tokens = len(output.token_ids)
                        usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )

                    chunk = self._build_stream_chunk(
                        request_id,
                        created_time,
                        model_name,
                        i,
                        delta_text,
                        output.finish_reason,
                        usage,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as e:
            logging.error("Error during streaming inference: %s", e)
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield "data: [DONE]\n\n"

    def preprocess(self, payload: dict, headers: dict[str, str] = None) -> dict:
        prompt = payload.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise InvalidInput("Expected a 'prompt' field (string) in the payload.")

        enable_thinking = strtobool(str(payload.get("reasoning", False)))
        defaults = SAMPLING_DEFAULTS[enable_thinking]

        max_tokens = int(payload.get("max_tokens", 32768))
        temperature = float(payload.get("temperature", defaults["temperature"]))
        top_p = float(payload.get("top_p", defaults["top_p"]))
        top_k = int(payload.get("top_k", 20))
        presence_penalty = float(
            payload.get("presence_penalty", defaults["presence_penalty"])
        )
        repetition_penalty = float(payload.get("repetition_penalty", 1.0))
        thinking_token_budget = self._validate_thinking_token_budget(
            payload.get("thinking_token_budget")
        )

        system = payload.get("system")
        messages = self._build_messages(prompt, system)

        text = self._apply_chat_template(messages, enable_thinking=enable_thinking)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            thinking_token_budget=thinking_token_budget,
        )

        return {
            "prompt": text,
            "sampling_params": sampling_params,
        }

    async def predict(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        try:
            prompt = inputs["prompt"]
            sampling_params = inputs["sampling_params"]

            request_id = uuid.uuid4().hex
            results_generator = self.model.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            completion = final_output.outputs[0]
            return {
                "model_name": self.name,
                "response": completion.text,
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(completion.token_ids),
            }

        except Exception as e:
            error_message = f"Error during inference: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "qwen36-27b")
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen3.6-27B-FP8")
    trust_remote_code = strtobool(os.environ.get("TRUST_REMOTE_CODE", "True"))
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "32768"))
    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "2"))
    dtype = os.environ.get("DTYPE", "auto")
    language_model_only = strtobool(os.environ.get("LANGUAGE_MODEL_ONLY", "True"))
    skip_mm_profiling = strtobool(os.environ.get("SKIP_MM_PROFILING", "True"))
    max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", "128"))
    max_num_batched_tokens = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "32768"))
    block_size = int(os.environ.get("BLOCK_SIZE", "64"))
    attention_backend = os.environ.get("ATTENTION_BACKEND", "TRITON_ATTN")
    kv_cache_dtype = os.environ.get("KV_CACHE_DTYPE", "auto")
    disable_custom_all_reduce = strtobool(
        os.environ.get("DISABLE_CUSTOM_ALL_REDUCE", "False")
    )
    enforce_eager = strtobool(os.environ.get("ENFORCE_EAGER", "False"))
    disable_log_stats = strtobool(os.environ.get("DISABLE_LOG_STATS", "False"))
    tool_calling_enabled = strtobool(os.environ.get("TOOL_CALLING_ENABLED", "False"))
    tool_call_parser = os.environ.get("TOOL_CALL_PARSER", "hermes")

    model = Qwen36Model(
        name=model_name,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        language_model_only_flag=language_model_only,
        skip_mm_profiling_flag=skip_mm_profiling,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
        attention_backend=attention_backend,
        kv_cache_dtype=kv_cache_dtype,
        disable_custom_all_reduce=disable_custom_all_reduce,
        enforce_eager=enforce_eager,
        disable_log_stats=disable_log_stats,
        tool_calling_enabled=tool_calling_enabled,
        tool_call_parser=tool_call_parser,
    )

    model.load()
    kserve.ModelServer().start([model])

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Union

import kserve
from kserve.errors import InferenceError, InvalidInput
from kserve.protocol.rest.openai import ChatPrompt, OpenAIChatAdapterModel
from kserve.protocol.rest.openai.types import (
    ChatCompletionRequest,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    UsageInfo,
)
from vllm import RequestOutput, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from python.type_utils import strtobool

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

# Sampling defaults keyed by enable_thinking (True = thinking, False = instruct).
# See https://huggingface.co/Qwen/Qwen3.6-27B-FP8#recommended-sampling-parameters
SAMPLING_DEFAULTS = {
    True: {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
    False: {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5},
}


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
        disable_custom_all_reduce: bool = False,
        enforce_eager: bool = False,
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
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.enforce_eager = enforce_eager
        self.model = None
        self.tokenizer = None
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
                tensor_parallel_size=self.tensor_parallel_size,
                disable_custom_all_reduce=self.disable_custom_all_reduce,
                enforce_eager=self.enforce_eager,
                dtype=self.dtype,
                language_model_only=self.language_model_only_flag,
                skip_mm_profiling=self.skip_mm_profiling_flag,
                enable_prefix_caching=True,
                reasoning_parser="qwen3",
            )
            self.model = AsyncLLMEngine.from_engine_args(engine_args)
            self.tokenizer = self.model.tokenizer
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

    def _apply_chat_template(self, messages: list, enable_thinking: bool = True) -> str:
        """Apply the tokenizer chat template with optional thinking mode.

        Falls back to calling without enable_thinking if the tokenizer doesn't
        support that parameter (e.g. older Qwen models).
        """
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def apply_chat_template(self, request: ChatCompletionRequest) -> ChatPrompt:
        messages = [dict(msg) for msg in request.messages]
        text = self._apply_chat_template(messages, enable_thinking=False)
        return ChatPrompt(prompt=text, response_role="assistant")

    def _build_sampling_params_from_request(
        self, request: CompletionRequest
    ) -> SamplingParams:
        """Extract sampling parameters from a CompletionRequest with defaults.

        Defaults use thinking-mode values (temperature=1.0, top_p=0.95,
        presence_penalty=0.0) since the OpenAI endpoint always enables thinking.
        """
        return SamplingParams(
            max_tokens=request.max_tokens or 32768,
            temperature=request.temperature if request.temperature is not None else 1.0,
            top_p=request.top_p if request.top_p is not None else 0.95,
            top_k=request.top_k or 20,
            presence_penalty=(
                request.presence_penalty
                if request.presence_penalty is not None
                else 0.0
            ),
            repetition_penalty=request.repetition_penalty or 1.0,
        )

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

    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request=None,
        context: dict | None = None,
    ) -> Union[AsyncGenerator[str, None], Completion]:
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = self.tokenizer.decode(prompt)

        sampling_params = self._build_sampling_params_from_request(request)
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

        final_output = await self._collect_generator(results_generator)
        return self._build_completion(
            final_output, request_id, int(time.time()), request.model
        )

    @staticmethod
    def _ensure_output_capacity(
        previous_texts: list,
        previous_num_tokens: list,
        index: int,
    ) -> None:
        """Grow tracking lists to accommodate output at the given index.

        vLLM can return results from multiple parallel outputs (e.g. n > 1).
        We lazily expand the tracking lists so we can compute per-output deltas.
        """
        if index >= len(previous_texts):
            gap = index - len(previous_texts) + 1
            previous_texts.extend([""] * gap)
            previous_num_tokens.extend([0] * gap)

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
            raise InferenceError(f"Error during streaming inference: {e}")

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
    disable_custom_all_reduce = strtobool(
        os.environ.get("DISABLE_CUSTOM_ALL_REDUCE", "False")
    )
    enforce_eager = strtobool(os.environ.get("ENFORCE_EAGER", "False"))

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
        disable_custom_all_reduce=disable_custom_all_reduce,
        enforce_eager=enforce_eager,
    )

    model.load()
    kserve.ModelServer().start([model])

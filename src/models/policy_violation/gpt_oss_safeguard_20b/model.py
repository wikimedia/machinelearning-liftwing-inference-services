import logging
import os
import uuid
from distutils.util import strtobool

import kserve
from kserve.errors import InferenceError, InvalidInput
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
)
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs import TokensPrompt

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class PolicyViolationModel(kserve.Model):
    def __init__(
        self,
        name: str,
        model_path: str,
        trust_remote_code: bool,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_batched_tokens: int,
        block_size: int,
        attention_backend: str,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.attention_backend = attention_backend
        self.model = None
        self.encoding = None
        self.ready = False

    def load(self) -> None:
        """
        Load Harmony encodings and vLLM engine
        """
        try:
            logging.info("Loading Harmony encodings...")
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

            logging.info("Loading Async vLLM engine...")
            # Define Async Engine Arguments
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                trust_remote_code=self.trust_remote_code,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_batched_tokens,
                block_size=self.block_size,
                attention_backend=self.attention_backend,
                compilation_config={
                    "use_inductor_graph_partition": True,
                },
            )
            # Initialize the Async Engine
            self.model = AsyncLLMEngine.from_engine_args(engine_args)
            self.ready = True
            logging.info("Model loaded successfully!")
        except Exception as e:
            error_message = f"Failed to load model or encodings. Reason: {e}"
            logging.critical(error_message)
            raise kserve.errors.ModelMissingError(error_message)

    def preprocess(self, payload: dict, headers: dict[str, str] = None) -> dict:
        """
        Preprocess the input data by validating and converting it to Harmony token IDs.
        """
        if "messages" not in payload:
            error_message = "Invalid payload format. Must contain a 'messages' list."
            logging.error(error_message)
            raise InvalidInput(error_message)

        messages_data = payload["messages"]
        developer_prompt = payload.get("developer_prompt")
        max_tokens = int(payload.get("max_tokens", 4096))
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.95))

        logging.info("Building Harmony conversation...")
        msgs = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new())]

        if developer_prompt:
            msgs.append(
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(developer_prompt),
                )
            )

        for m in messages_data:
            role = {"user": Role.USER, "assistant": Role.ASSISTANT}.get(
                m.get("role"), Role.USER
            )
            msgs.append(Message.from_role_and_content(role, m.get("content", "")))

        conversation = Conversation.from_messages(msgs)

        # Render conversation to token IDs
        prefill_ids = self.encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        # Configure sampling
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
        )

        prompt = TokensPrompt(prompt_token_ids=prefill_ids)

        return {
            "prompt": prompt,
            "sampling_params": sampling_params,
        }

    async def predict(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """
        Perform inference using vLLM
        """
        try:
            logging.info("Performing inference...")

            prompt = inputs["prompt"]
            sampling_params = inputs["sampling_params"]

            # Generate response using an async generator
            request_id = uuid.uuid4().hex
            results_generator = self.model.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            # Await the final output from the async generator
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            # Extract just the raw token IDs to pass to postprocess
            completion_token_ids = final_output.outputs[0].token_ids

            return {"completion_token_ids": completion_token_ids}

        except Exception as e:
            error_message = f"Error during inference: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)

    def postprocess(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """
        Parse the Harmony output channels
        """
        try:
            logging.info("Post-processing generated tokens...")
            completion_token_ids = inputs["completion_token_ids"]

            # Parse Harmony output channels
            response_messages = self.encoding.parse_messages_from_completion_tokens(
                completion_token_ids, Role.ASSISTANT
            )

            # Extract reasoning and verdict from channels
            reasoning = ""
            verdict = ""
            for msg in response_messages:
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                if msg.channel == "final":
                    verdict = content.strip()
                else:
                    reasoning += content

            # Fallback if no final channel is found
            if not verdict and response_messages:
                content = response_messages[-1].content
                verdict = content if isinstance(content, str) else str(content)

            return {
                "reasoning": reasoning.strip(),
                "verdict": verdict,
            }

        except Exception as e:
            error_message = f"Error during post-processing: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "policy-violation")
    # Ensure a TIKTOKEN_ENCODINGS_BASE env var exists in order for the isvc to use offline caches like MODEL_PATH below
    model_path = os.environ.get(
        "MODEL_PATH", "/mnt/models/snapshots/8a11e17b25c973a24099d4016bf2e17dd7ec1574"
    )
    trust_remote_code = strtobool(os.environ.get("TRUST_REMOTE_CODE", "True"))
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
    # Context length limit (gpt-oss-safeguard-20b supports up to 131k, default to 8192 if safe)
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", 8192))
    max_num_batched_tokens = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", 8192))
    block_size = int(os.environ.get("BLOCK_SIZE", 64))
    # Defaulting to Triton Attention backend since it supports both:
    # MI210 GPU: https://phabricator.wikimedia.org/P89093#L40 and
    # MI300x GPU: https://phabricator.wikimedia.org/P89862#L39
    attention_backend = os.environ.get("ATTENTION_BACKEND", "TRITON_ATTN")

    model = PolicyViolationModel(
        name=model_name,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
        attention_backend=attention_backend,
    )

    model.load()
    kserve.ModelServer().start([model])

import logging
import os
from distutils.util import strtobool

import kserve
from kserve.errors import InferenceError, InvalidInput
from vllm import LLM, SamplingParams

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

# CoPE-B prompt body. Unlike CoPE-A, there is NO "INSTRUCTIONS" header and NO
# "ANSWER" footer: the Gemma-4 chat template's role markers replace them. This
# string is passed as the user-turn content and rendered via the chat template.
PROMPT_TEMPLATE = """\
Examine the given POLICY and determine if the given CONTENT meets the criteria \
for ANY of the LABELS. Answer "1" if yes, and "0" if no.


POLICY
======

{policy}


CONTENT
=======

{content}
"""


class CoPEBModel(kserve.Model):
    def __init__(
        self,
        name: str,
        model_path: str,
        trust_remote_code: bool,
        dtype: str,
        quantization: str,
        max_model_len: int,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.model = None
        self.ready = False

    def load(self) -> None:
        """Load the vLLM engine with the CoPE-B-A4B model."""
        try:
            logging.info("Loading vLLM model...")
            self.model = LLM(
                model=self.model_path,
                trust_remote_code=self.trust_remote_code,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                quantization=self.quantization,
            )

            self.ready = True
            logging.info("Model loaded successfully!")
        except Exception as e:
            error_message = f"Failed to load model. Reason: {e}"
            logging.critical(error_message)
            raise kserve.errors.ModelMissingError(error_message)

    def preprocess(self, payload: dict, headers: dict[str, str] = None) -> dict:
        """Validate the payload and build the CoPE-B chat conversation."""
        if "content" not in payload:
            raise InvalidInput(
                "Invalid payload format. Must contain a 'content' field."
            )

        if "policy" not in payload:
            raise InvalidInput("Invalid payload format. Must contain a 'policy' field.")

        content = payload["content"]
        policy = payload["policy"]
        max_tokens = int(payload.get("max_tokens", 1))
        temperature = float(payload.get("temperature", 0.0))

        # CoPE-B is a binary classifier that emits a single 0/1 token, so a small
        # max_tokens is expected; the upper bound is a generous safety cap.
        if not 0 < max_tokens <= 256:
            raise InvalidInput(
                f"'max_tokens' must be between 1 and 256, got {max_tokens}."
            )

        if not 0.0 <= temperature <= 2.0:
            raise InvalidInput(
                f"'temperature' must be between 0.0 and 2.0, got {temperature}."
            )

        prompt = PROMPT_TEMPLATE.format(policy=policy, content=content)

        # CoPE-B requires the Gemma-4 chat template: the prompt is a user turn
        # and the verdict comes back as the assistant turn. Passing a messages
        # list to LLM.chat() makes vLLM apply the model's chat template (the
        # role markers that replace CoPE-A's INSTRUCTIONS/ANSWER text), unlike
        # the CoPE-A server which raw-concatenates its prompt for LLM.generate().
        # See: https://huggingface.co/zentropi-ai/cope-b-a4b#1-cope-b-uses-the-gemma-4-chat-template
        messages = [{"role": "user", "content": prompt}]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "messages": messages,
            "sampling_params": sampling_params,
        }

    def predict(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """Perform inference using vLLM."""
        try:
            logging.info("Performing inference...")

            output = self.model.chat(
                messages=inputs["messages"],
                sampling_params=inputs["sampling_params"],
            )

            generated_text = output[0].outputs[0].text.strip()
            return {"generated_text": generated_text}

        except Exception as e:
            error_message = f"Error during inference: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)

    def postprocess(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """Parse the model output into a binary verdict."""
        try:
            generated_text = inputs["generated_text"]

            # CoPE-B outputs "0" or "1". If the output is unexpectedly not "0"
            # or "1", log a warning and return None.
            if generated_text in ("0", "1"):
                violation = int(generated_text)
            else:
                logging.warning(
                    "Unexpected model output: '%s', defaulting to raw text",
                    generated_text,
                )
                violation = None

            return {
                "violation": violation,
            }

        except Exception as e:
            error_message = f"Error during post-processing: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "cope-b-a4b")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/snapshots/cope-b-a4b")
    trust_remote_code = strtobool(os.environ.get("TRUST_REMOTE_CODE", "True"))
    dtype = os.environ.get("DTYPE", "bfloat16")
    quantization = os.environ.get("QUANTIZATION", None)
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", 8192))

    model = CoPEBModel(
        name=model_name,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
    )

    model.load()
    kserve.ModelServer().start([model])

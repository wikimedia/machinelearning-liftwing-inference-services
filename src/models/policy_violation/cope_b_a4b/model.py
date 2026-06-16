import logging
import math
import os
from distutils.util import strtobool

import kserve
from kserve.errors import InferenceError, InvalidInput
from vllm import LLM, SamplingParams

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

# Number of top-token logprobs to request for the generated position. CoPE-B
# emits a single 0/1 token, so both classes are virtually always in the top
# few; 20 is a generous margin to ensure both "0" and "1" appear.
LOGPROBS_TOP_K = 20

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

        # Request top-k logprobs so postprocess can derive a confidence score
        # from the verdict token's probability. Confidence is only meaningful
        # for a deterministic verdict, i.e. temperature == 0.0 (greedy).
        # vLLM SamplingParams.logprobs (returns top-k token logprobs per
        # position): https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=LOGPROBS_TOP_K,
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

            completion = output[0].outputs[0]
            generated_text = completion.text.strip()

            # logprobs is a list with one entry per generated token. CoPE-B
            # generates a single verdict token, so the first position holds the
            # class logprobs: {token_id: Logprob(logprob=..., decoded_token=...)}.
            # vLLM output object (CompletionOutput.logprobs / Logprob):
            # https://docs.vllm.ai/en/latest/api/vllm/outputs.html
            first_token_logprobs = (
                completion.logprobs[0] if completion.logprobs else None
            )

            # Extract the logprob for the "0" and "1" tokens, if present.
            logprob_0 = None
            logprob_1 = None
            if first_token_logprobs:
                for logprob_obj in first_token_logprobs.values():
                    token = (logprob_obj.decoded_token or "").strip()
                    if token == "0":
                        logprob_0 = logprob_obj.logprob
                    elif token == "1":
                        logprob_1 = logprob_obj.logprob

            return {
                "generated_text": generated_text,
                "logprob_0": logprob_0,
                "logprob_1": logprob_1,
            }

        except Exception as e:
            error_message = f"Error during inference: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)

    def postprocess(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """Parse the model output into a binary verdict plus a confidence score."""
        try:
            generated_text = inputs["generated_text"]
            logprob_0 = inputs.get("logprob_0")
            logprob_1 = inputs.get("logprob_1")

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

            result = {"violation": violation}

            # Expose the positive-class (violation==1) probability and logprob so
            # callers can threshold to trade recall for precision. Both are
            # returned at full precision: CoPE-B is highly peaked, so exp(logprob)
            # saturates toward 1.0 and loses resolution at extreme confidence,
            # whereas the raw logprob preserves ordering and is the more robust
            # signal for thresholding. Thresholds should be calibrated against a
            # labeled sample of the caller's own traffic. Using the output token
            # probability/logprob as a confidence signal, and the need to
            # recalibrate it, are described in the CoPE-B model card:
            # https://huggingface.co/zentropi-ai/cope-b-a4b#2-recalibrate-confidence-thresholds
            if logprob_1 is not None:
                result["p_violation"] = math.exp(logprob_1)
                result["logprob_violation"] = logprob_1
            if logprob_0 is not None:
                result["p_safe"] = math.exp(logprob_0)
                result["logprob_safe"] = logprob_0

            # Confidence in the verdict actually returned.
            if violation == 1 and logprob_1 is not None:
                result["confidence"] = math.exp(logprob_1)
            elif violation == 0 and logprob_0 is not None:
                result["confidence"] = math.exp(logprob_0)

            return result

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

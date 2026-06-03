import logging
import os
from distutils.util import strtobool

import kserve
import torch
from kserve.errors import InferenceError, InvalidInput
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

# CoPE-B prompt body. Unlike CoPE-A, there is NO "INSTRUCTIONS" header and NO
# "ANSWER" footer: the Gemma-4 chat template's role markers replace them. This
# string is passed as the user-turn content and run through apply_chat_template.
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
        max_model_len: int,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.model = None
        self.tokenizer = None
        self.ready = False

    def load(self) -> None:
        """
        Load the CoPE-B-A4B model and tokenizer using HF transformers.
        """
        try:
            logging.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )

            logging.info("Loading model...")
            torch_dtype = getattr(torch, self.dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            self.model.eval()

            self.ready = True
            logging.info("Model loaded successfully!")
        except Exception as e:
            error_message = f"Failed to load model. Reason: {e}"
            logging.critical(error_message)
            raise kserve.errors.ModelMissingError(error_message)

    def preprocess(self, payload: dict, headers: dict[str, str] = None) -> dict:
        """
        Validate the payload and build the CoPE-B prompt via the chat template.
        """
        if "content" not in payload:
            error_message = "Invalid payload format. Must contain a 'content' field."
            logging.error(error_message)
            raise InvalidInput(error_message)

        if "policy" not in payload:
            error_message = "Invalid payload format. Must contain a 'policy' field."
            logging.error(error_message)
            raise InvalidInput(error_message)

        content = payload["content"]
        policy = payload["policy"]
        max_new_tokens = int(payload.get("max_tokens", 1))
        temperature = float(payload.get("temperature", 0.0))

        # CoPE-B is a binary classifier that emits a single 0/1 token, so a small
        # max_tokens is expected; the upper bound is a generous safety cap.
        if not 0 < max_new_tokens <= 256:
            error_message = (
                f"'max_tokens' must be between 1 and 256, got {max_new_tokens}."
            )
            logging.error(error_message)
            raise InvalidInput(error_message)

        if not 0.0 <= temperature <= 2.0:
            error_message = (
                f"'temperature' must be between 0.0 and 2.0, got {temperature}."
            )
            logging.error(error_message)
            raise InvalidInput(error_message)

        prompt = PROMPT_TEMPLATE.format(policy=policy, content=content)

        # CoPE-B requires the Gemma-4 chat template: the prompt is a user turn
        # and the verdict comes back as the assistant turn. apply_chat_template
        # adds the role markers that replace CoPE-A's INSTRUCTIONS/ANSWER text.
        # See details in: https://huggingface.co/zentropi-ai/cope-b-a4b#1-cope-b-uses-the-gemma-4-chat-template
        messages = [{"role": "user", "content": prompt}]
        # return_dict=True yields a BatchEncoding (input_ids + attention_mask)
        # which we unpack into generate() with **. Passing the BatchEncoding
        # positionally would make generate() treat it as the input tensor and
        # fail on `.shape`.
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        # Unlike the vLLM servers, the transformers path has no built-in
        # context-length enforcement, so guard against over-long inputs here to
        # avoid attempting a generation that could OOM the GPU.
        prompt_len = encoded["input_ids"].shape[1]
        if prompt_len > self.max_model_len:
            error_message = f"""
            Input is {prompt_len} tokens, exceeding the maximum of
            {self.max_model_len}. Shorten the policy and/or content.
            """
            logging.error(error_message)
            raise InvalidInput(error_message)

        return {
            "encoded": encoded,
            "prompt_len": prompt_len,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

    def predict(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """
        Perform inference using transformers.
        """
        try:
            logging.info("Performing inference...")

            encoded = inputs["encoded"]
            prompt_len = inputs["prompt_len"]
            temperature = inputs["temperature"]
            # temperature == 0.0 means greedy (do_sample=False), matching the
            # deterministic single-token verdict the model is designed for.
            do_sample = temperature > 0.0

            generate_kwargs = {
                "max_new_tokens": inputs["max_new_tokens"],
                "do_sample": do_sample,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature

            with torch.no_grad():
                output = self.model.generate(**encoded, **generate_kwargs)

            # Decode only the newly generated tokens (strip the prompt).
            generated_ids = output[0][prompt_len:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            return {"generated_text": generated_text}

        except Exception as e:
            error_message = f"Error during inference: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)

    def postprocess(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """
        Parse the model output into a binary verdict.
        """
        try:
            generated_text = inputs["generated_text"]

            # CoPE-B outputs "0" or "1". If the output is unexpectedly not "0" or "1", log a warning and return None.
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
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", 8192))

    model = CoPEBModel(
        name=model_name,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
    )

    model.load()
    kserve.ModelServer().start([model])

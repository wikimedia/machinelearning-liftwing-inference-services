import logging
import os
from distutils.util import strtobool

import kserve
import torch
import torch.nn.functional as F
from kserve.errors import InferenceError, InvalidInput
from vllm import LLM

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class EmbeddingModel(kserve.Model):
    def __init__(
        self,
        name: str,
        model_path: str,
        model_version: str,
        dtype: str,
        trust_remote_code: bool,
        gpu_memory_utilization: float,
        max_model_len: int,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.model_version = model_version
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.model = None
        self.ready = False

    def load(self) -> None:
        """
        Load the vLLM engine for embeddings.
        """
        try:
            logging.info("Loading vLLM model...")

            # Initialize vLLM without task="embed" as this was causing a crash due to our ROCm-specific setup.
            # Instead, we will call embed() directly in predict() which works without the task argument.
            # Unlike the previous transformers-based isvc, vLLM handles tokenizer, attention implementation, and pooling internally.
            self.model = LLM(
                model=self.model_path,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                enforce_eager=False,  # Allows CUDA graph capture for performance
            )

            self.ready = True
            logging.info("vLLM model loaded successfully!")
        except Exception as e:
            error_message = f"Failed to load vLLM model. Reason: {e}"
            logging.critical(error_message)
            raise kserve.errors.InferenceError(error_message)

    def preprocess(self, payload: dict, headers: dict[str, str]) -> list[str]:
        """
        Preprocess the input data. vLLM expects a list of strings.
        Supports OpenAI-compatible API request format. (see T412338#11482782)
        """
        if "input" in payload:
            inputs = payload["input"]
        else:
            error_message = "Invalid payload format. Use {'input': ['text1', 'text2']}"
            logging.error(error_message)
            raise InvalidInput(error_message)

        # Ensure input is a list
        if isinstance(inputs, str):
            inputs = [inputs]

        return inputs

    def predict(self, inputs: list[str], headers: dict[str, str] = None) -> dict:
        """
        Perform inference using vLLM to generate embeddings.
        Supports OpenAI-compatible API response format. (see T412338#11482782)
        """
        try:
            logging.info("Performing inference...")

            # vLLM inference
            # model.embed returns a list of RequestOutput objects
            outputs = self.model.embed(inputs)

            # Extract embeddings from vLLM output
            # Each output has an `outputs` attribute which contains the `embedding`
            raw_embeddings = [output.outputs.embedding for output in outputs]

            # Convert to tensor for normalization
            # We use the device of the first embedding (likely CPU return from vLLM)
            # or force to CPU for F.normalize calculation
            tensor_embeddings = torch.tensor(raw_embeddings)

            # Normalize embeddings (Important for cosine similarity)
            # vLLM returns raw embeddings, so normalization is still required manually
            normalized_embeddings = F.normalize(tensor_embeddings, p=2, dim=1)

            # Format response in OpenAI API format
            data = [
                {
                    "object": "embedding",
                    "embedding": embedding.tolist(),
                    "index": idx,
                }
                for idx, embedding in enumerate(normalized_embeddings)
            ]

            return {
                "object": "list",
                "data": data,
                "model": self.model_version or self.name,
            }

        except Exception as e:
            error_message = f"Error during inference: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "qwen3-embedding")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
    model_version = os.environ.get("MODEL_VERSION", "")
    dtype = os.environ.get(
        "DTYPE", "float16"
    )  # vLLM supports 'auto', 'float16', 'bfloat16'
    trust_remote_code = strtobool(os.environ.get("TRUST_REMOTE_CODE", "True"))
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))

    # Context length limit (Qwen3-Embedding supports up to 32k, default to 8192 if safe)
    # If not set, vLLM attempts to derive it from config.json
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", 8192))

    model = EmbeddingModel(
        name=model_name,
        model_path=model_path,
        model_version=model_version,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    model.load()
    kserve.ModelServer().start([model])

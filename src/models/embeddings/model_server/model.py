import logging
import os
from distutils.util import strtobool

import kserve
import torch
import torch.nn.functional as F
from kserve.errors import InferenceError, InvalidInput
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class EmbeddingModel(kserve.Model):
    def __init__(
        self,
        name: str,
        model_path: str,
        model_version: str,
        local_files_only: bool,
        dtype: torch.dtype,
        attn_implementation: str,
        max_length: int,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.model_version = model_version
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.ready = False
        self.local_files_only = local_files_only
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.max_length = max_length

    def load(self) -> None:
        """
        Load the tokenizer and model for embeddings.
        """
        try:
            logging.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )

            logging.info("Loading model...")
            self.model = AutoModel.from_pretrained(
                self.model_path,
                attn_implementation=self.attn_implementation,  # Qwen recommends enabling fa2 for better acceleration and memory saving
                torch_dtype=self.dtype,  # Use float16 for ROCm speed
                device_map=self.device,
                local_files_only=self.local_files_only,
            )
            self.model.eval()
            self.ready = True
        except Exception as e:
            error_message = f"Failed to load model or tokenizer. Reason: {e}"
            logging.critical(error_message)
            raise kserve.errors.InferenceError(error_message)

    def last_token_pooling(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts the embedding from the last token (EOS) position.
        This is common for Qwen/LLM-based embedding models.
        """
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def preprocess(self, payload: dict, headers: dict[str, str]) -> torch.Tensor:
        """
        Preprocess the input data by validating and tokenizing it.
        Supports OpenAI-compatible API request format. (see T412338#11482782)
        """
        if "input" in payload:
            inputs = payload["input"]
        else:
            error_message = "Invalid payload format. Use {'input': ['text1', 'text2']}"
            logging.error(error_message)
            raise InvalidInput(error_message)

        logging.info("Tokenizing inputs...")
        if self.max_length is not None:
            inputs = [input[: self.max_length] for input in inputs]
        encoded_input = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        ).to(self.device)
        return encoded_input

    def predict(
        self, encoded_input: torch.Tensor, headers: dict[str, str] = None
    ) -> dict:
        """
        Perform inference to generate embeddings.
        Supports OpenAI-compatible API response format. (see T412338#11482782)
        """
        try:
            # Perform inference
            logging.info("Performing inference...")
            with torch.no_grad():
                outputs = self.model(**encoded_input)

                # Check architecture type to decide pooling
                if hasattr(outputs, "last_hidden_state"):
                    embeddings = self.last_token_pooling(
                        outputs.last_hidden_state, encoded_input["attention_mask"]
                    )
                else:
                    # Fallback for some architectures
                    embeddings = outputs[0]

                # Normalize embeddings (Important for cosine similarity)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # Format response in OpenAI API format
            data = [
                {
                    "object": "embedding",
                    "embedding": embedding.tolist(),
                    "index": idx,
                }
                for idx, embedding in enumerate(embeddings)
            ]
            return {
                "object": "list",
                "data": data,
                "model": self.model_version
                or self.model.config._name_or_path
                or self.name,
            }

        except Exception as e:
            error_message = f"Error during inference: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "qwen3-embedding")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
    model_version = os.environ.get("MODEL_VERSION", "")
    local_files_only = strtobool(os.environ.get("LOCAL_FILES_ONLY", "True"))
    dtype = getattr(torch, os.environ.get("DTYPE", "float16"))
    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
    max_length = int(os.environ.get("MAX_LENGTH", 300))
    model = EmbeddingModel(
        model_name,
        model_path,
        model_version,
        local_files_only,
        dtype,
        attn_implementation,
        max_length,
    )
    model.load()
    kserve.ModelServer().start([model])

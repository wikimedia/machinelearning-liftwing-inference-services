import logging
import os

import kserve
import torch
from kserve.errors import InferenceError, InvalidInput
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

_GIB = 1024**3


def _bytes_to_gib(n: int | float) -> float:
    return n / _GIB


def _pct(n: int | float, total: int | float) -> float:
    return 100.0 * n / total if total else 0.0


def log_gpu_memory(label: str, device: str) -> None:
    """Log torch CUDA allocator stats (works on ROCm via the CUDA API shim)."""
    if device != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    free, total = torch.cuda.mem_get_info(0)
    used = total - free
    slack = reserved - allocated
    stats = torch.cuda.memory_stats(0)
    inactive_split = stats.get("inactive_split_bytes.all.current", 0)
    logging.info(
        "GPU mem [%s] device=%r: "
        "allocated=%.3f GiB (%.1f%%) reserved=%.3f GiB (%.1f%%) slack=%.3f GiB "
        "peak_allocated=%.3f GiB (%.1f%%) peak_reserved=%.3f GiB "
        "device_used=%.3f GiB (%.1f%%) device_free=%.3f GiB device_total=%.3f GiB "
        "inactive_split=%.3f GiB ooms=%d alloc_retries=%d",
        label,
        props.name,
        _bytes_to_gib(allocated),
        _pct(allocated, total),
        _bytes_to_gib(reserved),
        _pct(reserved, total),
        _bytes_to_gib(slack),
        _bytes_to_gib(peak_allocated),
        _pct(peak_allocated, total),
        _bytes_to_gib(peak_reserved),
        _bytes_to_gib(used),
        _pct(used, total),
        _bytes_to_gib(free),
        _bytes_to_gib(total),
        _bytes_to_gib(inactive_split),
        stats.get("num_ooms", 0),
        stats.get("num_alloc_retries", 0),
    )


def reset_peak_gpu_memory(device: str) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)


class EmbeddingModel(kserve.Model):
    """
    SentenceTransformer-based embedding server (no vLLM).

    Serves jinaai/jina-embeddings-v5-text-nano-retrieval via SentenceTransformer,
    the interface recommended on the model card. Unlike the vLLM backend
    (model.py) it does not capture HIP graphs at startup (which for Jina takes
    ~70min on ROCm and exceeds Knative's deadline), and unlike a raw AutoModel
    it applies the model's configured retrieval prompts and pooling.

    Retrieval prompts: jina-v5 is trained with distinct "query" and "document"
    prompts that are prepended before encoding and materially affect retrieval
    quality. The OpenAI-style {"input": [...]} request has no query/document
    notion, so callers may pass an optional "prompt_name" per request; it
    defaults to PROMPT_NAME (env, default "query").

    encode() is called with normalize_embeddings=True so the returned vectors are
    L2-normalized (consistent with the vLLM backend and ready for cosine/dot
    similarity).
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        model_version: str,
        dtype: torch.dtype,
        attn_implementation: str,
        default_prompt_name: str,
        truncate_dim: int | None,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.model_version = model_version
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.ready = False
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.default_prompt_name = default_prompt_name
        self.truncate_dim = truncate_dim

    def load(self) -> None:
        """
        Load the SentenceTransformer model.
        """
        try:
            logging.info("Loading SentenceTransformer model...")
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True,
                model_kwargs={"dtype": self.dtype},
                config_kwargs={"_attn_implementation": self.attn_implementation},
                truncate_dim=self.truncate_dim,
            )
            self.model.eval()
            self.ready = True
            logging.info("SentenceTransformer model loaded successfully!")
            log_gpu_memory("after_load", self.device)
        except Exception as e:
            error_message = f"Failed to load model. Reason: {e}"
            logging.critical(error_message)
            raise kserve.errors.InferenceError(error_message)

    def preprocess(self, payload: dict, headers: dict[str, str]) -> dict:
        """
        Validate the input and resolve the retrieval prompt.
        Supports OpenAI-compatible API request format. (see T412338#11482782)
        Optional "prompt_name" (e.g. "query" or "document") overrides the default.
        """
        if "input" not in payload:
            error_message = "Invalid payload format. Use {'input': ['text1', 'text2']}"
            logging.error(error_message)
            raise InvalidInput(error_message)

        inputs = payload["input"]
        # Ensure input is a list
        if isinstance(inputs, str):
            inputs = [inputs]

        prompt_name = payload.get("prompt_name", self.default_prompt_name) or None
        return {"sentences": inputs, "prompt_name": prompt_name}

    def predict(self, request: dict, headers: dict[str, str] = None) -> dict:
        """
        Perform inference to generate embeddings.
        Supports OpenAI-compatible API response format. (see T412338#11482782)
        """
        try:
            sentences = request["sentences"]
            n_sentences = len(sentences)
            total_chars = sum(len(s) for s in sentences)
            max_chars = max((len(s) for s in sentences), default=0)
            logging.info(
                "Performing inference (sentences=%d total_chars=%d max_chars=%d)...",
                n_sentences,
                total_chars,
                max_chars,
            )
            reset_peak_gpu_memory(self.device)
            log_gpu_memory("before_encode", self.device)
            embeddings = self.model.encode(
                sentences=sentences,
                prompt_name=request["prompt_name"],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            log_gpu_memory("after_encode", self.device)

            # encode() returns a 2D array (one row per input sentence).
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
                "model": self.model_version or self.name,
            }

        except Exception as e:
            error_message = f"Error during inference: {e}"
            logging.error(error_message)
            raise InferenceError(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "jina-embedding")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
    model_version = os.environ.get("MODEL_VERSION", "")
    # Jina model card recommends bfloat16 on GPUs.
    dtype = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
    # flash_attention_2 is disabled for now; use eager (or override via ATTN_IMPLEMENTATION).
    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION", "eager")
    # Default retrieval prompt; callers can override per request via "prompt_name".
    default_prompt_name = os.environ.get("PROMPT_NAME", "query").strip()
    # Optional Matryoshka truncation of the embedding dim (Jina v5 supports it).
    _truncate_dim = os.environ.get("TRUNCATE_DIM", "").strip()
    truncate_dim = int(_truncate_dim) if _truncate_dim else None

    model = EmbeddingModel(
        name=model_name,
        model_path=model_path,
        model_version=model_version,
        dtype=dtype,
        attn_implementation=attn_implementation,
        default_prompt_name=default_prompt_name,
        truncate_dim=truncate_dim,
    )
    model.load()
    kserve.ModelServer().start([model])

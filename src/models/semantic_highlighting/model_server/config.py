"""Settings for the semantic-highlighting model server, read from the environment.

``MODEL_NAME`` and ``MODEL_PATH`` carry their usual Lift Wing meanings: the
served model name (which becomes the ``/v1/models/<name>:predict`` path segment)
and where the weights are. As in the embeddings server, ``MODEL_PATH`` may be
either a directory (what the storage-initializer mounts a ``storageUri`` into) or
a Hugging Face model id, since ``from_pretrained`` accepts both -- so local runs
can point it straight at ``timpal0l/mdeberta-v3-base-squad2``.

Everything specific to this highlighter is ``SH_``-prefixed.
"""

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Reads variables as environment variables, falling back to the defaults."""

    # `model_` is a protected attribute namespace in pydantic v2; the Lift Wing
    # env var names are fixed, so opt out of the protection rather than rename.
    # populate_by_name lets tests construct a Settings directly by field name --
    # the SH_-aliased fields would otherwise only accept their alias.
    model_config = SettingsConfigDict(protected_namespaces=(), populate_by_name=True)

    # Served model name -- the URL path segment, and what shows up in logs.
    model_name: str = "semantic-highlighter"
    # Weights: a local directory or a Hugging Face model id.
    model_path: str = "/mnt/models/"
    # Weight precision. Half precision halves memory and is much faster on GPUs
    # with half-precision matrix units; it also shifts scores slightly, which
    # matters most around the abstention threshold -- compare highlight rates,
    # not just spans, before switching.
    dtype: Literal["float32", "bfloat16", "float16"] = Field(
        default="float32", validation_alias="SH_DTYPE"
    )

    # Max answer span length, in tokens.
    max_answer_len: int = Field(default=30, validation_alias="SH_MAX_ANSWER_LEN")
    # Candidate starts/ends considered when decoding a span.
    top_k: int = Field(default=20, validation_alias="SH_TOP_K")
    # Pairs per batched forward pass. OpenSearch sends up to
    # `max_inference_batch_size` (default 100) per call; we re-chunk to bound
    # peak memory.
    batch_size: int = Field(default=16, validation_alias="SH_BATCH_SIZE")
    # Drop answers below this confidence -> no highlight. Off by default.
    min_score: Optional[float] = Field(default=None, validation_alias="SH_MIN_SCORE")
    # Passages shorter than this skip inference and answer with no highlight:
    # the answer span is ~7 tokens whatever the passage length, so in a one-line
    # snippet the highlight covers most of the text and narrows nothing. Counted
    # in tokens, not characters, so the cut means the same in every script.
    # 0 disables it. The README has the numbers behind the default.
    min_context_tokens: int = Field(
        default=40, validation_alias="SH_MIN_CONTEXT_TOKENS"
    )


settings = Settings()

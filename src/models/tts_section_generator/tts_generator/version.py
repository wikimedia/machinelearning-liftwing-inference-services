"""generation_version construction and content hashing.

Contract definitions the storage/index design depends on:

``generation_version`` identifies everything that changes output audio for
identical input text: the Kokoro model version, the voice, and the
normalization identity. The normalization identity covers three things:
which normalizer ran (nemo vs the regex fallback, since they produce
different text), the hand-bumped ruleset tag (cleaning regex changes), and
a hash of the NeMo whitelist file (whitelist edits change pronunciation
without any code change). When ML bumps any component, existing artifacts
become outdated and a backfill is requested of the DE pipeline; the version
string in each index record is what makes that diff computable.

``content_sha256`` is hashed over the NORMALIZED text, not the raw section
text, because normalized text is what the voice actually speaks: two
revisions whose raw wikitext differs but normalizes identically (a citation
moved, a template touched) produce identical audio and should share an
artifact.

Consequence of hashing normalized text: the hash is only comparable within
one normalizer identity, which is why content_sha256 must always be read
alongside generation_version, never alone.
"""

import hashlib
from pathlib import Path

from tts_generator.config import (
    DEFAULT_VOICE,
    KOKORO_MODEL_VERSION,
    NEMO_WHITELIST,
    NORMALIZATION_RULESET,
)
from tts_generator.text import nemo_available


def _whitelist_hash() -> str:
    try:
        data = Path(NEMO_WHITELIST).read_bytes()
    except OSError:
        return "nowl"
    return hashlib.sha256(data).hexdigest()[:8]


def normalizer_version() -> str:
    engine = "nemo" if nemo_available() else "regex"
    return f"norm-{NORMALIZATION_RULESET}-{engine}-{_whitelist_hash()}"


def generation_version(voice: str = DEFAULT_VOICE) -> str:
    """E.g. ``kokoro-v1.0+af_heart+norm-2026.07-nemo-3f9a1c2b``."""
    return f"{KOKORO_MODEL_VERSION}+{voice}+{normalizer_version()}"


def content_sha256(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

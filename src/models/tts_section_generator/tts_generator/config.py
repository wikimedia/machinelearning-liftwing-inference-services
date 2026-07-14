"""Environment-driven configuration for the TTS Section Generator.

Every tunable lives here so deployment config (helm values / env vars)
is the single place operators look. Defaults target local development
against the ml-staging TTS isvc.
"""

import os
from pathlib import Path

# ── Identity ────────────────────────────────────────────────────────────────
USER_AGENT = os.environ.get(
    "TTS_GEN_USER_AGENT",
    "WMF-ML-TTS-Section-Generator/0.1 (https://phabricator.wikimedia.org/T430536)",
)

# ── MediaWiki fetch ─────────────────────────────────────────────────────────
FETCH_TIMEOUT_S = float(os.environ.get("TTS_GEN_FETCH_TIMEOUT_S", "30"))
FETCH_RETRIES = int(os.environ.get("TTS_GEN_FETCH_RETRIES", "3"))

# ── TTS inference service ───────────────────────────────────────────────────
ISVC_URL = os.environ.get(
    "TTS_ISVC_URL",
    "https://inference-staging.svc.codfw.wmnet:30443/v1/models/tts:predict",
)
# LiftWing routes on the Host header, not the URL authority.
ISVC_HOST_HEADER = os.environ.get("TTS_ISVC_HOST", "tts.experimental.wikimedia.org")
# Long sections synthesize for minutes; see the Phase 3 timeout spike before
# lowering this.
ISVC_TIMEOUT_S = float(os.environ.get("TTS_ISVC_TIMEOUT_S", "300"))
ISVC_VERIFY_TLS = os.environ.get("TTS_ISVC_VERIFY_TLS", "true").lower() == "true"

# ── Text processing ─────────────────────────────────────────────────────────
# v0's gate: cleaned section text at or below this length is skipped
# (deterministic skip, never retried).
MIN_TEXT_LENGTH = int(os.environ.get("TTS_GEN_MIN_TEXT_LENGTH", "50"))
# Segment size sent to the isvc. Anything <= 800 (Kokoro's practical input
# limit, MAX_SEGMENT_CHARS in the model-server) is valid; smaller chunks give
# the prosody a natural sentence cadence. Tune with listening in Phase 4.
MAX_SEGMENT_CHARS = int(os.environ.get("TTS_GEN_MAX_SEGMENT_CHARS", "400"))

# ── Generation identity (feeds generation_version, see version.py) ─────────
KOKORO_MODEL_VERSION = os.environ.get("TTS_GEN_KOKORO_VERSION", "kokoro-v1.0")
DEFAULT_VOICE = os.environ.get("TTS_GEN_DEFAULT_VOICE", "af_heart")
DEFAULT_LANG = os.environ.get("TTS_GEN_DEFAULT_LANG", "en-us")
# Bump manually whenever cleaning/normalization RULES change in a way that
# alters output text for identical input (regex fixes, blocklist changes,
# whitelist edits are covered separately by the whitelist hash).
NORMALIZATION_RULESET = os.environ.get("TTS_GEN_NORM_RULESET", "2026.07")

# ── NeMo ────────────────────────────────────────────────────────────────────
_PKG_DIR = Path(__file__).resolve().parent
NEMO_WHITELIST = os.environ.get(
    "TTS_GEN_NEMO_WHITELIST", str(_PKG_DIR / "nemo_whitelist.tsv")
)
NEMO_GRAMMAR_CACHE = os.environ.get(
    "TTS_GEN_NEMO_CACHE",
    "/tmp/tts-gen-nemo-grammars",  # noqa: S108
)

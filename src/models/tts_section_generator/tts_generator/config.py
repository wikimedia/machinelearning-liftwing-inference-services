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
# In the LiftWing cluster, Wikipedia API requests go through the envoy services-proxy at
# localhost:6500 (see https://phabricator.wikimedia.org/T348607) so that the pod never needs
# direct internet access.  Local development can override this to "" to hit en.wikipedia.org directly.
MW_API_PROXY = os.environ.get("TTS_GEN_MW_API_PROXY", "http://localhost:6500")

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
# Path to the CA bundle used when ISVC_VERIFY_TLS is true.  The default is the
# Wikimedia CA certificate bundle installed by the wmf-certificates apt package.
# certifi's bundle (used by requests when verify=True) only trusts public CAs
# and cannot verify the LiftWing-internal isvc TLS certificate.
ISVC_TLS_CA_BUNDLE = os.environ.get(
    "TTS_ISVC_TLS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt"
)
# Retries apply ONLY to transport failures and 5xx (the isvc is stateless
# and deterministic, so retries are always safe); 4xx never retries.
ISVC_RETRIES = int(os.environ.get("TTS_ISVC_RETRIES", "2"))
ISVC_BACKOFF_S = float(os.environ.get("TTS_ISVC_BACKOFF_S", "2.0"))

# ── Transcode (Phase 2) ─────────────────────────────────────────────────────
FFMPEG_PATH = os.environ.get("TTS_GEN_FFMPEG", "ffmpeg")
# Opus at 24-32 kbps mono is transparent for speech; "voip" tunes the
# encoder for speech intelligibility at low bitrates.
OPUS_BITRATE = os.environ.get("TTS_GEN_OPUS_BITRATE", "32k")
OPUS_APPLICATION = os.environ.get("TTS_GEN_OPUS_APPLICATION", "voip")
MP3_BITRATE = os.environ.get("TTS_GEN_MP3_BITRATE", "48k")

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
NORMALIZATION_RULESET = os.environ.get("TTS_GEN_NORM_RULESET", "2026.07.20")

# ── NeMo ────────────────────────────────────────────────────────────────────
_PKG_DIR = Path(__file__).resolve().parent
NEMO_WHITELIST = os.environ.get(
    "TTS_GEN_NEMO_WHITELIST", str(_PKG_DIR / "nemo_whitelist.tsv")
)
NEMO_GRAMMAR_CACHE = os.environ.get(
    "TTS_GEN_NEMO_CACHE",
    "/tmp/tts-gen-nemo-grammars",  # noqa: S108
)

# ── Artifact sink (Phase 3, Spike 2: blob-write mode) ──────────────────────
# inline: artifacts return as bytes_b64 (default; LAC-native, storage-free)
# file:   artifacts written under BLOB_SINK_DIR, response carries blob_uri
# s3:     object-storage mode; wiring lands with the Data Persistence bucket
BLOB_SINK = os.environ.get("TTS_GEN_BLOB_SINK", "inline")
BLOB_SINK_DIR = os.environ.get("TTS_GEN_BLOB_SINK_DIR", "/tmp/tts-artifacts")  # noqa: S108
S3_ENDPOINT = os.environ.get("TTS_GEN_S3_ENDPOINT", "")
S3_BUCKET = os.environ.get("TTS_GEN_S3_BUCKET", "")

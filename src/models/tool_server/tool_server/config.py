"""
Environment-driven configuration for the LiftWing tool-server.

Every tunable lives here so deployment config (helm values / env vars)
is the single place operators look. Defaults target local development
hitting Wikipedia directly.
"""

import os

# ── Identity ────────────────────────────────────────────────────────────────
USER_AGENT = os.environ.get(
    "TOOL_SERVER_USER_AGENT",
    "WMF-LiftWing-Tool-Server/0.1 (https://phabricator.wikimedia.org/T436892)",
)

# ── MediaWiki access ────────────────────────────────────────────────────────
# In the LiftWing cluster, Wikipedia API requests go through the envoy
# services-proxy at localhost:6500 (see https://phabricator.wikimedia.org/T348607)
# so that the pod never needs direct internet access. Local development can
# override this to "" to hit {language}.wikipedia.org directly.
MW_API_PROXY = os.environ.get("TOOL_MW_API_PROXY", "")
HTTP_TIMEOUT_S = float(os.environ.get("TOOL_HTTP_TIMEOUT_S", "10"))
HTTP_RETRIES = int(os.environ.get("TOOL_HTTP_RETRIES", "2"))

# ── wikipedia_semantic_search ───────────────────────────────────────────────
EXTRACT_MAX_CHARS = int(os.environ.get("TOOL_EXTRACT_MAX_CHARS", "500"))
DEFAULT_LANGUAGE = os.environ.get("TOOL_DEFAULT_LANGUAGE", "en")
DEFAULT_LIMIT = int(os.environ.get("TOOL_DEFAULT_LIMIT", "5"))
MAX_LIMIT = int(os.environ.get("TOOL_MAX_LIMIT", "10"))

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("TOOL_LOG_LEVEL", "INFO").upper()

"""HTTP client for the TTS inference service (T430536).

Phase 1 scope: a single synchronous call with a generous timeout, mapping
transport failures to SynthesisError. Phase 2 adds bounded retries with
backoff for transient 5xx (safe: the isvc is stateless and deterministic)
and per-artifact request shaping.

Contract notes pinned here so callers don't re-learn them:
* LiftWing routes on the Host header (ISVC_HOST_HEADER), not the URL.
* ``timestamps`` selects alignment cost per request: ``full`` (CTC),
  ``proportional`` (near-zero), ``none`` (audio-only, fastest). Audio-only
  artifacts must ride ``none``; captions need ``full`` (or ``proportional``
  if the Apps decision lands there).
* Responses default to int16 PCM (``pcm_s16le``), base64-encoded.
* The isvc serializes synthesis per pod (containerConcurrency=1); requests
  beyond replica count queue. The generator's caller owns concurrency
  discipline, not this client.
"""

import logging

import requests
from tts_generator.config import (
    ISVC_HOST_HEADER,
    ISVC_TIMEOUT_S,
    ISVC_URL,
    ISVC_VERIFY_TLS,
    USER_AGENT,
)

logger = logging.getLogger(__name__)


class SynthesisError(Exception):
    """isvc call failed (transport, HTTP error, or malformed response)."""


def synthesize(
    segments: list[dict],
    voice: str,
    lang: str,
    timestamps: str = "full",
    encoding: str = "pcm_s16le",
) -> dict:
    """POST one synthesis request; return the isvc response dict.

    Response shape: ``{audio_b64, encoding, timestamps_mode, sample_rate,
    duration_ms, timestamps: [{word, start_ms, end_ms}]}``.
    """
    payload = {
        "segments": segments,
        "default_voice": voice,
        "default_lang": lang,
        "timestamps": timestamps,
        "encoding": encoding,
    }
    try:
        resp = requests.post(
            ISVC_URL,
            json=payload,
            headers={"Host": ISVC_HOST_HEADER, "User-Agent": USER_AGENT},
            timeout=ISVC_TIMEOUT_S,
            verify=ISVC_VERIFY_TLS,
        )
    except requests.RequestException as e:
        raise SynthesisError(f"isvc request failed: {e}") from e

    if not resp.ok:
        raise SynthesisError(f"isvc returned {resp.status_code}: {resp.text[:500]}")

    try:
        return resp.json()
    except ValueError as e:
        raise SynthesisError("isvc returned non-JSON response") from e

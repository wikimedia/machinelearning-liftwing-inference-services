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
import time

import requests
from tts_generator.config import (
    ISVC_BACKOFF_S,
    ISVC_HOST_HEADER,
    ISVC_RETRIES,
    ISVC_TIMEOUT_S,
    ISVC_URL,
    ISVC_VERIFY_TLS,
    USER_AGENT,
)

logger = logging.getLogger(__name__)


class SynthesisError(Exception):
    """isvc call failed after retries (transport, 5xx, or malformed body)."""


class SynthesisRejected(Exception):
    """isvc rejected the request (4xx): deterministic, never retried.

    Reaching this means the generator built an invalid isvc request
    (over-long segment, bad parameter), which is a BUG here, not load: it
    surfaces as a 502 synthesis_error to the caller but is logged loudly.
    """


def synthesize(
    segments: list[dict],
    voice: str,
    lang: str,
    timestamps: str = "full",
    encoding: str = "pcm_s16le",
) -> dict:
    """POST one synthesis request; return the isvc response dict.

    Retries transport failures and 5xx up to ISVC_RETRIES times with
    linear backoff. Safe because the isvc is stateless and deterministic
    for fixed input. 4xx is never retried.

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

    last_error: Exception | None = None
    for attempt in range(1 + ISVC_RETRIES):
        if attempt:
            sleep_s = ISVC_BACKOFF_S * attempt
            logger.warning(
                "isvc retry %d/%d after %.1fs: %s",
                attempt,
                ISVC_RETRIES,
                sleep_s,
                last_error,
            )
            time.sleep(sleep_s)
        try:
            resp = requests.post(
                ISVC_URL,
                json=payload,
                headers={"Host": ISVC_HOST_HEADER, "User-Agent": USER_AGENT},
                timeout=ISVC_TIMEOUT_S,
                verify=ISVC_VERIFY_TLS,
            )
        except requests.RequestException as e:
            last_error = e
            continue

        if 400 <= resp.status_code < 500:
            logger.error(
                "isvc rejected request (%d): %s", resp.status_code, resp.text[:500]
            )
            raise SynthesisRejected(f"isvc {resp.status_code}: {resp.text[:500]}")
        if not resp.ok:
            last_error = SynthesisError(
                f"isvc returned {resp.status_code}: {resp.text[:500]}"
            )
            continue

        try:
            return resp.json()
        except ValueError as e:
            last_error = e
            continue

    raise SynthesisError(f"isvc failed after {1 + ISVC_RETRIES} attempts: {last_error}")

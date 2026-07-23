"""TTS Section Generator service: the ML-owned compute function.

Two endpoints, per the contract in openapi.yaml:

* ``GET /sections``: enumerate the valid, generatable sections of a pinned
  revision, with content hashes. This is what the DE pipeline diffs
  against the artifact index; section validity (blocklist, min-length) is
  TTS business logic and lives HERE so it never leaks into the DAG.
* ``POST /generate-section``: produce the artifact family for one section
  of one pinned revision.

Error taxonomy (machine-readable ``code`` in every error body): 4xx codes
are DETERMINISTIC. The same request will always produce the same skip, so
the pipeline records them and never retries. 5xx codes are transient and
retryable. This split is a contract promise the DE retry logic depends on.

Artifact set (Phase 2 complete): audio_opus (recommended), audio_mp3
(config-free alternative pending the Apps codec decision), captions_vtt,
timestamps_json, audio_pcm_s16le (raw passthrough, mostly for debugging).
Artifact choice drives isvc request shaping: requests with no timing
artifact ride the isvc's alignment-free path (RTF ~0.22 vs ~0.27).
"""

import base64
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from tts_generator import isvc_client
from tts_generator.chunking import split_text
from tts_generator.config import (
    DEFAULT_LANG,
    DEFAULT_VOICE,
    MAX_SEGMENT_CHARS,
    MIN_TEXT_LENGTH,
)
from tts_generator.fetch import FetchError, fetch_revision_html, fetch_revision_meta
from tts_generator.sections import extract_sections, find_section
from tts_generator.sinks import InlineSink, SinkWriteError, artifact_key, build_sink
from tts_generator.text import clean_spoken_text, init_nemo
from tts_generator.transcode import TranscodeError, pcm_to_mp3, pcm_to_opus
from tts_generator.version import content_sha256, generation_version
from tts_generator.vtt import timestamps_to_vtt

logger = logging.getLogger(__name__)

SUPPORTED_ARTIFACTS = {
    "audio_opus",
    "audio_mp3",
    "captions_vtt",
    "timestamps_json",
    "audio_pcm_s16le",
}
TIMING_ARTIFACTS = {"captions_vtt", "timestamps_json"}

MEDIA_TYPES = {
    "audio_opus": "audio/ogg; codecs=opus",
    "audio_mp3": "audio/mpeg",
    "captions_vtt": "text/vtt",
    "timestamps_json": "application/json",
    "audio_pcm_s16le": "audio/L16; rate=24000; channels=1",
}

# Safe default so the service functions even under runners that skip the
# lifespan (some test harnesses); startup replaces it with the configured
# sink, and a MISCONFIGURED sink still fails the deploy there.
_sink = InlineSink()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _sink
    # Sink first: a misconfigured sink must fail the deploy, not request #1.
    _sink = build_sink()
    logger.info("Artifact sink: %s", _sink.mode)
    # NeMo grammar compilation takes 60+ seconds cold; the deployment image
    # bakes the cache at image build (see Dockerfile) so this is fast in prod.
    init_nemo()
    yield


app = FastAPI(title="TTS Section Generator", version="0.3.0", lifespan=_lifespan)


# ── Error helper ────────────────────────────────────────────────────────────


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "message": message})


# Deterministic (never retried): revision_not_found, revision_page_mismatch,
# section_not_found_at_revision (blocklisted sections are never addressable),
# text_below_minimum, artifact_type_not_available, unsupported_wiki.
# Transient (retryable): upstream_fetch_error, synthesis_error,
# transcode_error, blob_write_error.


def _fetch_and_verify(wiki_id: str, page_id: int, rev_id: int):
    """Shared fetch + integrity path. Returns (meta, html) or JSONResponse."""
    try:
        meta = fetch_revision_meta(wiki_id, rev_id)
    except FetchError as e:
        if e.status == 404:
            return _error(404, "revision_not_found", str(e))
        if e.status == 400:
            return _error(400, "unsupported_wiki", str(e))
        return _error(502, "upstream_fetch_error", str(e))

    actual_page_id = (meta.get("page") or {}).get("id")
    if actual_page_id != page_id:
        # Generating under a mislabeled key would poison the index; refuse.
        return _error(
            409,
            "revision_page_mismatch",
            f"rev_id {rev_id} belongs to page_id {actual_page_id}, not {page_id}",
        )

    try:
        html = fetch_revision_html(wiki_id, rev_id)
    except FetchError as e:
        return _error(502, "upstream_fetch_error", str(e))
    return meta, html


# ── /sections ───────────────────────────────────────────────────────────────


@app.get("/sections")
def get_sections(
    wiki_id: str = Query(...),
    page_id: int = Query(...),
    rev_id: int = Query(...),
):
    result = _fetch_and_verify(wiki_id, page_id, rev_id)
    if isinstance(result, JSONResponse):
        return result
    meta, html = result

    out = []
    for s in extract_sections(html):
        cleaned = clean_spoken_text(s.raw_text)
        generatable = len(cleaned) > MIN_TEXT_LENGTH
        entry = {
            "section_id": s.section_id,
            "title": s.title,
            "level": s.level,
            "generatable": generatable,
            "char_count": len(cleaned),
        }
        if generatable:
            entry["content_sha256"] = content_sha256(cleaned)
        else:
            entry["skip_reason"] = "text_below_minimum"
        out.append(entry)

    return {
        "wiki_id": wiki_id,
        "page_id": page_id,
        "rev_id": rev_id,
        "revision_timestamp": meta.get("timestamp"),
        "generation_version": generation_version(),
        "sections": out,
    }


# ── /generate-section ───────────────────────────────────────────────────────


class GenerationConfig(BaseModel):
    voice: str = DEFAULT_VOICE
    lang: str = DEFAULT_LANG
    timestamps: str = Field(default="full", pattern="^(full|proportional|none)$")
    artifacts: list[str] = ["audio_opus", "captions_vtt", "timestamps_json"]


class GenerateRequest(BaseModel):
    wiki_id: str
    page_id: int
    rev_id: int
    section_id: str
    generation_config: GenerationConfig = GenerationConfig()


@app.post("/generate-section")
def generate_section(req: GenerateRequest):
    cfg = req.generation_config

    unknown = set(cfg.artifacts) - SUPPORTED_ARTIFACTS
    if unknown:
        return _error(
            400,
            "artifact_type_not_available",
            f"Unknown artifact types: {sorted(unknown)}",
        )
    if not cfg.artifacts:
        return _error(400, "artifact_type_not_available", "No artifacts requested")

    result = _fetch_and_verify(req.wiki_id, req.page_id, req.rev_id)
    if isinstance(result, JSONResponse):
        return result
    _meta, html = result

    section = find_section(extract_sections(html), req.section_id)
    if section is None:
        return _error(
            404,
            "section_not_found_at_revision",
            f"No section {req.section_id!r} at rev {req.rev_id} "
            "(blocklisted sections are never addressable)",
        )

    cleaned = clean_spoken_text(section.raw_text)
    if len(cleaned) <= MIN_TEXT_LENGTH:
        return _error(
            422,
            "text_below_minimum",
            f"Cleaned text is {len(cleaned)} chars (minimum {MIN_TEXT_LENGTH})",
        )

    segments = [{"text": c} for c in split_text(cleaned, MAX_SEGMENT_CHARS)]
    sha = content_sha256(cleaned)
    gv = generation_version(cfg.voice)

    # Request shaping: timing artifacts force the requested timestamps mode;
    # audio-only requests ride the isvc's alignment-free path (RTF ~0.22).
    ts_mode = cfg.timestamps if (set(cfg.artifacts) & TIMING_ARTIFACTS) else "none"

    t0 = time.perf_counter()
    try:
        isvc = isvc_client.synthesize(
            segments, voice=cfg.voice, lang=cfg.lang, timestamps=ts_mode
        )
    except isvc_client.SynthesisRejected as e:
        # A 4xx from the isvc means WE built a bad request: a generator bug,
        # not load. Surfaced as transient so the pipeline flags it, and
        # logged loudly for us.
        logger.error("isvc rejected generator-built request: %s", e)
        return _error(502, "synthesis_error", f"internal request rejected: {e}")
    except isvc_client.SynthesisError as e:
        return _error(502, "synthesis_error", str(e))
    synth_s = time.perf_counter() - t0

    pcm = base64.b64decode(isvc.pop("audio_b64"))
    sample_rate = isvc["sample_rate"]

    common = {
        "wiki_id": req.wiki_id,
        "page_id": req.page_id,
        "rev_id": req.rev_id,
        "section_id": req.section_id,
        "generation_version": gv,
        "content_sha256": sha,
        "duration_ms": isvc["duration_ms"],
    }

    def _emit(entry: dict, data: bytes) -> None:
        # Binary bytes go through the configured sink: inline -> bytes_b64
        # in the response; file/s3 -> written out, response carries blob_uri.
        key = artifact_key(
            req.wiki_id,
            req.page_id,
            req.rev_id,
            req.section_id,
            entry["artifact_type"],
        )
        entry.update(_sink.store(key, data, entry["media_type"]))

    artifacts = []
    transcode_s = 0.0
    try:
        for kind in cfg.artifacts:
            entry = {**common, "artifact_type": kind, "media_type": MEDIA_TYPES[kind]}
            if kind == "audio_opus":
                _t = time.perf_counter()
                _emit(entry, pcm_to_opus(pcm, sample_rate))
                transcode_s += time.perf_counter() - _t
            elif kind == "audio_mp3":
                _t = time.perf_counter()
                _emit(entry, pcm_to_mp3(pcm, sample_rate))
                transcode_s += time.perf_counter() - _t
            elif kind == "captions_vtt":
                entry["timestamps_mode"] = isvc["timestamps_mode"]
                _emit(entry, timestamps_to_vtt(isvc["timestamps"]).encode("utf-8"))
            elif kind == "timestamps_json":
                entry["timestamps"] = isvc["timestamps"]
                entry["timestamps_mode"] = isvc["timestamps_mode"]
            elif kind == "audio_pcm_s16le":
                entry["sample_rate"] = sample_rate
                entry["encoding"] = isvc["encoding"]
                _emit(entry, pcm)
            artifacts.append(entry)
    except TranscodeError as e:
        return _error(502, "transcode_error", str(e))
    except SinkWriteError as e:
        # Blob write failed after synthesis succeeded. Transient: the
        # whole request is retryable (generation is idempotent), and the
        # bytes are simply regenerated on retry.
        return _error(502, "blob_write_error", str(e))

    logger.info(
        "generated %s/%s/%s/%s: %d segments, %.1fms audio "
        "(synth %.2fs, transcode %.2fs, ts=%s)",
        req.wiki_id,
        req.page_id,
        req.rev_id,
        req.section_id,
        len(segments),
        isvc["duration_ms"],
        synth_s,
        transcode_s,
        ts_mode,
    )

    return {"artifacts": artifacts, "segment_count": len(segments)}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8081)  # noqa: S104

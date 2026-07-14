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

Phase 1 artifact scope: ``audio_pcm_s16le`` (raw isvc PCM passthrough) and
``timestamps_json``. ``audio_opus`` and ``captions_vtt`` land in Phase 2
(transcode + VTT formatting) and are declared in the spec now so the
contract does not change shape later.
"""

import logging
import time

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
from tts_generator.text import clean_spoken_text, init_nemo
from tts_generator.version import content_sha256, generation_version

logger = logging.getLogger(__name__)

PHASE1_ARTIFACTS = {"audio_pcm_s16le", "timestamps_json"}
PHASE2_ARTIFACTS = {"audio_opus", "captions_vtt"}

app = FastAPI(title="TTS Section Generator", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    # NeMo grammar compilation takes 60+ seconds cold; the deployment image
    # bakes the cache at build time (Phase 3) so this is fast in prod.
    init_nemo()


# ── Error helper ────────────────────────────────────────────────────────────


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "message": message})


# Deterministic (never retried): section_not_found_at_revision,
# section_blocklisted (implicit: blocklisted sections are absent from
# /sections), text_below_minimum, revision_page_mismatch,
# artifact_type_not_available, unsupported_wiki.
# Transient (retryable): upstream_fetch_error, synthesis_error.


# ── /sections ───────────────────────────────────────────────────────────────


@app.get("/sections")
def get_sections(
    wiki_id: str = Query(...),
    page_id: int = Query(...),
    rev_id: int = Query(...),
):
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
    artifacts: list[str] = ["audio_pcm_s16le", "timestamps_json"]


class GenerateRequest(BaseModel):
    wiki_id: str
    page_id: int
    rev_id: int
    section_id: str
    generation_config: GenerationConfig = GenerationConfig()


@app.post("/generate-section")
def generate_section(req: GenerateRequest):
    cfg = req.generation_config

    unknown = set(cfg.artifacts) - PHASE1_ARTIFACTS - PHASE2_ARTIFACTS
    if unknown:
        return _error(
            400,
            "artifact_type_not_available",
            f"Unknown artifact types: {sorted(unknown)}",
        )
    not_yet = set(cfg.artifacts) & PHASE2_ARTIFACTS
    if not_yet:
        return _error(
            400,
            "artifact_type_not_available",
            f"{sorted(not_yet)} land in Phase 2 (transcode + VTT)",
        )

    try:
        meta = fetch_revision_meta(req.wiki_id, req.rev_id)
    except FetchError as e:
        if e.status == 404:
            return _error(404, "revision_not_found", str(e))
        if e.status == 400:
            return _error(400, "unsupported_wiki", str(e))
        return _error(502, "upstream_fetch_error", str(e))

    actual_page_id = (meta.get("page") or {}).get("id")
    if actual_page_id != req.page_id:
        return _error(
            409,
            "revision_page_mismatch",
            f"rev_id {req.rev_id} belongs to page_id {actual_page_id}, not {req.page_id}",
        )

    try:
        html = fetch_revision_html(req.wiki_id, req.rev_id)
    except FetchError as e:
        return _error(502, "upstream_fetch_error", str(e))

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

    # Audio-only requests skip alignment cost entirely (isvc rides the
    # RTF-0.22 path); timestamps_json forces the requested timestamps mode.
    ts_mode = cfg.timestamps if "timestamps_json" in cfg.artifacts else "none"

    t0 = time.perf_counter()
    try:
        isvc = isvc_client.synthesize(
            segments, voice=cfg.voice, lang=cfg.lang, timestamps=ts_mode
        )
    except isvc_client.SynthesisError as e:
        return _error(502, "synthesis_error", str(e))
    synth_s = time.perf_counter() - t0

    logger.info(
        "generated %s/%s/%s/%s: %d segments, %.1fms audio in %.2fs",
        req.wiki_id,
        req.page_id,
        req.rev_id,
        req.section_id,
        len(segments),
        isvc["duration_ms"],
        synth_s,
    )

    common = {
        "wiki_id": req.wiki_id,
        "page_id": req.page_id,
        "rev_id": req.rev_id,
        "section_id": req.section_id,
        "generation_version": gv,
        "content_sha256": sha,
        "duration_ms": isvc["duration_ms"],
    }
    artifacts = []
    if "audio_pcm_s16le" in cfg.artifacts:
        artifacts.append(
            {
                **common,
                "artifact_type": "audio_pcm_s16le",
                "bytes_b64": isvc["audio_b64"],
                "sample_rate": isvc["sample_rate"],
                "encoding": isvc["encoding"],
            }
        )
    if "timestamps_json" in cfg.artifacts:
        artifacts.append(
            {
                **common,
                "artifact_type": "timestamps_json",
                "timestamps_mode": isvc["timestamps_mode"],
                "timestamps": isvc["timestamps"],
            }
        )

    return {"artifacts": artifacts, "segment_count": len(segments)}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8081)  # noqa: S104

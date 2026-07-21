# TTS Section Generator

The ML-owned **compute function** for the Wikipedia TTS Audio Archive: takes a
`(wiki_id, page_id, rev_id, section_id)` address and produces the audio
artifact family for that exact pinned revision. It is the layer that keeps all
TTS business logic (what to fetch, how text becomes speech, what an artifact
contains) out of the Data Engineering pipelines that invoke it.

Successor to the orchestration-adjacent parts of the v0 Toolforge prototype;
sits in front of the TTS inference service (T430536).

```
DE pipelines ──▶ THIS SERVICE ──▶ TTS isvc (Kokoro + Wav2Vec2, LiftWing)
(Airflow, Bento)    fetch pinned revision (MediaWiki REST, Parsoid HTML)
                    -> extract sections -> normalize -> chunk
                    -> synthesize + align -> transcode (Opus/MP3) -> VTT
```

Two endpoints; full spec in [`openapi.yaml`](openapi.yaml) (the artifact DE
reviews). `GET /sections` enumerates the valid, generatable sections of a
pinned revision (what the DE pipeline diffs against the artifact index);
`POST /generate-section` produces the artifact family for one section. The
service is stateless: a pure function of its inputs plus
`generation_version`, so retries are always safe. See [Input](#input) and
[Output](#output) for the request and response contracts, including the
error taxonomy.

## How to run locally

In order to run the section generator locally, please choose one of the two
options below. Once the server is running, see [Input](#input) and
[Output](#output) for how to query it.

<details>
<summary>1. Automated setup using docker compose</summary>

### 1.1. Build

From the repo root, in the first terminal run:
```console
docker compose build tts-section-generator
```

This bakes the NeMo grammar cache into the image at build time (the build
fails if NeMo did not initialize, so an image can never silently fall back
to regex normalization).

### 1.2. Run

On the same terminal run the service:
```console
docker compose up tts-section-generator
```

The service listens on `http://localhost:8080`. Note that generation
requests need a reachable TTS inference service (`TTS_ISVC_URL`); from
outside WMF infrastructure, point it at an ssh tunnel or a locally running
model-server.

### 1.3. Remove

If you would like to remove the setup run:
```console
docker compose down -v --rmi all
```
</details>

<details>
<summary>2. Manual setup</summary>

### 2.1. Build Python venv and install dependencies

From `src/models/tts_section_generator/`:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`nemo_text_processing` is heavy; without it the service falls back to regex
normalization and `generation_version` reflects that via the `regex` engine
tag (fine for development, never for generating real artifacts).

### 2.2. Run the tests

```console
python3 -m pytest tests/
```

### 2.3. Run the server

```console
python3 -m tts_generator.service
```

The service listens on `http://localhost:8081` (the in-module runner;
the deployed entrypoint serves on 8080).
</details>

## Environment variables

All tunables live in [`tts_generator/config.py`](tts_generator/config.py);
deployment values (helm) set them as env vars. Defaults target local
development against the ml-staging TTS isvc.

| Variable | Default | Description |
| --- | --- | --- |
| `TTS_ISVC_URL` | ml-staging tts `:predict` URL | TTS inference service endpoint. |
| `TTS_ISVC_HOST` | `tts.experimental.wikimedia.org` | Host header (LiftWing routes on it, not the URL). |
| `TTS_ISVC_TIMEOUT_S` | `300` | isvc request timeout. Staging/production set `600`: the corpus tail exceeds 300 s (see SPIKE_ANSWERS.md before lowering). |
| `TTS_ISVC_VERIFY_TLS` | `true` | Verify isvc TLS. |
| `TTS_ISVC_TLS_CA_BUNDLE` | `/etc/ssl/certs/ca-certificates.crt` | CA bundle for isvc TLS (certifi's bundle cannot verify the LiftWing-internal certificate). |
| `TTS_ISVC_RETRIES` | `2` | Retries for transport failures and 5xx only; 4xx never retries. |
| `TTS_ISVC_BACKOFF_S` | `2.0` | Linear backoff factor between retries. |
| `TTS_GEN_MW_API_PROXY` | `http://localhost:6500` | MediaWiki API via the envoy services-proxy (LiftWing pods have no general egress). Set empty for local dev to hit Wikipedia directly. |
| `TTS_GEN_FETCH_TIMEOUT_S` | `30` | MediaWiki fetch timeout. |
| `TTS_GEN_FETCH_RETRIES` | `3` | MediaWiki fetch retries (429/5xx). |
| `TTS_GEN_USER_AGENT` | WMF-ML UA string | User-Agent for all outbound requests. |
| `TTS_GEN_MIN_TEXT_LENGTH` | `50` | Cleaned text at or below this length is a deterministic skip. |
| `TTS_GEN_MAX_SEGMENT_CHARS` | `400` | Segment size sent to the isvc (isvc practical ceiling 800). Quality knob. |
| `TTS_GEN_KOKORO_VERSION` | `kokoro-v1.0` | Model component of `generation_version`. |
| `TTS_GEN_DEFAULT_VOICE` | `af_heart` | Default voice. |
| `TTS_GEN_DEFAULT_LANG` | `en-us` | Default language. |
| `TTS_GEN_NORM_RULESET` | `2026.07.20` | Hand-bumped normalization ruleset tag; bump whenever cleaning rules change output text for identical input. |
| `TTS_GEN_NEMO_WHITELIST` | packaged `nemo_whitelist.tsv` | Pronunciation whitelist; its hash is part of `generation_version`. |
| `TTS_GEN_NEMO_CACHE` | `/tmp/tts-gen-nemo-grammars` | NeMo grammar cache dir (baked into the image at build in deployment). |
| `TTS_GEN_FFMPEG` | `ffmpeg` | ffmpeg binary for transcoding. |
| `TTS_GEN_OPUS_BITRATE` | `32k` | Opus bitrate (mono speech). |
| `TTS_GEN_OPUS_APPLICATION` | `voip` | Opus encoder tuning (speech intelligibility). |
| `TTS_GEN_MP3_BITRATE` | `48k` | MP3 bitrate. |
| `TTS_GEN_BLOB_SINK` | `inline` | Artifact sink: `inline` (bytes_b64 in the response), `file` (writes under `TTS_GEN_BLOB_SINK_DIR`, returns `blob_uri`), `s3` (stub until the Data Persistence bucket exists; fails loudly at startup). |
| `TTS_GEN_BLOB_SINK_DIR` | `/tmp/tts-artifacts` | Root directory for the `file` sink. |
| `TTS_GEN_S3_ENDPOINT` / `TTS_GEN_S3_BUCKET` | empty | S3 sink wiring; lands with the Data Persistence PoC bucket. |

## Input

### `GET /sections`

Enumerates the valid, generatable sections of a pinned revision.
Blocklisted sections (References, See also, ...) are absent from the
response entirely; too-short sections appear with `generatable: false` so
"skipped" is distinguishable from "failed".

| Query parameter | Required | Description |
| --- | --- | --- |
| `wiki_id` | yes | e.g. `enwiki`. |
| `page_id` | yes | Page the revision must belong to (integrity-checked). |
| `rev_id` | yes | Exact revision to enumerate. |

```console
curl -s "http://localhost:8080/sections?wiki_id=enwiki&page_id=9228&rev_id=1362915217"
```

### `POST /generate-section`

Produces the artifact family for one section of one pinned revision.

| Body field | Required | Description |
| --- | --- | --- |
| `wiki_id` | yes | e.g. `enwiki`. |
| `page_id` | yes | Page the revision must belong to (integrity-checked). |
| `rev_id` | yes | Exact revision to generate from. |
| `section_id` | yes | Section address from `GET /sections` (lowercase heading slug; `lead` for the lead). |
| `generation_config.voice` | no (`af_heart`) | Voice. |
| `generation_config.lang` | no (`en-us`) | Language. |
| `generation_config.timestamps` | no (`full`) | `full`, `proportional`, or `none`; ignored when no timing artifact is requested (audio-only rides the isvc's cheapest path). |
| `generation_config.artifacts` | no | Any of `audio_opus`, `audio_mp3`, `captions_vtt`, `timestamps_json`, `audio_pcm_s16le`. Default: `["audio_opus", "captions_vtt", "timestamps_json"]`. |

```console
curl -s -X POST http://localhost:8080/generate-section \
  -H 'Content-Type: application/json' \
  -d '{
    "wiki_id": "enwiki",
    "page_id": 9228,
    "rev_id": 1362915217,
    "section_id": "atmosphere",
    "generation_config": {"artifacts": ["audio_opus", "captions_vtt"]}
  }'
```

## Output

### `GET /sections` response

`wiki_id` / `page_id` / `rev_id` echo the request; `revision_timestamp` is
the revision's save time; `generation_version` is the version this service
would stamp on artifacts generated now (the pipeline compares index records
against it to find outdated artifacts); `sections` is an array in document
order:

| Field | Description |
| --- | --- |
| `section_id` | Lowercase heading slug, `-N` ordinal for duplicate headings, `lead` for the lead. |
| `title`, `level` | Heading text; 1 = lead, 2-6 = heading level. |
| `generatable` | Whether `POST /generate-section` would produce audio. |
| `char_count` | Length of the cleaned (normalized) text. |
| `content_sha256` | SHA-256 of the normalized text (present iff generatable). Only comparable within one `generation_version`. |
| `skip_reason` | Present iff not generatable (e.g. `text_below_minimum`). |

### `POST /generate-section` response

`{artifacts: [...], segment_count}` where every artifact carries the full
index-record field set:

| Field | Description |
| --- | --- |
| `wiki_id`, `page_id`, `rev_id`, `section_id` | The address generated. |
| `generation_version` | Model + voice + normalizer identity that produced the artifact. |
| `content_sha256` | Hash of the normalized text spoken. |
| `artifact_type`, `media_type` | e.g. `audio_opus` / `audio/ogg; codecs=opus`. |
| `duration_ms` | Audio duration. |
| `bytes_b64` **or** `blob_uri` (+ `size_bytes`) | Inline bytes or a written-blob pointer, depending on the configured sink; the schema differs by exactly this field. |
| `timestamps`, `timestamps_mode` | On `timestamps_json` (and mode on `captions_vtt`). |
| `sample_rate`, `encoding` | On `audio_pcm_s16le`. |

### Errors

Every non-2xx body is `{code, message}` with a machine-readable code.
**4xx codes are deterministic** (same request, same outcome; record, never
retry). **5xx codes are transient** (retry with backoff). This split is a
contract promise the DE retry logic depends on.

| Code | Status | Class |
| --- | --- | --- |
| `revision_not_found` | 404 | deterministic |
| `revision_page_mismatch` | 409 | deterministic (rev belongs to a different page; generating would poison the index) |
| `section_not_found_at_revision` | 404 | deterministic (blocklisted sections are never addressable) |
| `text_below_minimum` | 422 | deterministic |
| `artifact_type_not_available` | 400 | deterministic |
| `unsupported_wiki` | 400 | deterministic |
| `upstream_fetch_error` | 502 | transient |
| `synthesis_error` | 502 | transient |
| `transcode_error` | 502 | transient |

## Contract decisions (Phase 1, locked)

| Decision | Definition | Rationale |
| --- | --- | --- |
| `section_id` | lowercase heading slug, `-N` ordinal for duplicates, `lead` for the lead | deterministic from heading text + document order; a renamed heading is a new section (a regeneration), accepted |
| `content_sha256` | SHA-256 of the **normalized** text | normalized text is what the voice speaks; markup-only edits hash identically and share artifacts. Only comparable within one `generation_version` |
| `generation_version` | `{kokoro_version}+{voice}+norm-{ruleset}-{engine}-{whitelist_sha8}` e.g. `kokoro-v1.0+af_heart+norm-2026.07.20-nemo-98d86449` | identifies everything that changes audio for identical input; a bump = ML requests a DE backfill. The engine tag (`nemo`/`regex`) matters because the fallback produces different text; the whitelist hash catches pronunciation changes with no code change |
| Fetch source | Parsoid HTML via `GET /w/rest.php/v1/revision/{id}/html` | templates expanded (unlike wikitext), old revisions addressable (unlike TextExtracts), sections structurally marked (`<section data-mw-section-id>`) |
| Chunk size | `MAX_SEGMENT_CHARS=400` default (isvc hard ceiling 800) | quality knob owned here; tune with listening in Phase 4 |

## Phase status

**Phase 1 (done) -- contract + core library.** OpenAPI spec; revision-pinned
fetcher with page/rev integrity check; section extraction from Parsoid HTML
(blocklist, noise stripping, section_id scheme); normalization (v0 port, NeMo
+ regex fallback); chunking (v0 behavior, reimplemented to its test suite);
`generation_version` / `content_sha256`; both endpoints wired.
Validated: live run against Earth pinned at two revisions showed 32/34
sections hash-identical across revisions (the content-reuse property) and
only genuinely edited sections differing.

**Phase 2 (done) -- artifacts + error taxonomy hardening.** Opus transcode
(ffmpeg, `-bitexact` + pinned Ogg serial for byte-deterministic output) with
MP3 as a supported alternative artifact type pending the Apps codec decision;
`captions_vtt` formatting (v0 port with its tests); `media_type` on every
artifact for the storage layer; isvc client retries with linear backoff
(transient 5xx/transport only; 4xx from the isvc means a generator bug and
raises loudly); artifact-driven request shaping (no timing artifact ->
isvc `timestamps=none`, the RTF-0.22 path); `transcode_error` added to the
taxonomy. Includes empirical transcode-determinism tests and the golden test
(same pinned input twice -> byte-identical artifacts).

**Phase 3 (done) -- deployment + de-risking spikes.** Image with the NeMo
grammar cache baked at build (a broken normalization stack fails the image
build, not the deploy; kills v0's 60s-cold-start / CrashLoopBackOff mode);
lifespan-API migration; artifact sinks behind one interface (see
`TTS_GEN_BLOB_SINK` above), closing the blob-write question as "both
supported, Data Persistence's call"; staging deployment; and the timeout
spike (see [SPIKE_ANSWERS.md](SPIKE_ANSWERS.md)): measured section-length
distribution, in-pod worst-case synthesis (242 s wall on the corpus-max
section), and the 600 s timeout chain provisioned and verified end to end.
The corpus scan also caught and fixed two real content bugs
(fallback-normalizer crash on 13-digit numbers; bibliography sections
reaching the voice through blocklist title variants, fixed structurally),
both regression-tested. The T426756 sup/superscript carry-over landed with
ruleset `2026.07.20` (markup-derived scientific notation is now spoken).

**Phase 4 (in progress, T432692) -- 50-article pilot run.** Produces the
measured numbers pack for the DE intake meeting: real artifact sizes,
latency distribution, skip/failure taxonomy rates, sections-per-article
stats, maxReplicas recommendation, and the permanent memory envelope.
Driver: `scripts/pilot_run.py`; pack: `scripts/pilot_summary.py`.

## What this service deliberately does NOT do

No queueing, no scheduling, no event consumption, no index diffing, no
storage ownership. The moment this service grows a work queue it has rebuilt
v0's Celery layer and un-drawn the ML/DE boundary. Those concerns belong to
the DE pipelines (Airflow batch, Bento edit-stream) and Data Persistence
(blob store + index), per the Prep Pantry intake document.

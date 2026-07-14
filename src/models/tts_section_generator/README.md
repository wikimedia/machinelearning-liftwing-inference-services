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
                    -> synthesize + align -> [Phase 2: transcode, VTT]
```

## Contract

Two endpoints; full spec in [`openapi.yaml`](openapi.yaml) (the artifact DE
reviews).

* `GET /sections?wiki_id=&page_id=&rev_id=` -- the valid, generatable
  sections of a pinned revision, with `content_sha256` per section. This is
  what the DE pipeline diffs against the artifact index. Blocklisted sections
  (References, See also, ...) are absent; too-short sections appear with
  `generatable: false` so "skipped" is distinguishable from "failed".
* `POST /generate-section` -- the artifact family for one section.

Error bodies carry a machine-readable `code`. **4xx codes are deterministic**
(same request, same outcome; record, never retry). **5xx are transient**
(retry with backoff). The service is stateless, so retries are always safe.

### Contract decisions (Phase 1, locked)

| Decision | Definition | Rationale |
| --- | --- | --- |
| `section_id` | lowercase heading slug, `-N` ordinal for duplicates, `lead` for the lead | deterministic from heading text + document order; a renamed heading is a new section (a regeneration), accepted |
| `content_sha256` | SHA-256 of the **normalized** text | normalized text is what the voice speaks; markup-only edits hash identically and share artifacts. Only comparable within one `generation_version` |
| `generation_version` | `{kokoro_version}+{voice}+norm-{ruleset}-{engine}-{whitelist_sha8}` e.g. `kokoro-v1.0+af_heart+norm-2026.07-nemo-98d86449` | identifies everything that changes audio for identical input; a bump = ML requests a DE backfill. The engine tag (`nemo`/`regex`) matters because the fallback produces different text; the whitelist hash catches pronunciation changes with no code change |
| Fetch source | Parsoid HTML via `GET /w/rest.php/v1/revision/{id}/html` | templates expanded (unlike wikitext), old revisions addressable (unlike TextExtracts), sections structurally marked (`<section data-mw-section-id>`) |
| Chunk size | `MAX_SEGMENT_CHARS=400` default (isvc hard ceiling 800) | quality knob owned here; tune with listening in Phase 4 |

## Run locally

```console
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # uncomment nemo_text_processing for full normalization
python3 -m pytest tests/               # 25 tests (v0 suites carried over + Parsoid fixtures)
python3 scripts/validate_phase1.py     # live pinning proof against en.wikipedia.org
python3 -m tts_generator.service       # serves on :8081
```

Configuration is env-driven; see `tts_generator/config.py` (isvc URL/Host
header, timeouts, chunk size, min text length, versions).

## Phase status

**Phase 1 (this) -- contract + core library.** OpenAPI spec; revision-pinned
fetcher with page/rev integrity check; section extraction from Parsoid HTML
(blocklist, noise stripping, section_id scheme); normalization (v0 port, NeMo
+ regex fallback); chunking (v0 behavior, reimplemented to its test suite);
`generation_version` / `content_sha256`; both endpoints wired; artifacts
limited to `audio_pcm_s16le` + `timestamps_json` (raw isvc passthrough).
Validated: 25 unit tests green; live run against Earth pinned at two
revisions showed 32/34 sections hash-identical across revisions (the
content-reuse property) and only genuinely edited sections differing.

**Phase 2 -- artifacts + error taxonomy hardening.** Opus/MP3 transcode
(ffmpeg), `captions_vtt` formatting (v0 port), isvc client retries/backoff,
golden-artifact regression tests (Kokoro is deterministic for fixed input).

**Phase 3 -- deployment + de-risking spikes.** Container with NeMo grammar
cache baked at image build; staging deploy; the timeout-ceiling spike
(longest-section synthesis vs gateway limits); blob-write mode behind the
artifact interface (`bytes_b64` now, `blob_uri` when storage exists).

**Phase 4 -- 50-article pilot run.** Produces the measured numbers pack for
the DE intake meeting: real artifact sizes, latency distribution, skip/failure
taxonomy rates, sections-per-article stats, maxReplicas recommendation.

## What this service deliberately does NOT do

No queueing, no scheduling, no event consumption, no index diffing, no
storage ownership. The moment this service grows a work queue it has rebuilt
v0's Celery layer and un-drawn the ML/DE boundary. Those concerns belong to
the DE pipelines (Airflow batch, Bento edit-stream) and Data Persistence
(blob store + index), per the Prep Pantry intake document.

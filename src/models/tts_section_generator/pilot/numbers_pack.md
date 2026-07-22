# TTS Section Generator: Phase 4 pilot measured numbers pack (T432692)

Run: 50 Featured Articles (pinned revisions from the committed scan snapshot), 588 sections generated, 85 deterministic skips, 1 failures. generation_version `kokoro-v1.0+af_heart+norm-2026.07.20-nemo-98d86449`. Full artifact family per section (Opus + MP3 + VTT + timestamps, full CTC alignment: worst-case tier).

## Latency (full path: fetch + normalize + synthesize + align + transcode, through the mesh ingress)

| | wall/section | audio/section |
| --- | --- | --- |
| p50 | 43.5 s | 131 s |
| p90 | 93.9 s | 272 s |
| p99 | 165.6 s | 529 s |
| max | 223.6 s | 650 s |

Calibration at pilot scale: **16.2 chars/audio-second**, effective end-to-end **RTF 0.339** (total 24.2 audio-hours in 8.2 wall-hours). 0 sections needed a transient retry before succeeding.

## Artifact sizes (measured per section)

| artifact | mean | p99 | total | per audio-hour |
| --- | --- | --- | --- | --- |
| audio_opus | 587 KB | 2100 KB | 345.3 MB | 14.3 MB |
| audio_mp3 | 889 KB | 3176 KB | 522.5 MB | 21.6 MB |
| captions_vtt | 14 KB | 53 KB | 8.5 MB | 0.3 MB |
| timestamps_json | 28 KB | 104 KB | 16.5 MB | 0.7 MB |

## Skips and failures by taxonomy code (observed)

- skip `text_below_minimum`: 85
- FAIL `502 synthesis_error`: 1

## Corpus extrapolation (pilot per-article means x 6,959 Featured Articles)

- sections/article: **11.8** generatable -> **~82k** corpus sections
- audio/article: **29.0 min** -> **~3,366 corpus audio-hours**
- storage: Opus **~48 GB**, MP3 ~73 GB, sidecars ~3 GB
- full-alignment corpus drain: **~48 pod-days** of isvc time (audio-only tier: derived by the isvc RTF ratio, ~39 pod-days)

## maxReplicas: corpus drain time vs isvc replica count

| replicas | full-alignment drain | audio-only drain (derived) |
| --- | --- | --- |
| 1 | 48 days | 39 days |
| 2 | 24 days | 19 days |
| 4 | 12 days | 10 days |
| 6 | 8 days | 6 days |
| 8 | 6 days | 5 days |

Recommendation slot: pick the replica count whose drain time fits the backfill window DE proposes, bounded by cluster capacity; edits-driven steady state needs ~1 replica (<<1 write/s).

## Memory envelope (Phase 3 finding #3 closure)
- generator `memory.peak` across the pilot (external driver, clean measurement): **1.26 GiB** -> recommended permanent limit 2 Gi

_Every figure above is measured from the pilot except rows labeled extrapolation/derived. Reproduce: pilot_run.py --scan corpus_scan.json --seed 43; the population and revisions are pinned by the committed scan snapshot._

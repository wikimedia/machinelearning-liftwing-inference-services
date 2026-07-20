# Phase 3 Spike Answers

Status: measured on a 100-article seeded sample (seed 43) of the 6,959
Featured Articles, 2026-07-12. Slots marked `[in-pod]` need the staging
isvc; slots marked `[infra]` need a config lookup.

## Spike 1: the request-timeout ceiling

**Question.** The generator makes one synchronous isvc call per section.
Does the longest real section fit under the gateway/revision timeout, or
do we need to split oversized sections across multiple isvc calls?

**Measured demand side** (scripts/scan_corpus.py, 100-article seeded
sample of the 6,959 FAs; estimates calibrated on T430536: ~15
chars/audio-second, RTF 0.27):

| Percentile | Cleaned chars | Est. audio | Est. isvc wall |
| --- | --- | --- | --- |
| p50 | 2,196 | ~146 s | ~40 s |
| p90 | 4,395 | ~293 s | ~79 s |
| p95 | 5,171 | ~345 s | ~93 s |
| p99 | 7,000 | ~467 s | ~126 s |
| sample max | 12,843 | ~856 s | ~231 s |

**Measured supply side** (scripts/spike_timeout.py, staging generator +
staging isvc, top-5 sections, full-alignment artifact family, 2026-07-20):

| Section | Chars | Audio | Wall | RTF |
| --- | --- | --- | --- | --- |
| Lockheed C-130 :: post-vietnam-tasks | 12,843 | 753 s | **242.2 s** | 0.322 |
| Æthelred :: early-rule | 9,280 | 552 s | 178.5 s | 0.323 |
| SMS Grosser Kurfürst :: battle-of-jutland | 8,338 | 496 s | 116.1 s | 0.234 |
| Belle Vue Zoological Gardens :: zoo | 8,190 | 478 s | 150.5 s | 0.315 |
| Texas Revolution :: background | 8,188 | 496 s | 116.0 s | 0.234 |

Recalibration: real prose speaks at **16.9 chars/audio-s** (assumed 15;
estimates were ~11% conservative on audio length). Effective end-to-end
RTF **0.286 mean, 0.32 worst** (isvc 0.27 + fetch/normalize/transcode).
The per-request RTF spread (0.234-0.323) matches the ~20% cross-pod
variance documented on the isvc in T430536; not a new effect. Estimate
vs measurement on the worst section: 231 s estimated, 242 s measured
(the two calibration errors nearly cancel).

**Answer: provision 600 s end to end; build no splitting.** The measured
sample max (242 s) is TIGHT against 300 s (<20% headroom); the
extrapolated corpus true max (1.5-2x sample chars, at measured
calibration) is **~365-490 s, which EXCEEDS 300 s and fits 600 s** with
1.2-1.6x headroom. Oversized-section splitting (paragraph-boundary joins
with a silence gap) remains the designed fallback if the full-corpus scan
finds a tail beyond ~800 s of wall; do not build it before then.

Config actions:
- `TTS_ISVC_TIMEOUT_S: "600"` in the generator values (the default 300 s
  was the tightest ceiling on the path and would be exceeded by the
  extrapolated corpus max).
- `[infra]` remaining lookups: generator inbound mesh route timeout
  (empirical check: the Lockheed section through
  tts-section-generator...:31443 must survive its 242 s), isvc Knative
  revision timeoutSeconds, istio gateway route timeout on
  inference-staging. The minimum of these must be >= 600 s.
- `[full-scan]` optional: scan_corpus.py --all replaces the extrapolated
  max with the true one (requires an internet-connected host or the
  MW_API_PROXY patch; the pod has no general egress).

**For DE:** at p50 ~40 s per section, batch task-level timeouts of 10
min/section with retry are comfortable; the 600 s per-request ceiling is
the constraint that matters.

## Spike 2: blob-write mode (intake open question #3)

**Question.** Does the compute function write artifacts to the blob store
and return URIs, or return bytes for the invoking layer to write?

**Answer: both are implemented behind one interface; it is now purely
Data Persistence's call.** `TTS_GEN_BLOB_SINK` selects at startup:
`inline` (default: `bytes_b64` in the response; LAC-native, zero storage
dependencies), `file` (writes under the canonical revision-scoped key
layout `{wiki}/{page}/{rev}/{section}.{ext}`, returns `blob_uri`; used by
the Phase 4 pilot and as the local stand-in), `s3` (interface-complete
stub; wiring lands with the Data Persistence PoC bucket, and selecting it
unconfigured fails at STARTUP, not first write). File-sink writes are
atomic (tmp + rename), and Phase 2's byte-determinism makes re-writes
idempotent overwrites. The response schema is identical except
`bytes_b64` vs `blob_uri`, so the DE pipeline's handling differs by one
field regardless of the eventual decision.

## Collateral findings (the scan paid for itself)

1. **Fallback normalizer crash on real text:** v0's `_int_to_words`
   raised IndexError on numbers beyond "billion"; a sampled FA contains a
   13+-digit figure. Fixed (scales extended; digit-by-digit beyond 10^18;
   a fallback must degrade, never crash) with a regression test.
2. **Bibliography sections were reaching the voice:** v0's exact-title
   blocklist misses FA variants ("Bibliography", "Sources", "Cited
   sources"...): 10-20k-char citation lists ranked as the longest
   "generatable" sections. Fixed structurally: `refbegin`/`reflist`
   containers and `citation`-class elements (rendered {{cite}} templates)
   are stripped, so citation lists empty out and fall under the
   min-length gate in ANY container under ANY heading; unambiguous title
   variants also added to the blocklist. Regression tests pin both. The
   scan's tail went from 19,725-char bibliographies to genuine prose.

3. **The corpus tail sizes the memory envelope, not the median.** The
   staging pod (1Gi limit) served normal-section traffic for 2 hours,
   then was OOMKilled by the FIRST corpus-tail request: measured
   `memory.peak` during worst-section generation is **1.13 GiB** (753 s
   of audio means a ~43 MB PCM decode plus a ~58 MB base64 response
   transiently stacked with NeMo FST paging and encoder buffers; the
   in-cgroup spike script added its own response copy). Limit raised to
   3Gi (2.6x measured peak); the Phase 4 pilot, with its external
   driver, will set the permanent value from a cleaner peak. Mitigation
   landed: the isvc base64 string is dropped immediately after decode.
   Both ceilings this spike exists to find (time and memory) are tail
   properties invisible to median-workload testing.

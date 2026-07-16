# Phase 3 Spike Answers

Status: measured on a 100-article seeded sample (seed 43) of the 6,959
Featured Articles, 2026-07-12. Slots marked `[in-pod]` need the staging
isvc; slots marked `[infra]` need a config lookup.

## Spike 1: the request-timeout ceiling (intake open question #7)

**Question.** The generator makes one synchronous isvc call per section.
Does the longest real section fit under the gateway/revision timeout, or
do we need to split oversized sections across multiple isvc calls?

**Measured demand side** (scripts/scan_corpus.py; estimates calibrated on
T430536: ~15 chars/audio-second, RTF 0.27 full alignment):

| Percentile | Cleaned chars | Est. audio | Est. isvc wall |
| --- | --- | --- | --- |
| p50 | 2,196 | ~146 s | ~40 s |
| p90 | 4,395 | ~293 s | ~79 s |
| p95 | 5,171 | ~345 s | ~93 s |
| p99 | 7,000 | ~467 s | ~126 s |
| sample max | 12,843 | ~856 s | ~231 s |

Sections exceeding candidate ceilings (estimated): >60s: 23.2%; >120s:
1.4%; **>300s: 0**; >600s: 0. Mean 12.2 generatable sections/article
(consistent with the intake's ~140k-section corpus estimate at 7k
articles). The sample max is a genuine prose section (Lockheed C-130,
"Post-Vietnam tasks"), not a data artifact: bibliography-type sections
that initially dominated the tail were content bugs and are now stripped
(see "collateral findings").

**Caveats.** A 100-of-6,959 sample under-observes the extreme tail; the
true corpus max plausibly exceeds the sample max by 1.5-2x (call it
~350-450 s estimated). Run `scan_corpus.py --all` once (slow, ~1-2 h
polite) to replace this guess with the true max: `[full-scan]`.

**Provisional answer.** A **600 s ceiling** covers the sampled max with
~2.6x headroom and the extrapolated true max with ~1.5x; a 300 s ceiling
is tight against the extrapolated tail. Recommendation: provision the
generator-to-isvc path (and any gateway in front of the generator) at
**600 s**, and do NOT build section splitting now: the sections that
would need it are 0% of the sample, and splitting complicates the
generator (crossfade/timestamp stitching across isvc calls) for a tail
that a config value covers. If the `[infra]` lookup finds a hard ceiling
below 600 s that cannot be raised, revisit; splitting at paragraph
boundaries with a silence gap is the designed fallback, and only then.

Remaining slots:
- `[in-pod]` scripts/spike_timeout.py on the scan's top-5: worst measured
  wall = ___ s; measured chars/audio-s = ___; effective RTF = ___.
- `[infra]` actual LiftWing/Knative values on the path: isvc revision
  timeoutSeconds = ___; gateway/Envoy route timeout = ___; anything in
  front of the generator = ___.

**For DE:** at p50 ~40 s per section, batch pipeline task-level timeouts
of 10 min/section with retry are comfortable; the per-request ceiling
above is the constraint that matters.

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

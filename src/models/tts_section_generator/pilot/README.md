# Phase 4 pilot run record (T432692)

Run 2026-07-21 from deploy2003 against staging through the mesh ingress:
50 Featured Articles (seed 43, pinned revisions from ../corpus_scan.json),
588/589 sections generated in 8.3 h, full artifact family, concurrency 1,
generation_version kokoro-v1.0+af_heart+norm-2026.07.20-nemo-98d86449.
The single failure is root-caused in P95036 (Kokoro 510-phoneme limit on
interlinear-gloss content; fixed structurally in a later ruleset).

- pilot_results.jsonl: one JSON record per section outcome (ok/skip/fail
  with timings, sizes, hashes). Regenerate the pack from it:
  python3 ../scripts/pilot_summary.py --log pilot_results.jsonl --memory-peak-gib 1.26
- numbers_pack.md: the measured numbers pack for the DE intake.
- memory_peak.txt: raw generator-pod memory.peak at run end (RESTARTS 0,
  3Gi limit; basis for the 2Gi permanent recommendation).

Reproduce: scripts/pilot_run.py --scan corpus_scan.json --articles 50
--seed 43 (population and revisions are pinned, so the run is repeatable).

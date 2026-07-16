#!/usr/bin/env python3
"""Spike 1, measurement half: synthesize the WORST sections the corpus
scan found, through a running generator against the real staging isvc,
and report wall time per section vs candidate ceilings.

Run from a WMF-internal host with the generator up (see README):

    python3 scripts/scan_corpus.py --sample 100 --out corpus_scan.json
    python3 scripts/spike_timeout.py --scan corpus_scan.json --top 5

Reports, per section: end-to-end generator wall time (fetch + normalize +
isvc + transcode) for the WORST-CASE artifact family (full CTC alignment),
audio duration, and the measured chars-per-audio-second and effective RTF,
which recalibrate the scan's estimate chain with real numbers.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CEILINGS_S = [60, 120, 300, 600]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="corpus_scan.json")
    ap.add_argument("--base", default="http://localhost:8081")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    scan = json.loads(Path(args.scan).read_text())
    targets = sorted(scan["sections"], key=lambda s: -s["chars"])[: args.top]

    print(
        f"Synthesizing the {len(targets)} longest sections "
        f"(full alignment, worst case)...\n"
    )
    results = []
    for t in targets:
        t0 = time.perf_counter()
        r = requests.post(
            f"{args.base}/generate-section",
            json={
                "wiki_id": "enwiki",
                "page_id": t["page_id"],
                "rev_id": t["rev_id"],
                "section_id": t["section_id"],
                "generation_config": {
                    "artifacts": ["audio_opus", "captions_vtt", "timestamps_json"]
                },
            },
            timeout=1800,
        )
        wall = time.perf_counter() - t0
        if r.status_code != 200:
            print(
                f"  FAILED {t['title']} :: {t['section_id']} "
                f"({r.status_code}): {r.text[:200]}"
            )
            continue
        dur_s = r.json()["artifacts"][0]["duration_ms"] / 1000
        results.append((t, wall, dur_s))
        print(
            f"  {wall:6.1f}s wall  {dur_s:6.0f}s audio  "
            f"chars/audio-s={t['chars'] / dur_s:5.1f}  RTF={wall / dur_s:.3f}  "
            f"{t['title']} :: {t['section_id']}"
        )

    if not results:
        print("No successful synths; nothing to conclude.")
        return 1

    worst_wall = max(w for _, w, _ in results)
    cps = sum(t["chars"] / d for t, _, d in results) / len(results)
    rtf = sum(w / d for _, w, d in results) / len(results)
    print(
        f"\nMeasured calibration: {cps:.1f} chars/audio-s, "
        f"effective end-to-end RTF {rtf:.3f}"
    )
    print(f"Worst measured wall: {worst_wall:.1f}s")
    for c in CEILINGS_S:
        verdict = (
            "FITS"
            if worst_wall < c * 0.8
            else ("TIGHT (<20% headroom)" if worst_wall < c else "EXCEEDS")
        )
        print(f"  vs {c:3d}s ceiling: {verdict}")
    print(
        "\nPaste these numbers into SPIKE_ANSWERS.md and check them "
        "against the actual LiftWing/Knative timeout configuration."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

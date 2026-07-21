#!/usr/bin/env python3
"""Spike 1 (intake open question #7): the request-timeout ceiling.

Scans Featured Articles for the section-length distribution, converts
lengths to ESTIMATED isvc wall time, and reports how many sections would
exceed candidate gateway ceilings. The estimate chain, calibrated on
T430536 measurements:

    audio_seconds  = cleaned_chars / CHARS_PER_AUDIO_SECOND (~15, measured:
                     245 chars -> 16.81s, 328 chars -> 21.4s)
    isvc_wall_est  = audio_seconds * RTF (0.27 full alignment, 0.22 none)

The generator makes ONE isvc call per section (all chunks in one request),
so section length maps directly onto single-request duration; the fetch/
normalize/transcode overhead around it is seconds, not the driver.

Estimates rank and scope the problem; the definitive numbers come from
scripts/spike_timeout.py, which synthesizes the WORST sections this scan
finds through the real staging isvc.

Usage:
    python3 scripts/scan_corpus.py --sample 100 --seed 43 --out scan.json
    python3 scripts/scan_corpus.py --all --delay 0.5   # full corpus (slow)
"""

import argparse
import json
import random
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_generator.config import MIN_TEXT_LENGTH, MW_API_PROXY
from tts_generator.fetch import (
    FetchError,
    _headers,
    _shared_session,
    fetch_revision_html,
    rest_base,
)
from tts_generator.sections import extract_sections
from tts_generator.text import clean_spoken_text, init_nemo

CHARS_PER_AUDIO_SECOND = 15.0
RTF_FULL = 0.27
CEILINGS_S = [60, 120, 300, 600]

# Action API (categorymembers) through the envoy services-proxy when
# configured (in-pod: no general egress), direct otherwise (dev hosts).
API = (
    f"{MW_API_PROXY}/w/api.php"
    if MW_API_PROXY
    else "https://en.wikipedia.org/w/api.php"
)


def fetch_fa_titles() -> list[str]:
    """All Featured Article titles via categorymembers (v0's extractor)."""
    titles, cont = [], None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Featured_articles",
            "cmnamespace": "0",
            "cmlimit": "max",
            "format": "json",
        }
        if cont:
            params["cmcontinue"] = cont
        url = f"{API}?{urllib.parse.urlencode(params)}"
        data = _shared_session.get(url, timeout=30, headers=_headers("enwiki")).json()
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            return sorted(set(titles))


def resolve(title: str) -> tuple[int, int]:
    enc = urllib.parse.quote(title.replace(" ", "_"), safe="")
    bare = _shared_session.get(
        f"{rest_base('enwiki')}/page/{enc}/bare",
        timeout=30,
        headers=_headers("enwiki"),
    ).json()
    return bare["id"], bare["latest"]["id"]


def pctl(sorted_vals: list, p: float):
    return sorted_vals[min(int(len(sorted_vals) * p), len(sorted_vals) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--out", default="corpus_scan.json")
    args = ap.parse_args()

    init_nemo()  # estimate on the same normalization path as generation

    print("Fetching Featured Article list...")
    fa = fetch_fa_titles()
    print(f"  {len(fa)} Featured Articles")

    if not args.all:
        random.seed(args.seed)
        fa = random.sample(fa, min(args.sample, len(fa)))
        print(f"  scanning seeded random sample of {len(fa)}")

    sections, per_article, failures, skipped_short = [], [], [], 0
    for i, title in enumerate(fa, 1):
        try:
            page_id, rev_id = resolve(title)
            html = fetch_revision_html("enwiki", rev_id)
            counts = []
            for s in extract_sections(html):
                n = len(clean_spoken_text(s.raw_text))
                if n <= MIN_TEXT_LENGTH:
                    skipped_short += 1
                    continue
                counts.append(n)
                sections.append(
                    {
                        "title": title,
                        "section_id": s.section_id,
                        "chars": n,
                        "page_id": page_id,
                        "rev_id": rev_id,
                    }
                )
            per_article.append(sum(counts))
        except (FetchError, KeyError, ValueError) as e:
            failures.append((title, str(e)[:80]))
        if i % 10 == 0:
            print(f"  {i}/{len(fa)} articles, {len(sections)} sections so far")
        time.sleep(args.delay)

    chars = sorted(s["chars"] for s in sections)
    est_wall = lambda c: (c / CHARS_PER_AUDIO_SECOND) * RTF_FULL  # noqa: E731

    print(
        f"\n=== SECTION LENGTH DISTRIBUTION ({len(sections)} generatable "
        f"sections, {len(per_article)} articles) ==="
    )
    rows = [("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]
    for label, p in rows:
        c = pctl(chars, p)
        print(
            f"  {label}: {c:6d} chars  ~{c / CHARS_PER_AUDIO_SECOND:6.0f}s audio  "
            f"~{est_wall(c):5.0f}s isvc wall (est)"
        )
    print(
        f"  max: {chars[-1]:6d} chars  ~{chars[-1] / CHARS_PER_AUDIO_SECOND:6.0f}s "
        f"audio  ~{est_wall(chars[-1]):5.0f}s isvc wall (est)"
    )

    print("\n=== SECTIONS EXCEEDING CANDIDATE CEILINGS (estimated) ===")
    for ceil in CEILINGS_S:
        n = sum(1 for c in chars if est_wall(c) > ceil)
        print(f"  > {ceil:3d}s: {n:4d} sections ({100 * n / len(chars):5.2f}%)")

    print(
        f"\n  sections/article: mean {len(sections) / len(per_article):.1f}; "
        f"skipped-short {skipped_short}; article failures {len(failures)}"
    )

    top = sorted(sections, key=lambda s: -s["chars"])[:10]
    print("\n=== LONGEST SECTIONS (spike_timeout.py targets) ===")
    for s in top:
        print(
            f"  {s['chars']:6d} chars  ~{est_wall(s['chars']):4.0f}s est  "
            f"{s['title']} :: {s['section_id']}"
        )

    Path(args.out).write_text(
        json.dumps(
            {
                "n_articles": len(per_article),
                "n_sections": len(sections),
                "skipped_short": skipped_short,
                "failures": failures,
                "sections": sections,
                "per_article_chars": per_article,
            },
            indent=1,
        )
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

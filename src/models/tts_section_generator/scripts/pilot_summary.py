#!/usr/bin/env python3
"""Phase 4 numbers-pack generator (T432692): results log -> the one-page
measured numbers pack for the DE intake, in Markdown.

Every number here is MEASURED from the pilot run except the corpus
extrapolations, which are (pilot per-article means) x (6,959 Featured
Articles) and are labeled as such.

Usage:
    python3 pilot_summary.py --log pilot_results.jsonl \\
        --scan corpus_scan.json [--corpus-articles 6959] \\
        [--memory-peak-gib X.XX] > numbers_pack.md
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def pctl(vals: list, p: float):
    s = sorted(vals)
    return s[min(int(len(s) * p), len(s) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="pilot_results.jsonl")
    ap.add_argument("--scan", default="corpus_scan.json")
    ap.add_argument("--corpus-articles", type=int, default=6959)
    ap.add_argument("--memory-peak-gib", type=float, default=None)
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.log).open()]
    ok = [r for r in records if r["status"] == "ok"]
    skips = [r for r in records if r["status"] == "skip"]
    fails = [r for r in records if r["status"] == "fail"]
    if not ok:
        print("No successful generations in the log; nothing to summarize.")
        return 1

    articles = {r["title"] for r in records if "title" in r}
    n_articles = len(articles)

    walls = [r["wall_s"] for r in ok]
    audio_s = [r["duration_ms"] / 1000 for r in ok]
    chars = [r["char_count"] for r in ok if r.get("char_count")]
    total_audio_s = sum(audio_s)
    total_wall_s = sum(walls)
    retried_ok = sum(1 for r in ok if r["attempts"] > 1)

    sizes = defaultdict(list)
    for r in ok:
        for kind, n in (r.get("sizes") or {}).items():
            sizes[kind].append(n)

    per_article_audio = defaultdict(float)
    per_article_secs = Counter()
    for r in ok:
        per_article_audio[r["title"]] += r["duration_ms"] / 1000
        per_article_secs[r["title"]] += 1

    mean_secs_art = len(ok) / n_articles
    mean_audio_art_s = total_audio_s / n_articles
    cps = sum(chars) / total_audio_s if chars else 0
    rtf = total_wall_s / total_audio_s

    C = args.corpus_articles
    corpus_audio_h = C * mean_audio_art_s / 3600
    opus_total = sum(sizes.get("audio_opus", []))
    mp3_total = sum(sizes.get("audio_mp3", []))
    opus_per_audio_s = opus_total / total_audio_s
    mp3_per_audio_s = mp3_total / total_audio_s
    corpus_opus_gb = corpus_audio_h * 3600 * opus_per_audio_s / 1e9
    corpus_mp3_gb = corpus_audio_h * 3600 * mp3_per_audio_s / 1e9
    corpus_wall_pod_days = corpus_audio_h * 3600 * rtf / 86400

    version = ok[0]["generation_version"]

    out = []
    w = out.append
    w("# TTS Section Generator: Phase 4 pilot measured numbers pack (T432692)")
    w("")
    w(
        f"Run: {n_articles} Featured Articles (pinned revisions from the "
        f"committed scan snapshot), {len(ok)} sections generated, "
        f"{len(skips)} deterministic skips, {len(fails)} failures. "
        f"generation_version `{version}`. Full artifact family per section "
        f"(Opus + MP3 + VTT + timestamps, full CTC alignment: worst-case tier)."
    )
    w("")
    w(
        "## Latency (full path: fetch + normalize + synthesize + align + "
        "transcode, through the mesh ingress)"
    )
    w("")
    w("| | wall/section | audio/section |")
    w("| --- | --- | --- |")
    for label, p in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        w(f"| {label} | {pctl(walls, p):.1f} s | {pctl(audio_s, p):.0f} s |")
    w(f"| max | {max(walls):.1f} s | {max(audio_s):.0f} s |")
    w("")
    w(
        f"Calibration at pilot scale: **{cps:.1f} chars/audio-second**, "
        f"effective end-to-end **RTF {rtf:.3f}** "
        f"(total {total_audio_s / 3600:.1f} audio-hours in "
        f"{total_wall_s / 3600:.1f} wall-hours). "
        f"{retried_ok} sections needed a transient retry before succeeding."
    )
    w("")
    w("## Artifact sizes (measured per section)")
    w("")
    w("| artifact | mean | p99 | total | per audio-hour |")
    w("| --- | --- | --- | --- | --- |")
    for kind in ("audio_opus", "audio_mp3", "captions_vtt", "timestamps_json"):
        v = sizes.get(kind)
        if not v:
            continue
        per_h = sum(v) / total_audio_s * 3600 / 1e6
        w(
            f"| {kind} | {sum(v) / len(v) / 1e3:.0f} KB | {pctl(v, 0.99) / 1e3:.0f} KB "
            f"| {sum(v) / 1e6:.1f} MB | {per_h:.1f} MB |"
        )
    w("")
    w("## Skips and failures by taxonomy code (observed)")
    w("")
    for code, n in Counter(r.get("code", "?") for r in skips).most_common():
        w(f"- skip `{code}`: {n}")
    if fails:
        for err, n in Counter(r.get("error", "?")[:60] for r in fails).most_common():
            w(f"- FAIL `{err}`: {n}")
    else:
        w("- transient failures surviving retries: **0**")
    w("")
    w(f"## Corpus extrapolation (pilot per-article means x {C:,} Featured Articles)")
    w("")
    w(
        f"- sections/article: **{mean_secs_art:.1f}** generatable "
        f"-> **~{C * mean_secs_art / 1000:.0f}k** corpus sections"
    )
    w(
        f"- audio/article: **{mean_audio_art_s / 60:.1f} min** "
        f"-> **~{corpus_audio_h:,.0f} corpus audio-hours**"
    )
    w(
        f"- storage: Opus **~{corpus_opus_gb:.0f} GB**, "
        f"MP3 ~{corpus_mp3_gb:.0f} GB, sidecars "
        f"~{(corpus_audio_h * 3600 * (sum(sizes.get('captions_vtt', [0])) + sum(sizes.get('timestamps_json', [0]))) / total_audio_s) / 1e9:.0f} GB"
    )
    w(
        f"- full-alignment corpus drain: **~{corpus_wall_pod_days:.0f} pod-days** "
        f"of isvc time (audio-only tier: derived by the isvc RTF ratio, "
        f"~{corpus_wall_pod_days * 0.22 / 0.27:.0f} pod-days)"
    )
    w("")
    w("## maxReplicas: corpus drain time vs isvc replica count")
    w("")
    w("| replicas | full-alignment drain | audio-only drain (derived) |")
    w("| --- | --- | --- |")
    for n in (1, 2, 4, 6, 8):
        w(
            f"| {n} | {corpus_wall_pod_days / n:.0f} days | "
            f"{corpus_wall_pod_days * 0.22 / 0.27 / n:.0f} days |"
        )
    w("")
    w(
        "Recommendation slot: pick the replica count whose drain time fits the "
        "backfill window DE proposes, bounded by cluster capacity; edits-driven "
        "steady state needs ~1 replica (<<1 write/s)."
    )
    w("")
    w("## Memory envelope (Phase 3 finding #3 closure)")
    if args.memory_peak_gib:
        w(
            f"- generator `memory.peak` across the pilot (external driver, clean "
            f"measurement): **{args.memory_peak_gib:.2f} GiB** -> recommended "
            f"permanent limit {max(2, round(args.memory_peak_gib * 1.5)):.0f} Gi"
        )
    else:
        w(
            "- [fill] generator memory.peak: read /sys/fs/cgroup/memory.peak in "
            "the pod after the run; recommended limit = peak x 1.5, rounded up."
        )
    w("")
    w(
        "_Every figure above is measured from the pilot except rows labeled "
        "extrapolation/derived. Reproduce: pilot_run.py --scan corpus_scan.json "
        "--seed 43; the population and revisions are pinned by the committed "
        "scan snapshot._"
    )
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

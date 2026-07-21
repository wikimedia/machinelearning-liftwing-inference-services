#!/usr/bin/env python3
"""Phase 4 pilot driver (T432692): deliberately a script, not a pipeline.

Drives the TTS Section Generator over ~50 Featured Articles and records
every outcome. This is the measured-numbers-pack producer and a reference
client for the generator's contract; it is NOT a component. It has no
queue, no scheduler, no index, and no event consumption: those are the DE
pipeline's job. The one convenience it allows itself is idempotent resume
(skip work its own results log already records), which is a restart
convenience, not orchestration.

Population: drawn from the committed corpus scan snapshot (pinned
page_id/rev_id per article), NOT re-derived from the live Featured
Article category. The Phase 3 rescan demonstrated that a seed alone does
not reproduce a sample when the category changes underneath it; pinning
the population file and the revisions makes this run reproducible
bit-for-bit regardless of edits or promotions.

Concurrency: bounded, default 1, sized to the isvc replica count so
requests never queue at the inference service (activator queue time
counts against the Knative revision timeout).

Retries: transient 5xx only, bounded, with linear backoff; deterministic
4xx outcomes are recorded and never retried (the taxonomy contract).

Usage (from deploy2003, against staging through the mesh ingress):

    python3 pilot_run.py \\
        --scan corpus_scan.json --articles 50 --seed 43 \\
        --base https://tts-section-generator.k8s-ml-staging.discovery.wmnet:31443 \\
        --out ./pilot-artifacts --log ./pilot_results.jsonl

Then: python3 pilot_summary.py --log ./pilot_results.jsonl --scan corpus_scan.json
"""

import argparse
import base64
import concurrent.futures
import datetime
import json
import random
import sys
import threading
import time
from pathlib import Path

import requests

ARTIFACTS = ["audio_opus", "audio_mp3", "captions_vtt", "timestamps_json"]
_EXT = {
    "audio_opus": "opus",
    "audio_mp3": "mp3",
    "captions_vtt": "vtt",
    "timestamps_json": "json",
}

TRANSIENT_RETRIES = 2
BACKOFF_S = 10.0

_log_lock = threading.Lock()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _append(log_path: Path, record: dict) -> None:
    with _log_lock:
        with log_path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _done_keys(log_path: Path) -> set[str]:
    """Keys already settled (ok or deterministic skip). Transient failures
    are NOT settled: a re-run retries them."""
    done = set()
    if not log_path.exists():
        return done
    with log_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") in ("ok", "skip"):
                done.add(r["key"])
    return done


def pick_articles(scan: dict, n: int, seed: int) -> list[dict]:
    """Draw n articles from the scan snapshot: one entry per article with
    its pinned (page_id, rev_id). Deterministic for a fixed scan file."""
    by_title: dict[str, dict] = {}
    for s in scan["sections"]:
        by_title.setdefault(
            s["title"],
            {"title": s["title"], "page_id": s["page_id"], "rev_id": s["rev_id"]},
        )
    titles = sorted(by_title)
    rng = random.Random(seed)
    chosen = rng.sample(titles, min(n, len(titles)))
    return [by_title[t] for t in sorted(chosen)]


def fetch_sections(base: str, art: dict, session: requests.Session) -> dict:
    r = session.get(
        f"{base}/sections",
        params={
            "wiki_id": "enwiki",
            "page_id": art["page_id"],
            "rev_id": art["rev_id"],
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def generate_one(
    base: str,
    art: dict,
    section: dict,
    out_root: Path,
    log_path: Path,
    session: requests.Session,
) -> str:
    """Generate one section; write artifacts; append exactly one record.
    Returns the record status."""
    key = f"enwiki/{art['page_id']}/{art['rev_id']}/{section['section_id']}"
    payload = {
        "wiki_id": "enwiki",
        "page_id": art["page_id"],
        "rev_id": art["rev_id"],
        "section_id": section["section_id"],
        "generation_config": {"artifacts": ARTIFACTS},
    }
    record = {
        "ts": _now(),
        "key": key,
        "title": art["title"],
        "page_id": art["page_id"],
        "rev_id": art["rev_id"],
        "section_id": section["section_id"],
        "char_count": section.get("char_count"),
        "attempts": 0,
    }

    last_err = None
    for attempt in range(1 + TRANSIENT_RETRIES):
        if attempt:
            time.sleep(BACKOFF_S * attempt)
        record["attempts"] = attempt + 1
        t0 = time.perf_counter()
        try:
            r = session.post(f"{base}/generate-section", json=payload, timeout=900)
        except requests.RequestException as e:
            last_err = f"transport: {e}"
            continue
        wall = time.perf_counter() - t0

        if r.status_code == 200:
            body = r.json()
            arts = {a["artifact_type"]: a for a in body["artifacts"]}
            sizes = {}
            for kind, a in arts.items():
                if "bytes_b64" in a:
                    data = base64.b64decode(a["bytes_b64"])
                elif kind == "timestamps_json":
                    data = json.dumps(a["timestamps"]).encode()
                else:
                    continue
                sizes[kind] = len(data)
                path = out_root / key
                path.parent.mkdir(parents=True, exist_ok=True)
                (path.parent / f"{section['section_id']}.{_EXT[kind]}").write_bytes(
                    data
                )
            any_art = body["artifacts"][0]
            record.update(
                status="ok",
                wall_s=round(wall, 2),
                duration_ms=any_art["duration_ms"],
                segment_count=body["segment_count"],
                generation_version=any_art["generation_version"],
                content_sha256=any_art["content_sha256"],
                sizes=sizes,
            )
            _append(log_path, record)
            return "ok"

        try:
            code = r.json().get("code", "unknown")
        except ValueError:
            code = "unknown"

        if 400 <= r.status_code < 500:
            # Deterministic: record once, never retry (taxonomy contract).
            record.update(
                status="skip",
                http_status=r.status_code,
                code=code,
                wall_s=round(wall, 2),
            )
            _append(log_path, record)
            return "skip"

        last_err = f"{r.status_code} {code}"

    record.update(status="fail", error=str(last_err))
    _append(log_path, record)
    return "fail"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scan",
        default="corpus_scan.json",
        help="committed corpus scan snapshot (the pinned population)",
    )
    ap.add_argument("--articles", type=int, default=50)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--out", default="./pilot-artifacts")
    ap.add_argument("--log", default="./pilot_results.jsonl")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="size to the isvc replica count; requests must not "
        "queue at the inference service",
    )
    args = ap.parse_args()

    scan = json.loads(Path(args.scan).read_text())
    articles = pick_articles(scan, args.articles, args.seed)
    out_root, log_path = Path(args.out), Path(args.log)
    out_root.mkdir(parents=True, exist_ok=True)
    done = _done_keys(log_path)
    session = requests.Session()

    print(
        f"Pilot: {len(articles)} articles from {args.scan} "
        f"(seed {args.seed}, pinned revisions); {len(done)} sections "
        f"already settled in {log_path}; concurrency {args.concurrency}"
    )

    # Enumerate first (serial, fast), then generate (bounded pool).
    tasks = []
    for art in articles:
        try:
            secs = fetch_sections(args.base, art, session)
        except requests.RequestException as e:
            _append(
                log_path,
                {
                    "ts": _now(),
                    "status": "fail",
                    "key": f"enwiki/{art['page_id']}/{art['rev_id']}/-",
                    "title": art["title"],
                    "error": f"sections: {e}",
                    "attempts": 1,
                },
            )
            print(f"  ENUM FAIL {art['title']}: {e}")
            continue
        for s in secs["sections"]:
            key = f"enwiki/{art['page_id']}/{art['rev_id']}/{s['section_id']}"
            if key in done:
                continue
            if not s["generatable"]:
                # Enumeration-level deterministic skip, recorded once.
                _append(
                    log_path,
                    {
                        "ts": _now(),
                        "key": key,
                        "title": art["title"],
                        "page_id": art["page_id"],
                        "rev_id": art["rev_id"],
                        "section_id": s["section_id"],
                        "status": "skip",
                        "code": s.get("skip_reason", "not_generatable"),
                        "char_count": s.get("char_count"),
                        "attempts": 0,
                    },
                )
                done.add(key)
                continue
            tasks.append((art, s))

    print(f"  {len(tasks)} sections to generate\n")
    t_start = time.perf_counter()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(generate_one, args.base, art, s, out_root, log_path, session): (
                art,
                s,
            )
            for art, s in tasks
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            art, s = futs[fut]
            status = fut.result()
            counts[status] += 1
            if i % 10 == 0 or status != "ok":
                elapsed = time.perf_counter() - t_start
                print(
                    f"  [{i}/{len(tasks)}] {status:4s} {art['title']} :: "
                    f"{s['section_id']}  ({elapsed / 60:.1f} min elapsed)"
                )

    print(
        f"\nDone: {counts['ok']} ok, {counts['skip']} skip, "
        f"{counts['fail']} fail in {(time.perf_counter() - t_start) / 60:.1f} min"
    )
    print(f"Results: {log_path}; artifacts: {out_root}")
    print(
        "Now read the generator pod's memory.peak (Phase 3 finding #3) and "
        "run pilot_summary.py."
    )
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

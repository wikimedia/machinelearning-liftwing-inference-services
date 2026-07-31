#!/usr/bin/env python3
"""One-shot batch generation for the TTS v1 experiment (T433594 workstream B).

Evolves the Phase 4 pilot driver (scripts/pilot_run.py, which stays frozen
as the committed pilot record's reproduction path) into the experiment's
run tool. Inherited unchanged from the pilot: pinned dataset with pinned
revisions, bounded concurrency sized to the isvc replica count, bounded
transient retries with linear backoff, deterministic-4xx
recorded-never-retried, idempotent resume from its own results log,
dead-letter = fail records in the log. Still deliberately a script, not a
pipeline: no queue, no scheduler, no event consumption.

New over the pilot:

* Artifacts pinned to the v1 experiment delivery set: MP3 + WebVTT (the
  service default leads with Opus; an unpinned driver would generate the
  wrong codec corpus-wide).
* REQUIRES a writing sink on the generator (s3 in the real run, file for
  local smoke): responses must carry blob_uri, and inline bytes_b64 is a
  hard error. A batch whose artifacts evaporate is not a batch.
* Per-article manifest writer: after every generatable section of an
  article settles ok (deterministic skips do not block completeness; any
  fail does), writes {wiki}/{page}/{rev}/manifest.json. Manifest presence
  is the "article fully generated" signal and the app's revision-match
  check (the v1 experiment architecture). Manifests carry relative object
  KEYS, not URLs: the serving domain is Traffic's design. Articles with a
  fail get no manifest and stay on the dead-letter list.
* Mixed-version guard: if an article's sections were generated under more
  than one generation_version (a mid-run redeploy), it gets NO manifest
  and is dead-lettered for regeneration: a manifest must describe one
  coherent generation.

Usage (from a deploy host, generator reachable with an s3 sink):

    python3 batch_generate.py \\
        --dataset articles.json \\
        --base https://tts-section-generator.discovery.wmnet:31443 \\
        --log ./batch_results.jsonl --concurrency 4

    # articles.json: [{"title": ..., "page_id": ..., "rev_id": ...}, ...]
    # Pin it FIRST if the product list arrives as titles:
    python3 batch_generate.py --resolve titles.txt --dataset articles.json

Manifest destination: --manifest-dir DIR (local, for smoke tests) or S3
via TTS_GEN_S3_ENDPOINT / TTS_GEN_S3_BUCKET + AWS env credentials (the
same pattern as the generator's sink; boto3 path-style).
"""

import argparse
import concurrent.futures
import datetime
import json
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import requests

SCHEMA_VERSION = 1
ARTIFACTS = ["audio_mp3", "captions_vtt"]  # the Apps codec decision; do not widen
ART_FIELD = {"audio_mp3": "audio", "captions_vtt": "captions"}

TRANSIENT_RETRIES = 2
BACKOFF_S = 10.0

_log_lock = threading.Lock()
_article_lock = threading.Lock()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _append(log_path: Path, record: dict) -> None:
    with _log_lock:
        with log_path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    with log_path.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _done_keys(records: list[dict]) -> set[str]:
    """Keys already settled (ok or deterministic skip). Transient failures
    are NOT settled: a re-run retries them."""
    return {r["key"] for r in records if r.get("status") in ("ok", "skip")}


def _key_from_uri(blob_uri: str) -> str:
    """Relative object key from a sink blob_uri.

    s3://bucket/enwiki/9/1/lead.mp3 -> enwiki/9/1/lead.mp3
    file:///root/enwiki/9/1/lead.mp3 -> enwiki/9/1/lead.mp3 (last 4 parts:
    the canonical {wiki}/{page}/{rev}/{section}.{ext} layout).
    """
    u = urllib.parse.urlparse(blob_uri)
    if u.scheme == "s3":
        return u.path.lstrip("/")
    return "/".join(u.path.split("/")[-4:])


# ── Manifest sinks ──────────────────────────────────────────────────────────


class DirManifestSink:
    """Write manifests under a local directory (smoke tests, file-sink runs)."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, body: bytes) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(body)
        tmp.rename(path)
        return path.as_uri()


class S3ManifestSink:
    """Write manifests to the artifact bucket: same endpoint/credential
    pattern as the generator's sink (path-style, AWS env creds), same
    fail-at-startup probe."""

    def __init__(self, endpoint: str, bucket: str):
        import boto3
        from botocore.config import Config as BotoConfig

        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name="us-east-1",
            config=BotoConfig(
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        self._client.head_bucket(Bucket=bucket)  # fail the run, not article #1

    def put(self, key: str, body: bytes) -> str:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"


# ── Dataset ─────────────────────────────────────────────────────────────────


def resolve_titles(titles_path: Path, base_api: str, out_path: Path) -> None:
    """Pin a titles-only product list to exact (page_id, rev_id) NOW, into a
    committed dataset file; the run consumes only the pinned file (the
    pilot's reproducibility lesson: a list alone does not reproduce a run)."""
    session = requests.Session()
    out = []
    for title in [t.strip() for t in titles_path.read_text().splitlines() if t.strip()]:
        enc = urllib.parse.quote(title.replace(" ", "_"), safe="")
        bare = session.get(f"{base_api}/page/{enc}/bare", timeout=30).json()
        out.append(
            {"title": title, "page_id": bare["id"], "rev_id": bare["latest"]["id"]}
        )
        print(f"  pinned {title}: page {bare['id']} rev {bare['latest']['id']}")
    out_path.write_text(json.dumps(out, indent=1))
    print(f"Wrote pinned dataset: {out_path} ({len(out)} articles)")


# ── Generation ──────────────────────────────────────────────────────────────


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
    doc_index: int,
    log_path: Path,
    session: requests.Session,
) -> dict:
    """Generate one section; append exactly one record; return it.

    The record carries everything the manifest needs (title, level,
    duration, hashes, artifact keys), so manifests are rebuildable from the
    log alone: that is what makes resume and manifest writing idempotent.
    """
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
        "section_title": section.get("title"),
        "level": section.get("level"),
        "doc_index": doc_index,
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
            missing_uri = [k for k, a in arts.items() if "blob_uri" not in a]
            if missing_uri:
                # Inline sink on the generator: the batch's artifacts would
                # evaporate. Hard, non-retryable operator error.
                record.update(
                    status="fail",
                    error=f"generator sink is inline (no blob_uri on "
                    f"{missing_uri}); configure a writing sink",
                )
                _append(log_path, record)
                return record
            any_art = body["artifacts"][0]
            record.update(
                status="ok",
                wall_s=round(wall, 2),
                duration_ms=any_art["duration_ms"],
                segment_count=body["segment_count"],
                generation_version=any_art["generation_version"],
                content_sha256=any_art["content_sha256"],
                **(
                    {"render_id": any_art["render_id"]}
                    if "render_id" in any_art
                    else {}
                ),
                artifacts={
                    k: {
                        "key": _key_from_uri(a["blob_uri"]),
                        "media_type": a["media_type"],
                        "size_bytes": a.get("size_bytes"),
                    }
                    for k, a in arts.items()
                },
            )
            _append(log_path, record)
            return record

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
            return record

        last_err = f"{r.status_code} {code}"

    record.update(status="fail", error=str(last_err))
    _append(log_path, record)
    return record


# ── Manifest ────────────────────────────────────────────────────────────────


def build_manifest(art: dict, enum: dict, ok_records: list[dict]) -> dict | None:
    """Assemble one article's manifest from its ok section records, or None
    with a reason printed if the article does not qualify."""
    versions = {r["generation_version"] for r in ok_records}
    if len(versions) > 1:
        print(
            f"  NO MANIFEST {art['title']}: mixed generation_versions "
            f"{sorted(versions)} (mid-run redeploy?); dead-lettered"
        )
        return None
    sections = []
    for r in sorted(ok_records, key=lambda r: r["doc_index"]):
        sections.append(
            {
                "section_id": r["section_id"],
                "title": r["section_title"],
                "level": r["level"],
                "duration_ms": r["duration_ms"],
                "content_sha256": r["content_sha256"],
                **{
                    ART_FIELD[k]: {"key": v["key"], "media_type": v["media_type"]}
                    for k, v in r["artifacts"].items()
                },
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "wiki_id": "enwiki",
        "page_id": art["page_id"],
        "rev_id": art["rev_id"],
        "generation_version": versions.pop(),
        "generated_at": _now(),
        "sections": sections,
    }
    rid = enum.get("render_id") or next(
        (r["render_id"] for r in ok_records if "render_id" in r), None
    )
    if rid:
        manifest["render_id"] = rid
    return manifest


def settle_article(
    art: dict, enum: dict, records_by_key: dict, manifest_sink, log_path: Path
) -> str:
    """Evaluate one article's completeness; write its manifest if earned.
    Returns 'manifest' | 'incomplete' | 'no_sections'."""
    gen_ids = [s["section_id"] for s in enum["sections"] if s["generatable"]]
    if not gen_ids:
        return "no_sections"
    recs = []
    for sid in gen_ids:
        r = records_by_key.get(f"enwiki/{art['page_id']}/{art['rev_id']}/{sid}")
        if r is None or r["status"] == "fail":
            return "incomplete"  # dead letter: fail records are in the log
        if r["status"] == "ok":
            recs.append(r)
        # status == "skip": deterministic, does not block, not in manifest
    if not recs:
        return "no_sections"  # every generatable section skipped at POST time
    manifest = build_manifest(art, enum, recs)
    if manifest is None:
        return "incomplete"
    key = f"enwiki/{art['page_id']}/{art['rev_id']}/manifest.json"
    uri = manifest_sink.put(key, json.dumps(manifest, indent=1).encode())
    _append(
        log_path,
        {
            "ts": _now(),
            "status": "manifest",
            "key": key,
            "title": art["title"],
            "uri": uri,
            "sections": len(manifest["sections"]),
        },
    )
    return "manifest"


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="articles.json",
        help="pinned article list: [{title, page_id, rev_id}]",
    )
    ap.add_argument(
        "--resolve",
        metavar="TITLES_TXT",
        help="pin a titles-only list into --dataset, then exit",
    )
    ap.add_argument(
        "--resolve-api",
        default="https://en.wikipedia.org/w/rest.php/v1",
        help="REST base used only by --resolve",
    )
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--log", default="./batch_results.jsonl")
    ap.add_argument(
        "--concurrency", type=int, default=1, help="size to the isvc replica count"
    )
    ap.add_argument(
        "--manifest-dir", help="write manifests under a local dir (smoke tests)"
    )
    ap.add_argument(
        "--s3-endpoint",
        default=None,
        help="manifest S3 endpoint (default: TTS_GEN_S3_ENDPOINT)",
    )
    ap.add_argument(
        "--s3-bucket",
        default=None,
        help="manifest S3 bucket (default: TTS_GEN_S3_BUCKET)",
    )
    args = ap.parse_args()

    if args.resolve:
        resolve_titles(Path(args.resolve), args.resolve_api, Path(args.dataset))
        return 0

    import os

    if args.manifest_dir:
        manifest_sink = DirManifestSink(args.manifest_dir)
    else:
        endpoint = args.s3_endpoint or os.environ.get("TTS_GEN_S3_ENDPOINT", "")
        bucket = args.s3_bucket or os.environ.get("TTS_GEN_S3_BUCKET", "")
        if not endpoint or not bucket:
            print(
                "Manifest destination unconfigured: pass --manifest-dir, or "
                "--s3-endpoint/--s3-bucket (or TTS_GEN_S3_ENDPOINT/_BUCKET "
                "env + AWS credentials).",
                file=sys.stderr,
            )
            return 2
        manifest_sink = S3ManifestSink(endpoint, bucket)

    articles = json.loads(Path(args.dataset).read_text())
    log_path = Path(args.log)
    prior = _read_log(log_path)
    done = _done_keys(prior)
    records_by_key = {r["key"]: r for r in prior if r.get("status") in ("ok", "skip")}
    session = requests.Session()

    print(
        f"Batch: {len(articles)} articles from {args.dataset} (pinned "
        f"revisions); {len(done)} sections already settled in {log_path}; "
        f"concurrency {args.concurrency}; artifacts {ARTIFACTS}"
    )

    # Enumerate first (serial, fast), then generate (bounded pool).
    tasks, enums = [], {}
    for art in articles:
        akey = (art["page_id"], art["rev_id"])
        try:
            enum = fetch_sections(args.base, art, session)
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
        enums[akey] = enum
        for i, s in enumerate(enum["sections"]):
            key = f"enwiki/{art['page_id']}/{art['rev_id']}/{s['section_id']}"
            if key in done:
                continue
            if not s["generatable"]:
                rec = {
                    "ts": _now(),
                    "key": key,
                    "title": art["title"],
                    "page_id": art["page_id"],
                    "rev_id": art["rev_id"],
                    "section_id": s["section_id"],
                    "status": "skip",
                    "code": s.get("skip_reason", "not_generatable"),
                    "char_count": s.get("char_count"),
                    "doc_index": i,
                    "attempts": 0,
                }
                _append(log_path, rec)
                records_by_key[key] = rec
                done.add(key)
                continue
            tasks.append((art, s, i))

    print(f"  {len(tasks)} sections to generate\n")
    t_start = time.perf_counter()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(generate_one, args.base, art, s, i, log_path, session): (art, s)
            for art, s, i in tasks
        }
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            art, s = futs[fut]
            rec = fut.result()
            counts[rec["status"]] += 1
            with _article_lock:
                records_by_key[rec["key"]] = rec
            if n % 10 == 0 or rec["status"] != "ok":
                elapsed = time.perf_counter() - t_start
                print(
                    f"  [{n}/{len(tasks)}] {rec['status']:4s} {art['title']} "
                    f":: {s['section_id']}  ({elapsed / 60:.1f} min elapsed)"
                )

    # Settle every enumerated article: completeness rule + manifest write.
    # Runs on EVERY invocation over the full log, which is what makes both
    # resume and manifest writing idempotent.
    outcomes = {"manifest": 0, "incomplete": 0, "no_sections": 0}
    for art in articles:
        enum = enums.get((art["page_id"], art["rev_id"]))
        if enum is None:
            outcomes["incomplete"] += 1
            continue
        outcomes[
            settle_article(art, enum, records_by_key, manifest_sink, log_path)
        ] += 1

    print(
        f"\nDone: {counts['ok']} ok, {counts['skip']} skip, "
        f"{counts['fail']} fail in {(time.perf_counter() - t_start) / 60:.1f} min"
    )
    print(
        f"Articles: {outcomes['manifest']} manifests written, "
        f"{outcomes['incomplete']} incomplete (dead letter: fail records "
        f"in {log_path}), {outcomes['no_sections']} with no generatable "
        f"sections"
    )
    return 0 if outcomes["incomplete"] == 0 and counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

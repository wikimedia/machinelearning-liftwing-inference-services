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
* REQUIRES that the bytes land somewhere. Either the generator writes
  them (its s3/file sink, response carries blob_uri) or this script
  writes them (--artifact-dir, response carries bytes_b64). Exactly one
  of the two must be configured: a batch whose artifacts evaporate is
  not a batch, so an inline response with no --artifact-dir is a hard,
  non-retryable error, and so is a missing payload of either kind.
* Writes index.json alongside the manifests: one entry per article that
  has audio, with its revision and artifact paths. Consumers fetch it
  once instead of probing a manifest per article (--no-index to skip).
* --sections limits generation to named sections (e.g. "lead" for the
  v1 experiment's lead-only corpus). The completeness rule and the
  manifest follow the same selection: a manifest means "every REQUESTED
  generatable section of this article is present", and records its
  scope so the reader knows which.
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

Manifest destination: --manifest-dir DIR (local; also the analytics
published tree) or S3 via TTS_GEN_S3_ENDPOINT / TTS_GEN_S3_BUCKET + AWS
env credentials (the same pattern as the generator's sink; boto3
path-style).

Analytics published-tree run (T436758), venv on a stat host:

    python3 batch_generate.py \\
        --dataset articles.json --sections lead \\
        --base https://tts-section-generator.discovery.wmnet:31443 \\
        --artifact-dir /srv/published/wmf-ml-models/tts/experiment-v1 \\
        --manifest-dir /srv/published/wmf-ml-models/tts/experiment-v1 \\
        --log ~/tts/batch_results.jsonl --concurrency 4

Keep the log and dataset OUTSIDE the published tree: everything under it
is served publicly.
"""

import argparse
import base64
import concurrent.futures
import datetime
import json
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import requests

SCHEMA_VERSION = 2  # 2 adds "scope" (which sections a manifest covers)
# The index is its own contract, versioned separately from the manifests.
INDEX_SCHEMA_VERSION = 1
INDEX_KEY = "index.json"

# Published-tree modes. Files inherit the process umask otherwise, and a
# group-only-readable file under /srv/published is invisible to Apache:
# the failure mode is a 403 in the app, not an error in this script.
FILE_MODE = 0o644
DIR_MODE = 0o755
ARTIFACTS = ["audio_mp3", "captions_vtt"]  # the Apps codec decision; do not widen
ART_FIELD = {"audio_mp3": "audio", "captions_vtt": "captions"}
# Mirrors tts_generator.sinks._EXT: the tree this script writes must be
# byte-for-byte the layout the generator's own file sink would produce.
_ARTIFACT_EXT = {"audio_mp3": "mp3", "captions_vtt": "vtt"}

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


def _write_public(path: Path, data: bytes) -> None:
    """Write atomically with world-readable modes.

    Dot-prefixed temp name: a partially written file under the published
    tree must never be fetchable, and ".lead.mp3.tmp" is both hidden and
    outside the {section}.{ext} naming the app composes.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.chmod(FILE_MODE)
    tmp.rename(path)  # atomic publish within one filesystem


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


class ArtifactWriter:
    """Write inline artifact bytes into a local tree (the analytics
    published directory, or any local root).

    Used when the generator cannot reach the destination filesystem, which
    is the case for the published tree: the deployed service returns
    bytes_b64 and this script is the writer. Keys are the canonical
    {wiki}/{page}/{rev}/{section}.{ext} layout, so the tree is identical
    to what the generator's own file sink would produce.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)

    def write(self, key: str, data: bytes) -> str:
        path = self.root / key
        _write_public(path, data)
        return path.as_uri()


class DirManifestSink:
    """Write manifests under a local directory (smoke tests, file-sink runs,
    and the analytics published tree)."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)

    def put(self, key: str, body: bytes) -> str:
        path = self.root / key
        _write_public(path, body)
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
    artifact_writer: "ArtifactWriter | None" = None,
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
            # The bytes must land somewhere: either the generator already
            # wrote them (blob_uri) or we write them here (bytes_b64 +
            # --artifact-dir). Anything else is a hard, non-retryable
            # operator error: artifacts would evaporate.
            unplaceable = [
                k
                for k, a in arts.items()
                if "blob_uri" not in a
                and not (artifact_writer is not None and "bytes_b64" in a)
            ]
            if unplaceable:
                record.update(
                    status="fail",
                    error=(
                        f"nowhere to put artifacts {unplaceable}: the "
                        "response carries neither blob_uri (generator sink) "
                        "nor bytes_b64 with --artifact-dir configured"
                    ),
                )
                _append(log_path, record)
                return record

            written: dict[str, str] = {}
            if artifact_writer is not None:
                try:
                    for k, a in arts.items():
                        if "blob_uri" in a:
                            continue  # generator already placed it
                        akey = f"{key}.{_ARTIFACT_EXT[k]}"
                        artifact_writer.write(akey, base64.b64decode(a["bytes_b64"]))
                        written[k] = akey
                except (OSError, ValueError) as e:
                    # Disk full, permissions, corrupt payload: transient
                    # from the batch's point of view (fix and re-run;
                    # resume skips what already settled).
                    record.update(status="fail", error=f"artifact write failed: {e}")
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
                        "key": (
                            written[k] if k in written else _key_from_uri(a["blob_uri"])
                        ),
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


def build_manifest(
    art: dict, enum: dict, ok_records: list[dict], scope: str = "all"
) -> dict | None:
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
        # Which sections this manifest claims to cover. "all" = every
        # generatable section of the article; "lead" = the lead only (the
        # v1 experiment). Readers must not assume completeness beyond it.
        "scope": scope,
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


def build_index(entries: list[dict], scope: str) -> dict:
    """One file listing every article that has audio, for consumers.

    The read path is static files, so without this a client has to fetch a
    manifest per article just to learn whether audio exists. With the index
    cached it can answer that offline: look up the page id, compare the
    revision it is displaying, and use the paths given here.

    Paths are relative to the index's own location, like the manifests', so
    the serving domain stays the CDN's concern and the storage layout stays
    ours to change: clients must not compose paths by convention.

    Entries mirror the manifest structure minus its integrity fields, so a
    client learns one shape. Sorted by page_id: stable diffs between runs,
    and a client can binary-search without building a map.
    """
    versions = sorted({e.pop("_generation_version") for e in entries})
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "scope": scope,
        "wiki_id": "enwiki",
        "generated_at": _now(),
        # A list: a run interrupted across a redeploy can legitimately span
        # more than one generation version, and hiding that would mislead.
        "generation_versions": versions,
        "count": len(entries),
        "articles": sorted(entries, key=lambda e: e["page_id"]),
    }


def _index_entry(art: dict, manifest: dict, manifest_key: str) -> dict:
    """Index entry for one article, from the manifest just written."""
    return {
        "title": art["title"],
        "page_id": manifest["page_id"],
        "rev_id": manifest["rev_id"],
        "duration_ms": round(sum(s["duration_ms"] for s in manifest["sections"]), 1),
        "manifest": manifest_key,
        "sections": [
            {
                "section_id": s["section_id"],
                "title": s["title"],
                "duration_ms": s["duration_ms"],
                **{
                    field: s[field]["key"] for field in ART_FIELD.values() if field in s
                },
            }
            for s in manifest["sections"]
        ],
        # Popped by build_index: used to report which versions the corpus
        # spans, not part of the per-article contract.
        "_generation_version": manifest["generation_version"],
    }


def settle_article(
    art: dict,
    enum: dict,
    records_by_key: dict,
    manifest_sink,
    log_path: Path,
    wanted: "set[str] | None" = None,
    scope: str = "all",
    index_entries: "list[dict] | None" = None,
) -> str:
    """Evaluate one article's completeness; write its manifest if earned.
    Returns 'manifest' | 'incomplete' | 'no_sections'.

    When ``index_entries`` is given, an entry is appended for each article
    that earns a manifest, so the index lists exactly the articles a
    consumer can play: incomplete and dead-lettered articles are absent.

    Completeness is scoped to the REQUESTED sections: under --sections
    lead, an article is complete when its lead is present, and the other
    sections' absence is by design rather than a dead letter.
    """
    gen_ids = [
        s["section_id"]
        for s in enum["sections"]
        if s["generatable"] and (wanted is None or s["section_id"] in wanted)
    ]
    if wanted is not None:
        # A requested section the article does not have at all: main() has
        # already written the fail record; refuse the manifest here too.
        present = {s["section_id"] for s in enum["sections"]}
        if wanted - present:
            return "incomplete"
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
    manifest = build_manifest(art, enum, recs, scope=scope)
    if manifest is None:
        return "incomplete"
    key = f"enwiki/{art['page_id']}/{art['rev_id']}/manifest.json"
    uri = manifest_sink.put(key, json.dumps(manifest, indent=1).encode())
    if index_entries is not None:
        index_entries.append(_index_entry(art, manifest, key))
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
        "--sections",
        default="",
        help='comma-separated section ids to generate (e.g. "lead"); '
        "default: every generatable section. The completeness rule and "
        "the manifest scope follow this selection.",
    )
    ap.add_argument(
        "--artifact-dir",
        help="write artifact bytes under this local root (required when the "
        "generator runs with an inline sink, e.g. the analytics published "
        "tree run from a stat host)",
    )
    ap.add_argument(
        "--no-index",
        action="store_true",
        help=f"skip writing {INDEX_KEY} (the per-article listing consumers "
        "read to find what has audio). The index is written at the end of a "
        "completed run, never mid-run: an interrupted run leaves the previous "
        "index in place rather than advertising a half-generated corpus.",
    )
    ap.add_argument(
        "--manifest-dir",
        help="write manifests under a local dir (smoke tests, published tree)",
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

    wanted = {s.strip() for s in args.sections.split(",") if s.strip()} or None
    scope = ",".join(sorted(wanted)) if wanted else "all"
    artifact_writer = ArtifactWriter(args.artifact_dir) if args.artifact_dir else None

    articles = json.loads(Path(args.dataset).read_text())
    log_path = Path(args.log)
    prior = _read_log(log_path)
    done = _done_keys(prior)
    records_by_key = {r["key"]: r for r in prior if r.get("status") in ("ok", "skip")}
    session = requests.Session()

    print(
        f"Batch: {len(articles)} articles from {args.dataset} (pinned "
        f"revisions); {len(done)} sections already settled in {log_path}; "
        f"concurrency {args.concurrency}; artifacts {ARTIFACTS}; "
        f"scope {scope}"
        + (f"; writing artifacts to {args.artifact_dir}" if artifact_writer else "")
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
        if wanted is not None:
            missing = wanted - {s["section_id"] for s in enum["sections"]}
            if missing:
                # Requested section absent from this revision. Recorded
                # loudly: silently generating nothing would leave the
                # corpus quietly short by however many articles.
                _append(
                    log_path,
                    {
                        "ts": _now(),
                        "status": "fail",
                        "key": f"enwiki/{art['page_id']}/{art['rev_id']}/"
                        f"{sorted(missing)[0]}",
                        "title": art["title"],
                        "error": f"requested section(s) {sorted(missing)} not "
                        f"present in this revision",
                        "attempts": 1,
                    },
                )
                print(f"  MISSING SECTION {art['title']}: {sorted(missing)}")
        for i, s in enumerate(enum["sections"]):
            if wanted is not None and s["section_id"] not in wanted:
                continue
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
            ex.submit(
                generate_one,
                args.base,
                art,
                s,
                i,
                log_path,
                session,
                artifact_writer,
            ): (art, s)
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
    index_entries: list[dict] = []
    for art in articles:
        enum = enums.get((art["page_id"], art["rev_id"]))
        if enum is None:
            outcomes["incomplete"] += 1
            continue
        outcomes[
            settle_article(
                art,
                enum,
                records_by_key,
                manifest_sink,
                log_path,
                wanted=wanted,
                scope=scope,
                index_entries=index_entries,
            )
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

    # The index goes last, after every manifest of this run has settled, and
    # only if there is something to list. A resumed run re-settles the whole
    # dataset from the log, so the index it writes covers earlier runs too.
    if not args.no_index and index_entries:
        index = build_index(index_entries, scope)
        uri = manifest_sink.put(
            INDEX_KEY, json.dumps(index, indent=1, ensure_ascii=False).encode()
        )
        print(f"Index: {index['count']} articles listed in {uri}")
    elif not args.no_index:
        print(f"Index: no articles to list, {INDEX_KEY} left unchanged")
    return 0 if outcomes["incomplete"] == 0 and counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

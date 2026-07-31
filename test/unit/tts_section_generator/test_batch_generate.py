"""Batch script tests against a stub generator (no network, no isvc).

The stub speaks just enough of the generator contract: /sections
enumeration with render_id and a mix of generatable/skip sections, and
/generate-section returning blob_uri artifacts (s3-shaped), a scripted
deterministic 4xx, a scripted persistent 5xx, and a scripted flip to a
second generation_version. Manifests land in a temp dir via
--manifest-dir (the S3 manifest sink is the same put() behind boto3,
MinIO-verified separately)."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).parent
# Script location: alongside the tests (sandbox) or in the service's
# scripts/ dir (repo layout: test/unit/tts_section_generator/ ->
# src/models/tts_section_generator/scripts/).
_CANDIDATES = [
    HERE / "batch_generate.py",
    HERE / "../../../src/models/tts_section_generator/scripts/batch_generate.py",
]
SCRIPT = next(c.resolve() for c in _CANDIDATES if c.exists())
GV1 = "kokoro-v1.0+af_heart+norm-2026.07.31-nemo1.2.0-98d86449"
GV2 = GV1.replace("07.31", "07.32")

# Scripted behavior per (page_id, section_id)
BEHAVIOR = {
    (1, "boom"): ("5xx", None),  # persistent transient -> fail
    (2, "tiny"): ("422", "text_below_minimum"),  # deterministic skip
    (3, "drift"): ("ok_v2", None),  # second generation_version
}


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # /sections?...page_id=N...
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(self.path).query)
        page = int(q["page_id"][0])
        rev = int(q["rev_id"][0])
        secs = [
            {
                "section_id": "lead",
                "title": "Lead",
                "level": 1,
                "generatable": True,
                "char_count": 900,
                "content_sha256": "a" * 64,
            },
            {
                "section_id": "empty",
                "title": "Empty",
                "level": 2,
                "generatable": False,
                "char_count": 0,
                "skip_reason": "text_below_minimum",
            },
        ]
        if page == 1:
            secs.append(
                {
                    "section_id": "boom",
                    "title": "Boom",
                    "level": 2,
                    "generatable": True,
                    "char_count": 500,
                    "content_sha256": "b" * 64,
                }
            )
        if page == 2:
            secs.append(
                {
                    "section_id": "tiny",
                    "title": "Tiny",
                    "level": 2,
                    "generatable": True,
                    "char_count": 60,
                    "content_sha256": "c" * 64,
                }
            )
        if page == 3:
            secs.append(
                {
                    "section_id": "drift",
                    "title": "Drift",
                    "level": 2,
                    "generatable": True,
                    "char_count": 700,
                    "content_sha256": "d" * 64,
                }
            )
        self._json(
            200,
            {
                "wiki_id": "enwiki",
                "page_id": page,
                "rev_id": rev,
                "generation_version": GV1,
                "render_id": f"rid-{page}",
                "sections": secs,
            },
        )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        page, sid, rev = body["page_id"], body["section_id"], body["rev_id"]
        kind, code = BEHAVIOR.get((page, sid), ("ok", None))
        if kind == "5xx":
            self._json(502, {"code": "synthesis_error", "message": "boom"})
            return
        if kind == "422":
            self._json(422, {"code": code, "message": "below minimum"})
            return
        gv = GV2 if kind == "ok_v2" else GV1
        key = f"enwiki/{page}/{rev}/{sid}"
        arts = [
            {
                "artifact_type": "audio_mp3",
                "media_type": "audio/mpeg",
                "generation_version": gv,
                "content_sha256": "e" * 64,
                "duration_ms": 12345.6,
                "render_id": f"rid-{page}",
                "blob_uri": f"s3://tts-artifacts/{key}.mp3",
                "size_bytes": 111,
                "wiki_id": "enwiki",
                "page_id": page,
                "rev_id": rev,
                "section_id": sid,
            },
            {
                "artifact_type": "captions_vtt",
                "media_type": "text/vtt",
                "generation_version": gv,
                "content_sha256": "e" * 64,
                "duration_ms": 12345.6,
                "render_id": f"rid-{page}",
                "blob_uri": f"s3://tts-artifacts/{key}.vtt",
                "size_bytes": 22,
                "wiki_id": "enwiki",
                "page_id": page,
                "rev_id": rev,
                "section_id": sid,
            },
        ]
        self._json(200, {"artifacts": arts, "segment_count": 2})


@pytest.fixture(scope="module")
def stub():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def run_batch(base, tmp, dataset, log="batch.jsonl"):
    ds = tmp / "articles.json"
    ds.write_text(json.dumps(dataset))
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            str(ds),
            "--base",
            base,
            "--log",
            str(tmp / log),
            "--manifest-dir",
            str(tmp / "manifests"),
            "--concurrency",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc, tmp / "manifests", tmp / log


def _manifest(mdir, page, rev):
    p = mdir / f"enwiki/{page}/{rev}/manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_clean_article_gets_manifest_with_correct_shape(stub, tmp_path):
    proc, mdir, _ = run_batch(
        stub, tmp_path, [{"title": "Clean", "page_id": 9, "rev_id": 90}]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    m = _manifest(mdir, 9, 90)
    assert m is not None
    assert m["schema_version"] == 1
    assert m["generation_version"] == GV1
    assert m["render_id"] == "rid-9"
    assert [s["section_id"] for s in m["sections"]] == ["lead"]  # skip absent
    lead = m["sections"][0]
    assert lead["audio"] == {"key": "enwiki/9/90/lead.mp3", "media_type": "audio/mpeg"}
    assert lead["captions"]["key"] == "enwiki/9/90/lead.vtt"
    assert "s3://" not in json.dumps(m)  # keys, never URLs


def test_failed_section_blocks_manifest_dead_letter(stub, tmp_path):
    proc, mdir, log = run_batch(
        stub, tmp_path, [{"title": "HasFail", "page_id": 1, "rev_id": 10}]
    )
    assert proc.returncode == 1  # dead letter present
    assert _manifest(mdir, 1, 10) is None
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    fails = [r for r in recs if r.get("status") == "fail"]
    assert len(fails) == 1 and fails[0]["section_id"] == "boom"
    assert fails[0]["attempts"] == 3  # bounded retries exhausted


def test_deterministic_skip_does_not_block_manifest(stub, tmp_path):
    proc, mdir, _ = run_batch(
        stub, tmp_path, [{"title": "HasSkip", "page_id": 2, "rev_id": 20}]
    )
    assert proc.returncode == 0
    m = _manifest(mdir, 2, 20)
    assert m is not None
    assert [s["section_id"] for s in m["sections"]] == ["lead"]  # tiny absent


def test_mixed_generation_version_blocks_manifest(stub, tmp_path):
    proc, mdir, _ = run_batch(
        stub, tmp_path, [{"title": "Drift", "page_id": 3, "rev_id": 30}]
    )
    assert proc.returncode == 1
    assert _manifest(mdir, 3, 30) is None
    assert "mixed generation_versions" in proc.stdout


def test_resume_is_idempotent_and_rewrites_manifest(stub, tmp_path):
    ds = [{"title": "Clean", "page_id": 9, "rev_id": 90}]
    proc1, mdir, log = run_batch(stub, tmp_path, ds)
    m1 = _manifest(mdir, 9, 90)
    n_records_1 = len(log.read_text().splitlines())
    proc2, _, _ = run_batch(stub, tmp_path, ds)  # same log: full resume
    assert proc2.returncode == 0
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    gen = [r for r in recs if r.get("status") == "ok"]
    assert len(gen) == 1  # NOT regenerated on resume
    m2 = _manifest(mdir, 9, 90)
    assert m2["sections"] == m1["sections"]  # rebuilt identically from log
    assert len(recs) == n_records_1 + 1  # one new manifest record only

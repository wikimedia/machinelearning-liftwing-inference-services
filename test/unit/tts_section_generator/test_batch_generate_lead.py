"""Lead-only + local artifact writing tests (T436758 analytics storage).

Stub generator speaking the real contract, in two modes: blob_uri
(generator wrote the bytes) and bytes_b64 (this script writes them).
"""

import base64
import json
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

HERE = Path(__file__).parent
# Script location: repo layout test/unit/tts_section_generator/ ->
# src/models/tts_section_generator/scripts/ (matches test_batch_generate.py).
_CANDIDATES = [
    HERE / "batch_generate.py",
    HERE / "../../../src/models/tts_section_generator/scripts/batch_generate.py",
]
SCRIPT = next(c.resolve() for c in _CANDIDATES if c.exists())
GV = "kokoro-v1.0+af_heart+norm-2026.08.10-nemo1.2.0-c350b336"

# page 7 has no lead section at all (the missing-requested-section case)
NO_LEAD_PAGE = 7


class Stub(BaseHTTPRequestHandler):
    inline = True  # class switch: bytes_b64 vs blob_uri

    def log_message(self, *a):
        pass

    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        page, rev = int(q["page_id"][0]), int(q["rev_id"][0])
        secs = [
            {
                "section_id": "lead",
                "title": "Lead",
                "level": 1,
                "generatable": True,
                "char_count": 900,
            },
            {
                "section_id": "history",
                "title": "History",
                "level": 2,
                "generatable": True,
                "char_count": 4000,
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
        if page == NO_LEAD_PAGE:
            secs = [s for s in secs if s["section_id"] != "lead"]
        self._json(
            200,
            {
                "wiki_id": "enwiki",
                "page_id": page,
                "rev_id": rev,
                "generation_version": GV,
                "render_id": f"rid-{page}",
                "sections": secs,
            },
        )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        page, sid, rev = body["page_id"], body["section_id"], body["rev_id"]
        key = f"enwiki/{page}/{rev}/{sid}"
        arts = []
        for atype, mtype, payload in (
            ("audio_mp3", "audio/mpeg", b"ID3-fake-mp3-bytes"),
            ("captions_vtt", "text/vtt", b"WEBVTT\n\n00:00.000 --> 00:01.000\nhi\n"),
        ):
            a = {
                "artifact_type": atype,
                "media_type": mtype,
                "generation_version": GV,
                "content_sha256": "e" * 64,
                "duration_ms": 12345.6,
                "render_id": f"rid-{page}",
                "wiki_id": "enwiki",
                "page_id": page,
                "rev_id": rev,
                "section_id": sid,
            }
            if Stub.inline:
                a["bytes_b64"] = base64.b64encode(payload).decode()
            else:
                ext = "mp3" if atype == "audio_mp3" else "vtt"
                a["blob_uri"] = f"s3://bucket/{key}.{ext}"
                a["size_bytes"] = len(payload)
            arts.append(a)
        self._json(200, {"artifacts": arts, "segment_count": 2})


@pytest.fixture(scope="module")
def stub():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def run(base, tmp, dataset, *extra, log="batch.jsonl"):
    ds = tmp / "articles.json"
    ds.write_text(json.dumps(dataset))
    pub = tmp / "published"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--dataset",
        str(ds),
        "--base",
        base,
        "--log",
        str(tmp / log),
        "--manifest-dir",
        str(pub),
        "--concurrency",
        "2",
        *extra,
    ]
    return (
        subprocess.run(cmd, capture_output=True, text=True, timeout=120),
        pub,
        tmp / log,
    )


def manifest(pub, page, rev):
    p = pub / f"enwiki/{page}/{rev}/manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_lead_only_generates_one_section_and_scoped_manifest(stub, tmp_path):
    Stub.inline = True
    proc, pub, _ = run(
        stub,
        tmp_path,
        [{"title": "A", "page_id": 9, "rev_id": 90}],
        "--sections",
        "lead",
        "--artifact-dir",
        str(tmp_path / "published"),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    m = manifest(pub, 9, 90)
    assert m["schema_version"] == 2
    assert m["scope"] == "lead"
    assert [s["section_id"] for s in m["sections"]] == ["lead"]
    # history was NOT generated, and its absence does not block the manifest
    assert not (pub / "enwiki/9/90/history.mp3").exists()


def test_inline_bytes_are_written_to_the_tree_with_public_modes(stub, tmp_path):
    Stub.inline = True
    proc, pub, _ = run(
        stub,
        tmp_path,
        [{"title": "A", "page_id": 9, "rev_id": 90}],
        "--sections",
        "lead",
        "--artifact-dir",
        str(tmp_path / "published"),
    )
    assert proc.returncode == 0
    mp3 = pub / "enwiki/9/90/lead.mp3"
    vtt = pub / "enwiki/9/90/lead.vtt"
    assert mp3.read_bytes() == b"ID3-fake-mp3-bytes"
    assert vtt.read_bytes().startswith(b"WEBVTT")
    for f in (mp3, vtt, pub / "enwiki/9/90/manifest.json"):
        assert stat.S_IMODE(f.stat().st_mode) == 0o644, f
    assert stat.S_IMODE((pub / "enwiki/9/90").stat().st_mode) == 0o755
    # no temp files left behind, and none of them were servable names
    assert not list(pub.rglob("*.tmp"))
    # manifest keys stay relative (never file:// or absolute paths)
    m = manifest(pub, 9, 90)
    assert m["sections"][0]["audio"]["key"] == "enwiki/9/90/lead.mp3"


def test_inline_without_artifact_dir_is_a_hard_fail(stub, tmp_path):
    """The artifacts-must-land rule, preserved: inline bytes with nowhere
    to write them is a non-retryable operator error, not a silent pass."""
    Stub.inline = True
    proc, pub, log = run(
        stub,
        tmp_path,
        [{"title": "A", "page_id": 9, "rev_id": 90}],
        "--sections",
        "lead",
    )
    assert proc.returncode == 1
    assert manifest(pub, 9, 90) is None
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    fails = [r for r in recs if r.get("status") == "fail"]
    assert len(fails) == 1 and "nowhere to put artifacts" in fails[0]["error"]
    assert fails[0]["attempts"] == 1  # not retried


def test_blob_uri_mode_still_works_unchanged(stub, tmp_path):
    """Generator-writes mode (s3/file sink) is untouched by the new path."""
    Stub.inline = False
    proc, pub, _ = run(
        stub,
        tmp_path,
        [{"title": "A", "page_id": 9, "rev_id": 90}],
        "--sections",
        "lead",
    )
    assert proc.returncode == 0
    m = manifest(pub, 9, 90)
    assert m["sections"][0]["audio"]["key"] == "enwiki/9/90/lead.mp3"
    assert m["scope"] == "lead"


def test_missing_requested_section_is_recorded_not_silent(stub, tmp_path):
    """An article without a lead must dead-letter loudly: silently
    generating nothing would leave the corpus quietly short."""
    Stub.inline = True
    proc, pub, log = run(
        stub,
        tmp_path,
        [{"title": "NoLead", "page_id": NO_LEAD_PAGE, "rev_id": 70}],
        "--sections",
        "lead",
        "--artifact-dir",
        str(tmp_path / "published"),
    )
    assert proc.returncode == 1
    assert manifest(pub, NO_LEAD_PAGE, 70) is None
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    fails = [r for r in recs if r.get("status") == "fail"]
    assert len(fails) == 1
    assert "not present in this revision" in fails[0]["error"]
    assert "MISSING SECTION" in proc.stdout


def test_resume_is_idempotent_under_lead_only(stub, tmp_path):
    Stub.inline = True
    ds = [{"title": "A", "page_id": 9, "rev_id": 90}]
    args = ("--sections", "lead", "--artifact-dir", str(tmp_path / "published"))
    proc1, pub, log = run(stub, tmp_path, ds, *args)
    m1 = manifest(pub, 9, 90)
    n1 = len(log.read_text().splitlines())
    proc2, _, _ = run(stub, tmp_path, ds, *args)
    assert proc2.returncode == 0
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert len([r for r in recs if r.get("status") == "ok"]) == 1  # not regenerated
    assert manifest(pub, 9, 90)["sections"] == m1["sections"]
    assert len(recs) == n1 + 1  # one new manifest record only


# ── index.json (consumer listing) ────────────────────────────────────────


def _index(pub):
    p = pub / "index.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_index_lists_every_article_with_audio(stub, tmp_path):
    Stub.inline = True
    ds = [
        {"title": "A", "page_id": 9, "rev_id": 90},
        {"title": "B", "page_id": 5, "rev_id": 50},
    ]
    proc, pub, _ = run(
        stub,
        tmp_path,
        ds,
        "--sections",
        "lead",
        "--artifact-dir",
        str(tmp_path / "published"),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    idx = _index(pub)
    assert idx["schema_version"] == 1
    assert idx["scope"] == "lead"
    assert idx["wiki_id"] == "enwiki"
    assert idx["count"] == 2
    assert idx["generation_versions"] == [GV]
    # sorted by page_id for stable diffs and binary search
    assert [a["page_id"] for a in idx["articles"]] == [5, 9]
    entry = next(a for a in idx["articles"] if a["page_id"] == 9)
    assert entry["title"] == "A"
    assert entry["rev_id"] == 90
    assert entry["manifest"] == "enwiki/9/90/manifest.json"
    assert entry["sections"][0]["audio"] == "enwiki/9/90/lead.mp3"
    assert entry["sections"][0]["captions"] == "enwiki/9/90/lead.vtt"
    assert entry["duration_ms"] > 0
    # relative paths only: never absolute, never a URI scheme
    blob = json.dumps(idx)
    assert "file://" not in blob and "s3://" not in blob
    assert "/srv/" not in blob and "_generation_version" not in blob
    assert f"{idx['count']} articles listed" in proc.stdout


def test_index_excludes_incomplete_articles(stub, tmp_path):
    """An article whose requested section failed gets no manifest, so it
    must not appear in the index: the index lists what can be played."""
    Stub.inline = True
    ds = [
        {"title": "Good", "page_id": 9, "rev_id": 90},
        {"title": "NoLead", "page_id": NO_LEAD_PAGE, "rev_id": 70},
    ]
    proc, pub, _ = run(
        stub,
        tmp_path,
        ds,
        "--sections",
        "lead",
        "--artifact-dir",
        str(tmp_path / "published"),
    )
    assert proc.returncode == 1  # dead letter present
    idx = _index(pub)
    assert idx["count"] == 1
    assert [a["page_id"] for a in idx["articles"]] == [9]


def test_no_index_flag_skips_the_file(stub, tmp_path):
    Stub.inline = True
    proc, pub, _ = run(
        stub,
        tmp_path,
        [{"title": "A", "page_id": 9, "rev_id": 90}],
        "--sections",
        "lead",
        "--no-index",
        "--artifact-dir",
        str(tmp_path / "published"),
    )
    assert proc.returncode == 0
    assert _index(pub) is None


def test_index_is_complete_after_a_resumed_run(stub, tmp_path):
    """The settle pass re-runs over the whole log, so a resumed run's index
    covers articles generated by earlier runs too."""
    Stub.inline = True
    ds = [{"title": "A", "page_id": 9, "rev_id": 90}]
    args = ("--sections", "lead", "--artifact-dir", str(tmp_path / "published"))
    run(stub, tmp_path, ds, *args)
    ds2 = ds + [{"title": "B", "page_id": 5, "rev_id": 50}]
    proc2, pub, log = run(stub, tmp_path, ds2, *args)
    assert proc2.returncode == 0
    idx = _index(pub)
    assert idx["count"] == 2, idx
    assert [a["page_id"] for a in idx["articles"]] == [5, 9]
    # the already-settled article was not regenerated
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert len([r for r in recs if r.get("status") == "ok"]) == 2


def test_index_written_in_blob_uri_mode_too(stub, tmp_path):
    Stub.inline = False
    proc, pub, _ = run(
        stub,
        tmp_path,
        [{"title": "A", "page_id": 9, "rev_id": 90}],
        "--sections",
        "lead",
    )
    assert proc.returncode == 0
    idx = _index(pub)
    assert idx["articles"][0]["sections"][0]["audio"] == "enwiki/9/90/lead.mp3"

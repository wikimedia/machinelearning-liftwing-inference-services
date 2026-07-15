"""Service-level tests with the network mocked out.

Covers the Phase 2 contract mechanics end to end minus real upstreams:
artifact assembly for the full family, isvc request SHAPING (audio-only
requests must ride timestamps="none"), the deterministic-vs-transient
error taxonomy, and retry semantics of the isvc client.
"""

import base64
import math
import struct

import pytest
from fastapi.testclient import TestClient
from tts_generator import fetch as fetch_mod

from src.models.tts_section_generator.tts_generator import isvc_client, service

SR = 24000


def _pcm(seconds: float = 1.0) -> bytes:
    n = int(SR * seconds)
    return struct.pack(
        f"<{n}h", *(int(20000 * math.sin(2 * math.pi * 220 * i / SR)) for i in range(n))
    )


FIXTURE_HTML = """
<html><body>
<section data-mw-section-id="0">
  <p>Earth is the third planet from the Sun and the only astronomical
  object known to harbor life, which is a fact repeated here to comfortably
  clear the minimum text length gate for generation.</p>
</section>
<section data-mw-section-id="1"><h2>Stub</h2><p>Too short.</p></section>
</body></html>
"""


@pytest.fixture()
def client(monkeypatch):
    calls = {}

    def fake_meta(wiki_id, rev_id):
        if rev_id == 404404:
            raise fetch_mod.FetchError("no such revision", status=404)
        return {"page": {"id": 9228}, "timestamp": "2026-07-01T00:00:00Z"}

    def fake_html(wiki_id, rev_id):
        return FIXTURE_HTML

    def fake_synthesize(segments, voice, lang, timestamps="full", encoding="pcm_s16le"):
        calls["timestamps"] = timestamps
        calls["segments"] = segments
        pcm = _pcm()
        return {
            "audio_b64": base64.b64encode(pcm).decode("ascii"),
            "encoding": "pcm_s16le",
            "timestamps_mode": timestamps,
            "sample_rate": SR,
            "duration_ms": 1000.0,
            "timestamps": []
            if timestamps == "none"
            else [
                {"word": "Earth", "start_ms": 80.0, "end_ms": 280.0},
                {"word": "is", "start_ms": 300.0, "end_ms": 400.0},
            ],
        }

    monkeypatch.setattr(service, "fetch_revision_meta", fake_meta)
    monkeypatch.setattr(service, "fetch_revision_html", fake_html)
    monkeypatch.setattr(service.isvc_client, "synthesize", fake_synthesize)
    return TestClient(service.app), calls


def _req(artifacts, section_id="lead"):
    return {
        "wiki_id": "enwiki",
        "page_id": 9228,
        "rev_id": 12345,
        "section_id": section_id,
        "generation_config": {"artifacts": artifacts},
    }


def test_full_artifact_family(client):
    c, _ = client
    r = c.post(
        "/generate-section",
        json=_req(["audio_opus", "audio_mp3", "captions_vtt", "timestamps_json"]),
    )
    assert r.status_code == 200, r.text
    arts = {a["artifact_type"]: a for a in r.json()["artifacts"]}
    assert set(arts) == {"audio_opus", "audio_mp3", "captions_vtt", "timestamps_json"}

    opus = base64.b64decode(arts["audio_opus"]["bytes_b64"])
    assert opus[:4] == b"OggS"
    mp3 = base64.b64decode(arts["audio_mp3"]["bytes_b64"])
    assert mp3[0] == 0xFF

    vtt = base64.b64decode(arts["captions_vtt"]["bytes_b64"]).decode()
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.080 --> 00:00:00.280\nEarth" in vtt

    assert arts["timestamps_json"]["timestamps"][0]["word"] == "Earth"
    for a in arts.values():
        assert a["generation_version"].startswith("kokoro-v1.0+af_heart+norm-")
        assert len(a["content_sha256"]) == 64
        assert a["media_type"]


def test_audio_only_rides_alignment_free_path(client):
    c, calls = client
    r = c.post("/generate-section", json=_req(["audio_opus"]))
    assert r.status_code == 200
    assert calls["timestamps"] == "none"  # RTF-0.22 path, no alignment cost


def test_timing_artifact_forces_full_alignment(client):
    c, calls = client
    r = c.post("/generate-section", json=_req(["audio_opus", "captions_vtt"]))
    assert r.status_code == 200
    assert calls["timestamps"] == "full"


def test_deterministic_skips(client):
    c, _ = client
    r = c.post("/generate-section", json=_req(["audio_opus"], section_id="stub"))
    assert r.status_code == 422
    assert r.json()["code"] == "text_below_minimum"

    r = c.post("/generate-section", json=_req(["audio_opus"], section_id="missing"))
    assert r.status_code == 404
    assert r.json()["code"] == "section_not_found_at_revision"

    r = c.post("/generate-section", json=_req(["audio_wav"]))
    assert r.status_code == 400
    assert r.json()["code"] == "artifact_type_not_available"

    bad = _req(["audio_opus"])
    bad["page_id"] = 1111
    r = c.post("/generate-section", json=bad)
    assert r.status_code == 409
    assert r.json()["code"] == "revision_page_mismatch"


def test_revision_not_found(client):
    c, _ = client
    bad = _req(["audio_opus"])
    bad["rev_id"] = 404404
    r = c.post("/generate-section", json=bad)
    assert r.status_code == 404
    assert r.json()["code"] == "revision_not_found"


def test_sections_endpoint_marks_short_sections(client):
    c, _ = client
    r = c.get("/sections", params={"wiki_id": "enwiki", "page_id": 9228, "rev_id": 1})
    assert r.status_code == 200
    by_id = {s["section_id"]: s for s in r.json()["sections"]}
    assert by_id["lead"]["generatable"] is True
    assert "content_sha256" in by_id["lead"]
    assert by_id["stub"]["generatable"] is False
    assert by_id["stub"]["skip_reason"] == "text_below_minimum"


# ── isvc client retry semantics ────────────────────────────────────────────


def test_isvc_client_retries_transient_then_succeeds(monkeypatch):
    attempts = []

    class FakeResp:
        def __init__(self, status):
            self.status_code = status
            self.ok = status == 200
            self.text = "err"

        def json(self):
            return {"audio_b64": "", "sample_rate": SR}

    def fake_post(*a, **k):
        attempts.append(1)
        return FakeResp(503 if len(attempts) < 2 else 200)

    monkeypatch.setattr(isvc_client.requests, "post", fake_post)
    monkeypatch.setattr(isvc_client.time, "sleep", lambda s: None)
    out = isvc_client.synthesize([{"text": "x"}], voice="v", lang="l")
    assert len(attempts) == 2
    assert out["sample_rate"] == SR


def test_isvc_client_never_retries_4xx(monkeypatch):
    attempts = []

    class FakeResp:
        status_code = 400
        ok = False
        text = "bad request"

    def fake_post(*a, **k):
        attempts.append(1)
        return FakeResp()

    monkeypatch.setattr(isvc_client.requests, "post", fake_post)
    with pytest.raises(isvc_client.SynthesisRejected):
        isvc_client.synthesize([{"text": "x"}], voice="v", lang="l")
    assert len(attempts) == 1

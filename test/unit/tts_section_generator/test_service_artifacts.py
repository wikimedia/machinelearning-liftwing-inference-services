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
from tts_generator import fetch as fetch_mod, isvc_client, service

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
        return FIXTURE_HTML, None  # (html, render_id); None = no header

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


def test_render_id_on_artifacts_and_sections(monkeypatch):
    """render_id from X-MediaWiki-Render-ID is stamped on every artifact
    and on the /sections response when present; absent when missing."""

    def fake_meta(w, r):
        return {"page": {"id": 9228}, "timestamp": "2026-07-01T00:00:00Z"}

    def fake_html_present(w, r):
        return FIXTURE_HTML, "abc-def-123"

    def fake_synth(segments, voice, lang, timestamps="full", encoding="pcm_s16le"):
        pcm = _pcm()
        return {
            "audio_b64": base64.b64encode(pcm).decode("ascii"),
            "encoding": "pcm_s16le",
            "timestamps_mode": timestamps,
            "sample_rate": SR,
            "duration_ms": 1000.0,
            "timestamps": [],
        }

    monkeypatch.setattr(service, "fetch_revision_meta", fake_meta)
    monkeypatch.setattr(service, "fetch_revision_html", fake_html_present)
    monkeypatch.setattr(service.isvc_client, "synthesize", fake_synth)

    c = TestClient(service.app)

    # /sections carries render_id when header present
    r = c.get("/sections", params={"wiki_id": "enwiki", "page_id": 9228, "rev_id": 1})
    assert r.status_code == 200
    assert r.json()["render_id"] == "abc-def-123"

    # /generate-section stamps render_id on every artifact
    r = c.post("/generate-section", json=_req(["audio_opus"]))
    assert r.status_code == 200, r.text
    for a in r.json()["artifacts"]:
        assert a["render_id"] == "abc-def-123"

    # Absent header -> key omitted
    monkeypatch.setattr(
        service, "fetch_revision_html", lambda w, r: (FIXTURE_HTML, None)
    )
    r2 = c.get("/sections", params={"wiki_id": "enwiki", "page_id": 9228, "rev_id": 1})
    assert "render_id" not in r2.json()


def test_synthesis_not_possible_maps_to_422(monkeypatch):
    """SynthesisNotPossible -> 422 text_not_synthesizable (deterministic skip)."""

    def fake_meta(w, r):
        return {"page": {"id": 9228}, "timestamp": "2026-07-01T00:00:00Z"}

    def fake_html(w, r):
        return FIXTURE_HTML, None

    def fake_synth(*a, **k):
        raise isvc_client.SynthesisNotPossible("phoneme limit 510 exceeded")

    monkeypatch.setattr(service, "fetch_revision_meta", fake_meta)
    monkeypatch.setattr(service, "fetch_revision_html", fake_html)
    monkeypatch.setattr(service.isvc_client, "synthesize", fake_synth)

    c = TestClient(service.app)
    r = c.post("/generate-section", json=_req(["audio_opus"]))
    assert r.status_code == 422
    assert r.json()["code"] == "text_not_synthesizable"


def test_isvc_client_never_retries_4xx(monkeypatch):
    attempts = []

    class FakeResp:
        status_code = 400
        ok = False
        text = "bad request"

        def json(self):
            return {}

    def fake_post(*a, **k):
        attempts.append(1)
        return FakeResp()

    monkeypatch.setattr(isvc_client.requests, "post", fake_post)
    with pytest.raises(isvc_client.SynthesisRejected):
        isvc_client.synthesize([{"text": "x"}], voice="v", lang="l")
    assert len(attempts) == 1


def test_isvc_client_raises_synthesis_not_possible_for_phoneme_code(monkeypatch):
    """4xx with code=text_not_synthesizable -> SynthesisNotPossible, not
    SynthesisRejected. Deterministic skip, never retried."""

    class FakeResp:
        status_code = 422
        ok = False
        text = '{"code": "text_not_synthesizable"}'

        def json(self):
            return {"code": "text_not_synthesizable"}

    monkeypatch.setattr(isvc_client.requests, "post", lambda *a, **k: FakeResp())
    with pytest.raises(isvc_client.SynthesisNotPossible):
        isvc_client.synthesize([{"text": "x"}], voice="v", lang="l")


def test_kserve_error_envelope_is_recognised(monkeypatch):
    """The model server returns KServe's {"error": "<message>"} with no
    "code" key. Matching only on "code" reported a deterministic rejection
    as a generic 502, which the batch retried three times and then
    dead-lettered: 81 articles in the first production run (T438647)."""

    class FakeResp:
        status_code = 400
        ok = False
        text = '{"error": "text_not_synthesizable: segment 3 phonemizes to 551..."}'

        def json(self):
            return {
                "error": "text_not_synthesizable: segment 3 phonemizes to 551 "
                "phonemes, over the Kokoro context limit (510); rejecting "
                "rather than truncating"
            }

    monkeypatch.setattr(isvc_client.requests, "post", lambda *a, **k: FakeResp())
    with pytest.raises(isvc_client.SynthesisNotPossible):
        isvc_client.synthesize([{"text": "x"}], voice="v", lang="l")


def test_other_4xx_still_raises_the_loud_path(monkeypatch):
    class FakeResp:
        status_code = 400
        ok = False
        text = '{"error": "some other problem"}'

        def json(self):
            return {"error": "some other problem"}

    monkeypatch.setattr(isvc_client.requests, "post", lambda *a, **k: FakeResp())
    with pytest.raises(isvc_client.SynthesisRejected):
        isvc_client.synthesize([{"text": "x"}], voice="v", lang="l")


# ── Long-section splitting (T441983) ──────────────────────────────────────


def _recording_client(monkeypatch, chunks, budget, pcm_fn=None):
    """Patch fetch/split_text/synthesize so a section's chunk list and
    batch budget are controlled, and every isvc call is recorded.

    ``split_text`` is stubbed (its behaviour is covered by test_chunking.py)
    so the split tests exercise the batch loop wiring deterministically:
    one call per batch, the per-batch segment lists, PCM concatenation, and
    timestamp offsets. Returns (TestClient, calls) where ``calls`` is the
    ordered list of ``segments`` argument lists passed to synthesize.
    """
    calls = []

    def fake_meta(wiki_id, rev_id):
        return {"page": {"id": 9228}, "timestamp": "2026-07-01T00:00:00Z"}

    def fake_html(wiki_id, rev_id):
        return FIXTURE_HTML, None

    def fake_synthesize(segments, voice, lang, timestamps="full", encoding="pcm_s16le"):
        pcm = pcm_fn(len(calls)) if pcm_fn else _pcm()
        calls.append(segments)
        return {
            "audio_b64": base64.b64encode(pcm).decode("ascii"),
            "encoding": "pcm_s16le",
            "timestamps_mode": timestamps,
            "sample_rate": SR,
            "duration_ms": 1000.0,
            "timestamps": []
            if timestamps == "none"
            else [
                {"word": s["text"].split()[0], "start_ms": 0.0, "end_ms": 100.0}
                for s in segments
            ],
        }

    monkeypatch.setattr(service, "fetch_revision_meta", fake_meta)
    monkeypatch.setattr(service, "fetch_revision_html", fake_html)
    monkeypatch.setattr(service, "split_text", lambda text, n: chunks)
    monkeypatch.setattr(service, "SPLIT_BATCH_CHARS", budget)
    monkeypatch.setattr(service.isvc_client, "synthesize", fake_synthesize)
    return TestClient(service.app), calls


def test_long_section_issues_one_call_per_batch(monkeypatch):
    """A section over the budget issues one isvc call per batch, in order.
    Fails against the pre-split single-call path, so it discriminates the
    change from a no-op."""
    chunks = [
        "One sentence.",
        "Two sentence.",
        "Three sentence.",
        "Four sentence.",
    ]
    c, calls = _recording_client(monkeypatch, chunks, budget=25)
    r = c.post("/generate-section", json=_req(["audio_opus"]))
    assert r.status_code == 200, r.text
    assert len(calls) > 1
    # Every chunk sent, in order, never dropped or reordered.
    assert [s["text"] for call in calls for s in call] == chunks


def test_below_threshold_issues_exactly_one_call(monkeypatch):
    """The no-op path: a section under the budget is a single request and
    byte-identical to pre-split output."""
    chunks = ["One sentence.", "Two sentence."]
    c, calls = _recording_client(monkeypatch, chunks, budget=1000)
    r = c.post("/generate-section", json=_req(["audio_opus"]))
    assert r.status_code == 200, r.text
    assert len(calls) == 1


def test_split_concatenates_pcm_with_silence_gap(monkeypatch):
    """Batch PCM is joined as b0 + gap + b1 + ... with the configured
    silence between batches (no crossfade across request boundaries)."""
    chunks = ["One sentence.", "Two sentence.", "Three sentence."]
    pcm0 = b"\x01\x02" * 100
    pcm1 = b"\x03\x04" * 100
    c, calls = _recording_client(
        monkeypatch, chunks, budget=20, pcm_fn=lambda i: [pcm0, pcm1][i]
    )
    r = c.post("/generate-section", json=_req(["audio_pcm_s16le"]))
    assert r.status_code == 200, r.text
    assert len(calls) == 2
    raw = base64.b64decode(
        [a for a in r.json()["artifacts"] if a["artifact_type"] == "audio_pcm_s16le"][
            0
        ]["bytes_b64"]
    )
    gap = service._silence_pcm(service.SPLIT_GAP_MS, SR)
    assert raw == pcm0 + gap + pcm1


def test_split_offsets_timestamps_by_cumulative_duration(monkeypatch):
    """Each batch's word timestamps are shifted by the duration of every
    preceding batch plus its gap, so captions stay in sync with the
    concatenated audio."""
    chunks = ["One sentence.", "Two sentence.", "Three sentence."]
    # budget 20 -> [["One", "Two"], ["Three"]]: two batches, one gap.
    c, calls = _recording_client(monkeypatch, chunks, budget=20)
    r = c.post("/generate-section", json=_req(["audio_opus", "timestamps_json"]))
    assert r.status_code == 200, r.text
    ts = [a for a in r.json()["artifacts"] if a["artifact_type"] == "timestamps_json"][
        0
    ]["timestamps"]

    assert [t["word"] for t in ts] == ["One", "Two", "Three"]
    # Batch 0 words are unshifted.
    assert ts[0]["start_ms"] == 0.0
    # Batch 1 word ("Three") is shifted by pcm0 (1000 ms) + gap (250 ms).
    assert ts[2]["start_ms"] == pytest.approx(1250.0)

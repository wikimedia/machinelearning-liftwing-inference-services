"""Sink tests (Spike 2): key layout, file-sink behavior, service
integration in file mode, and the loud-failure property of the s3 stub."""

import base64
import math
import struct

import pytest

from src.models.tts_section_generator.tts_generator.sinks import (
    FileSink,
    InlineSink,
    S3Sink,
    artifact_key,
)


def test_artifact_key_layout():
    assert (
        artifact_key("enwiki", 9228, 123, "atmosphere", "audio_opus")
        == "enwiki/9228/123/atmosphere.opus"
    )
    assert artifact_key("enwiki", 9228, 123, "lead", "captions_vtt").endswith(
        "lead.vtt"
    )


def test_inline_sink_returns_b64():
    out = InlineSink().store("k", b"hello", "audio/ogg")
    assert base64.b64decode(out["bytes_b64"]) == b"hello"
    assert "blob_uri" not in out


def test_file_sink_writes_and_returns_uri(tmp_path):
    sink = FileSink(str(tmp_path))
    out = sink.store("enwiki/9228/123/lead.opus", b"opusdata", "audio/ogg")
    assert out["blob_uri"].startswith("file://")
    assert out["size_bytes"] == 8
    written = tmp_path / "enwiki/9228/123/lead.opus"
    assert written.read_bytes() == b"opusdata"
    # No .tmp remnants: publish is atomic.
    assert not list(tmp_path.rglob("*.tmp"))


def test_file_sink_overwrite_is_idempotent(tmp_path):
    sink = FileSink(str(tmp_path))
    sink.store("a/b.opus", b"one", "audio/ogg")
    out = sink.store("a/b.opus", b"one", "audio/ogg")
    assert (tmp_path / "a/b.opus").read_bytes() == b"one"
    assert out["size_bytes"] == 3


def test_s3_stub_fails_loudly_without_config():
    with pytest.raises(RuntimeError, match="requires"):
        S3Sink("", "")


def test_s3_stub_fails_loudly_even_with_config():
    with pytest.raises(NotImplementedError, match="Data Persistence"):
        S3Sink("https://endpoint", "bucket")


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


def test_service_file_mode_returns_blob_uri(monkeypatch, tmp_path):
    """End-to-end through the service with the file sink: response carries
    blob_uri instead of bytes_b64, and the bytes land on disk."""
    from fastapi.testclient import TestClient

    from src.models.tts_section_generator.tts_generator import service

    monkeypatch.setattr(
        service,
        "fetch_revision_meta",
        lambda w, r: {"page": {"id": 9228}, "timestamp": "2026-07-01T00:00:00Z"},
    )
    monkeypatch.setattr(service, "fetch_revision_html", lambda w, r: FIXTURE_HTML)

    def fake_synthesize(segments, voice, lang, timestamps="full", encoding="pcm_s16le"):
        return {
            "audio_b64": base64.b64encode(_pcm()).decode("ascii"),
            "encoding": "pcm_s16le",
            "timestamps_mode": timestamps,
            "sample_rate": SR,
            "duration_ms": 1000.0,
            "timestamps": [{"word": "Earth", "start_ms": 0.0, "end_ms": 200.0}],
        }

    monkeypatch.setattr(service.isvc_client, "synthesize", fake_synthesize)
    monkeypatch.setattr(service, "_sink", FileSink(str(tmp_path)))

    c = TestClient(service.app)
    r = c.post(
        "/generate-section",
        json={
            "wiki_id": "enwiki",
            "page_id": 9228,
            "rev_id": 777,
            "section_id": "lead",
            "generation_config": {"artifacts": ["audio_opus", "captions_vtt"]},
        },
    )
    assert r.status_code == 200, r.text
    arts = {a["artifact_type"]: a for a in r.json()["artifacts"]}
    for a in arts.values():
        assert "blob_uri" in a and "bytes_b64" not in a
    opus_path = tmp_path / "enwiki/9228/777/lead.opus"
    assert opus_path.read_bytes()[:4] == b"OggS"
    vtt_path = tmp_path / "enwiki/9228/777/lead.vtt"
    assert vtt_path.read_text().startswith("WEBVTT")

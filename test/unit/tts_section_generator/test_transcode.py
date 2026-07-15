"""Transcode tests.

Determinism is verified EMPIRICALLY (same input twice -> identical bytes)
because golden-artifact tests and content-addressed dedup depend on it,
and it is exactly the property an ffmpeg upgrade could silently break
(the Ogg muxer's random serial number is pinned by -serial_offset; the
bitexact flags suppress version-stamped metadata).
"""

import hashlib
import math
import shutil
import struct
import subprocess

import pytest

from src.models.tts_section_generator.tts_generator.config import FFMPEG_PATH
from src.models.tts_section_generator.tts_generator.transcode import (
    pcm_to_mp3,
    pcm_to_opus,
)

pytestmark = pytest.mark.skipif(
    shutil.which(FFMPEG_PATH) is None, reason="ffmpeg not installed"
)

SR = 24000


def _sine_pcm(seconds: float = 2.0, freq: float = 220.0) -> bytes:
    """Deterministic int16 mono test tone (speech-band, isvc sample rate)."""
    n = int(SR * seconds)
    samples = (int(20000 * math.sin(2 * math.pi * freq * i / SR)) for i in range(n))
    return struct.pack(f"<{n}h", *samples)


def _decoded_duration_s(data: bytes) -> float:
    """Decode back to 24kHz int16 PCM and count samples: exact duration,
    works on piped input (ffprobe cannot seek a pipe to find Ogg duration),
    and doubles as a round-trip decodability check."""
    proc = subprocess.run(  # noqa: S603
        [
            FFMPEG_PATH,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            str(SR),
            "-ac",
            "1",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=True,
    )
    return len(proc.stdout) / 2 / SR


def test_transcode_preserves_duration():
    pcm = _sine_pcm(seconds=2.0)
    for codec in (pcm_to_opus, pcm_to_mp3):
        dur = _decoded_duration_s(codec(pcm, SR))
        # Codec framing adds small padding; anything beyond ~120ms drift
        # would desync the word timestamps.
        assert abs(dur - 2.0) < 0.12, f"{codec.__name__}: {dur}"


def test_opus_output_is_deterministic():
    pcm = _sine_pcm()
    a, b = pcm_to_opus(pcm, SR), pcm_to_opus(pcm, SR)
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_mp3_output_is_deterministic():
    pcm = _sine_pcm()
    a, b = pcm_to_mp3(pcm, SR), pcm_to_mp3(pcm, SR)
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_opus_is_ogg_and_smaller_than_pcm():
    pcm = _sine_pcm()
    out = pcm_to_opus(pcm, SR)
    assert out[:4] == b"OggS"
    assert len(out) < len(pcm) / 4  # 32kbps vs 384kbps PCM


def test_mp3_has_sync_and_is_smaller_than_pcm():
    pcm = _sine_pcm()
    out = pcm_to_mp3(pcm, SR)
    # MPEG frame sync: 11 set bits at the start (no ID3 header requested).
    assert out[0] == 0xFF and (out[1] & 0xE0) == 0xE0
    assert len(out) < len(pcm) / 4


def test_different_input_different_output():
    a = pcm_to_opus(_sine_pcm(freq=220.0), SR)
    b = pcm_to_opus(_sine_pcm(freq=440.0), SR)
    assert a != b

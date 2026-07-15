"""PCM -> delivery-codec transcoding via ffmpeg.

The isvc returns raw PCM (int16 by default); this module produces the
delivery artifacts. Opus is the recommended codec for speech (better
quality than MP3 at half the bitrate, native on iOS 17+/Android); MP3 is
kept as a supported alternative so the Apps codec decision never blocks
generation work. Both are exposed as distinct artifact types
(audio_opus / audio_mp3) rather than a config switch, so the request
records what was produced.

DETERMINISM is a requirement here, not an accident: golden-artifact tests
byte-compare output, and content-addressed storage dedup benefits from it.
Two ffmpeg behaviors would silently break it:

* the Ogg muxer embeds a RANDOM stream serial number per invocation;
  pinned with ``-serial_offset``.
* muxers stamp encoder version/creation metadata; suppressed with
  ``-fflags +bitexact -flags:a +bitexact`` on the output.

tests/test_transcode.py verifies determinism empirically (same input
twice -> identical bytes) so an ffmpeg upgrade that breaks it fails CI
instead of production.
"""

import logging
import subprocess

from tts_generator.config import (
    FFMPEG_PATH,
    MP3_BITRATE,
    OPUS_APPLICATION,
    OPUS_BITRATE,
)

logger = logging.getLogger(__name__)


class TranscodeError(Exception):
    """ffmpeg failed; carries stderr tail for diagnosis."""


def _run_ffmpeg(args: list[str], pcm: bytes) -> bytes:
    cmd = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error", *args]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd, input=pcm, capture_output=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise TranscodeError(f"ffmpeg execution failed: {e}") from e
    if proc.returncode != 0 or not proc.stdout:
        raise TranscodeError(
            f"ffmpeg exited {proc.returncode}: {proc.stderr.decode()[-500:]}"
        )
    return proc.stdout


def _input_args(sample_rate: int) -> list[str]:
    # Raw int16 little-endian mono on stdin, as the isvc emits (pcm_s16le).
    return ["-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0"]


def pcm_to_opus(pcm: bytes, sample_rate: int) -> bytes:
    """int16 PCM -> Opus in Ogg, deterministic output."""
    return _run_ffmpeg(
        [
            *_input_args(sample_rate),
            "-c:a",
            "libopus",
            "-b:a",
            OPUS_BITRATE,
            "-application",
            OPUS_APPLICATION,
            "-vbr",
            "on",
            "-fflags",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            "-serial_offset",
            "1",  # pin the Ogg stream serial (random otherwise)
            "-f",
            "ogg",
            "pipe:1",
        ],
        pcm,
    )


def pcm_to_mp3(pcm: bytes, sample_rate: int) -> bytes:
    """int16 PCM -> MP3 (libmp3lame), deterministic output."""
    return _run_ffmpeg(
        [
            *_input_args(sample_rate),
            "-c:a",
            "libmp3lame",
            "-b:a",
            MP3_BITRATE,
            "-fflags",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            "-id3v2_version",
            "0",  # no ID3 header: nothing to carry timestamps
            "-f",
            "mp3",
            "pipe:1",
        ],
        pcm,
    )

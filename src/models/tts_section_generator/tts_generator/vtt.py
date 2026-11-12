"""WebVTT / JSON formatting for word-level timestamps.

Direct port of the output-formatting half of v0 ``wiki_tts/timestamps.py``
(the alignment half lives in the isvc). v0's tests carry over verbatim in
tests/test_vtt.py.
"""

import json


def timestamps_to_vtt(timestamps: list[dict]) -> str:
    """Format word timestamps as WebVTT (one cue per word)."""
    lines = ["WEBVTT\n"]
    for t in timestamps:
        start = _ms_to_vtt(t["start_ms"])
        end = _ms_to_vtt(t["end_ms"])
        lines.append(f"{start} --> {end}\n{t['word']}\n")
    return "\n".join(lines)


def paragraphs_to_vtt(paragraphs: list[dict]) -> str:
    """Format paragraph time ranges as WebVTT (one cue per paragraph).

    The cue text is the paragraph's pruned prose -- citation markers and
    other non-spoken noise removed -- not its normalized form, so "1000"
    stays "1000" rather than "one thousand". It matches what the voice
    speaks modulo normalization, not the fully-rendered page. Word cues
    carry normalized text instead, which is exactly why they cannot be
    matched that way.
    """
    lines = ["WEBVTT\n"]
    for p in paragraphs:
        start = _ms_to_vtt(p["start_ms"])
        end = _ms_to_vtt(p["end_ms"])
        lines.append(f"{start} --> {end}\n{p['excerpt']}\n")
    return "\n".join(lines)


def timestamps_to_json(timestamps: list[dict]) -> str:
    """Format word timestamps as a JSON array."""
    return json.dumps(timestamps, indent=2)


def _ms_to_vtt(ms: float) -> str:
    """Convert milliseconds to WebVTT timestamp (``HH:MM:SS.mmm``)."""
    total_seconds = ms / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

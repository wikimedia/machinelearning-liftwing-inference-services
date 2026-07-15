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

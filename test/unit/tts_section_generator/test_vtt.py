"""v0's timestamp output-formatting tests, carried over verbatim."""

from src.models.tts_section_generator.tts_generator.vtt import (
    _ms_to_vtt,
    paragraphs_to_vtt,
    timestamps_to_json,
    timestamps_to_vtt,
)


def test_ms_to_vtt_formats_hours_minutes_seconds_and_milliseconds():
    assert _ms_to_vtt(3_723_456) == "01:02:03.456"


def test_timestamps_to_vtt_emits_webvtt_cues():
    result = timestamps_to_vtt([{"word": "Hello", "start_ms": 0, "end_ms": 1250}])

    assert result == "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHello\n"


def test_timestamps_to_json_pretty_prints_timestamp_data():
    result = timestamps_to_json([{"word": "Hi", "start_ms": 0, "end_ms": 20}])

    assert '"word": "Hi"' in result
    assert result.startswith("[\n")


def test_vtt_multiple_cues_are_blank_line_separated():
    result = timestamps_to_vtt(
        [
            {"word": "Hello", "start_ms": 80.0, "end_ms": 280.0},
            {"word": "world.", "start_ms": 420.0, "end_ms": 720.0},
        ]
    )
    assert "00:00:00.080 --> 00:00:00.280\nHello" in result
    assert "00:00:00.420 --> 00:00:00.720\nworld." in result
    assert result.count("-->") == 2


def test_paragraphs_to_vtt_emits_one_cue_per_paragraph():
    result = paragraphs_to_vtt(
        [
            {
                "start_ms": 0.0,
                "end_ms": 1250.0,
                "excerpt": "Earth is the third planet.",
            },
            {"start_ms": 1250.0, "end_ms": 2500.0, "excerpt": "It is the only place."},
        ]
    )

    assert "00:00:00.000 --> 00:00:01.250\nEarth is the third planet." in result
    assert "00:00:01.250 --> 00:00:02.500\nIt is the only place." in result
    assert result.count("-->") == 2

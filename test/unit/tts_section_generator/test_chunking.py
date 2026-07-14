"""v0's _split_text test suite, carried over verbatim as the behavioral
spec for the reimplementation (v0's worker.py was not portable)."""

from src.models.tts_section_generator.tts_generator.chunking import split_text


def test_split_text_returns_short_text_as_single_chunk():
    assert split_text("Short text.", max_chars=50) == ["Short text."]


def test_split_text_prefers_sentence_boundaries():
    chunks = split_text(
        "First sentence. Second sentence. Third sentence.", max_chars=32
    )

    assert chunks == ["First sentence. Second sentence.", "Third sentence."]
    assert all(len(chunk) <= 32 for chunk in chunks)


def test_split_text_splits_long_sentence_on_word_boundaries():
    chunks = split_text("alpha beta gamma delta", max_chars=10)

    assert chunks == ["alpha beta", "gamma", "delta"]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_split_text_ignores_empty_sentence_fragments():
    chunks = split_text("One.   Two?\n\nThree!", max_chars=10)

    assert chunks == ["One. Two?", "Three!"]


def test_split_text_handles_empty_and_blank():
    assert split_text("", max_chars=10) == []
    assert split_text("   \n  ", max_chars=10) == []

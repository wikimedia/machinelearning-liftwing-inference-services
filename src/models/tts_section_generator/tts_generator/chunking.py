"""Sentence-aware text chunking for TTS segments.

Reimplementation of v0 ``wiki_tts/worker._split_text`` to the behavior its
test suite defines (the v0 tests carry over verbatim in
tests/test_chunking.py):

* text at or under the limit returns as a single chunk;
* chunks break at sentence boundaries first, packing greedily;
* a single sentence over the limit splits greedily at word boundaries;
* empty fragments and stray whitespace are dropped.

The isvc enforces nothing here; it only warns above 800 chars (Kokoro's
practical input limit, where the model may silently truncate). Chunk size
is therefore a quality knob owned by this service; see MAX_SEGMENT_CHARS
in config.py.
"""

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = f"{current} {sentence}" if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        flush()

        if len(sentence) <= max_chars:
            current = sentence
            continue

        # Single sentence over the limit: greedy word-boundary packing.
        for word in sentence.split(" "):
            candidate = f"{current} {word}" if current else word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                current = word

    flush()
    return chunks

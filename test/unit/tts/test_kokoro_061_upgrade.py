"""Behaviour pinned across the kokoro-onnx 0.5.0 -> 0.6.1 upgrade.

0.6.1 changes two things that matter here:

* ``Tokenizer.phonemize`` serializes espeak internally (kokoro-onnx#197),
  which is the fix for the corruption that forced this service's own
  synthesis lock (T430536);
* text whose phonemization exceeds ``MAX_PHONEME_LENGTH`` is split into
  balanced batches instead of being silently truncated.

The second only applies past the limit, and ``check_phoneme_limits``
rejects over-limit segments before synthesis, so every segment this
service actually synthesizes is a single batch — the same path 0.5.0
took. These tests pin that, so a future change that lets over-limit text
through has to say so explicitly.

These are integration tests: they load the real kokoro-onnx, onnxruntime,
transformers and scipy (absent from ``requirements-test.txt``, so the
module imports are gated with ``importorskip``) plus the model files
(skipped when ``MODEL_PATH`` is unset).
"""

import os
import threading

import numpy as np
import pytest

# inference.py imports onnxruntime, transformers, scipy and kokoro_onnx at
# module import time; none of those are in requirements-test.txt, so the
# module must be imported skippably rather than at collection time.
# importorskip also skips the whole module cleanly on a runner that lacks
# kokoro-onnx (e.g. CI, which stubs it for test_inference.py instead).
inference = pytest.importorskip("src.models.tts.model_server.inference")
MAX_PHONEME_LENGTH = pytest.importorskip("kokoro_onnx.config").MAX_PHONEME_LENGTH

MODEL_PATH = os.environ.get("MODEL_PATH")
needs_models = pytest.mark.skipif(
    not MODEL_PATH, reason="MODEL_PATH unset; needs kokoro + wav2vec2 model files"
)


@pytest.fixture(scope="module")
def pipeline():
    return inference.TTSInferencePipeline(
        f"{MODEL_PATH}/kokoro/kokoro-v1.0.onnx",
        f"{MODEL_PATH}/kokoro/voices-v1.0.bin",
        f"{MODEL_PATH}/wav2vec2",
        kokoro_threads=2,
        w2v2_threads=1,
    )


# ── The guard still rejects over-limit text ──────────────────────────────


@needs_models
def test_over_limit_segment_is_still_rejected(pipeline):
    """0.6.1 could synthesize this by splitting it, but the service's
    contract is to reject it so the caller records a skip. Removing the
    guard is a separate, deliberate decision (T441983)."""
    long_text = "The quick brown fox jumps over the lazy dog. " * 40
    with pytest.raises(inference.TextNotSynthesizable):
        inference.check_phoneme_limits(
            [{"text": long_text}], pipeline.kokoro.tokenizer, "en-us"
        )


@needs_models
def test_in_limit_segment_passes_the_guard(pipeline):
    inference.check_phoneme_limits(
        [{"text": "A short sentence."}], pipeline.kokoro.tokenizer, "en-us"
    )


@needs_models
def test_guard_counts_phonemes_not_characters(pipeline):
    """Short text with many unpronounceable characters can still exceed
    the limit: espeak spells them out (T438647)."""
    phonemes = pipeline.kokoro.tokenizer.phonemize("Nguyễn", lang="en-us")
    assert len(phonemes) > len("Nguyễn")


# ── Synthesis still works, and is deterministic ──────────────────────────


@needs_models
def test_predict_returns_audio_and_timestamps(pipeline):
    result = pipeline.predict([{"text": "Hello world."}])
    assert result["sample_rate"] == 24000
    assert result["audio"].dtype == np.float32
    assert result["audio"].size > 0
    assert result["timestamps"]


@needs_models
def test_same_input_gives_same_audio(pipeline):
    """Synthesis must be deterministic: the index's content hash assumes
    identical input yields identical output."""
    a = pipeline.predict([{"text": "Determinism matters here."}])["audio"]
    b = pipeline.predict([{"text": "Determinism matters here."}])["audio"]
    assert np.array_equal(a, b)


# ── The upstream thread-safety fix ───────────────────────────────────────


@needs_models
def test_phonemization_is_thread_safe(pipeline):
    """The corruption that forced the service lock (T430536) is fixed
    upstream in 0.6.1: the tokenizer serializes espeak itself.

    Two properties this test needs in order to detect the bug at all,
    both established by running it against 0.5.0:

    * DISTINCT texts per thread — identical inputs would pass even under
      corruption, since every thread would corrupt alike;
    * REALISTIC segment length — short sentences phonemize too quickly to
      interleave. Four-sentence texts of ~400 characters, the size this
      service actually receives, corrupt reliably on 0.5.0 and pass on
      0.6.1.
    """
    texts = [
        "The covered bridge carries traffic over the creek in the township. "
        "It was constructed with a Burr arch truss and listed on the register. "
        "The structure measures approximately thirty metres in total span. "
        "A rehabilitation of the historic district followed some years later.",
        "The county seat lies south of the river and north of the ridge. "
        "Several places in the district were listed together in one filing. "
        "The survey measured each span and recorded its construction date. "
        "Traffic was carried by a single lane for most of its working life.",
        "Approximately half the structures in the survey remain standing. "
        "The register entry describes the truss, the abutments and the deck. "
        "Rehabilitation work replaced the floor beams but kept the arches. "
        "The bridge continues to carry local traffic across the creek today.",
        "A historic district in the borough contains eleven listed buildings. "
        "The earliest was constructed before the county was formally organised. "
        "Later additions followed the railway and the lumber industry there. "
        "The district boundary measures roughly one kilometre end to end.",
    ]
    tokenizer = pipeline.kokoro.tokenizer
    expected = {t: tokenizer.phonemize(t, lang="en-us") for t in texts}

    got: dict[str, str] = {}
    errors: list[Exception] = []
    lock = threading.Lock()

    def run(text: str) -> None:
        try:
            phonemes = tokenizer.phonemize(text, lang="en-us")
        except Exception as e:  # noqa: BLE001 - the failure IS the result
            with lock:
                errors.append(e)
            return
        with lock:
            got[text] = phonemes

    threads = [threading.Thread(target=run, args=(t,)) for t in texts * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"phonemization raised under concurrency: {errors[:2]}"
    for text, phonemes in got.items():
        assert phonemes == expected[text], f"corrupted phonemes for {text!r}"


def test_phoneme_limit_constant_is_unchanged():
    """Pinned because the service's guard, the generator's chunk size and
    the published skip reasons are all calibrated to this number."""
    assert MAX_PHONEME_LENGTH == 510

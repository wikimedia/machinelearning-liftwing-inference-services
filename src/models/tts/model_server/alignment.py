"""
Wav2Vec2-CTC forced alignment for word-level timestamps.

Extracted from v0 ``wiki_tts/timestamps.py`` and packaged as a standalone
module so the KServe model-server has no dependency on the wiki_tts package.

A summary on the v0 implementation can be seen here:
https://phabricator.wikimedia.org/T424378#12068767
"""

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy.signal
from transformers import Wav2Vec2Processor

logger = logging.getLogger(__name__)

ALIGNER_SR = 16000  # Wav2Vec2 expects 16 kHz input
FRAME_DURATION_MS = 20  # Wav2Vec2 frame stride at 16 kHz (20 ms per frame)


class Aligner:
    """
    Lazy-loading Wav2Vec2-CTC ONNX aligner.

    Load model + processor once at startup, then call ``.align()`` per chunk.

    Args:
        model_dir: Path to a directory containing ``model.onnx`` and a
            ``processor/`` subdirectory with Wav2Vec2 tokenizer config files.
    """

    def __init__(self, model_dir: str):
        self.model_path = Path(model_dir) / "model.onnx"
        self.processor_path = Path(model_dir) / "processor"

        if not self.model_path.exists() or not self.processor_path.exists():
            raise FileNotFoundError(
                f"Wav2Vec2 model or processor not found at {model_dir}"
            )

        logger.info("Loading Wav2Vec2-CTC ONNX session...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(self.model_path), sess_options)

        logger.info("Loading Wav2Vec2 processor from %s...", self.processor_path)
        self.processor = Wav2Vec2Processor.from_pretrained(str(self.processor_path))
        logger.info("Wav2Vec2-CTC aligner ready.")

    def align(self, audio: np.ndarray, sample_rate: int, text: str) -> list[dict]:
        """
        Run forced alignment on a single chunk of synthesized audio.

        Resamples to 16 kHz, runs the Wav2Vec2-CTC ONNX model, CTC-decodes
        against the known text, and maps the result to per-word frame ranges.

        Args:
            audio: Float32 PCM audio samples at ``sample_rate``.
            sample_rate: Sample rate of ``audio`` in Hz (e.g. 24000).
            text: The known text that was spoken in ``audio``.

        Returns:
            A list of ``{"word": str, "start_ms": float, "end_ms": float}``
            dicts, or an empty list if ``text`` is blank or alignment produces
            no CTC segments.
        """
        if not text.strip():
            return []

        _t = time.perf_counter()
        audio_16k = _resample(audio, sample_rate, ALIGNER_SR)
        logger.debug("align.resample: %.3fs", time.perf_counter() - _t)

        _t = time.perf_counter()
        inputs = self.processor(
            audio_16k,
            sampling_rate=ALIGNER_SR,
            return_tensors="np",
            padding=True,
        )

        logits = self.session.run(None, {"input_values": inputs.input_values})[0]
        logger.debug("align.w2v2_onnx: %.3fs", time.perf_counter() - _t)

        _t = time.perf_counter()
        result = _ctc_word_alignment(logits, text, self.processor)
        logger.debug("align.ctc_decode: %.3fs", time.perf_counter() - _t)
        return result


# ── Resampling ──────────────────────────────────────────────────────────────


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """FFT-based resample to avoid aliasing artifacts."""
    num_samples = int(round(len(audio) * target_sr / orig_sr))
    return scipy.signal.resample(audio, num_samples)


# ── CTC forced alignment ────────────────────────────────────────────────────


def _ctc_word_alignment(
    logits: np.ndarray,
    text: str,
    processor: Wav2Vec2Processor,
) -> list[dict]:
    """CTC forced alignment against known text → per-word frame ranges."""

    blank_id = processor.tokenizer.pad_token_id
    vocab = processor.tokenizer.get_vocab()
    id2char = {v: k for k, v in vocab.items()}

    ids = np.argmax(logits[0], axis=-1)

    # Collapse consecutive duplicates, remove blanks, track frame ranges
    segments: list[tuple[int, int, int]] = []
    prev = blank_id
    seg_start: int | None = None

    for t, cid in enumerate(ids):
        if cid == blank_id:
            if prev != blank_id and seg_start is not None:
                segments.append((prev, seg_start, t))
                seg_start = None
        else:
            if cid != prev:
                if prev != blank_id and seg_start is not None:
                    segments.append((prev, seg_start, t))
                seg_start = t
        prev = cid

    if prev != blank_id and seg_start is not None:
        segments.append((prev, seg_start, len(ids)))

    if not segments:
        return []

    recognised = "".join(id2char.get(s[0], "") for s in segments).upper()

    # Build clean reference.  Keep every token so ``clean_words`` stays index-
    # aligned with ``words``; tokens that contain no alphanumeric characters
    # (e.g. em-dashes) become empty strings and contribute zero characters to
    # ``clean_ref``.
    words = text.split()
    clean_words: list[str] = []
    for w in words:
        cleaned = "".join(c for c in w if c.isalnum())
        clean_words.append(cleaned.upper())

    clean_ref = "".join(clean_words)

    # Sanity check: if recognised text is too short, fall back to proportional
    if len(recognised) < len(clean_ref) * 0.5:
        logger.warning(
            "CTC alignment too short (%d chars vs %d expected); falling back to proportional timing",
            len(recognised),
            len(clean_ref),
        )
        return _proportional_timestamps(text, len(ids) * FRAME_DURATION_MS)

    alignment = _character_align(recognised, clean_ref)

    return _assign_frames_to_words(segments, alignment, words, clean_words, len(ids))


def _character_align(recognised: str, reference: str) -> list[int | None]:
    """Needleman-Wunsch alignment of recognised chars to reference chars."""
    m, n = len(recognised), len(reference)
    if m == 0 or n == 0:
        return []

    score = np.zeros((m + 1, n + 1), dtype=np.int32)
    score[0, :] = np.arange(n + 1)
    score[:, 0] = np.arange(m + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if recognised[i - 1] == reference[j - 1] else 1
            score[i, j] = min(
                score[i - 1, j] + 1,
                score[i, j - 1] + 1,
                score[i - 1, j - 1] + cost,
            )

    alignment: list[int | None] = []
    i, j = m, n
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and score[i, j]
            == score[i - 1, j - 1] + (0 if recognised[i - 1] == reference[j - 1] else 1)
        ):
            alignment.append(j - 1 if recognised[i - 1] == reference[j - 1] else None)
            i -= 1
            j -= 1
        elif i > 0 and score[i, j] == score[i - 1, j] + 1:
            alignment.append(None)
            i -= 1
        else:
            j -= 1

    alignment.reverse()
    return alignment


def _assign_frames_to_words(
    segments: list[tuple[int, int, int]],
    alignment: list[int | None],
    words: list[str],
    clean_words: list[str],
    total_frames: int,
) -> list[dict]:
    """Map CTC segments → alignment → word timestamps."""

    word_timestamps: list[dict] = []
    word_idx = 0
    char_in_word = 0
    word_start_global: int | None = None

    for seg_idx in range(min(len(segments), len(alignment))):
        # Emit zero-duration entries for non-alphanumeric tokens (e.g. em-dashes)
        # so they stay index-aligned with the recognised words.
        while word_idx < len(clean_words) and len(clean_words[word_idx]) == 0:
            zero_ms = word_timestamps[-1]["end_ms"] if word_timestamps else 0.0
            word_timestamps.append(
                {"word": words[word_idx], "start_ms": zero_ms, "end_ms": zero_ms}
            )
            word_idx += 1

        if word_idx >= len(clean_words):
            break

        ref_char_idx = alignment[seg_idx]
        if ref_char_idx is None:
            continue

        word_len = len(clean_words[word_idx])

        if char_in_word == 0:
            word_start_global = segments[seg_idx][1]

        char_in_word += 1

        if char_in_word >= word_len:
            word_end_frame = segments[seg_idx][2]
            word_timestamps.append(
                {
                    "word": words[word_idx],
                    "start_ms": word_start_global * FRAME_DURATION_MS,
                    "end_ms": word_end_frame * FRAME_DURATION_MS,
                }
            )
            word_idx += 1
            char_in_word = 0
            word_start_global = None

    # Flush remaining
    if char_in_word > 0 and word_idx < len(clean_words):
        final_frame = segments[-1][2] if segments else total_frames
        word_timestamps.append(
            {
                "word": words[word_idx],
                "start_ms": word_start_global * FRAME_DURATION_MS,
                "end_ms": final_frame * FRAME_DURATION_MS,
            }
        )
        word_idx += 1

    # Flush any trailing zero-length words (non-alnum tokens after the last
    # recognised word).
    while word_idx < len(clean_words) and len(clean_words[word_idx]) == 0:
        zero_ms = word_timestamps[-1]["end_ms"] if word_timestamps else 0.0
        word_timestamps.append(
            {"word": words[word_idx], "start_ms": zero_ms, "end_ms": zero_ms}
        )
        word_idx += 1

    # Any words we couldn't assign — append at end
    final_ms = total_frames * FRAME_DURATION_MS
    while word_idx < len(words):
        start_ms = word_timestamps[-1]["end_ms"] if word_timestamps else final_ms
        word_timestamps.append(
            {"word": words[word_idx], "start_ms": start_ms, "end_ms": final_ms}
        )
        word_idx += 1

    return word_timestamps


def _proportional_timestamps(text: str, total_duration_ms: float) -> list[dict]:
    """Fallback: distribute time across words by character count."""
    words = text.split()
    if not words:
        return []

    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []

    timestamps: list[dict] = []
    current_ms = 0.0
    for word in words:
        word_duration = (len(word) / total_chars) * total_duration_ms
        timestamps.append(
            {
                "word": word,
                "start_ms": current_ms,
                "end_ms": current_ms + word_duration,
            }
        )
        current_ms += word_duration
    return timestamps

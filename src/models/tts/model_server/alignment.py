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
        w2v2_threads: intra_op_num_threads for the ONNX session. Kept
            explicit for the same reason as Kokoro's (T430536): the ORT
            default (0) sizes the thread pool from the host's cores, not
            the pod's cgroup quota.
    """

    def __init__(self, model_dir: str, w2v2_threads: int = 1):
        self.model_path = Path(model_dir) / "model.onnx"
        self.processor_path = Path(model_dir) / "processor"

        if not self.model_path.exists() or not self.processor_path.exists():
            raise FileNotFoundError(
                f"Wav2Vec2 model or processor not found at {model_dir}"
            )

        logger.info(
            "Loading Wav2Vec2-CTC ONNX session (intra_op_num_threads=%d)...",
            w2v2_threads,
        )
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = w2v2_threads
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
    """Map CTC segments -> alignment -> word timestamps.

    Each entry of ``alignment`` says which REFERENCE character the
    corresponding recognised segment matched (or ``None`` for an
    insertion). We attribute segments to words through that index, so a
    run of characters the recogniser missed cannot shift the words that
    follow it: the mapping re-synchronises at the next matched character.

    Walking words and segments in lockstep instead (the original
    approach) meant one missed run displaced every later word and the
    tail words ran out of segments, collapsing onto a single frame. On a
    two-minute section that produced captions up to a second behind the
    audio plus a cluster of zero-length cues (T436758).

    Words with no matched segment (missed by the recogniser, or tokens
    with no alphanumeric characters such as em-dashes) are interpolated
    between their anchored neighbours in proportion to their length.
    Those timings are estimates, not measurements, but they are
    monotonic and non-degenerate, which cue-stepping players require.

    Guarantees, asserted in the tests:
      * one timestamp per input word, in input order (a word is omitted
        only where the audio leaves literally no room for a cue);
      * ``start_ms`` non-decreasing; ``end_ms`` > ``start_ms``;
      * output for a fully matched alignment is unchanged from the
        lockstep implementation (to within one frame on ``end_ms``).
    """
    if not words:
        return []

    # Reference character index -> index of the word that owns it.
    char_owner: list[int] = []
    for word_idx, clean in enumerate(clean_words):
        char_owner.extend([word_idx] * len(clean))

    # Frame span of the segments matched to each word.
    spans: dict[int, tuple[int, int]] = {}
    for seg_idx in range(min(len(segments), len(alignment))):
        ref_char_idx = alignment[seg_idx]
        if ref_char_idx is None or not 0 <= ref_char_idx < len(char_owner):
            continue  # insertion, or an index past the reference text
        word_idx = char_owner[ref_char_idx]
        _symbol, frame_start, frame_end = segments[seg_idx]
        if word_idx in spans:
            known_start, known_end = spans[word_idx]
            spans[word_idx] = (min(known_start, frame_start), max(known_end, frame_end))
        else:
            spans[word_idx] = (frame_start, frame_end)

    # Keep only anchors that advance: a word whose matched frames start
    # before the previous anchor ended is a mis-match, not a measurement.
    timestamps: list[dict | None] = [None] * len(words)
    last_end_frame = -1
    for word_idx in sorted(spans):
        frame_start, frame_end = spans[word_idx]
        if frame_start < last_end_frame:
            continue
        timestamps[word_idx] = {
            "word": words[word_idx],
            "start_ms": frame_start * FRAME_DURATION_MS,
            "end_ms": max(
                frame_end * FRAME_DURATION_MS,
                frame_start * FRAME_DURATION_MS + FRAME_DURATION_MS,
            ),
        }
        last_end_frame = frame_end

    _interpolate_unanchored(timestamps, words, clean_words, total_frames)
    # A cue with no duration cannot be rendered (players skip it), so it is
    # dropped rather than emitted. This only happens where the audio leaves
    # literally no room, e.g. punctuation between two adjacent measurements.
    return [t for t in timestamps if t is not None and t["end_ms"] > t["start_ms"]]


def _interpolate_unanchored(
    timestamps: list[dict | None],
    words: list[str],
    clean_words: list[str],
    total_frames: int,
) -> None:
    """Fill runs of unanchored words in place, between their neighbours.

    Unanchored words are those the recogniser missed, plus tokens with no
    alphanumeric characters (em-dashes, bare quotes) which produce no
    reference characters to match. They are laid out across the gap
    between the surrounding anchors in proportion to their length.

    Each cue gets at least one frame, and cues never overlap: a player
    highlighting two words at once is as wrong as one highlighting none.
    When the gap is too narrow to give every word a frame, the run
    borrows from the preceding anchor's END (shortening a measured cue by
    a few frames is imperceptible, and the alternatives are overlapping
    or degenerate cues). Anchor START times are never moved: they are the
    measurements this whole module exists to produce.
    """
    total_ms = float(total_frames * FRAME_DURATION_MS)
    idx = 0
    while idx < len(timestamps):
        if timestamps[idx] is not None:
            idx += 1
            continue

        run_end = idx
        while run_end < len(timestamps) and timestamps[run_end] is None:
            run_end += 1

        left = timestamps[idx - 1] if idx > 0 else None
        right = timestamps[run_end] if run_end < len(timestamps) else None
        window_start = float(left["end_ms"]) if left else 0.0
        window_end = float(right["start_ms"]) if right else total_ms
        if window_end < window_start:
            window_end = window_start

        run_length = run_end - idx
        needed = run_length * FRAME_DURATION_MS
        if window_end - window_start < needed:
            # Borrow a few frames so every word keeps a visible cue.
            # Prefer shortening the preceding anchor's END; if there is no
            # preceding anchor (a missed word at the very start), delay the
            # following anchor's START instead. Either way the change is
            # bounded by the run length and never inverts a cue.
            if left is not None:
                floor_ms = float(left["start_ms"]) + FRAME_DURATION_MS
                window_start = max(floor_ms, window_end - needed)
                left["end_ms"] = window_start
            elif right is not None:
                ceiling_ms = float(right["end_ms"]) - FRAME_DURATION_MS
                window_end = min(ceiling_ms, window_start + needed)
                if window_end < window_start:
                    window_end = window_start
                right["start_ms"] = window_end

        weights = [max(len(clean_words[i]), 1) for i in range(idx, run_end)]
        weight_total = sum(weights)
        span = max(window_end - window_start, 0.0)
        cursor = window_start
        for i, weight in zip(range(idx, run_end), weights):
            share = span * weight / weight_total
            step = max(share, float(FRAME_DURATION_MS))
            timestamps[i] = {
                "word": words[i],
                "start_ms": cursor,
                "end_ms": cursor + step,
            }
            cursor += step
        # Last resort for a gap so tight that even borrowing left no room:
        # trim the run back so it cannot cross the next measurement.
        if right is not None and cursor > right["start_ms"]:
            _trim_run(timestamps, idx, run_end, float(right["start_ms"]))
        idx = run_end


def _trim_run(
    timestamps: list[dict | None], start_idx: int, stop_idx: int, limit_ms: float
) -> None:
    """Squeeze cues [start_idx, stop_idx) so none passes ``limit_ms``."""
    run = [t for t in timestamps[start_idx:stop_idx] if t is not None]
    if not run:
        return
    begin = float(run[0]["start_ms"])
    available = max(limit_ms - begin, 0.0)
    slice_ms = available / len(run) if available else 0.0
    cursor = begin
    for t in run:
        t["start_ms"] = cursor
        t["end_ms"] = cursor + slice_ms
        cursor += slice_ms


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

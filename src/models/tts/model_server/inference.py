"""
Kokoro TTS + Wav2Vec2 forced alignment inference pipeline.

This is the core inference engine extracted from v0 ``wiki_tts/worker.py``.
It takes pre-normalized text segments, runs TTS and alignment on each, and
returns concatenated float32 PCM audio with accumulated word-level timestamps.

A summary on the v0 implementation can be seen here:
https://phabricator.wikimedia.org/T424378#12068767
"""

import logging

import numpy as np
import onnxruntime as ort
from alignment import Aligner

logger = logging.getLogger(__name__)

FADE_LEN = 120  # samples for crossfade envelope
MAX_SEGMENT_CHARS = 800  # Kokoro's practical input-length limit


class TTSInferencePipeline:
    """
    Thin wrapper around Kokoro ONNX + Wav2Vec2 CTC aligner.

    Models are loaded once at construction time (triggered by the KServe
    startup hook).  ``.predict()`` is called per inference request.
    """

    def __init__(
        self,
        kokoro_model: str,
        kokoro_voices: str,
        wav2vec2_model_dir: str,
        kokoro_threads: int = 2,
    ):
        logger.info("Loading Kokoro ONNX model from %s...", kokoro_model)
        from kokoro_onnx import Kokoro

        # kokoro-onnx's default constructor leaves intra_op_num_threads at 0,
        # which makes ONNX Runtime size its thread pool from the HOST's core
        # count (96 on our k8s nodes), not the pod's cgroup quota (8 CPUs).
        # ~96 threads contending for 8 CPUs means coordination dominates the
        # small per-op work: synthesis ran ~10x slower than real-time.
        # An explicit thread count fixes it (RTF 4.96 -> 0.46 at intra=2 on an
        # 8-CPU pod). Graph optimization is not a factor: ORT_ENABLE_ALL is
        # already the onnxruntime default, verified to have no effect.
        so = ort.SessionOptions()
        so.intra_op_num_threads = kokoro_threads
        so.inter_op_num_threads = 1
        session = ort.InferenceSession(
            kokoro_model, so, providers=["CPUExecutionProvider"]
        )
        self.kokoro = Kokoro.from_session(session, kokoro_voices)
        self.sample_rate = 24000

        logger.info("Loading Wav2Vec2 aligner from %s...", wav2vec2_model_dir)
        self.aligner = Aligner(wav2vec2_model_dir)

    def predict(
        self,
        segments: list[dict],
        default_voice: str = "af_heart",
        default_speed: float = 1.0,
        default_lang: str = "en-us",
    ) -> dict:
        """
        Generate concatenated audio + word timestamps for a sequence of text segments.

        Segments are crossfaded together for natural-sounding transitions.
        Timestamps are accumulated so they refer to positions in the final
        concatenated audio.

        Each segment's ``text`` should be pre-chunked by the caller (e.g. via
        ``_split_text``) to stay under ~800 characters, the practical input
        limit for Kokoro.  Longer text may be silently truncated by the model.

        Args:
            segments: List of ``{"text": str, "voice"?: str, "speed"?: float, "lang"?: str}``.
            default_voice: Voice to use for segments that don't specify one.
            default_speed: Speaking rate (1.0 = normal).
            default_lang: Language code passed to Kokoro.

        Returns:
            ``{"audio": np.ndarray (float32), "sample_rate": int, "timestamps": list[dict]}``
        """
        audio_chunks: list[np.ndarray] = []
        all_timestamps: list[dict] = []
        current_time_ms = 0.0

        fade_out = np.linspace(1, 0, FADE_LEN, dtype=np.float32)
        fade_in = np.linspace(0, 1, FADE_LEN, dtype=np.float32)
        prev_tail: np.ndarray | None = None

        for i, seg in enumerate(segments):
            text = seg["text"]
            voice = seg.get("voice", default_voice)
            speed = seg.get("speed", default_speed)
            lang = seg.get("lang", default_lang)

            chunk_audio, _ = self.kokoro.create(
                text, voice=voice, speed=speed, lang=lang
            )
            # Defensive copy: kokoro.create() may return a view, cached buffer,
            # or read-only array, in-place crossfade ops must own the memory.
            chunk_audio = np.array(chunk_audio, dtype=np.float32, copy=True)

            # Word-level alignment
            chunk_ts = self.aligner.align(chunk_audio, self.sample_rate, text)
            for t in chunk_ts:
                t["start_ms"] += current_time_ms
                t["end_ms"] += current_time_ms
                all_timestamps.append(t)

            # Crossfade between consecutive chunks.
            # Chunks shorter than FADE_LEN (~5 ms) skip crossfade as they're
            # too short for the envelope and would produce misaligned tails.
            if len(chunk_audio) < FADE_LEN:
                audio_chunks.append(chunk_audio)
                contributed_samples = len(chunk_audio)
                prev_tail = None  # break the crossfade chain
            elif len(segments) == 1:
                audio_chunks.append(chunk_audio)
                contributed_samples = len(chunk_audio)
            elif i == 0:
                chunk_audio[-FADE_LEN:] *= fade_out
                prev_tail = chunk_audio[-FADE_LEN:].copy()
                audio_chunks.append(chunk_audio[:-FADE_LEN])
                contributed_samples = len(chunk_audio) - FADE_LEN
            elif i < len(segments) - 1:
                if prev_tail is not None:
                    chunk_audio[:FADE_LEN] *= fade_in
                    chunk_audio[:FADE_LEN] += prev_tail
                chunk_audio[-FADE_LEN:] *= fade_out
                prev_tail = chunk_audio[-FADE_LEN:].copy()
                audio_chunks.append(chunk_audio[:-FADE_LEN])
                contributed_samples = len(chunk_audio) - FADE_LEN
            else:
                if prev_tail is not None:
                    chunk_audio[:FADE_LEN] *= fade_in
                    chunk_audio[:FADE_LEN] += prev_tail
                audio_chunks.append(chunk_audio)
                contributed_samples = len(chunk_audio)

            # Advance the timestamp clock by what was actually contributed
            # to the output (post-crossfade), not the full chunk length.
            current_time_ms += (contributed_samples / self.sample_rate) * 1000

        audio = (
            np.concatenate(audio_chunks)
            if audio_chunks
            else np.array([], dtype=np.float32)
        )

        return {
            "audio": audio,
            "sample_rate": self.sample_rate,
            "timestamps": all_timestamps,
        }

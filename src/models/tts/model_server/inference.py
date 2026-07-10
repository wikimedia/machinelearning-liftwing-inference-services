"""
Kokoro TTS + Wav2Vec2 forced alignment inference pipeline.

This is the core inference engine extracted from v0 ``wiki_tts/worker.py``.
It takes pre-normalized text segments, runs TTS and alignment on each, and
returns concatenated float32 PCM audio with accumulated word-level timestamps.

A summary on the v0 implementation can be seen here:
https://phabricator.wikimedia.org/T424378#12068767
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from alignment import Aligner, _proportional_timestamps

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
        w2v2_threads: int = 1,
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

        # Concurrent predict() calls corrupt synthesis output: four
        # concurrent requests with identical input produced 14.49-26.28s
        # of audio; the same four serialized produced exactly 16.81s each.
        # Suspected shared mutable state in kokoro-onnx's espeak phonemizer
        # (espeak-ng is not thread-safe). Serialize the pipeline until
        # root-caused. See T430536.
        self._synth_lock = threading.Lock()

        logger.info("Loading Wav2Vec2 aligner from %s...", wav2vec2_model_dir)
        self.aligner = Aligner(wav2vec2_model_dir, w2v2_threads=w2v2_threads)

    def predict(
        self,
        segments: list[dict],
        default_voice: str = "af_heart",
        default_speed: float = 1.0,
        default_lang: str = "en-us",
        timestamps_mode: str = "full",
    ) -> dict:
        """
        Serialized entry point for the inference pipeline.

        Concurrent execution corrupts synthesis output (see the
        ``_synth_lock`` comment in ``__init__``), so requests are
        processed one at a time; concurrent callers queue on the lock.
        """
        with self._synth_lock:
            return self._predict_impl(
                segments,
                default_voice=default_voice,
                default_speed=default_speed,
                default_lang=default_lang,
                timestamps_mode=timestamps_mode,
            )

    def _predict_impl(
        self,
        segments: list[dict],
        default_voice: str = "af_heart",
        default_speed: float = 1.0,
        default_lang: str = "en-us",
        timestamps_mode: str = "full",
    ) -> dict:
        """
        Generate concatenated audio + word timestamps for a sequence of text segments.

        Segments are crossfaded together for natural-sounding transitions.
        Timestamps are accumulated so they refer to positions in the final
        concatenated audio.

        Alignment of chunk i runs in a single background worker while this
        thread synthesizes chunk i+1, hiding most alignment time on
        multi-segment requests. Crossfade bookkeeping and the timestamp
        clock stay on this thread in segment order; each chunk's timestamp
        offset is captured at dispatch time, and alignment runs on a
        snapshot of the raw chunk audio (the crossfade mutates the original
        in place afterwards).

        Each segment's ``text`` should be pre-chunked by the caller (e.g. via
        ``_split_text``) to stay under ~800 characters, the practical input
        limit for Kokoro.  Longer text may be silently truncated by the model.

        Args:
            segments: List of ``{"text": str, "voice"?: str, "speed"?: float, "lang"?: str}``.
            default_voice: Voice to use for segments that don't specify one.
            default_speed: Speaking rate (1.0 = normal).
            default_lang: Language code passed to Kokoro.
            timestamps_mode: "full" (CTC forced alignment, default),
                "proportional" (char-count-weighted timing, near-zero
                cost), or "none" (skip timestamps entirely).

        Returns:
            ``{"audio": np.ndarray (float32), "sample_rate": int, "timestamps": list[dict]}``
        """
        audio_chunks: list[np.ndarray] = []
        all_timestamps: list[dict] = []
        current_time_ms = 0.0
        kokoro_s = 0.0
        align_s = 0.0
        t_request = time.perf_counter()

        # Overlap: alignment of chunk i runs in a single background worker
        # while the main thread synthesizes chunk i+1. Safe because the
        # thread-safety corruption is confined to kokoro's synthesis path
        # (verified: 4x concurrent align during locked synth -> uniform
        # output, T430536); synthesis itself stays on this thread only.
        # max_workers=1: align-vs-align concurrency is untested; one worker
        # also keeps completion order deterministic.
        # Thread budget: overlapped stages contend for the pod's CPUs, so
        # fewer aligner threads can be net-faster (hidden time is cheap,
        # exposed synthesis time is not): swept kokoro/w2v2 8/4=RTF 0.34,
        # 8/2=0.32, 6/2=0.34 on a 6-segment request. W2V2_THREADS=2 is set
        # in the deployment config.
        align_pool = (
            ThreadPoolExecutor(max_workers=1) if timestamps_mode == "full" else None
        )
        pending: list[tuple] = []  # (future, offset_ms) in segment order

        def _timed_align(a: np.ndarray, sr: int, txt: str):
            t0 = time.perf_counter()
            ts = self.aligner.align(a, sr, txt)
            return ts, time.perf_counter() - t0

        fade_out = np.linspace(1, 0, FADE_LEN, dtype=np.float32)
        fade_in = np.linspace(0, 1, FADE_LEN, dtype=np.float32)
        prev_tail: np.ndarray | None = None

        for i, seg in enumerate(segments):
            text = seg["text"]
            voice = seg.get("voice", default_voice)
            speed = seg.get("speed", default_speed)
            lang = seg.get("lang", default_lang)

            _t = time.perf_counter()
            chunk_audio, _ = self.kokoro.create(
                text, voice=voice, speed=speed, lang=lang
            )
            kokoro_s += time.perf_counter() - _t
            # Defensive copy: kokoro.create() may return a view, cached buffer,
            # or read-only array, in-place crossfade ops must own the memory.
            chunk_audio = np.array(chunk_audio, dtype=np.float32, copy=True)

            # Word-level timestamps, per requested mode:
            #   full         — CTC forced alignment on a snapshot of the raw
            #                  chunk (the crossfade below mutates chunk_audio
            #                  in place), dispatched to the background worker
            #                  so it overlaps the next chunk's synthesis.
            #   proportional — char-count-weighted timing, computed inline
            #                  (microseconds; no model call).
            #   none         — skip entirely; audio-only batch generation
            #                  pays synthesis cost alone.
            # The timestamp offset is captured NOW in all modes, before the
            # clock advances.
            if timestamps_mode == "full":
                pending.append(
                    (
                        align_pool.submit(
                            _timed_align, chunk_audio.copy(), self.sample_rate, text
                        ),
                        current_time_ms,
                    )
                )
            elif timestamps_mode == "proportional":
                chunk_ms = (len(chunk_audio) / self.sample_rate) * 1000
                for t in _proportional_timestamps(text, chunk_ms):
                    t["start_ms"] += current_time_ms
                    t["end_ms"] += current_time_ms
                    all_timestamps.append(t)
            # "none": nothing to do

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

        # Gather alignments in segment order and apply per-chunk offsets.
        # fut.result() re-raises any alignment exception on this thread, so
        # the caller's error handling is unchanged. (Non-"full" modes have
        # no pool and produced their timestamps inline; align_s stays 0.)
        if align_pool is not None:
            for fut, offset_ms in pending:
                chunk_ts, spent = fut.result()
                align_s += spent
                for t in chunk_ts:
                    t["start_ms"] += offset_ms
                    t["end_ms"] += offset_ms
                    all_timestamps.append(t)
            align_pool.shutdown(wait=False)

        audio = (
            np.concatenate(audio_chunks)
            if audio_chunks
            else np.array([], dtype=np.float32)
        )

        total_s = time.perf_counter() - t_request
        audio_s = len(audio) / self.sample_rate if len(audio) else 0.0
        logger.info(
            "predict: %d segments -> %.2fs audio in %.2fs "
            "(kokoro %.2fs, align %.2fs, RTF %.2f)",
            len(segments),
            audio_s,
            total_s,
            kokoro_s,
            align_s,
            (total_s / audio_s) if audio_s else -1.0,
        )

        return {
            "audio": audio,
            "sample_rate": self.sample_rate,
            "timestamps": all_timestamps,
        }

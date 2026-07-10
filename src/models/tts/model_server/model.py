import asyncio
import base64
import logging
import os
import time

import kserve
import numpy as np
from inference import MAX_SEGMENT_CHARS, TTSInferencePipeline
from kserve.errors import InferenceError, InvalidInput, ModelMissingError

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)
logger = logging.getLogger(__name__)


class TTSModel(kserve.Model):
    def __init__(
        self,
        name: str,
        kokoro_model: str,
        kokoro_voices: str,
        wav2vec2_model_dir: str,
        kokoro_threads: int = 2,
        w2v2_threads: int = 1,
    ) -> None:
        super().__init__(name)
        self.name = name
        self.kokoro_model = kokoro_model
        self.kokoro_voices = kokoro_voices
        self.wav2vec2_model_dir = wav2vec2_model_dir
        self.kokoro_threads = kokoro_threads
        self.w2v2_threads = w2v2_threads
        self.pipeline: TTSInferencePipeline | None = None
        self.ready = False

    def load(self) -> None:
        """
        Load Kokoro ONNX and Wav2Vec2 ONNX models into memory.

        Called once by KServe at startup.  Sets ``self.ready = True`` on
        success so the readiness probe returns 200.

        Raises:
            ModelMissingError: If either model fails to load.
        """
        try:
            logger.info("Loading inference pipeline (Kokoro + Wav2Vec2)...")
            self.pipeline = TTSInferencePipeline(
                self.kokoro_model,
                self.kokoro_voices,
                self.wav2vec2_model_dir,
                kokoro_threads=self.kokoro_threads,
                w2v2_threads=self.w2v2_threads,
            )
            # Warm-up: the first synthesis pays one-time costs (espeak/G2P
            # init, ONNX first-inference). Pay them at startup so the first
            # real request doesn't. A warm-up failure fails load()
            # intentionally: a server that cannot synthesize should not go
            # ready.
            logger.info("Warming up TTS pipeline...")
            _t = time.perf_counter()
            self.pipeline.predict([{"text": "Warm up."}])
            logger.info("Warm-up complete in %.2fs", time.perf_counter() - _t)
            self.ready = True
            logger.info("Model loaded successfully!")
        except Exception as e:
            error_message = f"Failed to load TTS models. Reason: {e}"
            logger.critical(error_message)
            raise ModelMissingError(error_message)

    def preprocess(self, payload: dict, headers: dict[str, str] = None) -> dict:
        """
        Validate the incoming JSON payload and apply per-request defaults.

        Args:
            payload: The raw JSON body from the KServe request.  Must contain a
                non-empty ``segments`` list; each segment requires a non-empty
                ``text`` field.  Optional top-level keys ``default_voice``,
                ``default_speed``, and ``default_lang`` override the built-in
                defaults.

        Returns:
            A dict with keys ``segments``, ``default_voice``, ``default_speed``,
            and ``default_lang``, ready to pass to :meth:`predict`.

        Raises:
            InvalidInput: If ``segments`` is missing, empty, or contains a
                segment with a missing or blank ``text`` field.
        """
        segments = payload.get("segments")
        if not segments:
            error_message = "`segments` is required and must be a non-empty list."
            logger.error(error_message)
            raise InvalidInput(error_message)

        if not isinstance(segments, list):
            raise InvalidInput("`segments` must be a list.")

        for i, seg in enumerate(segments):
            if "text" not in seg:
                raise InvalidInput(f"Segment {i} is missing `text`.")
            if not isinstance(seg["text"], str) or not seg["text"].strip():
                raise InvalidInput(f"Segment {i} has empty or non-string `text`.")
            # Kokoro has a practical input-length limit (~800 chars).  The
            # orchestrator is expected to pre-chunk via _split_text().  This
            # is a non-fatal warning as the model may silently truncate.
            if len(seg["text"]) > MAX_SEGMENT_CHARS:
                logger.warning(
                    "Segment %d is %d chars (max recommended: %d); "
                    "text may be truncated by Kokoro.  Pre-chunk via _split_text().",
                    i,
                    len(seg["text"]),
                    MAX_SEGMENT_CHARS,
                )
            # MIN_TEXT_LENGTH filtering (v0's 50-char gate) is the
            # orchestrator's responsibility since it owns text cleaning.

        return {
            "segments": segments,
            "default_voice": payload.get("default_voice", "af_heart"),
            "default_speed": float(payload.get("default_speed", 1.0)),
            "default_lang": payload.get("default_lang", "en-us"),
        }

    async def predict(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """
        Run TTS inference + forced alignment on the preprocessed segments.

        Offloaded to a threadpool executor so the event loop stays responsive
        to liveness / readiness probes during long synthesis runs.

        Args:
            inputs: Preprocessed dict from :meth:`preprocess` with keys
                ``segments``, ``default_voice``, ``default_speed``, and
                ``default_lang``.

        Returns:
            ``{"audio": np.ndarray (float32), "sample_rate": int, "timestamps": list[dict]}``

        Raises:
            InferenceError: If TTS synthesis or alignment fails for any segment.
        """
        try:
            logger.info(
                "Running inference on %d segments (voice=%s)...",
                len(inputs["segments"]),
                inputs["default_voice"],
            )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.pipeline.predict(
                    segments=inputs["segments"],
                    default_voice=inputs["default_voice"],
                    default_speed=inputs["default_speed"],
                    default_lang=inputs["default_lang"],
                ),
            )
            return result

        except Exception as e:
            error_message = f"Error during TTS inference: {e}"
            logger.error(error_message)
            raise InferenceError(error_message)

    def postprocess(self, inputs: dict, headers: dict[str, str] = None) -> dict:
        """
        Encode float32 PCM audio as base64 and attach metadata.

        Args:
            inputs: Dict from :meth:`predict` with keys ``audio``
                (float32 np.ndarray), ``sample_rate`` (int), and ``timestamps``
                (list[dict]).

        Returns:
            ``{"audio_b64": str, "sample_rate": int, "duration_ms": float, "timestamps": list[dict]}``

        Raises:
            InferenceError: If base64 encoding fails.
        """
        try:
            audio: np.ndarray = inputs["audio"]
            sample_rate: int = inputs["sample_rate"]
            timestamps: list[dict] = inputs["timestamps"]

            audio_bytes = audio.tobytes()
            duration_ms = (len(audio) / sample_rate) * 1000

            return {
                "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                "sample_rate": sample_rate,
                "duration_ms": round(duration_ms, 1),
                "timestamps": timestamps,
            }

        except Exception as e:
            error_message = f"Error during post-processing: {e}"
            logger.error(error_message)
            raise InferenceError(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "tts")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models")

    kokoro_model = os.environ.get(
        "KOKORO_MODEL", os.path.join(model_path, "kokoro", "kokoro-v1.0.onnx")
    )
    kokoro_voices = os.environ.get(
        "KOKORO_VOICES", os.path.join(model_path, "kokoro", "voices-v1.0.bin")
    )
    wav2vec2_model_dir = os.environ.get(
        "WAV2VEC2_MODEL_DIR", os.path.join(model_path, "wav2vec2")
    )
    kokoro_threads = int(os.environ.get("KOKORO_THREADS", "2"))
    w2v2_threads = int(os.environ.get("W2V2_THREADS", "1"))

    model = TTSModel(
        name=model_name,
        kokoro_model=kokoro_model,
        kokoro_voices=kokoro_voices,
        wav2vec2_model_dir=wav2vec2_model_dir,
        kokoro_threads=kokoro_threads,
        w2v2_threads=w2v2_threads,
    )

    model.load()
    kserve.ModelServer().start([model])

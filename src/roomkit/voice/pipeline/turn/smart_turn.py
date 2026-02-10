"""Audio-native turn detector using pipecat-ai/smart-turn ONNX model.

Analyzes prosody and intonation from raw speech audio to decide
whether a user's conversational turn is complete — without needing
text.  The model is a ~8M param Whisper Tiny encoder + classifier
(~8MB ONNX) that operates on the last 8 seconds of audio.

Requires optional dependencies::

    pip install roomkit[smart-turn]

This installs ``numpy``, ``onnxruntime``, and ``transformers``.

Model download
--------------
Download the ONNX model from HuggingFace (no auto-download)::

    # CPU (int8 quantized, ~8MB — recommended)
    wget https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx

    # GPU (fp32, ~32MB — use with provider="cuda")
    wget https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx

Quick start
-----------
::

    from roomkit.voice.pipeline import AudioPipelineConfig
    from roomkit.voice.pipeline.turn import SmartTurnConfig, SmartTurnDetector

    detector = SmartTurnDetector(SmartTurnConfig(
        model_path="smart-turn-v3.2-cpu.onnx",
        threshold=0.5,   # probability above which the turn is complete
    ))

    pipeline = AudioPipelineConfig(
        vad=my_vad,
        turn_detector=detector,
    )

The detector receives raw speech audio via ``TurnContext.audio_bytes``
(populated automatically by ``VoiceChannel`` from VAD SPEECH_END events).
When a turn is incomplete, audio accumulates across segments so the
model always sees the full conversation context (up to 8 seconds).

Tuning the threshold
--------------------
- **Lower values** (e.g. 0.3): more eager to complete turns, less waiting.
  Good for command-style interactions or noisy environments.
- **Higher values** (e.g. 0.7): waits longer for natural turn endings.
  Better for conversational, multi-sentence responses from the user.
- **Default (0.5)**: balanced for general conversational use.

See ``examples/voice_smart_turn.py`` for a full working example.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from roomkit.voice.pipeline.turn.base import TurnContext, TurnDecision, TurnDetector

logger = logging.getLogger("roomkit.voice.turn.smart_turn")

# Whisper Tiny expects 8-second chunks at 16 kHz
_CHUNK_SECONDS = 8
_EXPECTED_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = _CHUNK_SECONDS * _EXPECTED_SAMPLE_RATE


@dataclass
class SmartTurnConfig:
    """Configuration for :class:`SmartTurnDetector`.

    Attributes:
        model_path: Local path to the ONNX model file (required, no auto-download).
        threshold: Completion probability above which the turn is considered done.
        num_threads: Number of CPU threads for the ONNX runtime session.
        provider: ONNX execution provider (``"cpu"`` or ``"cuda"``).
        fallback_on_no_audio: When ``True`` (default), return ``is_complete=True``
            if no audio is available (e.g. continuous STT mode).  When ``False``,
            return ``is_complete=False``.
    """

    model_path: str
    threshold: float = 0.5
    num_threads: int = 1
    provider: str = "cpu"
    fallback_on_no_audio: bool = True


class SmartTurnDetector(TurnDetector):
    """Audio-native turn detector backed by a smart-turn ONNX model.

    Usage::

        from roomkit.voice.pipeline.turn import SmartTurnConfig, SmartTurnDetector

        detector = SmartTurnDetector(SmartTurnConfig(model_path="model.onnx"))
        pipeline_config = AudioPipelineConfig(turn_detector=detector, ...)
    """

    def __init__(self, config: SmartTurnConfig) -> None:
        if not config.model_path:
            raise ValueError("SmartTurnConfig.model_path must be a non-empty string")

        # Eagerly check that required packages are importable
        try:
            import numpy as _np  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SmartTurnDetector requires numpy. Install with: pip install roomkit[smart-turn]"
            ) from exc

        try:
            import onnxruntime as _ort  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SmartTurnDetector requires onnxruntime. "
                "Install with: pip install roomkit[smart-turn]"
            ) from exc

        self._config = config
        self._session = None  # lazy-init ONNX InferenceSession
        self._feature_extractor: Any = None  # lazy-init WhisperFeatureExtractor

    @property
    def name(self) -> str:
        return "SmartTurnDetector"

    # ------------------------------------------------------------------
    # Lazy initialisation (defers transformers import to first use)
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Lazily create the ONNX session and Whisper feature extractor."""
        if self._session is not None:
            return

        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if self._config.provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = self._config.num_threads
        opts.intra_op_num_threads = self._config.num_threads

        self._session = ort.InferenceSession(
            self._config.model_path,
            sess_options=opts,
            providers=providers,
        )

        from transformers import WhisperFeatureExtractor

        self._feature_extractor = WhisperFeatureExtractor(  # type: ignore[no-untyped-call]
            chunk_length=_CHUNK_SECONDS
        )

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pcm_int16_to_float32(pcm: bytes) -> Any:
        """Convert int16 PCM bytes to float32 numpy array in [-1, 1]."""
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        return samples

    @staticmethod
    def _truncate_or_pad(
        samples: Any,
        target_length: int,
    ) -> Any:
        """Keep last *target_length* samples; zero-pad at start if shorter."""
        import numpy as np

        if len(samples) >= target_length:
            return samples[-target_length:]
        pad = np.zeros(target_length - len(samples), dtype=samples.dtype)
        return np.concatenate([pad, samples])

    # ------------------------------------------------------------------
    # Core evaluate
    # ------------------------------------------------------------------

    def evaluate(self, context: TurnContext) -> TurnDecision:  # noqa: C901
        """Evaluate turn completion from audio.

        Falls back gracefully when audio is unavailable or on errors.
        """
        audio = context.audio_bytes

        # No audio available (e.g. continuous STT mode)
        if not audio:
            return TurnDecision(
                is_complete=self._config.fallback_on_no_audio,
                confidence=0.0,
                reason="no audio available",
            )

        try:
            self._ensure_initialized()
            if self._session is None or self._feature_extractor is None:
                raise RuntimeError("SmartTurnDetector failed to initialize")

            import numpy as np

            # Convert int16 PCM → float32
            samples = self._pcm_int16_to_float32(audio)

            # Resample if needed
            if context.audio_sample_rate != _EXPECTED_SAMPLE_RATE:
                ratio = _EXPECTED_SAMPLE_RATE / context.audio_sample_rate
                target_len = int(len(samples) * ratio)
                indices = np.linspace(0, len(samples) - 1, target_len).astype(int)
                samples = samples[indices]

            # Truncate/pad to 8s window
            samples = self._truncate_or_pad(samples, _CHUNK_SAMPLES)

            # Extract mel features
            features = self._feature_extractor(
                samples,
                sampling_rate=_EXPECTED_SAMPLE_RATE,
                return_tensors="np",
            )
            input_features = features["input_features"]  # (1, 80, T)

            # ONNX inference
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: input_features})
            logit = float(output[0][0][0])

            # Sigmoid
            prob = 1.0 / (1.0 + np.exp(-logit))
            is_complete = prob >= self._config.threshold

            return TurnDecision(
                is_complete=is_complete,
                confidence=float(prob) if is_complete else float(1.0 - prob),
                reason=f"smart-turn prob={prob:.3f} threshold={self._config.threshold}",
            )

        except Exception:
            logger.exception("SmartTurnDetector inference error — failing open")
            return TurnDecision(
                is_complete=True,
                confidence=0.1,
                reason="inference error — fail-open",
            )

    def close(self) -> None:
        """Release the ONNX session."""
        self._session = None
        self._feature_extractor = None

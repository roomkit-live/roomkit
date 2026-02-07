"""sherpa-onnx neural VAD provider (TEN-VAD / Silero VAD).

Uses sherpa-onnx's VoiceActivityDetector with frame-level
``is_speech_detected()`` plus our own state machine for instant
SPEECH_START events and pre-roll buffering.
"""

from __future__ import annotations

import logging
import struct
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType, VADProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger(__name__)


def _pcm_s16le_to_float32(data: bytes) -> list[float]:
    """Convert PCM signed 16-bit little-endian bytes to float32 list in [-1, 1]."""
    n = len(data) // 2
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    return [s / 32768.0 for s in samples]


@dataclass
class SherpaOnnxVADConfig:
    """Configuration for the sherpa-onnx VAD provider.

    Attributes:
        model: Path to the ``.onnx`` model file.
        model_type: Model architecture — ``"ten"`` for TEN-VAD or
            ``"silero"`` for Silero VAD.
        threshold: Speech probability threshold (0–1).
        silence_threshold_ms: Consecutive silence in ms to trigger SPEECH_END.
        min_speech_duration_ms: Minimum speech duration to emit; shorter
            segments are silently discarded.
        speech_pad_ms: Pre-roll buffer duration in ms.
        max_speech_duration: Maximum speech segment length in seconds before
            forcing a segment break inside sherpa.
        sample_rate: Expected audio sample rate.
        num_threads: Number of CPU threads for inference.
        provider: ONNX execution provider (``"cpu"`` or ``"cuda"``).
    """

    model: str = ""
    model_type: str = "ten"
    threshold: float = 0.5
    silence_threshold_ms: float = 500
    min_speech_duration_ms: float = 250
    speech_pad_ms: float = 300
    max_speech_duration: float = 20.0
    sample_rate: int = 16000
    num_threads: int = 1
    provider: str = "cpu"


class SherpaOnnxVADProvider(VADProvider):
    """Neural VAD provider using sherpa-onnx (TEN-VAD or Silero VAD).

    The detector is created lazily on the first call to :meth:`process`.
    sherpa-onnx must be installed (``pip install roomkit[sherpa-onnx]``).

    Parameters:
        config: Provider configuration.
    """

    def __init__(self, config: SherpaOnnxVADConfig) -> None:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx is required for SherpaOnnxVADProvider. "
                "Install it with: pip install roomkit[sherpa-onnx]"
            ) from exc

        self._config = config
        self._sherpa: Any = __import__("sherpa_onnx")
        self._detector: Any = None

        # State machine
        self._speaking = False
        self._silence_ms: float = 0.0
        self._speech_ms: float = 0.0
        self._speech_buf = bytearray()

        # Pre-roll buffer
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_ms: float = 0.0

    @property
    def name(self) -> str:
        return "SherpaOnnxVAD"

    def _ensure_detector(self) -> None:
        """Lazily create the sherpa-onnx VoiceActivityDetector."""
        if self._detector is not None:
            return

        cfg = self._config
        sherpa = self._sherpa

        vad_config = sherpa.VadModelConfig()

        if cfg.model_type == "silero":
            vad_config.silero_vad.model = cfg.model
            vad_config.silero_vad.threshold = cfg.threshold
            vad_config.silero_vad.max_speech_duration = cfg.max_speech_duration
        else:
            # Default to TEN-VAD
            vad_config.ten_vad.model = cfg.model
            vad_config.ten_vad.threshold = cfg.threshold
            vad_config.ten_vad.max_speech_duration = cfg.max_speech_duration

        vad_config.sample_rate = cfg.sample_rate
        vad_config.num_threads = cfg.num_threads
        vad_config.provider = cfg.provider

        self._detector = sherpa.VoiceActivityDetector(vad_config)
        logger.debug(
            "SherpaOnnxVAD: created detector model_type=%s model=%s",
            cfg.model_type,
            cfg.model,
        )

    def _frame_duration_ms(self, frame: AudioFrame) -> float:
        """Duration of a single frame in milliseconds."""
        n_samples = len(frame.data) // (frame.sample_width * frame.channels)
        return (n_samples / frame.sample_rate) * 1000.0

    def _push_pre_roll(self, data: bytes, duration_ms: float) -> None:
        """Maintain a rolling buffer of recent frames for pre-speech padding."""
        self._pre_roll.append(data)
        self._pre_roll_ms += duration_ms
        while self._pre_roll_ms > self._config.speech_pad_ms and len(self._pre_roll) > 1:
            removed = self._pre_roll.popleft()
            n_samples = len(removed) // 2  # int16
            self._pre_roll_ms -= (n_samples / self._config.sample_rate) * 1000.0

    def process(self, frame: AudioFrame) -> VADEvent | None:  # noqa: C901
        self._ensure_detector()

        duration_ms = self._frame_duration_ms(frame)

        # Feed audio to sherpa detector
        float_samples = _pcm_s16le_to_float32(frame.data)
        self._detector.accept_waveform(float_samples)

        # Drain completed segments to prevent unbounded memory
        while not self._detector.empty():
            self._detector.pop()

        is_speech = self._detector.is_speech_detected()

        if not self._speaking:
            # --- Idle state ---
            self._push_pre_roll(frame.data, duration_ms)

            if is_speech:
                self._speaking = True
                self._silence_ms = 0.0
                self._speech_ms = duration_ms
                # Start accumulating with pre-roll
                self._speech_buf = bytearray()
                for chunk in self._pre_roll:
                    self._speech_buf.extend(chunk)
                self._pre_roll.clear()
                self._pre_roll_ms = 0.0
                return VADEvent(type=VADEventType.SPEECH_START, confidence=1.0)
        else:
            # --- Speaking state ---
            self._speech_buf.extend(frame.data)
            self._speech_ms += duration_ms

            if is_speech:
                self._silence_ms = 0.0
            else:
                self._silence_ms += duration_ms

                if self._silence_ms >= self._config.silence_threshold_ms:
                    # Transition to idle
                    self._speaking = False
                    speech_ms = self._speech_ms
                    audio = bytes(self._speech_buf)

                    # Reset accumulators
                    self._speech_buf = bytearray()
                    self._speech_ms = 0.0
                    self._silence_ms = 0.0

                    if speech_ms >= self._config.min_speech_duration_ms:
                        return VADEvent(
                            type=VADEventType.SPEECH_END,
                            audio_bytes=audio,
                            duration_ms=speech_ms,
                        )
                    # Too short — discard silently

        return None

    def reset(self) -> None:
        """Reset all internal state."""
        self._speaking = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._speech_buf = bytearray()
        self._pre_roll.clear()
        self._pre_roll_ms = 0.0
        if self._detector is not None:
            self._detector.reset()

    def close(self) -> None:
        """Release resources."""
        if self._detector is not None:
            self._detector.flush()
            self._detector = None

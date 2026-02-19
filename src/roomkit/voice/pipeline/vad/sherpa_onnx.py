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


_DEBUG_SUMMARY_INTERVAL = 50  # frames (~1s at 20ms/frame)


def _rms_int16(data: bytes) -> float:
    """Compute RMS of int16 little-endian PCM data."""
    n_samples = len(data) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", data[: n_samples * 2])
    sum_sq = sum(s * s for s in samples)
    return float((sum_sq / n_samples) ** 0.5)


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
        threshold: Speech probability threshold (0–1).  Default 0.35
            works well with denoised audio; raise to 0.5 without denoiser.
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
    threshold: float = 0.35
    silence_threshold_ms: float = 500
    min_speech_duration_ms: float = 250
    speech_pad_ms: float = 300
    max_speech_duration: float = 20.0
    # Energy-based fast exit: if RMS drops below this threshold for
    # silence_threshold_ms, force SPEECH_END even if the model still
    # reports speech.  Addresses model inertia where is_speech_detected()
    # stays True on silence after speech.  Set to 0 to disable.
    energy_silence_rms: float = 20.0
    # sherpa-onnx internal hysteresis — keep low so is_speech_detected()
    # transitions quickly; our own silence_threshold_ms handles debounce.
    sherpa_min_silence_duration: float = 0.05
    sherpa_min_speech_duration: float = 0.1
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
        self._energy_silence_ms: float = 0.0
        self._speech_ms: float = 0.0
        self._speech_buf = bytearray()

        # Pre-roll buffer
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_ms: float = 0.0

        # Debug logging counters
        self._debug_frame_count = 0
        self._debug_rms_sum = 0.0
        self._debug_rms_max = 0.0
        self._debug_speech_count = 0

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
            vad_config.silero_vad.min_silence_duration = cfg.sherpa_min_silence_duration
            vad_config.silero_vad.min_speech_duration = cfg.sherpa_min_speech_duration
        else:
            # Default to TEN-VAD
            vad_config.ten_vad.model = cfg.model
            vad_config.ten_vad.threshold = cfg.threshold
            vad_config.ten_vad.max_speech_duration = cfg.max_speech_duration
            vad_config.ten_vad.min_silence_duration = cfg.sherpa_min_silence_duration
            vad_config.ten_vad.min_speech_duration = cfg.sherpa_min_speech_duration

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

        # Debug logging: accumulate stats and emit periodic summary
        if logger.isEnabledFor(logging.DEBUG):
            rms = _rms_int16(frame.data)
            self._debug_frame_count += 1
            self._debug_rms_sum += rms
            if rms > self._debug_rms_max:
                self._debug_rms_max = rms
            if is_speech:
                self._debug_speech_count += 1
            if self._debug_frame_count >= _DEBUG_SUMMARY_INTERVAL:
                avg = self._debug_rms_sum / self._debug_frame_count
                state = "speaking" if self._speaking else "idle"
                logger.debug(
                    "VAD: state=%s is_speech=%d/%d rms_avg=%.0f rms_max=%.0f"
                    " silence_ms=%.0f speech_ms=%.0f",
                    state,
                    self._debug_speech_count,
                    self._debug_frame_count,
                    avg,
                    self._debug_rms_max,
                    self._silence_ms,
                    self._speech_ms,
                )
                self._debug_frame_count = 0
                self._debug_rms_sum = 0.0
                self._debug_rms_max = 0.0
                self._debug_speech_count = 0

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
                return VADEvent(
                    type=VADEventType.SPEECH_START,
                    confidence=1.0,
                    audio_bytes=bytes(self._speech_buf),
                )
        else:
            # --- Speaking state ---
            self._speech_buf.extend(frame.data)
            self._speech_ms += duration_ms

            # Force SPEECH_END if buffer exceeds max duration (safety cap)
            max_ms = self._config.max_speech_duration * 1000
            if self._speech_ms >= max_ms:
                logger.warning(
                    "Speech duration %.0fms exceeded max (%.0fms); forcing SPEECH_END",
                    self._speech_ms,
                    max_ms,
                )
                audio = bytes(self._speech_buf)
                duration = self._speech_ms
                self._speaking = False
                self._silence_ms = 0.0
                self._energy_silence_ms = 0.0
                self._speech_buf = bytearray()
                if self._detector is not None:
                    self._detector.reset()
                return VADEvent(
                    type=VADEventType.SPEECH_END,
                    confidence=1.0,
                    audio_bytes=audio,
                    duration_ms=duration,
                )

            if is_speech:
                self._silence_ms = 0.0
            else:
                self._silence_ms += duration_ms

            # Energy-based fast exit: the model may stay in speech state
            # long after the user stops speaking (model inertia).  Track
            # consecutive low-energy frames independently and force
            # SPEECH_END when the audio is clearly silence.
            rms_threshold = self._config.energy_silence_rms
            if rms_threshold > 0:
                rms = _rms_int16(frame.data)
                if rms < rms_threshold:
                    self._energy_silence_ms += duration_ms
                else:
                    self._energy_silence_ms = 0.0
            else:
                self._energy_silence_ms = 0.0

            silence_triggered = self._silence_ms >= self._config.silence_threshold_ms
            energy_triggered = self._energy_silence_ms >= self._config.silence_threshold_ms

            if silence_triggered or energy_triggered:
                # Transition to idle
                if energy_triggered and not silence_triggered:
                    logger.debug(
                        "VAD: energy-based speech end (rms < %.0f for %.0fms)",
                        rms_threshold,
                        self._energy_silence_ms,
                    )
                    # Reset sherpa detector to clear stuck internal state,
                    # otherwise is_speech_detected() stays True and
                    # immediately re-triggers a false SPEECH_START.
                    if self._detector is not None:
                        self._detector.reset()
                self._speaking = False
                speech_ms = self._speech_ms
                audio = bytes(self._speech_buf)

                # Reset accumulators
                self._speech_buf = bytearray()
                self._speech_ms = 0.0
                self._silence_ms = 0.0
                self._energy_silence_ms = 0.0

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
        self._energy_silence_ms = 0.0
        self._speech_ms = 0.0
        self._speech_buf = bytearray()
        self._pre_roll.clear()
        self._pre_roll_ms = 0.0
        self._debug_frame_count = 0
        self._debug_rms_sum = 0.0
        self._debug_rms_max = 0.0
        self._debug_speech_count = 0
        if self._detector is not None:
            self._detector.reset()

    def close(self) -> None:
        """Release resources."""
        if self._detector is not None:
            self._detector.flush()
            self._detector = None

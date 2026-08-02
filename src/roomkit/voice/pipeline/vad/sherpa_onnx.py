"""sherpa-onnx neural VAD provider (TEN-VAD / Silero VAD).

Uses sherpa-onnx's VoiceActivityDetector with frame-level
``is_speech_detected()`` plus our own state machine for instant
SPEECH_START events and pre-roll buffering.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType, VADProvider
from roomkit.voice.utils import _get_np

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger(__name__)


_DEBUG_SUMMARY_INTERVAL = 50  # frames (~1s at 20ms/frame)


def _rms_int16(data: bytes) -> float:
    """Compute RMS of int16 little-endian PCM data.

    Vectorised: this runs per frame on the energy-silence path, and
    sherpa-onnx already guarantees numpy is installed.
    """
    n_samples = len(data) // 2
    if n_samples == 0:
        return 0.0
    np = _get_np()
    samples = np.frombuffer(data[: n_samples * 2], dtype="<i2").astype(np.int64)
    return float(np.sqrt((samples @ samples) / n_samples))


def _pcm_s16le_to_float32(data: bytes) -> Any:
    """Convert PCM signed 16-bit little-endian bytes to float32 in [-1, 1].

    Returns a numpy float32 array — ``accept_waveform`` converts it to its
    ``std::vector<float>`` at C speed, where a Python list pays a per-element
    conversion on every frame.
    """
    np = _get_np()
    n = len(data) // 2
    return np.frombuffer(data[: n * 2], dtype="<i2").astype(np.float32) / 32768.0


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


@dataclass
class _StreamState:
    """One speaker's detector and detection state.

    The sherpa detector accumulates waveform and holds its own speech
    probability history, so it cannot be shared: two speakers through one
    detector make silence from one close the other's utterance.
    """

    detector: Any = None
    speaking: bool = False
    silence_ms: float = 0.0
    energy_silence_ms: float = 0.0
    speech_ms: float = 0.0
    speech_buf: bytearray = field(default_factory=bytearray)
    pre_roll: deque[bytes] = field(default_factory=deque)
    pre_roll_ms: float = 0.0
    debug_frame_count: int = 0
    debug_rms_sum: float = 0.0
    debug_rms_max: float = 0.0
    debug_speech_count: int = 0


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
        self._streams: dict[str, _StreamState] = {}

    @property
    def name(self) -> str:
        return "SherpaOnnxVAD"

    def _state_for(self, stream: str) -> _StreamState:
        """Get or create this stream's detection state."""
        return self._streams.setdefault(stream, _StreamState())

    def _ensure_detector(self, st: _StreamState) -> None:
        """Lazily create this stream's sherpa-onnx VoiceActivityDetector."""
        if st.detector is not None:
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

        st.detector = sherpa.VoiceActivityDetector(vad_config)
        logger.debug(
            "SherpaOnnxVAD: created detector model_type=%s model=%s",
            cfg.model_type,
            cfg.model,
        )

    def _frame_duration_ms(self, frame: AudioFrame) -> float:
        """Duration of a single frame in milliseconds."""
        n_samples = len(frame.data) // (frame.sample_width * frame.channels)
        return (n_samples / frame.sample_rate) * 1000.0

    def _push_pre_roll(self, st: _StreamState, data: bytes, duration_ms: float) -> None:
        """Maintain a rolling buffer of recent frames for pre-speech padding."""
        st.pre_roll.append(data)
        st.pre_roll_ms += duration_ms
        while st.pre_roll_ms > self._config.speech_pad_ms and len(st.pre_roll) > 1:
            removed = st.pre_roll.popleft()
            n_samples = len(removed) // 2  # int16
            st.pre_roll_ms -= (n_samples / self._config.sample_rate) * 1000.0

    def process(self, frame: AudioFrame, stream: str) -> VADEvent | None:  # noqa: C901
        st = self._state_for(stream)
        self._ensure_detector(st)

        duration_ms = self._frame_duration_ms(frame)

        # Feed audio to sherpa detector
        float_samples = _pcm_s16le_to_float32(frame.data)
        st.detector.accept_waveform(float_samples)

        # Drain completed segments to prevent unbounded memory
        while not st.detector.empty():
            st.detector.pop()

        is_speech = st.detector.is_speech_detected()

        # Debug logging: accumulate stats and emit periodic summary
        if logger.isEnabledFor(logging.DEBUG):
            rms = _rms_int16(frame.data)
            st.debug_frame_count += 1
            st.debug_rms_sum += rms
            if rms > st.debug_rms_max:
                st.debug_rms_max = rms
            if is_speech:
                st.debug_speech_count += 1
            if st.debug_frame_count >= _DEBUG_SUMMARY_INTERVAL:
                avg = st.debug_rms_sum / st.debug_frame_count
                state = "speaking" if st.speaking else "idle"
                logger.debug(
                    "VAD: state=%s is_speech=%d/%d rms_avg=%.0f rms_max=%.0f"
                    " silence_ms=%.0f speech_ms=%.0f",
                    state,
                    st.debug_speech_count,
                    st.debug_frame_count,
                    avg,
                    st.debug_rms_max,
                    st.silence_ms,
                    st.speech_ms,
                )
                st.debug_frame_count = 0
                st.debug_rms_sum = 0.0
                st.debug_rms_max = 0.0
                st.debug_speech_count = 0

        if not st.speaking:
            # --- Idle state ---
            self._push_pre_roll(st, frame.data, duration_ms)

            if is_speech:
                st.speaking = True
                st.silence_ms = 0.0
                st.speech_ms = duration_ms
                # Start accumulating with pre-roll
                st.speech_buf = bytearray()
                for chunk in st.pre_roll:
                    st.speech_buf.extend(chunk)
                st.pre_roll.clear()
                st.pre_roll_ms = 0.0
                return VADEvent(
                    type=VADEventType.SPEECH_START,
                    confidence=1.0,
                    audio_bytes=bytes(st.speech_buf),
                )
        else:
            # --- Speaking state ---
            st.speech_buf.extend(frame.data)
            st.speech_ms += duration_ms

            # Force SPEECH_END if buffer exceeds max duration (safety cap)
            max_ms = self._config.max_speech_duration * 1000
            if st.speech_ms >= max_ms:
                logger.warning(
                    "Speech duration %.0fms exceeded max (%.0fms); forcing SPEECH_END",
                    st.speech_ms,
                    max_ms,
                )
                audio = bytes(st.speech_buf)
                duration = st.speech_ms
                st.speaking = False
                st.silence_ms = 0.0
                st.energy_silence_ms = 0.0
                st.speech_buf = bytearray()
                st.detector.reset()
                return VADEvent(
                    type=VADEventType.SPEECH_END,
                    confidence=1.0,
                    audio_bytes=audio,
                    duration_ms=duration,
                )

            if is_speech:
                st.silence_ms = 0.0
            else:
                st.silence_ms += duration_ms

            # Energy-based fast exit: the model may stay in speech state
            # long after the user stops speaking (model inertia).  Track
            # consecutive low-energy frames independently and force
            # SPEECH_END when the audio is clearly silence.
            rms_threshold = self._config.energy_silence_rms
            if rms_threshold > 0:
                rms = _rms_int16(frame.data)
                if rms < rms_threshold:
                    st.energy_silence_ms += duration_ms
                else:
                    st.energy_silence_ms = 0.0
            else:
                st.energy_silence_ms = 0.0

            silence_triggered = st.silence_ms >= self._config.silence_threshold_ms
            energy_triggered = st.energy_silence_ms >= self._config.silence_threshold_ms

            if silence_triggered or energy_triggered:
                # Transition to idle
                if energy_triggered and not silence_triggered:
                    logger.debug(
                        "VAD: energy-based speech end (rms < %.0f for %.0fms)",
                        rms_threshold,
                        st.energy_silence_ms,
                    )
                    # Reset sherpa detector to clear stuck internal state,
                    # otherwise is_speech_detected() stays True and
                    # immediately re-triggers a false SPEECH_START.
                    st.detector.reset()
                st.speaking = False
                speech_ms = st.speech_ms
                audio = bytes(st.speech_buf)

                # Reset accumulators
                st.speech_buf = bytearray()
                st.speech_ms = 0.0
                st.silence_ms = 0.0
                st.energy_silence_ms = 0.0

                if speech_ms >= self._config.min_speech_duration_ms:
                    return VADEvent(
                        type=VADEventType.SPEECH_END,
                        audio_bytes=audio,
                        duration_ms=speech_ms,
                    )
                # Too short — discard silently

        return None

    def reset(self, stream: str) -> None:
        """Drop this stream's detector and detection state.

        Not flushed: the pending segment belongs to a stream that is over, and
        flushing would push it out for a caller that is no longer listening.
        """
        self._streams.pop(stream, None)

    def close(self) -> None:
        """Release every stream's detector."""
        for st in self._streams.values():
            if st.detector is not None:
                st.detector.flush()
        self._streams.clear()

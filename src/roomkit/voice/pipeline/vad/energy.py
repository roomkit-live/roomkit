"""Energy-based Voice Activity Detection provider.

Uses RMS amplitude thresholding to detect speech — no external dependencies.
Suitable for local testing and simple deployments where a neural VAD
(e.g. Silero) is not available.
"""

from __future__ import annotations

import logging
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType, VADProvider

logger = logging.getLogger(__name__)

_DEBUG_SUMMARY_INTERVAL = 50  # frames (~1s at 20ms/frame)

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


def _rms_int16(data: bytes) -> float:
    """Compute RMS of int16 little-endian PCM data."""
    n_samples = len(data) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", data[: n_samples * 2])
    sum_sq = sum(s * s for s in samples)
    return float((sum_sq / n_samples) ** 0.5)


@dataclass
class _StreamState:
    """One speaker's detection state.

    Per stream, because two speakers share a provider but not a voice: letting
    one advance the other's state makes silence from one close the other's
    utterance.
    """

    speaking: bool = False
    silence_ms: float = 0.0
    speech_ms: float = 0.0
    speech_buf: bytearray = field(default_factory=bytearray)
    pre_roll: deque[bytes] = field(default_factory=deque)
    pre_roll_ms: float = 0.0
    debug_frame_count: int = 0
    debug_rms_sum: float = 0.0
    debug_rms_max: float = 0.0
    debug_speech_count: int = 0


class EnergyVADProvider(VADProvider):
    """VAD provider that detects speech by RMS energy thresholding.

    Parameters:
        energy_threshold: RMS threshold for speech detection (int16 scale, 0–32768).
        silence_threshold_ms: Milliseconds of consecutive silence to end speech.
        min_speech_duration_ms: Minimum speech duration to emit SPEECH_END.
            Shorter segments are silently discarded.
        speech_pad_ms: Pre-speech audio padding. A rolling buffer of recent
            frames is kept so the start of speech isn't clipped.
    """

    def __init__(
        self,
        *,
        energy_threshold: float = 300.0,
        silence_threshold_ms: float = 500,
        min_speech_duration_ms: float = 200,
        speech_pad_ms: float = 300,
        max_speech_duration_ms: float = 60_000,
    ) -> None:
        self._energy_threshold = energy_threshold
        self._silence_threshold_ms = silence_threshold_ms
        self._min_speech_duration_ms = min_speech_duration_ms
        self._speech_pad_ms = speech_pad_ms
        self._max_speech_duration_ms = max_speech_duration_ms

        self._streams: dict[str, _StreamState] = {}

    @property
    def name(self) -> str:
        return "EnergyVADProvider"

    def _frame_duration_ms(self, frame: AudioFrame) -> float:
        """Duration of a single frame in milliseconds."""
        n_samples = len(frame.data) // (frame.sample_width * frame.channels)
        return (n_samples / frame.sample_rate) * 1000.0

    def _push_pre_roll(
        self, st: _StreamState, data: bytes, duration_ms: float, sample_rate: int
    ) -> None:
        """Maintain a rolling buffer of recent frames for pre-speech padding."""
        st.pre_roll.append(data)
        st.pre_roll_ms += duration_ms
        while st.pre_roll_ms > self._speech_pad_ms and len(st.pre_roll) > 1:
            removed = st.pre_roll.popleft()
            n_samples = len(removed) // 2  # int16
            st.pre_roll_ms -= (n_samples / sample_rate) * 1000.0

    def process(self, frame: AudioFrame, stream: str) -> VADEvent | None:
        st = self._streams.setdefault(stream, _StreamState())
        rms = _rms_int16(frame.data)
        duration_ms = self._frame_duration_ms(frame)
        is_speech = rms >= self._energy_threshold

        # Debug logging: accumulate stats and emit periodic summary
        if logger.isEnabledFor(logging.DEBUG):
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
            self._push_pre_roll(st, frame.data, duration_ms, frame.sample_rate)

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

            if is_speech:
                st.silence_ms = 0.0
            else:
                st.silence_ms += duration_ms

            # Force speech-end if max duration exceeded (safety cap)
            force_end = st.speech_ms >= self._max_speech_duration_ms
            if force_end:
                logger.warning(
                    "Speech duration %.0fms exceeded max (%.0fms); forcing SPEECH_END",
                    st.speech_ms,
                    self._max_speech_duration_ms,
                )

            if st.silence_ms >= self._silence_threshold_ms or force_end:
                # Transition to idle
                st.speaking = False
                speech_ms = st.speech_ms
                audio = bytes(st.speech_buf)

                # Reset accumulators
                st.speech_buf = bytearray()
                st.speech_ms = 0.0
                st.silence_ms = 0.0

                if speech_ms >= self._min_speech_duration_ms:
                    return VADEvent(
                        type=VADEventType.SPEECH_END,
                        audio_bytes=audio,
                        duration_ms=speech_ms,
                    )
                # Too short — discard silently

        return None

    def reset(self, stream: str) -> None:
        """Drop this stream's state."""
        self._streams.pop(stream, None)

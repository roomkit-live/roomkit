"""Energy-based Voice Activity Detection provider.

Uses RMS amplitude thresholding to detect speech — no external dependencies.
Suitable for local testing and simple deployments where a neural VAD
(e.g. Silero) is not available.
"""

from __future__ import annotations

import logging
import struct
from collections import deque
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
    ) -> None:
        self._energy_threshold = energy_threshold
        self._silence_threshold_ms = silence_threshold_ms
        self._min_speech_duration_ms = min_speech_duration_ms
        self._speech_pad_ms = speech_pad_ms

        # State
        self._speaking = False
        self._silence_ms: float = 0.0
        self._speech_ms: float = 0.0
        self._speech_buf = bytearray()

        # Pre-roll buffer (deque of raw bytes)
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_ms: float = 0.0

        # Debug logging counters
        self._debug_frame_count = 0
        self._debug_rms_sum = 0.0
        self._debug_rms_max = 0.0
        self._debug_speech_count = 0

    @property
    def name(self) -> str:
        return "EnergyVADProvider"

    def _frame_duration_ms(self, frame: AudioFrame) -> float:
        """Duration of a single frame in milliseconds."""
        n_samples = len(frame.data) // (frame.sample_width * frame.channels)
        return (n_samples / frame.sample_rate) * 1000.0

    def _push_pre_roll(self, data: bytes, duration_ms: float) -> None:
        """Maintain a rolling buffer of recent frames for pre-speech padding."""
        self._pre_roll.append(data)
        self._pre_roll_ms += duration_ms
        while self._pre_roll_ms > self._speech_pad_ms and len(self._pre_roll) > 1:
            removed = self._pre_roll.popleft()
            n_samples = len(removed) // 2  # int16
            # Approximate: assume same sample rate as current stream
            self._pre_roll_ms -= (n_samples / 16000) * 1000.0

    def process(self, frame: AudioFrame) -> VADEvent | None:
        rms = _rms_int16(frame.data)
        duration_ms = self._frame_duration_ms(frame)
        is_speech = rms >= self._energy_threshold

        # Debug logging: accumulate stats and emit periodic summary
        if logger.isEnabledFor(logging.DEBUG):
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
                return VADEvent(type=VADEventType.SPEECH_START, confidence=1.0)
        else:
            # --- Speaking state ---
            self._speech_buf.extend(frame.data)
            self._speech_ms += duration_ms

            if is_speech:
                self._silence_ms = 0.0
            else:
                self._silence_ms += duration_ms

                if self._silence_ms >= self._silence_threshold_ms:
                    # Transition to idle
                    self._speaking = False
                    speech_ms = self._speech_ms
                    audio = bytes(self._speech_buf)

                    # Reset accumulators
                    self._speech_buf = bytearray()
                    self._speech_ms = 0.0
                    self._silence_ms = 0.0

                    if speech_ms >= self._min_speech_duration_ms:
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
        self._debug_frame_count = 0
        self._debug_rms_sum = 0.0
        self._debug_rms_max = 0.0
        self._debug_speech_count = 0

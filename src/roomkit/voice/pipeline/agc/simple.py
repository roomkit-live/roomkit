"""Dependency-free automatic gain control for signed 16-bit PCM."""

from __future__ import annotations

import logging
import math
import struct
import threading
from dataclasses import dataclass, field

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.agc.base import AGCConfig, AGCProvider

logger = logging.getLogger("roomkit.voice.pipeline.agc.simple")


@dataclass
class _StreamState:
    """Adaptive gain belonging to one input stream."""

    gain_db: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class SimpleAGCProvider(AGCProvider):
    """Adaptive RMS-based gain control for PCM16 audio.

    The controller measures each frame's RMS level, moves a stream-local gain
    toward ``AGCConfig.target_level_dbfs`` using the configured attack/release
    time constants, and applies a peak limiter before converting back to PCM.
    Near-silence is never amplified, which avoids turning an idle microphone's
    noise floor into apparent speech.

    Provider-specific settings may be supplied through ``AGCConfig.metadata``:

    - ``silence_threshold_dbfs`` (default ``-60``): frames below this RMS level
      pass through at unity gain.
    - ``min_gain_db`` (default ``-30``): maximum attenuation.
    """

    def __init__(self, config: AGCConfig | None = None) -> None:
        self._config = config or AGCConfig()
        self._silence_threshold_dbfs = self._metadata_float("silence_threshold_dbfs", -60.0)
        self._min_gain_db = self._metadata_float("min_gain_db", -30.0)
        if self._silence_threshold_dbfs > 0:
            raise ValueError("silence_threshold_dbfs must be at most 0")
        if self._min_gain_db > 0:
            raise ValueError("min_gain_db must be at most 0")

        self._streams: dict[str, _StreamState] = {}
        self._streams_lock = threading.Lock()
        self._warned_formats: set[tuple[int, int, int, int]] = set()
        self._closed = False

    @property
    def name(self) -> str:
        return "simple_agc"

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        """Normalize one PCM16 frame without sharing gain across streams."""
        if not self._matches_format(frame):
            return frame
        state = self._state_for(stream)
        if state is None:
            return frame

        sample_count = len(frame.data) // 2
        metadata = dict(frame.metadata)
        if sample_count == 0:
            metadata["gain_applied_db"] = 0.0
            return self._copy_frame(frame, frame.data, metadata)

        samples = struct.unpack(f"<{sample_count}h", frame.data)
        energy = sum(sample * sample for sample in samples)
        rms = math.sqrt(energy / sample_count)
        if rms <= 0:
            metadata["gain_applied_db"] = 0.0
            return self._copy_frame(frame, frame.data, metadata)

        level_dbfs = 20.0 * math.log10(rms / 32768.0)
        if level_dbfs < self._silence_threshold_dbfs:
            metadata["gain_applied_db"] = 0.0
            return self._copy_frame(frame, frame.data, metadata)

        desired_db = min(
            self._config.max_gain_db,
            max(self._min_gain_db, self._config.target_level_dbfs - level_dbfs),
        )
        frame_samples = sample_count / frame.channels
        duration_ms = frame_samples * 1000.0 / frame.sample_rate

        with state.lock:
            time_constant_ms = (
                self._config.attack_ms if desired_db > state.gain_db else self._config.release_ms
            )
            if time_constant_ms == 0:
                smoothed_db = desired_db
            else:
                alpha = 1.0 - math.exp(-duration_ms / time_constant_ms)
                smoothed_db = state.gain_db + alpha * (desired_db - state.gain_db)

            peak = max(abs(sample) for sample in samples)
            if peak:
                limiter_db = 20.0 * math.log10(32767.0 / peak)
                applied_db = min(smoothed_db, limiter_db)
            else:
                applied_db = smoothed_db
            state.gain_db = applied_db

        gain = 10.0 ** (applied_db / 20.0)
        output = [max(-32768, min(32767, round(sample * gain))) for sample in samples]
        metadata["gain_applied_db"] = applied_db
        return self._copy_frame(frame, struct.pack(f"<{sample_count}h", *output), metadata)

    def reset(self, stream: str) -> None:
        """Forget one stream's adaptive gain."""
        with self._streams_lock:
            self._streams.pop(stream, None)

    def close(self) -> None:
        """Release all stream state and reject future state creation."""
        with self._streams_lock:
            self._closed = True
            self._streams.clear()

    def _metadata_float(self, key: str, default: float) -> float:
        value = self._config.metadata.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"AGC metadata {key!r} must be a finite number")
        return float(value)

    def _state_for(self, stream: str) -> _StreamState | None:
        with self._streams_lock:
            if self._closed:
                return None
            return self._streams.setdefault(stream, _StreamState())

    def _matches_format(self, frame: AudioFrame) -> bool:
        valid = (
            frame.sample_rate > 0
            and frame.channels > 0
            and frame.sample_width == 2
            and len(frame.data) % (2 * frame.channels) == 0
        )
        if valid:
            return True
        key = (frame.sample_rate, frame.channels, frame.sample_width, len(frame.data))
        if key not in self._warned_formats:
            self._warned_formats.add(key)
            logger.warning(
                "Simple AGC requires aligned PCM16: got %dHz/%dch/%d-byte with %d bytes; "
                "frame ignored",
                *key,
            )
        return False

    @staticmethod
    def _copy_frame(frame: AudioFrame, data: bytes, metadata: dict[str, object]) -> AudioFrame:
        return AudioFrame(
            data=data,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=metadata,
        )

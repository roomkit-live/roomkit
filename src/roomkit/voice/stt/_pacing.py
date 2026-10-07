"""Pacing the audio sent to a realtime STT service that limits how far ahead of
real time it may run (RFC §12.2, audio between streams)."""

from __future__ import annotations

import time
from collections.abc import Callable


class AudioPace:
    """A token bucket over seconds of audio: at most ``burst`` seconds at once,
    refilled at ``speed`` times real time.

    A stream that opens on a backlog sends its first ``burst`` seconds at once,
    then catches up at ``speed``; live audio, arriving in real time, never waits
    once ``speed`` is above 1.
    """

    def __init__(
        self, burst: float, speed: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._burst = burst
        self._speed = speed
        self._clock = clock
        self._tokens = burst
        self._at = clock()

    def delay(self, seconds: float) -> float:
        """How long to wait before sending *seconds* of audio, counted as sent."""
        now = self._clock()
        self._tokens = min(self._burst, self._tokens + (now - self._at) * self._speed)
        self._at = now
        self._tokens -= seconds
        return max(0.0, -self._tokens / self._speed)

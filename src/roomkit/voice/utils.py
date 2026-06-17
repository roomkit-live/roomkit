"""Shared audio utilities for the voice subsystem."""

from __future__ import annotations

import math
from typing import Any

_np: Any = None


def _get_np() -> Any:
    """Import numpy lazily, cached at module level.

    NumPy is an optional dependency (voice extras): base installs must
    import roomkit without it, and the DSP hot paths must not pay a
    per-call import.
    """
    global _np  # noqa: PLW0603
    if _np is None:
        import numpy

        _np = numpy
    return _np


def rms_db(data: bytes) -> float:
    """Compute RMS level in dB from 16-bit little-endian PCM bytes.

    Returns a value in the range [-60.0, 0.0] where 0 dB is full scale
    and -60 dB is silence (or empty input).

    Vectorised with NumPy: this runs per audio chunk on the event loop,
    where a per-sample Python loop holds the GIL long enough to starve
    realtime RTP pacing.
    """
    n = len(data) // 2
    if n == 0:
        return -60.0
    np = _get_np()
    samples = np.frombuffer(data[: n * 2], dtype="<i2").astype(np.float64)
    rms = math.sqrt(float(np.mean(samples * samples))) / 32768.0
    if rms < 1e-10:
        return -60.0
    return max(-60.0, 20.0 * math.log10(rms))

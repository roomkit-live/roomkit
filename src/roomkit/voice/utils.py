"""Shared audio utilities for the voice subsystem."""

from __future__ import annotations

import math
import struct


def rms_db(data: bytes) -> float:
    """Compute RMS level in dB from 16-bit little-endian PCM bytes.

    Returns a value in the range [-60.0, 0.0] where 0 dB is full scale
    and -60 dB is silence (or empty input).
    """
    n = len(data) // 2
    if n == 0:
        return -60.0
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    rms = math.sqrt(sum(s * s for s in samples) / n) / 32768.0
    if rms < 1e-10:
        return -60.0
    return max(-60.0, 20.0 * math.log10(rms))

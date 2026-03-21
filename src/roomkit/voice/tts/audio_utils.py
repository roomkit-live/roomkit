"""Shared audio utilities for TTS providers."""

from __future__ import annotations

import struct
from typing import Any


def wrap_wav(pcm_data: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw PCM S16LE data in a minimal WAV header."""
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data


def numpy_to_pcm_s16le(samples: Any) -> bytes:
    """Convert a numpy float32 array in [-1, 1] to PCM signed 16-bit LE bytes."""
    import numpy as np  # optional dependency

    arr = np.clip(samples, -1.0, 1.0)
    int_samples = (arr * 32767).astype(np.int16)
    return bytes(int_samples.tobytes())

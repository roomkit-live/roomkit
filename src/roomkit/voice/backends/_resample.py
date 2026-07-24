"""Shared streaming resampler for voice backends (soxr preferred).

Backends that carry a fixed-rate wire format (Twilio mu-law at 8 kHz, Buzz
huddles at 48 kHz) resample to/from the pipeline or provider rates
themselves. They all want the same thing: a stateful, chunk-by-chunk s16le
mono converter that keeps latency at ~one frame and does not smear the
spectrum the way plain linear interpolation does on upsampling.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Callable

logger = logging.getLogger("roomkit.voice.backends.resample")


def build_streaming_resampler(in_rate: int, out_rate: int) -> Callable[[bytes], bytes]:
    """Build a stateful s16le mono resampler (soxr preferred, linear fallback).

    quality="QQ" (Quick) keeps per-chunk latency at ~one frame. soxr's
    default VHQ buffers ~120 ms of filter history before emitting, which
    silently drops the first six 20 ms frames of a stream. QQ's polyphase
    filter is still far cleaner than the linear fallback, whose imaging
    artifacts make synthesized voices sound harsh on upsampling.
    """
    if in_rate == out_rate:
        return lambda data: data
    try:
        import numpy as np
        import soxr

        stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype=np.int16, quality="QQ")
        logger.info("Resampler: soxr stream QQ (%d -> %d Hz)", in_rate, out_rate)

        def _soxr(data: bytes) -> bytes:
            out = stream.resample_chunk(np.frombuffer(data, dtype=np.int16))
            return bytes(out.tobytes())

        return _soxr
    except ImportError:
        logger.info("Resampler: linear (%d -> %d Hz)", in_rate, out_rate)
        ratio = out_rate / in_rate

        def _linear(data: bytes) -> bytes:
            n_in = len(data) // 2
            if n_in == 0:
                return data
            samples = struct.unpack(f"<{n_in}h", data)
            n_out = int(n_in * ratio)
            out = [0] * n_out
            for i in range(n_out):
                src = i / ratio
                idx = int(src)
                frac = src - idx
                s0 = samples[min(idx, n_in - 1)]
                s1 = samples[min(idx + 1, n_in - 1)]
                out[i] = max(-32768, min(32767, int(s0 + frac * (s1 - s0))))
            return struct.pack(f"<{n_out}h", *out)

        return _linear

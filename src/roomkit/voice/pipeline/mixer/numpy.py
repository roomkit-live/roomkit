"""NumPy-accelerated PCM mixer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from roomkit.voice.pipeline.mixer.base import MixerProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class NumpyMixerProvider(MixerProvider):
    """Mixer using NumPy vectorized operations.

    Uses ``np.frombuffer`` for near-zero-cost decoding, int32
    accumulation to avoid overflow, and vectorized clipping.
    Roughly 20x faster than the pure-Python mixer for typical
    frame sizes (960 samples at 48 kHz).

    Requires ``numpy`` (``pip install numpy``).
    """

    @property
    def name(self) -> str:
        return "numpy"

    def mix(self, frames: list[AudioFrame]) -> AudioFrame:
        from roomkit.voice.audio_frame import AudioFrame as _AudioFrame

        if len(frames) == 1:
            return frames[0]

        ref = frames[0]
        n = len(frames)
        min_samples = min(len(f.data) // 2 for f in frames)

        # Decode to int16 views, truncate to shortest
        arrays = [np.frombuffer(f.data[: min_samples * 2], dtype=np.int16) for f in frames]

        # Accumulate in int32 to avoid int16 overflow
        acc = arrays[0].astype(np.int32)
        for a in arrays[1:]:
            acc += a.astype(np.int32)

        # Headroom: average for 3+ sources, direct sum for 2
        if n >= 3:
            acc //= n

        np.clip(acc, -32768, 32767, out=acc)
        out_data = acc.astype(np.int16).tobytes()

        return _AudioFrame(
            data=out_data,
            sample_rate=ref.sample_rate,
            channels=ref.channels,
            sample_width=ref.sample_width,
            timestamp_ms=ref.timestamp_ms,
        )

"""Pure-Python PCM mixer using struct."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from roomkit.voice.pipeline.mixer.base import MixerProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class PythonMixerProvider(MixerProvider):
    """Mixer using pure-Python ``struct`` pack/unpack.

    No external dependencies.  Suitable for small frame sizes and low
    participant counts.  For higher throughput, use
    :class:`~roomkit.voice.pipeline.mixer.numpy.NumpyMixerProvider`.
    """

    @property
    def name(self) -> str:
        return "python"

    def mix(self, frames: list[AudioFrame]) -> AudioFrame:
        from roomkit.voice.audio_frame import AudioFrame as _AudioFrame

        if len(frames) == 1:
            return frames[0]

        ref = frames[0]
        n = len(frames)
        min_samples = min(len(f.data) // 2 for f in frames)

        decoded: list[list[int]] = []
        for f in frames:
            decoded.append(list(struct.unpack(f"<{min_samples}h", f.data[: min_samples * 2])))

        mixed: list[int] = []
        if n == 2:
            s0, s1 = decoded[0], decoded[1]
            for i in range(min_samples):
                val = s0[i] + s1[i]
                mixed.append(max(-32768, min(32767, val)))
        else:
            for i in range(min_samples):
                total = sum(d[i] for d in decoded)
                val = total // n
                mixed.append(max(-32768, min(32767, val)))

        out_data = struct.pack(f"<{min_samples}h", *mixed)
        return _AudioFrame(
            data=out_data,
            sample_rate=ref.sample_rate,
            channels=ref.channels,
            sample_width=ref.sample_width,
            timestamp_ms=ref.timestamp_ms,
        )

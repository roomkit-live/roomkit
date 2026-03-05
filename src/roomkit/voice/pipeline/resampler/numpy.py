"""NumPy-accelerated linear interpolation resampler provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from roomkit.voice.pipeline.resampler.base import ResamplerProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

_DTYPE_MAP: dict[int, type[np.int8] | type[np.int16] | type[np.int32]] = {
    1: np.int8,
    2: np.int16,
    4: np.int32,
}


class NumpyResamplerProvider(ResamplerProvider):
    """Resampler using NumPy vectorized linear interpolation.

    Drop-in replacement for :class:`LinearResamplerProvider` with
    significantly lower CPU usage — ``np.interp`` replaces Python-level
    sample loops, and ``np.frombuffer``/``ndarray.tobytes`` replace
    ``struct.unpack``/``struct.pack``.
    """

    @property
    def name(self) -> str:
        return "numpy"

    def resample(
        self,
        frame: AudioFrame,
        target_rate: int,
        target_channels: int,
        target_width: int,
    ) -> AudioFrame:
        if (
            frame.sample_rate == target_rate
            and frame.channels == target_channels
            and frame.sample_width == target_width
        ):
            return frame

        from roomkit.voice.audio_frame import AudioFrame as AudioFrameClass

        src_dtype = _DTYPE_MAP.get(frame.sample_width)
        if src_dtype is None:
            return frame  # unsupported width, pass-through

        samples = np.frombuffer(frame.data, dtype=src_dtype)
        src_channels = frame.channels
        src_rate = frame.sample_rate

        # --- Channel conversion ---
        if src_channels != target_channels:
            if src_channels == 2 and target_channels == 1:
                # Stereo -> mono: average L+R
                samples = (samples[0::2].astype(np.int32) + samples[1::2].astype(np.int32)) // 2
                samples = samples.astype(src_dtype)
            elif src_channels == 1 and target_channels == 2:
                # Mono -> stereo: duplicate
                samples = np.repeat(samples, 2)
            src_channels = target_channels

        # --- Sample rate conversion (vectorized linear interpolation) ---
        if src_rate != target_rate:
            frames_per_channel = len(samples) // src_channels

            if src_channels == 1:
                new_frames = int(frames_per_channel * target_rate / src_rate)
                x_old = np.arange(frames_per_channel, dtype=np.float64)
                x_new = np.linspace(0, frames_per_channel - 1, new_frames)
                samples = np.interp(x_new, x_old, samples.astype(np.float64))
                samples = np.rint(samples).astype(np.int32)
            else:
                new_frames = int(frames_per_channel * target_rate / src_rate)
                x_old = np.arange(frames_per_channel, dtype=np.float64)
                x_new = np.linspace(0, frames_per_channel - 1, new_frames)
                # De-interleave, resample each channel, re-interleave
                reshaped = samples.reshape(-1, src_channels).T.astype(np.float64)
                resampled_channels = np.empty((src_channels, new_frames), dtype=np.float64)
                for ch in range(src_channels):
                    resampled_channels[ch] = np.interp(x_new, x_old, reshaped[ch])
                samples = np.rint(resampled_channels.T.ravel()).astype(np.int32)

        # --- Sample width conversion ---
        src_width = frame.sample_width
        if src_width != target_width:
            max_src = (1 << (src_width * 8 - 1)) - 1
            max_tgt = (1 << (target_width * 8 - 1)) - 1
            if max_src > 0:
                samples = (samples.astype(np.int64) * max_tgt // max_src).astype(np.int32)

        # --- Encode back to bytes ---
        tgt_dtype = _DTYPE_MAP.get(target_width)
        if tgt_dtype is None:
            return frame
        # Clamp to valid range
        info = np.iinfo(tgt_dtype)
        samples = np.clip(samples, info.min, info.max).astype(tgt_dtype)

        return AudioFrameClass(
            data=samples.tobytes(),
            sample_rate=target_rate,
            channels=target_channels,
            sample_width=target_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=dict(frame.metadata),
        )

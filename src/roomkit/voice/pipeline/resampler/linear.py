"""Linear interpolation resampler provider."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from roomkit.voice.pipeline.resampler.base import ResamplerProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

_FMT_MAP = {1: "b", 2: "h", 4: "i"}


class LinearResamplerProvider(ResamplerProvider):
    """Resampler using linear interpolation in pure Python.

    Handles channel conversion, sample rate conversion, and sample width
    conversion.  This is the default resampler auto-created by the pipeline
    engine when a contract is set but no explicit resampler is provided.
    """

    @property
    def name(self) -> str:
        return "linear"

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

        data = frame.data
        src_width = frame.sample_width
        src_channels = frame.channels
        src_rate = frame.sample_rate

        # --- Decode raw bytes into list of integer samples ---
        src_fmt = _FMT_MAP.get(src_width)
        if src_fmt is None:
            return frame  # unsupported width, pass-through
        num_samples = len(data) // src_width
        samples = list(
            struct.unpack(f"<{num_samples}{src_fmt}", data[: num_samples * src_width])
        )

        # --- Channel conversion ---
        if src_channels != target_channels:
            if src_channels == 2 and target_channels == 1:
                # Stereo -> mono: average L+R
                samples = [
                    (samples[i] + samples[i + 1]) // 2
                    for i in range(0, len(samples), 2)
                ]
            elif src_channels == 1 and target_channels == 2:
                # Mono -> stereo: duplicate
                stereo: list[int] = []
                for s in samples:
                    stereo.append(s)
                    stereo.append(s)
                samples = stereo
            num_samples = len(samples)
            src_channels = target_channels

        # --- Sample rate conversion (linear interpolation) ---
        if src_rate != target_rate:
            frames_per_channel = num_samples // src_channels
            new_frames = int(frames_per_channel * target_rate / src_rate)
            resampled: list[int] = []
            for ch in range(src_channels):
                ch_samples = [
                    samples[i * src_channels + ch] for i in range(frames_per_channel)
                ]
                for i in range(new_frames):
                    src_pos = i * (frames_per_channel - 1) / max(new_frames - 1, 1)
                    idx = int(src_pos)
                    frac = src_pos - idx
                    if idx + 1 < frames_per_channel:
                        val = ch_samples[idx] * (1 - frac) + ch_samples[idx + 1] * frac
                    else:
                        val = ch_samples[idx]
                    resampled.append(int(val))
            # Interleave channels
            if src_channels > 1:
                interleaved: list[int] = []
                for i in range(new_frames):
                    for ch in range(src_channels):
                        interleaved.append(resampled[ch * new_frames + i])
                samples = interleaved
            else:
                samples = resampled
            num_samples = len(samples)

        # --- Sample width conversion ---
        if src_width != target_width:
            max_src = (1 << (src_width * 8 - 1)) - 1
            max_tgt = (1 << (target_width * 8 - 1)) - 1
            if max_src > 0:
                samples = [int(s * max_tgt / max_src) for s in samples]

        # --- Encode back to bytes ---
        tgt_fmt = _FMT_MAP.get(target_width)
        if tgt_fmt is None:
            return frame
        # Clamp values to valid range for target width
        min_val = -(1 << (target_width * 8 - 1))
        max_val = (1 << (target_width * 8 - 1)) - 1
        samples = [max(min_val, min(max_val, s)) for s in samples]
        out_data = struct.pack(f"<{num_samples}{tgt_fmt}", *samples)

        return AudioFrameClass(
            data=out_data,
            sample_rate=target_rate,
            channels=target_channels,
            sample_width=target_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=dict(frame.metadata),
        )

"""Windowed sinc interpolation resampler provider."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

from roomkit.voice.pipeline.resampler.base import ResamplerProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

_FMT_MAP = {1: "b", 2: "h", 4: "i"}


def _sinc_resample_channel(
    extended: list[int],
    hist_len: int,
    frames_per_channel: int,
    target_rate: int,
    src_rate: int,
    taps: int,
) -> list[int]:
    """Resample a single channel using windowed sinc interpolation.

    Args:
        extended: history + current_samples (+ optional look-ahead)
        hist_len: number of history samples prepended
        frames_per_channel: number of NEW input samples (excluding history/look-ahead)
        target_rate: output sample rate
        src_rate: input sample rate
        taps: kernel half-width

    Returns:
        List of resampled output samples.
    """
    new_frames = int(frames_per_channel * target_rate / src_rate)
    ratio = src_rate / target_rate
    cutoff = min(src_rate, target_rate) / (2.0 * src_rate)
    ext_len = len(extended)

    out: list[int] = []
    for i in range(new_frames):
        src_pos = i * ratio
        center = int(src_pos) + hist_len
        frac = src_pos - int(src_pos)

        acc = 0.0
        wsum = 0.0
        for j in range(-taps + 1, taps + 1):
            idx = center + j
            if idx < 0 or idx >= ext_len:
                continue
            x = frac - j
            sx = x * 2.0 * cutoff
            if abs(sx) < 1e-9:
                sinc_val = 1.0
            else:
                pi_sx = math.pi * sx
                sinc_val = math.sin(pi_sx) / pi_sx
            t = (j - frac) / taps
            if abs(t) >= 1.0:
                continue
            window = 0.5 + 0.5 * math.cos(math.pi * t)
            w = sinc_val * window * 2.0 * cutoff
            acc += extended[idx] * w
            wsum += w

        if wsum > 1e-9:
            out.append(int(acc / wsum + 0.5))
        else:
            out.append(0)

    return out


class SincResamplerProvider(ResamplerProvider):
    """High-quality resampler using windowed sinc interpolation.

    Uses a Hann-windowed sinc kernel for band-limited interpolation.
    This avoids the aliasing artifacts of naive linear interpolation
    and is suitable for speech audio (8kHz/16kHz/24kHz/48kHz conversions).

    Maintains a one-frame delay so the kernel has full context on both
    sides of every frame boundary (no crackling at 20ms edges).

    Pure Python — only ``struct`` and ``math`` from the standard library.

    Args:
        taps: Number of sinc kernel taps (half-width). More taps = better
              stopband rejection but more computation. Default 16 is good
              for speech (>80 dB stopband attenuation with Hann window).
    """

    def __init__(self, taps: int = 16) -> None:
        self._taps = taps
        # Per-direction state: keyed by (src_rate, target_rate, channels).
        # Value: (tail, pending) where:
        #   tail = last `taps` samples per channel from the frame before pending
        #   pending = full per-channel samples from the previous frame
        self._state: dict[
            tuple[int, int, int],
            tuple[list[list[int]], list[list[int]]],
        ] = {}

    @property
    def name(self) -> str:
        return "sinc"

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
        samples = list(struct.unpack(f"<{num_samples}{src_fmt}", data[: num_samples * src_width]))

        # --- Channel conversion ---
        if src_channels != target_channels:
            if src_channels == 2 and target_channels == 1:
                samples = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
            elif src_channels == 1 and target_channels == 2:
                stereo: list[int] = []
                for s in samples:
                    stereo.append(s)
                    stereo.append(s)
                samples = stereo
            num_samples = len(samples)
            src_channels = target_channels

        # --- Sample rate conversion (windowed sinc interpolation) ---
        if src_rate != target_rate:
            frames_per_channel = num_samples // src_channels
            taps = self._taps

            state_key = (src_rate, target_rate, src_channels)
            prev_state = self._state.get(state_key)

            # Extract per-channel samples for current frame
            cur_per_ch: list[list[int]] = []
            for ch in range(src_channels):
                cur_per_ch.append(
                    [samples[i * src_channels + ch] for i in range(frames_per_channel)]
                )

            resampled: list[int] = []

            if prev_state is None:
                # First frame: no look-ahead available yet.  Buffer it as
                # pending so the next call can resample it with full kernel
                # context.  Outputting it here *and* as pending on the next
                # call would duplicate the first chunk (audible as a repeat
                # of the first ~20 ms of speech).
                new_tail: list[list[int]] = [[] for _ in range(src_channels)]
                self._state[state_key] = (new_tail, cur_per_ch)
                resampled = []
            else:
                # Subsequent frames: output for PENDING frame using
                # current frame as look-ahead context.
                tail_per_ch, pending_per_ch = prev_state
                pending_fpc = len(pending_per_ch[0]) if pending_per_ch else 0

                for ch in range(src_channels):
                    # Build: tail + pending + current
                    # Kernel has tail on the left, current on the right → full support
                    prefix = tail_per_ch[ch] if ch < len(tail_per_ch) else []
                    extended = prefix + pending_per_ch[ch] + cur_per_ch[ch]
                    resampled.extend(
                        _sinc_resample_channel(
                            extended,
                            len(prefix),
                            pending_fpc,
                            target_rate,
                            src_rate,
                            taps,
                        )
                    )

                # Update state: tail = end of pending, pending = current
                new_tail = []
                for ch in range(src_channels):
                    pch = pending_per_ch[ch]
                    new_tail.append(pch[-taps:] if len(pch) >= taps else pch[:])
                self._state[state_key] = (new_tail, cur_per_ch)

            # Interleave channels
            if src_channels > 1:
                new_frames = len(resampled) // src_channels
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

    def flush(
        self,
        target_rate: int,
        target_channels: int,
        target_width: int,
    ) -> AudioFrame | None:
        """Flush the pending frame using silence as look-ahead.

        Returns ``None`` if no pending frame exists for the given direction.
        After flushing, the state entry is deleted so the next response
        starts fresh.
        """
        from roomkit.voice.audio_frame import AudioFrame as AudioFrameClass

        # Find the matching state entry
        state_key: tuple[int, int, int] | None = None
        for key in self._state:
            src_rate, tgt_rate, channels = key
            if tgt_rate == target_rate and channels == target_channels:
                state_key = key
                break

        if state_key is None:
            return None

        prev_state = self._state.pop(state_key)
        tail_per_ch, pending_per_ch = prev_state

        if not pending_per_ch or not pending_per_ch[0]:
            return None

        src_rate, _, src_channels = state_key
        pending_fpc = len(pending_per_ch[0])
        taps = self._taps

        # Use silence as look-ahead context
        silence = [0] * pending_fpc

        resampled: list[int] = []
        for ch in range(src_channels):
            prefix = tail_per_ch[ch] if ch < len(tail_per_ch) else []
            extended = prefix + pending_per_ch[ch] + silence
            resampled.extend(
                _sinc_resample_channel(
                    extended,
                    len(prefix),
                    pending_fpc,
                    target_rate,
                    src_rate,
                    taps,
                )
            )

        # Interleave channels
        if src_channels > 1:
            new_frames = len(resampled) // src_channels
            interleaved: list[int] = []
            for i in range(new_frames):
                for ch in range(src_channels):
                    interleaved.append(resampled[ch * new_frames + i])
            samples = interleaved
        else:
            samples = resampled

        num_samples = len(samples)
        if num_samples == 0:
            return None

        tgt_fmt = _FMT_MAP.get(target_width)
        if tgt_fmt is None:
            return None
        min_val = -(1 << (target_width * 8 - 1))
        max_val = (1 << (target_width * 8 - 1)) - 1
        samples = [max(min_val, min(max_val, s)) for s in samples]
        out_data = struct.pack(f"<{num_samples}{tgt_fmt}", *samples)

        return AudioFrameClass(
            data=out_data,
            sample_rate=target_rate,
            channels=target_channels,
            sample_width=target_width,
        )

    def reset(self) -> None:
        """Reset internal state (clears history buffers)."""
        self._state.clear()

    def close(self) -> None:
        """Release resources."""
        self._state.clear()

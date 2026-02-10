"""Tests for windowed sinc interpolation resampler provider."""

from __future__ import annotations

import math
import struct

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.resampler.sinc import SincResamplerProvider


def _frame(
    data: bytes = b"\x00\x00",
    rate: int = 16000,
    channels: int = 1,
    width: int = 2,
) -> AudioFrame:
    return AudioFrame(data=data, sample_rate=rate, channels=channels, sample_width=width)


def _decode(data: bytes, width: int = 2) -> list[int]:
    fmt = {1: "b", 2: "h", 4: "i"}[width]
    n = len(data) // width
    return list(struct.unpack(f"<{n}{fmt}", data))


class TestSincResamplerProvider:
    def test_name(self):
        assert SincResamplerProvider().name == "sinc"

    def test_noop_when_format_matches(self):
        """Returns the exact same frame object when no conversion is needed."""
        provider = SincResamplerProvider()
        frame = _frame(b"\x01\x00\x02\x00")
        result = provider.resample(frame, 16000, 1, 2)
        assert result is frame

    def test_first_frame_buffered(self):
        """First frame is buffered (one-frame delay), producing no output."""
        provider = SincResamplerProvider()
        n = 160
        samples = [int(1000 * math.sin(2 * math.pi * 400 * i / 8000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=8000)

        result = provider.resample(frame, 16000, 1, 2)
        assert result.sample_rate == 16000
        assert len(result.data) == 0

    def test_upsample_8k_to_16k_length(self):
        """8kHz -> 16kHz produces 2x output samples (after one-frame delay)."""
        provider = SincResamplerProvider()
        # 160 samples at 8kHz = 20ms frame
        n = 160
        samples = [int(1000 * math.sin(2 * math.pi * 400 * i / 8000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=8000)

        provider.resample(frame, 16000, 1, 2)  # buffered
        result = provider.resample(frame, 16000, 1, 2)  # outputs first frame
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.sample_width == 2
        out = _decode(result.data)
        assert len(out) == 320  # 160 * 16000/8000

    def test_upsample_8k_to_16k_no_nan_or_clipping(self):
        """Upsampled signal has no extreme values (NaN mapped to int, overflow)."""
        provider = SincResamplerProvider()
        n = 160
        samples = [int(1000 * math.sin(2 * math.pi * 400 * i / 8000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=8000)

        provider.resample(frame, 16000, 1, 2)  # buffered
        result = provider.resample(frame, 16000, 1, 2)  # outputs first frame
        out = _decode(result.data)
        for s in out:
            assert -32768 <= s <= 32767
            # No NaN-like garbage values — output should be in the signal's range
            assert abs(s) <= 1500  # input peak is ~1000, some overshoot OK

    def test_downsample_16k_to_8k(self):
        """16kHz -> 8kHz halves the sample count."""
        provider = SincResamplerProvider()
        n = 320
        samples = [int(1000 * math.sin(2 * math.pi * 400 * i / 16000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=16000)

        provider.resample(frame, 8000, 1, 2)  # buffered
        result = provider.resample(frame, 8000, 1, 2)  # outputs first frame
        assert result.sample_rate == 8000
        out = _decode(result.data)
        assert len(out) == 160  # 320 * 8000/16000

    def test_downsample_48k_to_16k(self):
        """48kHz -> 16kHz reduces sample count by 3x."""
        provider = SincResamplerProvider()
        n = 960  # 20ms at 48kHz
        samples = [int(1000 * math.sin(2 * math.pi * 400 * i / 48000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=48000)

        provider.resample(frame, 16000, 1, 2)  # buffered
        result = provider.resample(frame, 16000, 1, 2)  # outputs first frame
        assert result.sample_rate == 16000
        out = _decode(result.data)
        assert len(out) == 320  # 960 * 16000/48000

    def test_channel_conversion_stereo_to_mono(self):
        """Stereo -> mono averages L+R channels."""
        provider = SincResamplerProvider()
        # 2 stereo frames: (100, 200), (300, 400)
        samples = [100, 200, 300, 400]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=16000, channels=2)

        result = provider.resample(frame, 16000, 1, 2)
        assert result.channels == 1
        out = _decode(result.data)
        assert out == [150, 350]

    def test_channel_and_rate_conversion_combined(self):
        """48kHz stereo -> 16kHz mono."""
        provider = SincResamplerProvider()
        # 6 stereo frames at 48kHz = 12 samples
        samples = [100, 100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=48000, channels=2)

        result = provider.resample(frame, 16000, 1, 2)
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.sample_width == 2

    def test_sample_width_conversion_2_to_4(self):
        """16-bit -> 32-bit scales sample values."""
        provider = SincResamplerProvider()
        samples = [1000, -1000]
        data = struct.pack("<2h", *samples)
        frame = _frame(data, rate=16000, channels=1, width=2)

        result = provider.resample(frame, 16000, 1, 4)
        assert result.sample_width == 4
        out = _decode(result.data, width=4)
        assert len(out) == 2
        assert out[0] > 1000
        assert out[1] < -1000

    def test_preserves_timestamp(self):
        provider = SincResamplerProvider()
        frame = AudioFrame(
            data=struct.pack("<2h", 100, 200),
            sample_rate=48000,
            channels=1,
            sample_width=2,
            timestamp_ms=42.0,
        )
        result = provider.resample(frame, 16000, 1, 2)
        assert result.timestamp_ms == 42.0

    def test_preserves_metadata(self):
        provider = SincResamplerProvider()
        frame = AudioFrame(
            data=struct.pack("<2h", 100, 200),
            sample_rate=48000,
            channels=1,
            sample_width=2,
            metadata={"key": "value"},
        )
        result = provider.resample(frame, 16000, 1, 2)
        assert result.metadata["key"] == "value"

    def test_reset_and_close_clear_state(self):
        """reset() and close() clear state and don't raise."""
        provider = SincResamplerProvider()
        # Build up some state
        n = 160
        samples = [int(1000 * math.sin(2 * math.pi * 400 * i / 8000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=8000)
        provider.resample(frame, 16000, 1, 2)
        assert len(provider._state) > 0

        provider.reset()
        assert len(provider._state) == 0

        provider.resample(frame, 16000, 1, 2)
        provider.close()
        assert len(provider._state) == 0

    def test_custom_taps(self):
        """Custom tap count works without error."""
        provider = SincResamplerProvider(taps=8)
        n = 160
        samples = [int(1000 * math.sin(2 * math.pi * 400 * i / 8000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=8000)

        provider.resample(frame, 16000, 1, 2)  # buffered
        result = provider.resample(frame, 16000, 1, 2)  # outputs first frame
        out = _decode(result.data)
        assert len(out) == 320

    def test_dc_signal_preserved(self):
        """A constant (DC) signal should be preserved through resampling."""
        provider = SincResamplerProvider()
        n = 160
        samples = [500] * n
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=8000)

        provider.resample(frame, 16000, 1, 2)  # buffered
        result = provider.resample(frame, 16000, 1, 2)  # outputs first frame
        out = _decode(result.data)
        # All output samples should be close to 500 (windowing at edges may differ)
        mid = out[20:-20]  # skip edges where kernel doesn't have full support
        for s in mid:
            assert abs(s - 500) <= 5, f"DC sample drifted to {s}"

    def test_streaming_no_boundary_crackling(self):
        """Multiple consecutive frames produce seamless output with no boundary clicks.

        The one-frame delay gives the sinc kernel full context on both sides
        of every frame boundary.  Frame 0 is buffered (no output), frames
        1-9 each output the resampled *previous* frame.
        """
        provider = SincResamplerProvider()
        freq = 400
        n = 160  # 20ms at 8kHz

        # Generate 10 consecutive frames of a sine wave
        all_output: list[int] = []
        for frame_idx in range(10):
            offset = frame_idx * n
            samples = [
                int(10000 * math.sin(2 * math.pi * freq * (offset + i) / 8000)) for i in range(n)
            ]
            data = struct.pack(f"<{n}h", *samples)
            frame = _frame(data, rate=8000)
            result = provider.resample(frame, 16000, 1, 2)
            out = _decode(result.data)
            all_output.extend(out)

        # Frame 0 → nothing, frames 1-9 → resampled frames 0-8
        # Total: 9 * 320 = 2880 samples at 16kHz
        assert len(all_output) == 2880

        # One-shot reference: frames 0-8 as a single chunk, flushed by frame 9
        ref_samples: list[int] = []
        for frame_idx in range(9):
            offset = frame_idx * n
            ref_samples.extend(
                [int(10000 * math.sin(2 * math.pi * freq * (offset + i) / 8000)) for i in range(n)]
            )
        ref_data = struct.pack(f"<{n * 9}h", *ref_samples)
        ref_frame = _frame(ref_data, rate=8000)
        ref_provider = SincResamplerProvider()
        ref_provider.resample(ref_frame, 16000, 1, 2)  # buffered

        # Flush with frame 9 to get the resampled output for frames 0-8
        flush_offset = 9 * n
        flush_samples = [
            int(10000 * math.sin(2 * math.pi * freq * (flush_offset + i) / 8000)) for i in range(n)
        ]
        flush_data = struct.pack(f"<{n}h", *flush_samples)
        flush_frame = _frame(flush_data, rate=8000)
        ref_result = ref_provider.resample(flush_frame, 16000, 1, 2)
        ref_output = _decode(ref_result.data)

        # ref_output = resampled frames 0-8 = 2880 samples at 16kHz
        assert len(ref_output) == 2880

        # Compare streaming frames 1-8 vs reference frames 1-8
        # (skip frame 0 where left context differs slightly)
        stream_slice = all_output[320:2560]
        ref_slice = ref_output[320:2560]
        max_diff = max(abs(a - b) for a, b in zip(stream_slice, ref_slice, strict=True))
        # Allow tiny rounding differences (integer arithmetic)
        assert max_diff <= 1, f"Max diff between streaming and one-shot: {max_diff}"

    def test_flush_outputs_pending_frame(self):
        """flush() after first (buffered) frame outputs the pending frame."""
        provider = SincResamplerProvider()
        n = 160
        samples = [int(1000 * math.sin(2 * math.pi * 400 * i / 8000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=8000)

        # First frame is buffered (no output)
        result = provider.resample(frame, 16000, 1, 2)
        assert len(result.data) == 0

        # Flush should emit the pending frame
        flushed = provider.flush(16000, 1, 2)
        assert flushed is not None
        assert flushed.sample_rate == 16000
        assert flushed.channels == 1
        assert flushed.sample_width == 2
        out = _decode(flushed.data)
        assert len(out) == 320  # 160 * 16000/8000

    def test_flush_clears_state(self):
        """State is empty after flush."""
        provider = SincResamplerProvider()
        n = 160
        samples = [int(1000 * math.sin(2 * math.pi * 400 * i / 8000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=8000)

        provider.resample(frame, 16000, 1, 2)
        assert len(provider._state) > 0

        provider.flush(16000, 1, 2)
        assert len(provider._state) == 0

    def test_flush_no_pending_returns_none(self):
        """flush() returns None when no pending frame exists."""
        provider = SincResamplerProvider()
        assert provider.flush(16000, 1, 2) is None

    def test_flush_after_multiple_frames(self):
        """flush() after 3 frames returns last pending frame."""
        provider = SincResamplerProvider()
        n = 160

        for i in range(3):
            offset = i * n
            samples = [
                int(1000 * math.sin(2 * math.pi * 400 * (offset + j) / 8000)) for j in range(n)
            ]
            data = struct.pack(f"<{n}h", *samples)
            frame = _frame(data, rate=8000)
            provider.resample(frame, 16000, 1, 2)

        flushed = provider.flush(16000, 1, 2)
        assert flushed is not None
        out = _decode(flushed.data)
        assert len(out) == 320  # one frame worth at 16kHz

    def test_flush_signal_quality(self):
        """Flushed output contains no garbage values."""
        provider = SincResamplerProvider()
        n = 160
        samples = [int(500 * math.sin(2 * math.pi * 400 * i / 8000)) for i in range(n)]
        data = struct.pack(f"<{n}h", *samples)
        frame = _frame(data, rate=8000)

        provider.resample(frame, 16000, 1, 2)
        flushed = provider.flush(16000, 1, 2)
        assert flushed is not None
        out = _decode(flushed.data)
        for s in out:
            assert -32768 <= s <= 32767
            # Signal peak is ~500, allow some overshoot from sinc interpolation
            assert abs(s) <= 1000, f"Garbage value in flushed output: {s}"

    def test_reimport_from_package(self):
        """SincResamplerProvider is importable from the resampler package."""
        from roomkit.voice.pipeline.resampler import SincResamplerProvider as Cls

        assert Cls is SincResamplerProvider

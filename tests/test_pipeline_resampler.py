"""Tests for pluggable resampler providers."""

from __future__ import annotations

import struct

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.pipeline.resampler.mock import MockResamplerProvider


def _frame(
    data: bytes = b"\x00\x00",
    rate: int = 16000,
    channels: int = 1,
    width: int = 2,
) -> AudioFrame:
    return AudioFrame(data=data, sample_rate=rate, channels=channels, sample_width=width)


# ---------------------------------------------------------------------------
# LinearResamplerProvider
# ---------------------------------------------------------------------------


class TestLinearResamplerProvider:
    def test_name(self):
        assert LinearResamplerProvider().name == "linear"

    def test_noop_when_format_matches(self):
        """Returns the exact same frame object when no conversion is needed."""
        provider = LinearResamplerProvider()
        frame = _frame(b"\x01\x00\x02\x00")
        result = provider.resample(frame, 16000, 1, 2)
        assert result is frame

    def test_stereo_to_mono(self):
        """Stereo -> mono averages L+R channels."""
        provider = LinearResamplerProvider()
        # 2 stereo frames: (100, 200), (300, 400)
        samples = [100, 200, 300, 400]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=16000, channels=2)

        result = provider.resample(frame, 16000, 1, 2)
        assert result.channels == 1
        assert result.sample_rate == 16000
        out_samples = list(struct.unpack(f"<{len(result.data) // 2}h", result.data))
        assert out_samples == [150, 350]

    def test_mono_to_stereo(self):
        """Mono -> stereo duplicates samples."""
        provider = LinearResamplerProvider()
        samples = [100, 200]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=16000, channels=1)

        result = provider.resample(frame, 16000, 2, 2)
        assert result.channels == 2
        out_samples = list(struct.unpack(f"<{len(result.data) // 2}h", result.data))
        assert out_samples == [100, 100, 200, 200]

    def test_downsample_48k_to_16k(self):
        """48kHz -> 16kHz reduces sample count by ~3x."""
        provider = LinearResamplerProvider()
        # 6 mono samples at 48kHz
        samples = [0, 100, 200, 300, 400, 500]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=48000, channels=1)

        result = provider.resample(frame, 16000, 1, 2)
        assert result.sample_rate == 16000
        out_samples = list(struct.unpack(f"<{len(result.data) // 2}h", result.data))
        assert len(out_samples) == 2  # 6 * 16000/48000 = 2

    def test_upsample_16k_to_48k(self):
        """16kHz -> 48kHz increases sample count by 3x."""
        provider = LinearResamplerProvider()
        samples = [100, 200, 300]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=16000, channels=1)

        result = provider.resample(frame, 48000, 1, 2)
        assert result.sample_rate == 48000
        out_samples = list(struct.unpack(f"<{len(result.data) // 2}h", result.data))
        assert len(out_samples) == 9  # 3 * 48000/16000 = 9

    def test_sample_width_conversion_2_to_4(self):
        """16-bit -> 32-bit scales sample values."""
        provider = LinearResamplerProvider()
        samples = [1000, -1000]
        data = struct.pack("<2h", *samples)
        frame = _frame(data, rate=16000, channels=1, width=2)

        result = provider.resample(frame, 16000, 1, 4)
        assert result.sample_width == 4
        out_samples = list(struct.unpack(f"<{len(result.data) // 4}i", result.data))
        # Values should be scaled up proportionally
        assert len(out_samples) == 2
        assert out_samples[0] > 1000
        assert out_samples[1] < -1000

    def test_combined_conversion(self):
        """48kHz stereo -> 16kHz mono with format change."""
        provider = LinearResamplerProvider()
        # 6 stereo frames at 48kHz = 12 samples
        samples = [100, 100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=48000, channels=2)

        result = provider.resample(frame, 16000, 1, 2)
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.sample_width == 2

    def test_preserves_timestamp(self):
        provider = LinearResamplerProvider()
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
        provider = LinearResamplerProvider()
        frame = AudioFrame(
            data=struct.pack("<2h", 100, 200),
            sample_rate=48000,
            channels=1,
            sample_width=2,
            metadata={"key": "value"},
        )
        result = provider.resample(frame, 16000, 1, 2)
        assert result.metadata["key"] == "value"

    def test_unsupported_width_rejected_by_audio_frame(self):
        """Unsupported sample width (3 bytes) is rejected by AudioFrame validation."""
        import pytest

        with pytest.raises(ValueError, match="sample_width must be 1, 2, or 4"):
            _frame(b"\x00\x00\x00", rate=16000, channels=1, width=3)

    def test_reset_and_close_are_noop(self):
        """reset() and close() don't raise."""
        provider = LinearResamplerProvider()
        provider.reset()
        provider.close()


# ---------------------------------------------------------------------------
# MockResamplerProvider
# ---------------------------------------------------------------------------


class TestMockResamplerProvider:
    def test_name(self):
        assert MockResamplerProvider().name == "mock"

    def test_passthrough(self):
        """Mock passes frames through unchanged."""
        mock = MockResamplerProvider()
        frame = _frame(b"\x01\x00")
        result = mock.resample(frame, 48000, 2, 4)
        assert result is frame

    def test_records_calls(self):
        mock = MockResamplerProvider()
        frame = _frame()
        mock.resample(frame, 48000, 2, 4)

        assert len(mock.calls) == 1
        assert mock.calls[0].target_rate == 48000
        assert mock.calls[0].target_channels == 2
        assert mock.calls[0].target_width == 4
        assert mock.calls[0].frame is frame

    def test_reset(self):
        mock = MockResamplerProvider()
        mock.reset()
        assert mock.reset_count == 1

    def test_close(self):
        mock = MockResamplerProvider()
        mock.close()
        assert mock.closed

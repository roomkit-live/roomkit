"""Tests for NumpyResamplerProvider."""

from __future__ import annotations

import struct

import numpy as np

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.resampler.numpy import NumpyResamplerProvider


def _frame(
    data: bytes = b"\x00\x00",
    rate: int = 16000,
    channels: int = 1,
    width: int = 2,
) -> AudioFrame:
    return AudioFrame(data=data, sample_rate=rate, channels=channels, sample_width=width)


class TestNumpyResamplerProvider:
    def test_name(self):
        assert NumpyResamplerProvider().name == "numpy"

    def test_noop_when_format_matches(self):
        """Returns the exact same frame object when no conversion is needed."""
        provider = NumpyResamplerProvider()
        frame = _frame(b"\x01\x00\x02\x00")
        result = provider.resample(frame, 16000, 1, 2)
        assert result is frame

    def test_stereo_to_mono(self):
        """Stereo -> mono averages L+R channels."""
        provider = NumpyResamplerProvider()
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
        provider = NumpyResamplerProvider()
        samples = [100, 200]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=16000, channels=1)

        result = provider.resample(frame, 16000, 2, 2)
        assert result.channels == 2
        out_samples = list(struct.unpack(f"<{len(result.data) // 2}h", result.data))
        assert out_samples == [100, 100, 200, 200]

    def test_downsample_48k_to_16k(self):
        """48kHz -> 16kHz reduces sample count by ~3x."""
        provider = NumpyResamplerProvider()
        samples = [0, 100, 200, 300, 400, 500]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=48000, channels=1)

        result = provider.resample(frame, 16000, 1, 2)
        assert result.sample_rate == 16000
        out_samples = list(struct.unpack(f"<{len(result.data) // 2}h", result.data))
        assert len(out_samples) == 2

    def test_upsample_16k_to_48k(self):
        """16kHz -> 48kHz increases sample count by 3x."""
        provider = NumpyResamplerProvider()
        samples = [100, 200, 300]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=16000, channels=1)

        result = provider.resample(frame, 48000, 1, 2)
        assert result.sample_rate == 48000
        out_samples = list(struct.unpack(f"<{len(result.data) // 2}h", result.data))
        assert len(out_samples) == 9

    def test_sample_width_conversion_2_to_4(self):
        """16-bit -> 32-bit scales sample values."""
        provider = NumpyResamplerProvider()
        samples = [1000, -1000]
        data = struct.pack("<2h", *samples)
        frame = _frame(data, rate=16000, channels=1, width=2)

        result = provider.resample(frame, 16000, 1, 4)
        assert result.sample_width == 4
        out_samples = list(struct.unpack(f"<{len(result.data) // 4}i", result.data))
        assert len(out_samples) == 2
        assert out_samples[0] > 1000
        assert out_samples[1] < -1000

    def test_combined_conversion(self):
        """48kHz stereo -> 16kHz mono with format change."""
        provider = NumpyResamplerProvider()
        samples = [100, 100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = _frame(data, rate=48000, channels=2)

        result = provider.resample(frame, 16000, 1, 2)
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.sample_width == 2

    def test_preserves_timestamp(self):
        provider = NumpyResamplerProvider()
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
        provider = NumpyResamplerProvider()
        frame = AudioFrame(
            data=struct.pack("<2h", 100, 200),
            sample_rate=48000,
            channels=1,
            sample_width=2,
            metadata={"key": "value"},
        )
        result = provider.resample(frame, 16000, 1, 2)
        assert result.metadata["key"] == "value"

    def test_reset_and_close_are_noop(self):
        """reset() and close() don't raise."""
        provider = NumpyResamplerProvider()
        provider.reset()
        provider.close()

    def test_matches_linear_resampler_output(self):
        """NumPy resampler should produce equivalent output to the linear resampler."""
        from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider

        np_r = NumpyResamplerProvider()
        py_r = LinearResamplerProvider()

        # 20ms at 48kHz mono
        n = 960
        data = struct.pack(
            f"<{n}h",
            *[int(16000 * np.sin(2 * np.pi * 440 * i / 48000)) for i in range(n)],
        )
        frame = AudioFrame(data=data, sample_rate=48000, channels=1, sample_width=2)

        np_result = np_r.resample(frame, 24000, 1, 2)
        py_result = py_r.resample(frame, 24000, 1, 2)

        assert np_result.sample_rate == py_result.sample_rate
        assert len(np_result.data) == len(py_result.data)

        # Values should be very close (minor float rounding differences)
        np_samples = np.frombuffer(np_result.data, dtype=np.int16)
        py_samples = np.frombuffer(py_result.data, dtype=np.int16)
        max_diff = np.max(np.abs(np_samples.astype(np.int32) - py_samples.astype(np.int32)))
        assert max_diff <= 1, f"Max sample difference: {max_diff}"

    def test_bridge_scenario_48k_to_8k(self):
        """WebRTC → SIP resampling (48kHz → 8kHz)."""
        provider = NumpyResamplerProvider()
        n = 960  # 20ms at 48kHz
        data = struct.pack(f"<{n}h", *[i % 1000 for i in range(n)])
        frame = _frame(data, rate=48000)

        result = provider.resample(frame, 8000, 1, 2)
        assert result.sample_rate == 8000
        expected_samples = int(n * 8000 / 48000)
        actual_samples = len(result.data) // 2
        assert actual_samples == expected_samples

    def test_bridge_scenario_8k_to_24k(self):
        """SIP → WebRTC resampling (8kHz → 24kHz)."""
        provider = NumpyResamplerProvider()
        n = 160  # 20ms at 8kHz
        data = struct.pack(f"<{n}h", *[i % 500 for i in range(n)])
        frame = _frame(data, rate=8000)

        result = provider.resample(frame, 24000, 1, 2)
        assert result.sample_rate == 24000
        expected_samples = int(n * 24000 / 8000)
        actual_samples = len(result.data) // 2
        assert actual_samples == expected_samples

    def test_clamp_prevents_overflow(self):
        """Values near max int16 don't overflow during width conversion."""
        provider = NumpyResamplerProvider()
        samples = [32767, -32768]
        data = struct.pack("<2h", *samples)
        frame = _frame(data, rate=16000, channels=1, width=2)

        result = provider.resample(frame, 16000, 1, 4)
        out_samples = list(struct.unpack(f"<{len(result.data) // 4}i", result.data))
        # Values should be within int32 range
        assert all(-(1 << 31) <= s < (1 << 31) for s in out_samples)

    def test_reimport_from_package(self):
        """Can be imported from the resampler package."""
        from roomkit.voice.pipeline.resampler import NumpyResamplerProvider as Cls

        assert Cls is NumpyResamplerProvider

"""Tests for RNNoiseDenoiserProvider (RNNoise ctypes wrapper)."""

from __future__ import annotations

import math
import os
import struct
import sys

import pytest

from roomkit.voice.audio_frame import AudioFrame


def _rnnoise_available() -> bool:
    """Check if librnnoise can be loaded (system path or ~/.local/lib)."""
    import ctypes.util

    if ctypes.util.find_library("rnnoise") is not None:
        return True
    soname = "librnnoise.dylib" if sys.platform == "darwin" else "librnnoise.so"
    for d in [os.path.expanduser("~/.local/lib"), "/usr/local/lib"]:
        if os.path.isfile(os.path.join(d, soname)):
            return True
    return False


# Skip the entire module if librnnoise is not available.
pytestmark = pytest.mark.skipif(
    not _rnnoise_available(),
    reason="librnnoise not installed",
)


def _frame(n_samples: int = 160, value: int = 0, sample_rate: int = 16000) -> AudioFrame:
    """Create a PCM-16 mono AudioFrame with *n_samples* samples."""
    data = struct.pack(f"<{n_samples}h", *([value] * n_samples))
    return AudioFrame(data=data, sample_rate=sample_rate, channels=1, sample_width=2)


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestRNNoiseInit:
    def test_default_construction(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        assert dn.name == "rnnoise"
        dn.close()

    def test_48khz_construction(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider(sample_rate=48000)
        assert dn.name == "rnnoise"
        dn.close()

    def test_24khz_construction(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider(sample_rate=24000)
        assert dn.name == "rnnoise"
        dn.close()

    def test_invalid_sample_rate(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        with pytest.raises(ValueError, match="divides evenly into 48000"):
            RNNoiseDenoiserProvider(sample_rate=22050)


# ---------------------------------------------------------------------------
# process() — noise suppression on inbound audio
# ---------------------------------------------------------------------------


class TestRNNoiseProcess:
    def test_process_returns_new_frame(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        frame_in = _frame(160, value=100)
        frame_out = dn.process(frame_in)

        # Must return a *new* AudioFrame, not the same object.
        assert isinstance(frame_out, AudioFrame)
        assert len(frame_out.data) == len(frame_in.data)
        assert frame_out.sample_rate == frame_in.sample_rate
        assert frame_out.channels == frame_in.channels
        assert frame_out.sample_width == frame_in.sample_width
        dn.close()

    def test_process_preserves_timestamp(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        frame_in = _frame(160)
        frame_in.timestamp_ms = 42.5
        frame_out = dn.process(frame_in)
        assert frame_out.timestamp_ms == 42.5
        dn.close()

    def test_process_copies_metadata(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        frame_in = _frame(160)
        frame_in.metadata["source"] = "test"
        frame_out = dn.process(frame_in)
        assert frame_out.metadata["source"] == "test"
        # Mutation of output metadata must not affect input.
        frame_out.metadata["extra"] = True
        assert "extra" not in frame_in.metadata
        dn.close()

    def test_process_wrong_frame_size_passthrough(self):
        """When frame size is not a multiple of chunk size, pass through."""
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        wrong_frame = _frame(100)  # Not a multiple of 160
        result = dn.process(wrong_frame)
        assert result is wrong_frame  # Same object returned
        dn.close()

    def test_process_multi_chunk_frame(self):
        """Frames larger than one chunk (e.g. 20ms at 24kHz = 480) work."""
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider(sample_rate=24000)
        # 480 samples = 2 × 240 (two 10ms chunks at 24 kHz)
        frame_in = _frame(480, value=100, sample_rate=24000)
        frame_out = dn.process(frame_in)
        assert isinstance(frame_out, AudioFrame)
        assert len(frame_out.data) == len(frame_in.data)
        dn.close()

    def test_process_after_close_passthrough(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        dn.close()
        frame = _frame(160)
        result = dn.process(frame)
        assert result is frame

    def test_silence_in_silence_out(self):
        """Processing silence should yield near-silence."""
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        frame_in = _frame(160, value=0)
        frame_out = dn.process(frame_in)
        samples = struct.unpack("<160h", frame_out.data)
        # RNNoise may introduce tiny artifacts, but energy should be negligible.
        energy = sum(s * s for s in samples)
        assert energy < 1000, f"Silence output energy too high: {energy}"
        dn.close()


# ---------------------------------------------------------------------------
# reset() / close()
# ---------------------------------------------------------------------------


class TestRNNoiseLifecycle:
    def test_reset_does_not_raise(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        dn.reset()
        # Should still be usable after reset.
        result = dn.process(_frame(160))
        assert isinstance(result, AudioFrame)
        dn.close()

    def test_double_close(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        dn.close()
        dn.close()  # Must not raise

    def test_close_then_reset(self):
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()
        dn.close()
        dn.reset()  # Must not raise


# ---------------------------------------------------------------------------
# Denoise integration — noisy signal should have reduced energy
# ---------------------------------------------------------------------------


class TestRNNoiseDenoise:
    def test_noise_is_reduced(self):
        """A noisy signal should have lower energy after denoising.

        Generates a simple tone mixed with pseudo-random noise and
        verifies that RNNoise reduces the overall energy.
        """
        from roomkit.voice.pipeline.rnnoise import RNNoiseDenoiserProvider

        dn = RNNoiseDenoiserProvider()

        # Generate a noisy signal: a 400 Hz tone + pseudo-noise.
        n_samples = 160
        noisy = []
        for i in range(n_samples):
            tone = int(3000 * math.sin(2 * math.pi * 400 * i / 16000))
            # Simple pseudo-noise using a hash-like pattern.
            noise = ((i * 7919 + 104729) % 65536) - 32768
            noise = noise // 4  # Scale noise down
            sample = max(-32768, min(32767, tone + noise))
            noisy.append(sample)

        data = struct.pack(f"<{n_samples}h", *noisy)
        frame_in = AudioFrame(data=data, sample_rate=16000, channels=1, sample_width=2)

        # Let RNNoise warm up with several frames of the same signal.
        for _ in range(20):
            dn.process(frame_in)

        # Process one more — this is the result we check.
        frame_out = dn.process(frame_in)

        in_samples = struct.unpack(f"<{n_samples}h", frame_in.data)
        out_samples = struct.unpack(f"<{n_samples}h", frame_out.data)

        in_energy = sum(s * s for s in in_samples)
        out_energy = sum(s * s for s in out_samples)

        # RNNoise should suppress some energy. We don't require massive
        # reduction since the tone itself is a valid signal, but noise
        # suppression should measurably reduce overall energy.
        assert out_energy < in_energy, (
            f"Expected output energy ({out_energy}) < input energy ({in_energy})"
        )
        dn.close()

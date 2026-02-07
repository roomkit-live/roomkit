"""Tests for SpeexAECProvider (SpeexDSP ctypes wrapper)."""

from __future__ import annotations

import ctypes
import ctypes.util
import struct

import pytest

from roomkit.voice.audio_frame import AudioFrame

# Skip the entire module if libspeexdsp is not available.
pytestmark = pytest.mark.skipif(
    ctypes.util.find_library("speexdsp") is None,
    reason="libspeexdsp not installed",
)


def _frame(n_samples: int = 320, value: int = 0) -> AudioFrame:
    """Create a PCM-16 mono AudioFrame with *n_samples* samples."""
    data = struct.pack(f"<{n_samples}h", *([value] * n_samples))
    return AudioFrame(data=data, sample_rate=16000, channels=1, sample_width=2)


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestSpeexAECInit:
    def test_default_construction(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider()
        assert aec.name == "speex_aec"
        aec.close()

    def test_custom_params(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=160, filter_length=1600, sample_rate=8000)
        assert aec.name == "speex_aec"
        aec.close()


# ---------------------------------------------------------------------------
# process() — echo cancellation on inbound audio
# ---------------------------------------------------------------------------


class TestSpeexAECProcess:
    def test_process_returns_new_frame(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        frame_in = _frame(320, value=100)
        frame_out = aec.process(frame_in)

        # Must return a *new* AudioFrame, not the same object.
        assert isinstance(frame_out, AudioFrame)
        assert len(frame_out.data) == len(frame_in.data)
        assert frame_out.sample_rate == frame_in.sample_rate
        assert frame_out.channels == frame_in.channels
        assert frame_out.sample_width == frame_in.sample_width
        aec.close()

    def test_process_preserves_timestamp(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        frame_in = _frame(320)
        frame_in.timestamp_ms = 42.5
        frame_out = aec.process(frame_in)
        assert frame_out.timestamp_ms == 42.5
        aec.close()

    def test_process_copies_metadata(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        frame_in = _frame(320)
        frame_in.metadata["source"] = "test"
        frame_out = aec.process(frame_in)
        assert frame_out.metadata["source"] == "test"
        # Mutation of output metadata must not affect input.
        frame_out.metadata["extra"] = True
        assert "extra" not in frame_in.metadata
        aec.close()

    def test_process_wrong_frame_size_passthrough(self):
        """When frame size doesn't match, frame should pass through unchanged."""
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        wrong_frame = _frame(160)  # Half the expected size
        result = aec.process(wrong_frame)
        assert result is wrong_frame  # Same object returned
        aec.close()

    def test_process_after_close_passthrough(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        aec.close()
        frame = _frame(320)
        result = aec.process(frame)
        assert result is frame

    def test_silence_in_silence_out(self):
        """Processing silence with no reference should yield silence."""
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        frame_in = _frame(320, value=0)
        frame_out = aec.process(frame_in)
        samples = struct.unpack(f"<{320}h", frame_out.data)
        assert all(s == 0 for s in samples)
        aec.close()


# ---------------------------------------------------------------------------
# feed_reference() — outbound playback frames
# ---------------------------------------------------------------------------


class TestSpeexAECFeedReference:
    def test_feed_reference_accepts_frame(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        ref = _frame(320, value=500)
        aec.feed_reference(ref)  # Should not raise
        aec.close()

    def test_feed_reference_wrong_size_ignored(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        wrong = _frame(160)
        aec.feed_reference(wrong)  # Should log warning, not raise
        aec.close()

    def test_feed_reference_after_close(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        aec.close()
        aec.feed_reference(_frame(320))  # Should not raise


# ---------------------------------------------------------------------------
# reset() / close()
# ---------------------------------------------------------------------------


class TestSpeexAECLifecycle:
    def test_reset_does_not_raise(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        aec.reset()
        # Should still be usable after reset.
        result = aec.process(_frame(320))
        assert isinstance(result, AudioFrame)
        aec.close()

    def test_double_close(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        aec.close()
        aec.close()  # Must not raise

    def test_close_then_reset(self):
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320)
        aec.close()
        aec.reset()  # Must not raise


# ---------------------------------------------------------------------------
# Echo cancellation integration — feed reference then process
# ---------------------------------------------------------------------------


class TestSpeexAECEchoCancellation:
    def test_echo_is_attenuated(self):
        """When reference matches capture, AEC should reduce the output energy.

        Uses the split API: interleave feed_reference() (playback) and
        process() (capture) so the adaptive filter can converge.
        """
        from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

        aec = SpeexAECProvider(frame_size=320, filter_length=3200)

        # Simulate a loud tone playing through speakers (reference)
        # and arriving at the mic (capture) simultaneously.
        tone_value = 10000
        ref = _frame(320, value=tone_value)
        capture = _frame(320, value=tone_value)

        # Interleave feed_reference + process so the adaptive filter
        # converges on the echo path.
        for _ in range(50):
            aec.feed_reference(ref)
            aec.process(capture)

        # One more round — this is the result we check.
        aec.feed_reference(ref)
        result = aec.process(capture)

        # The output energy should be lower than the input energy.
        in_samples = struct.unpack("<320h", capture.data)
        out_samples = struct.unpack("<320h", result.data)

        in_energy = sum(s * s for s in in_samples)
        out_energy = sum(s * s for s in out_samples)

        assert out_energy < in_energy, (
            f"Expected output energy ({out_energy}) < input energy ({in_energy})"
        )
        aec.close()

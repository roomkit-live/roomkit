"""LocalAudioBackend AEC tests.

Covers transport-level AEC reference feeding, rate mismatch resampling,
and the feeds_aec_reference property.

Tests are skipped when sounddevice/numpy are not installed.
"""

from __future__ import annotations

import pytest

sd = pytest.importorskip("sounddevice")
np = pytest.importorskip("numpy")

from roomkit.voice.backends.base import VoiceBackend  # noqa: E402
from roomkit.voice.backends.local import LocalAudioBackend  # noqa: E402
from roomkit.voice.pipeline.aec.mock import MockAECProvider  # noqa: E402


class TestFeedsAECReferenceProperty:
    """Tests for VoiceBackend.feeds_aec_reference default and LocalAudioBackend override."""

    def test_abc_defaults_false(self):
        """VoiceBackend subclass without override returns False."""

        class MinimalBackend(VoiceBackend):
            @property
            def name(self):
                return "minimal"

            async def connect(self, room_id, participant_id, channel_id, *, metadata=None):
                pass

            async def disconnect(self, session):
                pass

            async def send_audio(self, session, audio):
                pass

        backend = MinimalBackend()
        assert backend.feeds_aec_reference is False

    def test_local_backend_false_without_aec(self):
        """LocalAudioBackend without AEC returns False."""
        backend = LocalAudioBackend(aec=None)
        assert backend.feeds_aec_reference is False

    def test_local_backend_true_with_aec(self):
        """LocalAudioBackend with AEC returns True."""
        aec = MockAECProvider()
        backend = LocalAudioBackend(aec=aec)
        assert backend.feeds_aec_reference is True


class TestAECFeedPlayed:
    """Tests for LocalAudioBackend._aec_feed_played transport-level reference feeding."""

    def test_feed_played_with_rate_mismatch(self):
        """AEC reference resampled when output_rate != input_rate."""
        aec = MockAECProvider()
        backend = LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=24000,
            channels=1,
            block_duration_ms=20,
            aec=aec,
        )

        # Output block size: 24000 * 20/1000 * 1 * 2 = 960 bytes
        output_block = bytearray(b"\x01\x00" * (24000 * 20 // 1000))
        backend._aec_feed_played(output_block)

        assert len(aec.reference_frames) == 1
        assert aec.reference_frames[0].sample_rate == 16000

    def test_feed_played_without_rate_mismatch(self):
        """AEC reference fed directly when output_rate == input_rate."""
        aec = MockAECProvider()
        backend = LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=16000,
            channels=1,
            block_duration_ms=20,
            aec=aec,
        )

        # Block size: 16000 * 20/1000 * 1 * 2 = 640 bytes
        block = bytearray(b"\x01\x00" * (16000 * 20 // 1000))
        backend._aec_feed_played(block)

        assert len(aec.reference_frames) == 1
        assert aec.reference_frames[0].sample_rate == 16000

    def test_feed_played_accumulates_partial_blocks(self):
        """_aec_feed_played accumulates data until a full block is available."""
        aec = MockAECProvider()
        backend = LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=16000,
            channels=1,
            block_duration_ms=20,
            aec=aec,
        )

        # Block size = 640 bytes; send less than one block
        backend._aec_feed_played(bytearray(b"\x01\x00" * 100))  # 200 bytes
        assert len(aec.reference_frames) == 0

        # Send more to complete the block
        backend._aec_feed_played(bytearray(b"\x01\x00" * 220))  # 440 bytes, total 640
        assert len(aec.reference_frames) == 1

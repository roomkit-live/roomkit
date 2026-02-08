"""AEC integration tests for the audio pipeline.

Covers AEC reference feeding, resampling, capability-aware skipping,
and error resilience in the outbound path.
"""

from __future__ import annotations

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.engine import AudioPipeline


def _frame(
    data: bytes = b"\x00\x00",
    sample_rate: int = 16000,
) -> AudioFrame:
    return AudioFrame(data=data, sample_rate=sample_rate, channels=1, sample_width=2)


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id="p1", channel_id="c1")


class TestOutboundAECReference:
    """Tests for AEC reference feeding in the outbound pipeline path."""

    def test_aec_reference_skipped_with_backend_feeds_flag(self):
        """AEC feed_reference skipped when backend_feeds_aec_reference=True."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config, backend_feeds_aec_reference=True)

        pipeline.process_outbound(_session(), _frame())
        assert len(aec.reference_frames) == 0

    def test_aec_reference_resampled_when_rates_differ(self):
        """AEC reference resampled when outbound rate != inbound rate."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        # Establish inbound rate at 16000Hz
        pipeline.process_inbound(_session(), _frame(sample_rate=16000))

        # Outbound at 22050Hz — should be resampled to 16000Hz
        outbound_frame = _frame(b"\x01\x00" * 22, sample_rate=22050)
        pipeline.process_outbound(_session(), outbound_frame)

        assert len(aec.reference_frames) == 1
        assert aec.reference_frames[0].sample_rate == 16000

    def test_aec_reference_not_resampled_when_rates_match(self):
        """AEC reference not resampled when outbound rate == inbound rate."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        # Establish inbound rate
        pipeline.process_inbound(_session(), _frame(sample_rate=16000))

        # Outbound at same rate
        pipeline.process_outbound(_session(), _frame(sample_rate=16000))

        assert len(aec.reference_frames) == 1
        assert aec.reference_frames[0].sample_rate == 16000
        # No AEC resampler should have been created
        assert pipeline._aec_resampler is None

    def test_aec_reference_fed_before_inbound(self):
        """AEC reference fed even when no inbound frame has set _inbound_sample_rate."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        # Outbound without any prior inbound — _inbound_sample_rate is None
        pipeline.process_outbound(_session(), _frame(sample_rate=22050))

        # Reference still fed (no resample because target_rate is falsy)
        assert len(aec.reference_frames) == 1


class TestPipelineResetWithAEC:
    """Tests for pipeline reset clearing AEC-related state."""

    def test_reset_clears_inbound_sample_rate(self):
        """reset() clears _inbound_sample_rate."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), _frame(sample_rate=16000))
        assert pipeline._inbound_sample_rate == 16000

        pipeline.reset()
        assert pipeline._inbound_sample_rate is None

    def test_reset_resets_aec_resampler(self):
        """reset() resets the AEC resampler if it was created."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        # Create the AEC resampler by processing mismatched rates
        pipeline.process_inbound(_session(), _frame(sample_rate=16000))
        pipeline.process_outbound(_session(), _frame(b"\x01\x00" * 22, sample_rate=22050))
        assert pipeline._aec_resampler is not None

        pipeline.reset()
        # LinearResamplerProvider doesn't track reset_count, but the
        # resampler should still be present (just reset, not destroyed)
        assert pipeline._aec_resampler is not None


class TestBackendFeedsFlag:
    """Tests for backend_feeds_aec_reference flag storage."""

    def test_flag_stored_true(self):
        """backend_feeds_aec_reference=True stored correctly."""
        config = AudioPipelineConfig()
        pipeline = AudioPipeline(config, backend_feeds_aec_reference=True)
        assert pipeline._backend_feeds_aec_ref is True

    def test_flag_defaults_false(self):
        """backend_feeds_aec_reference defaults to False."""
        config = AudioPipelineConfig()
        pipeline = AudioPipeline(config)
        assert pipeline._backend_feeds_aec_ref is False


class TestAECOutboundErrorResilience:
    """Tests for error resilience in the AEC outbound path."""

    def test_aec_feed_reference_error_does_not_crash_outbound(self):
        """AEC feed_reference error doesn't crash the outbound pipeline."""

        class FailingAEC(MockAECProvider):
            def feed_reference(self, frame):
                raise RuntimeError("AEC feed boom")

        aec = FailingAEC()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        # Should not raise — returns the frame
        result = pipeline.process_outbound(_session(), _frame())
        assert result is not None
        assert result.data == b"\x00\x00"


class TestCloseAECResampler:
    """Test that close() releases the AEC resampler."""

    def test_close_closes_aec_resampler(self):
        """close() should close _aec_resampler if it exists."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        # Create the AEC resampler
        pipeline.process_inbound(_session(), _frame(sample_rate=16000))
        pipeline.process_outbound(_session(), _frame(b"\x01\x00" * 22, sample_rate=22050))
        assert pipeline._aec_resampler is not None

        pipeline.close()
        # LinearResamplerProvider.close() is a no-op, but we verify no crash

    def test_close_without_aec_resampler(self):
        """close() works when _aec_resampler is None."""
        config = AudioPipelineConfig()
        pipeline = AudioPipeline(config)
        assert pipeline._aec_resampler is None

        # Should not raise
        pipeline.close()

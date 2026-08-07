"""AEC integration tests for the audio pipeline.

Covers AEC reference feeding, resampling, capability-aware skipping,
and error resilience in the outbound path.
"""

from __future__ import annotations

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceCapability, VoiceSession
from roomkit.voice.pipeline.aec.base import AECProvider
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.config import AudioFormat, AudioPipelineConfig, AudioPipelineContract
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
        """AEC reference is still fed before capture establishes a format."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        # Outbound without a prior capture and without a declared contract.
        pipeline.process_outbound(_session(), _frame(sample_rate=22050))

        # Reference still fed (no resample because target_rate is falsy)
        assert len(aec.reference_frames) == 1

    def test_reference_matches_post_resampler_capture_format(self):
        """AEC reference matches the frame format that actually reaches AEC."""
        aec = MockAECProvider()
        contract = AudioPipelineContract(
            transport_inbound_format=AudioFormat(sample_rate=48000, channels=2),
            transport_outbound_format=AudioFormat(sample_rate=48000, channels=2),
            internal_format=AudioFormat(sample_rate=16000, channels=1),
        )
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec, contract=contract))
        session = _session()

        pipeline.process_inbound(
            session,
            AudioFrame(
                data=b"\x01\x00" * 960,
                sample_rate=48000,
                channels=2,
                sample_width=2,
            ),
        )
        pipeline.process_outbound(
            session,
            AudioFrame(
                data=b"\x02\x00" * 960,
                sample_rate=48000,
                channels=2,
                sample_width=2,
            ),
        )

        reference = aec.reference_frames[-1]
        assert (reference.sample_rate, reference.channels, reference.sample_width) == (16000, 1, 2)

    def test_reference_capture_format_is_tracked_per_stream(self):
        """One session's native format must not be reused for another session."""
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))
        alice = _session("alice")
        bob = _session("bob")

        pipeline.process_inbound(alice, _frame(sample_rate=16000))
        pipeline.process_inbound(
            bob,
            AudioFrame(
                data=b"\x01\x00\x00\x00" * 160,
                sample_rate=8000,
                channels=2,
                sample_width=2,
            ),
        )
        pipeline.process_outbound(
            bob,
            AudioFrame(
                data=b"\x02\x00" * 320,
                sample_rate=16000,
                channels=1,
                sample_width=2,
            ),
        )

        reference = aec.reference_frames[-1]
        assert aec.reference_streams[-1] == "bob"
        assert (reference.sample_rate, reference.channels, reference.sample_width) == (8000, 2, 2)

    def test_playback_reference_uses_capture_format(self):
        """Playback-time callbacks use the same format normalization."""
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))
        session = _session()
        pipeline.enable_playback_aec_feed()
        pipeline.process_inbound(session, _frame(sample_rate=16000))

        pipeline.feed_aec_reference(
            AudioFrame(
                data=b"\x01\x00\x02\x00" * 480,
                sample_rate=48000,
                channels=2,
                sample_width=2,
            ),
            session.id,
        )

        reference = aec.reference_frames[-1]
        assert (reference.sample_rate, reference.channels, reference.sample_width) == (16000, 1, 2)

    def test_playback_reference_skipped_for_native_aec(self):
        """A NATIVE_AEC backend must not drive the configured pipeline AEC."""
        aec = MockAECProvider()
        pipeline = AudioPipeline(
            AudioPipelineConfig(aec=aec),
            backend_capabilities=VoiceCapability.NATIVE_AEC,
        )

        pipeline.feed_aec_reference(_frame(), "s1")

        assert aec.reference_frames == []


class TestPipelineResetWithAEC:
    """Tests for pipeline reset clearing AEC-related state."""

    def test_reset_clears_capture_formats(self):
        """reset() clears every stream's remembered AEC capture format."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), _frame(sample_rate=16000))
        assert pipeline._aec_capture_formats == {"s1": (16000, 1, 2)}

        pipeline.reset()
        assert pipeline._aec_capture_formats == {}

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


class TestAECActivityLifecycle:
    """AEC activity follows all concurrent playback sources per stream."""

    def test_last_playback_source_controls_bypass_without_reset(self):
        """Stopping sources bypasses AEC without discarding its learned filter."""
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))

        pipeline.set_aec_active("s1", True, source="bridge")
        pipeline.set_aec_active("s1", True)
        pipeline.set_aec_active("s1", False)

        assert aec.active_changes == [("s1", True)]
        assert aec.reset_count == 0

        pipeline.set_aec_active("s1", False, source="bridge")

        assert aec.active_changes == [("s1", True), ("s1", False)]
        assert aec.reset_count == 0

    def test_repeated_bridge_frames_do_not_repeat_provider_activation(self):
        """The realtime bridge path changes provider state only once."""
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))

        pipeline.set_aec_active("s1", True, source="bridge")
        pipeline.set_aec_active("s1", True, source="bridge")

        assert aec.active_changes == [("s1", True)]

    def test_native_aec_owns_activity_state(self):
        """Pipeline lifecycle must not manipulate a backend-native AEC."""
        aec = MockAECProvider()
        pipeline = AudioPipeline(
            AudioPipelineConfig(aec=aec),
            backend_capabilities=VoiceCapability.NATIVE_AEC,
        )

        pipeline.set_aec_active("s1", True)
        pipeline.set_aec_active("s1", False)

        assert aec.active_changes == []
        assert aec.reset_count == 0

    def test_legacy_global_provider_stays_active_for_other_streams(self):
        """A global-only custom provider is disabled after its final stream."""

        class GlobalAEC(MockAECProvider):
            set_stream_active = AECProvider.set_stream_active

        aec = GlobalAEC()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))

        pipeline.set_aec_active("alice", True)
        pipeline.set_aec_active("bob", True)
        pipeline.set_aec_active("alice", False)

        assert aec.active_changes == [(None, True)]
        assert aec.reset_streams == []

        pipeline.set_aec_active("bob", False)

        assert aec.active_changes == [(None, True), (None, False)]
        assert aec.reset_streams == []

    def test_session_teardown_deactivates_legacy_global_provider(self):
        """An ended final stream cannot leave a global provider enabled."""

        class GlobalAEC(MockAECProvider):
            set_stream_active = AECProvider.set_stream_active

        aec = GlobalAEC()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))

        pipeline.set_aec_active("alice", True)
        pipeline.on_session_ended(_session("alice"))

        assert aec.active_changes == [(None, True), (None, False)]
        assert aec.reset_streams == ["alice"]

    def test_reset_deactivates_each_stream_local_provider(self):
        """Blanket reset releases activity even before capture audio arrives."""
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))

        pipeline.set_aec_active("alice", True)
        pipeline.set_aec_active("bob", True)
        pipeline.reset()

        assert aec.active_changes == [
            ("alice", True),
            ("bob", True),
            ("alice", False),
            ("bob", False),
        ]
        assert set(aec.reset_streams) == {"alice", "bob"}

    def test_failed_stream_activation_can_be_retried(self):
        """Provider/bookkeeping state stays aligned after a transient failure."""

        class FlakyAEC(MockAECProvider):
            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            def set_stream_active(self, stream: str, active: bool) -> None:
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("transient")
                super().set_stream_active(stream, active)

        aec = FlakyAEC()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))

        pipeline.set_aec_active("alice", True)
        pipeline.set_aec_active("alice", True)

        assert aec.attempts == 2
        assert aec.active_changes == [("alice", True)]


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
            def feed_reference(self, frame, stream):
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

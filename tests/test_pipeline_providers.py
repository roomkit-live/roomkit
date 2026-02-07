"""Tests for pipeline v2 provider ABCs and data types."""

from __future__ import annotations

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.aec.base import AECProvider
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.agc.base import AGCConfig, AGCProvider
from roomkit.voice.pipeline.agc.mock import MockAGCProvider
from roomkit.voice.pipeline.backchannel.base import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)
from roomkit.voice.pipeline.backchannel.mock import MockBackchannelDetector
from roomkit.voice.pipeline.config import AudioFormat, AudioPipelineContract
from roomkit.voice.pipeline.dtmf.base import DTMFDetector, DTMFEvent
from roomkit.voice.pipeline.dtmf.mock import MockDTMFDetector
from roomkit.voice.pipeline.recorder.base import (
    AudioRecorder,
    RecordingChannelMode,
    RecordingConfig,
    RecordingMode,
    RecordingTrigger,
)
from roomkit.voice.pipeline.recorder.mock import MockAudioRecorder
from roomkit.voice.pipeline.turn.base import (
    TurnContext,
    TurnDecision,
    TurnDetector,
    TurnEntry,
)
from roomkit.voice.pipeline.turn.mock import MockTurnDetector


def _frame(data: bytes = b"\x00\x00") -> AudioFrame:
    return AudioFrame(data=data, sample_rate=16000, channels=1, sample_width=2)


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id="p1", channel_id="c1")


# ---------------------------------------------------------------------------
# AGCProvider
# ---------------------------------------------------------------------------


class TestAGCProvider:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            AGCProvider()  # type: ignore[abstract]

    def test_mock_process(self):
        agc = MockAGCProvider()
        frame = _frame(b"\x01\x02")
        result = agc.process(frame)
        assert result is frame
        assert len(agc.frames) == 1

    def test_mock_reset(self):
        agc = MockAGCProvider()
        agc.reset()
        assert agc.reset_count == 1

    def test_mock_close(self):
        agc = MockAGCProvider()
        agc.close()
        assert agc.closed

    def test_agc_config_defaults(self):
        cfg = AGCConfig()
        assert cfg.target_level_dbfs == -3.0
        assert cfg.max_gain_db == 30.0
        assert cfg.attack_ms == 10.0
        assert cfg.release_ms == 100.0
        assert cfg.metadata == {}


# ---------------------------------------------------------------------------
# AECProvider
# ---------------------------------------------------------------------------


class TestAECProvider:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            AECProvider()  # type: ignore[abstract]

    def test_mock_process(self):
        aec = MockAECProvider()
        frame = _frame()
        result = aec.process(frame)
        assert result is frame
        assert len(aec.frames) == 1

    def test_mock_feed_reference(self):
        aec = MockAECProvider()
        frame = _frame()
        aec.feed_reference(frame)
        assert len(aec.reference_frames) == 1

    def test_mock_reset(self):
        aec = MockAECProvider()
        aec.reset()
        assert aec.reset_count == 1

    def test_mock_close(self):
        aec = MockAECProvider()
        aec.close()
        assert aec.closed


# ---------------------------------------------------------------------------
# DTMFDetector
# ---------------------------------------------------------------------------


class TestDTMFDetector:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            DTMFDetector()  # type: ignore[abstract]

    def test_mock_process_with_event(self):
        event = DTMFEvent(digit="5", duration_ms=100.0, confidence=0.99)
        detector = MockDTMFDetector(events=[event])
        result = detector.process(_frame())
        assert result is event
        assert result.digit == "5"

    def test_mock_process_no_event(self):
        detector = MockDTMFDetector()
        assert detector.process(_frame()) is None

    def test_mock_reset(self):
        detector = MockDTMFDetector(events=[DTMFEvent(digit="1", duration_ms=50.0)])
        detector.process(_frame())
        detector.reset()
        assert detector.reset_count == 1
        # After reset, sequence replays
        result = detector.process(_frame())
        assert result is not None and result.digit == "1"

    def test_dtmf_event_defaults(self):
        event = DTMFEvent(digit="#", duration_ms=80.0)
        assert event.confidence == 1.0


# ---------------------------------------------------------------------------
# AudioRecorder
# ---------------------------------------------------------------------------


class TestAudioRecorder:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            AudioRecorder()  # type: ignore[abstract]

    def test_mock_lifecycle(self):
        recorder = MockAudioRecorder()
        session = _session()
        config = RecordingConfig()

        handle = recorder.start(session, config)
        assert handle.id == "rec_1"
        assert handle.session_id == "s1"
        assert handle.state == "recording"

        recorder.tap_inbound(handle, _frame(b"\x01"))
        recorder.tap_outbound(handle, _frame(b"\x02"))
        assert len(recorder.inbound_frames) == 1
        assert len(recorder.outbound_frames) == 1

        result = recorder.stop(handle)
        assert result.id == "rec_1"
        assert result.duration_seconds == 1.0

    def test_recording_config_defaults(self):
        cfg = RecordingConfig()
        assert cfg.mode == RecordingMode.BOTH
        assert cfg.trigger == RecordingTrigger.ALWAYS
        assert cfg.channels == RecordingChannelMode.MIXED
        assert cfg.format == "wav"
        assert cfg.storage == ""
        assert cfg.retention_days is None
        assert cfg.metadata == {}

    def test_recording_mode_values(self):
        assert RecordingMode.INBOUND_ONLY == "inbound_only"
        assert RecordingMode.OUTBOUND_ONLY == "outbound_only"
        assert RecordingMode.BOTH == "both"

    def test_recording_mode_backwards_compat(self):
        """Legacy 'always' and 'speech_only' strings map to BOTH."""
        assert RecordingMode("always") == RecordingMode.BOTH
        assert RecordingMode("speech_only") == RecordingMode.BOTH

    def test_recording_trigger_values(self):
        assert RecordingTrigger.ALWAYS == "always"
        assert RecordingTrigger.SPEECH_ONLY == "speech_only"

    def test_recording_channel_mode_values(self):
        assert RecordingChannelMode.MIXED == "mixed"
        assert RecordingChannelMode.SEPARATE == "separate"
        assert RecordingChannelMode.STEREO == "stereo"


# ---------------------------------------------------------------------------
# TurnDetector
# ---------------------------------------------------------------------------


class TestTurnDetector:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            TurnDetector()  # type: ignore[abstract]

    def test_mock_evaluate_complete(self):
        decision = TurnDecision(is_complete=True, confidence=0.9, reason="long pause")
        detector = MockTurnDetector(decisions=[decision])

        ctx = TurnContext(
            conversation_history=[TurnEntry(text="Hello")],
            silence_duration_ms=1000.0,
            session_id="s1",
        )
        result = detector.evaluate(ctx)
        assert result.is_complete
        assert result.reason == "long pause"
        assert len(detector.evaluations) == 1

    def test_mock_evaluate_incomplete(self):
        decision = TurnDecision(is_complete=False, confidence=0.7, reason="trailing")
        detector = MockTurnDetector(decisions=[decision])
        result = detector.evaluate(TurnContext())
        assert not result.is_complete

    def test_mock_default_decision(self):
        detector = MockTurnDetector()
        result = detector.evaluate(TurnContext())
        assert result.is_complete  # Default is complete
        assert result.reason is None  # Default reason is None

    def test_turn_entry_defaults(self):
        entry = TurnEntry(text="hi")
        assert entry.role is None
        assert entry.duration_ms is None

    def test_turn_context_with_multiple_entries(self):
        """TurnContext should hold multiple conversation_history entries."""
        entries = [
            TurnEntry(text="Hello"),
            TurnEntry(text="How are you"),
            TurnEntry(text="I need help"),
        ]
        ctx = TurnContext(
            conversation_history=entries, silence_duration_ms=500.0, session_id="s1"
        )
        assert len(ctx.conversation_history) == 3
        assert ctx.conversation_history[2].text == "I need help"

    def test_turn_entry_role_and_duration(self):
        """TurnEntry fields should carry through to detector."""
        entry = TurnEntry(text="Hi there", role="user", duration_ms=1200.0)
        ctx = TurnContext(conversation_history=[entry], session_id="s1")

        detector = MockTurnDetector()
        detector.evaluate(ctx)

        assert len(detector.evaluations) == 1
        evaluated_entry = detector.evaluations[0].conversation_history[0]
        assert evaluated_entry.role == "user"
        assert evaluated_entry.duration_ms == 1200.0

    def test_turn_context_silence_duration_ms(self):
        """silence_duration_ms should be passed through to the detector."""
        ctx = TurnContext(
            conversation_history=[TurnEntry(text="Done")],
            silence_duration_ms=2500.0,
            session_id="s1",
        )

        detector = MockTurnDetector()
        detector.evaluate(ctx)

        assert detector.evaluations[0].silence_duration_ms == 2500.0

    def test_turn_context_new_fields(self):
        """TurnContext exposes transcript, is_final, speech_duration_ms, metadata."""
        ctx = TurnContext(
            transcript="Hello world",
            is_final=True,
            speech_duration_ms=1500.0,
            metadata={"source": "test"},
        )
        assert ctx.transcript == "Hello world"
        assert ctx.is_final is True
        assert ctx.speech_duration_ms == 1500.0
        assert ctx.metadata == {"source": "test"}

    def test_turn_decision_suggested_wait_ms(self):
        """TurnDecision has suggested_wait_ms field."""
        decision = TurnDecision(
            is_complete=False, confidence=0.5, suggested_wait_ms=500.0
        )
        assert decision.suggested_wait_ms == 500.0


# ---------------------------------------------------------------------------
# BackchannelDetector
# ---------------------------------------------------------------------------


class TestBackchannelDetector:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            BackchannelDetector()  # type: ignore[abstract]

    def test_mock_classify_backchannel(self):
        decision = BackchannelDecision(is_backchannel=True, confidence=0.95)
        detector = MockBackchannelDetector(decisions=[decision])

        ctx = BackchannelContext(transcript="uh-huh", speech_duration_ms=300.0, session_id="s1")
        result = detector.classify(ctx)
        assert result.is_backchannel
        assert result.confidence == 0.95

    def test_mock_classify_not_backchannel(self):
        detector = MockBackchannelDetector()
        ctx = BackchannelContext(transcript="Actually, I disagree", speech_duration_ms=2000.0)
        result = detector.classify(ctx)
        assert not result.is_backchannel

    def test_backchannel_context_defaults(self):
        ctx = BackchannelContext(transcript="yeah", speech_duration_ms=200.0)
        assert ctx.session_id == ""


# ---------------------------------------------------------------------------
# Denoiser & PostProcessor reset()
# ---------------------------------------------------------------------------


class TestDenoiserReset:
    def test_denoiser_has_reset(self):
        from roomkit.voice.pipeline.denoiser.mock import MockDenoiserProvider

        d = MockDenoiserProvider()
        d.reset()
        assert d.reset_count == 1


class TestPostProcessorReset:
    def test_postprocessor_has_reset(self):
        from roomkit.voice.pipeline.postprocessor.base import AudioPostProcessor

        # Verify reset exists on ABC
        assert hasattr(AudioPostProcessor, "reset")


# ---------------------------------------------------------------------------
# AudioPipelineContract — 3-format model
# ---------------------------------------------------------------------------


class TestAudioPipelineContract:
    def test_three_format_fields(self):
        """Contract exposes transport_inbound, transport_outbound, internal formats."""
        fmt_in = AudioFormat(sample_rate=48000, channels=2, sample_width=2, codec="opus")
        fmt_out = AudioFormat(sample_rate=48000, channels=2, sample_width=2, codec="opus")
        fmt_int = AudioFormat(sample_rate=16000, channels=1, sample_width=2, codec="pcm_s16le")
        contract = AudioPipelineContract(
            transport_inbound_format=fmt_in,
            transport_outbound_format=fmt_out,
            internal_format=fmt_int,
        )
        assert contract.transport_inbound_format is fmt_in
        assert contract.transport_outbound_format is fmt_out
        assert contract.internal_format is fmt_int

    def test_backwards_compat_aliases(self):
        """input_format/output_format properties map to transport formats."""
        fmt_in = AudioFormat(sample_rate=48000)
        fmt_out = AudioFormat(sample_rate=24000)
        contract = AudioPipelineContract(
            transport_inbound_format=fmt_in,
            transport_outbound_format=fmt_out,
        )
        assert contract.input_format is fmt_in
        assert contract.output_format is fmt_out

    def test_default_formats(self):
        """Default contract uses 16kHz mono pcm_s16le for all three."""
        contract = AudioPipelineContract()
        for fmt in (
            contract.transport_inbound_format,
            contract.transport_outbound_format,
            contract.internal_format,
        ):
            assert fmt.sample_rate == 16000
            assert fmt.channels == 1
            assert fmt.sample_width == 2
            assert fmt.codec == "pcm_s16le"

    def test_codec_field_on_audio_format(self):
        """AudioFormat has a codec field with default pcm_s16le."""
        fmt = AudioFormat()
        assert fmt.codec == "pcm_s16le"
        fmt2 = AudioFormat(codec="opus")
        assert fmt2.codec == "opus"

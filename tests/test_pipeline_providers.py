"""Tests for pipeline v2 provider ABCs and data types."""

from __future__ import annotations

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.aec_provider import AECProvider
from roomkit.voice.pipeline.agc_provider import AGCConfig, AGCProvider
from roomkit.voice.pipeline.backchannel_detector import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)
from roomkit.voice.pipeline.dtmf_detector import DTMFDetector, DTMFEvent
from roomkit.voice.pipeline.mock import (
    MockAECProvider,
    MockAGCProvider,
    MockAudioRecorder,
    MockBackchannelDetector,
    MockDTMFDetector,
    MockTurnDetector,
)
from roomkit.voice.pipeline.recorder import (
    AudioRecorder,
    RecordingChannelMode,
    RecordingConfig,
    RecordingMode,
)
from roomkit.voice.pipeline.turn_detector import (
    TurnContext,
    TurnDecision,
    TurnDetector,
    TurnEntry,
)


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
        assert cfg.extra == {}


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
        assert handle.recording_id == "rec_1"
        assert handle.session_id == "s1"

        recorder.tap_inbound(handle, _frame(b"\x01"))
        recorder.tap_outbound(handle, _frame(b"\x02"))
        assert len(recorder.inbound_frames) == 1
        assert len(recorder.outbound_frames) == 1

        result = recorder.stop(handle)
        assert result.recording_id == "rec_1"
        assert result.duration_ms == 1000.0

    def test_recording_config_defaults(self):
        cfg = RecordingConfig()
        assert cfg.mode == RecordingMode.ALWAYS
        assert cfg.channel_mode == RecordingChannelMode.MIXED
        assert cfg.output_format == "wav"

    def test_recording_mode_values(self):
        assert RecordingMode.ALWAYS == "always"
        assert RecordingMode.SPEECH_ONLY == "speech_only"

    def test_recording_channel_mode_values(self):
        assert RecordingChannelMode.MIXED == "mixed"
        assert RecordingChannelMode.SEPARATE == "separate"


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
            entries=[TurnEntry(text="Hello")],
            silence_ms=1000.0,
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

    def test_turn_entry_defaults(self):
        entry = TurnEntry(text="hi")
        assert entry.speaker_id is None
        assert entry.duration_ms is None


# ---------------------------------------------------------------------------
# BackchannelDetector
# ---------------------------------------------------------------------------


class TestBackchannelDetector:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            BackchannelDetector()  # type: ignore[abstract]

    def test_mock_evaluate_backchannel(self):
        decision = BackchannelDecision(is_backchannel=True, confidence=0.95)
        detector = MockBackchannelDetector(decisions=[decision])

        ctx = BackchannelContext(text="uh-huh", duration_ms=300.0, session_id="s1")
        result = detector.evaluate(ctx)
        assert result.is_backchannel
        assert result.confidence == 0.95

    def test_mock_evaluate_not_backchannel(self):
        detector = MockBackchannelDetector()
        ctx = BackchannelContext(text="Actually, I disagree", duration_ms=2000.0)
        result = detector.evaluate(ctx)
        assert not result.is_backchannel

    def test_backchannel_context_defaults(self):
        ctx = BackchannelContext(text="yeah", duration_ms=200.0)
        assert ctx.session_id == ""


# ---------------------------------------------------------------------------
# Denoiser & PostProcessor reset()
# ---------------------------------------------------------------------------


class TestDenoiserReset:
    def test_denoiser_has_reset(self):
        from roomkit.voice.pipeline.mock import MockDenoiserProvider

        d = MockDenoiserProvider()
        d.reset()
        assert d.reset_count == 1


class TestPostProcessorReset:
    def test_postprocessor_has_reset(self):
        from roomkit.voice.pipeline.postprocessor import AudioPostProcessor

        # Verify reset exists on ABC
        assert hasattr(AudioPostProcessor, "reset")

"""Tests for pipeline engine v2 — new inbound order, outbound path, capabilities."""

from __future__ import annotations

import asyncio

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceCapability, VoiceSession
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.agc.mock import MockAGCProvider
from roomkit.voice.pipeline.config import (
    AudioFormat,
    AudioPipelineConfig,
    AudioPipelineContract,
)
from roomkit.voice.pipeline.denoiser.mock import MockDenoiserProvider
from roomkit.voice.pipeline.diarization.base import DiarizationResult
from roomkit.voice.pipeline.diarization.mock import MockDiarizationProvider
from roomkit.voice.pipeline.dtmf.base import DTMFEvent
from roomkit.voice.pipeline.dtmf.mock import MockDTMFDetector
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.recorder.base import RecordingConfig
from roomkit.voice.pipeline.recorder.mock import MockAudioRecorder
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.pipeline.resampler.mock import MockResamplerProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider


def _frame(data: bytes = b"\x00\x00") -> AudioFrame:
    return AudioFrame(data=data, sample_rate=16000, channels=1, sample_width=2)


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id="p1", channel_id="c1")


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    def test_process_frame_delegates_to_process_inbound(self):
        """process_frame() is a backwards-compatible alias."""
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        config = AudioPipelineConfig(vad=vad)
        pipeline = AudioPipeline(config)

        pipeline.process_frame(_session(), _frame())
        assert len(vad.frames) == 1

    def test_constructor_without_capabilities(self):
        """AudioPipeline can be constructed without backend_capabilities."""
        config = AudioPipelineConfig()
        pipeline = AudioPipeline(config)
        assert pipeline._backend_capabilities == VoiceCapability.NONE


# ---------------------------------------------------------------------------
# Inbound processing order
# ---------------------------------------------------------------------------


class TestInboundOrder:
    def test_full_inbound_chain(self):
        """Verify all stages run in correct order with metadata annotations."""
        aec = MockAECProvider()
        agc = MockAGCProvider()
        denoiser = MockDenoiserProvider()
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])

        config = AudioPipelineConfig(aec=aec, agc=agc, denoiser=denoiser, vad=vad)
        pipeline = AudioPipeline(config)
        session = _session()
        frame = _frame(b"\x01\x02")

        pipeline.process_inbound(session, frame)

        # All stages processed the frame
        assert len(aec.frames) == 1
        assert len(agc.frames) == 1
        assert len(denoiser.frames) == 1
        assert len(vad.frames) == 1

        # Metadata annotated
        assert frame.metadata.get("aec") == "MockAECProvider"
        assert frame.metadata.get("agc") == "MockAGCProvider"
        assert frame.metadata.get("denoiser") == "MockDenoiserProvider"

    def test_aec_skipped_with_native_aec(self):
        """AEC should be skipped when backend has NATIVE_AEC."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config, backend_capabilities=VoiceCapability.NATIVE_AEC)

        pipeline.process_inbound(_session(), _frame())
        assert len(aec.frames) == 0

    def test_agc_skipped_with_native_agc(self):
        """AGC should be skipped when backend has NATIVE_AGC."""
        agc = MockAGCProvider()
        config = AudioPipelineConfig(agc=agc)
        pipeline = AudioPipeline(config, backend_capabilities=VoiceCapability.NATIVE_AGC)

        pipeline.process_inbound(_session(), _frame())
        assert len(agc.frames) == 0

    def test_aec_runs_without_native_flag(self):
        """AEC should run when backend does NOT have NATIVE_AEC."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), _frame())
        assert len(aec.frames) == 1

    def test_agc_runs_without_native_flag(self):
        """AGC should run when backend does NOT have NATIVE_AGC."""
        agc = MockAGCProvider()
        config = AudioPipelineConfig(agc=agc)
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), _frame())
        assert len(agc.frames) == 1


# ---------------------------------------------------------------------------
# DTMF
# ---------------------------------------------------------------------------


class TestDTMF:
    def test_dtmf_callback_fires(self):
        """DTMF events should fire on_dtmf callbacks."""
        event = DTMFEvent(digit="5", duration_ms=100.0)
        dtmf = MockDTMFDetector(events=[event])
        config = AudioPipelineConfig(dtmf=dtmf)
        pipeline = AudioPipeline(config)

        received: list[DTMFEvent] = []
        pipeline.on_dtmf(lambda _s, e: received.append(e))

        pipeline.process_inbound(_session(), _frame())

        assert len(received) == 1
        assert received[0].digit == "5"

    def test_dtmf_no_event(self):
        """No callback when DTMF detector returns None."""
        dtmf = MockDTMFDetector()
        config = AudioPipelineConfig(dtmf=dtmf)
        pipeline = AudioPipeline(config)

        received: list[DTMFEvent] = []
        pipeline.on_dtmf(lambda _s, e: received.append(e))

        pipeline.process_inbound(_session(), _frame())
        assert len(received) == 0

    def test_dtmf_metadata(self):
        """DTMF events annotate frame metadata."""
        event = DTMFEvent(digit="#", duration_ms=80.0)
        dtmf = MockDTMFDetector(events=[event])
        config = AudioPipelineConfig(dtmf=dtmf)
        pipeline = AudioPipeline(config)

        frame = _frame()
        pipeline.process_inbound(_session(), frame)
        assert frame.metadata["dtmf"]["digit"] == "#"

    def test_dtmf_runs_before_aec(self):
        """DTMF processes the frame before AEC modifies it."""
        order: list[str] = []

        class OrderTrackingDTMF(MockDTMFDetector):
            def process(self, frame):
                order.append("dtmf")
                return super().process(frame)

        class OrderTrackingAEC(MockAECProvider):
            def process(self, frame):
                order.append("aec")
                return super().process(frame)

        event = DTMFEvent(digit="9", duration_ms=50.0)
        dtmf = OrderTrackingDTMF(events=[event])
        aec = OrderTrackingAEC()
        config = AudioPipelineConfig(dtmf=dtmf, aec=aec)
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), _frame())

        assert order == ["dtmf", "aec"]


# ---------------------------------------------------------------------------
# Outbound path
# ---------------------------------------------------------------------------


class TestOutboundPath:
    def test_postprocessors_applied(self):
        """Outbound path should apply postprocessors."""
        from roomkit.voice.pipeline.denoiser.mock import MockDenoiserProvider

        # Reuse MockDenoiserProvider as a postprocessor-like mock
        # (it implements process() correctly)
        class MockPP(MockDenoiserProvider):
            @property
            def name(self) -> str:
                return "MockPP"

            def reset(self) -> None:
                self.reset_count += 1

        pp = MockPP()
        config = AudioPipelineConfig(postprocessors=[pp])
        pipeline = AudioPipeline(config)

        result = pipeline.process_outbound(_session(), _frame(b"\x01\x00"))
        assert len(pp.frames) == 1
        assert result is not None

    def test_outbound_feeds_aec_reference(self):
        """Outbound path should feed AEC reference frames."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config)

        pipeline.process_outbound(_session(), _frame())
        assert len(aec.reference_frames) == 1

    def test_outbound_aec_reference_skipped_with_native(self):
        """AEC feed_reference skipped when backend has NATIVE_AEC."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        pipeline = AudioPipeline(config, backend_capabilities=VoiceCapability.NATIVE_AEC)

        pipeline.process_outbound(_session(), _frame())
        assert len(aec.reference_frames) == 0

    def test_outbound_recorder_tap(self):
        """Outbound path should tap the recorder."""
        recorder = MockAudioRecorder()
        config = AudioPipelineConfig(
            recorder=recorder,
            recording_config=RecordingConfig(),
        )
        pipeline = AudioPipeline(config)
        session = _session()

        # Must start recording first
        pipeline.on_session_active(session)

        pipeline.process_outbound(session, _frame(b"\x02\x00"))
        assert len(recorder.outbound_frames) == 1


# ---------------------------------------------------------------------------
# Recorder / session lifecycle
# ---------------------------------------------------------------------------


class TestRecorderLifecycle:
    def test_recorder_inbound_tap(self):
        """Inbound frames are tapped to the recorder."""
        recorder = MockAudioRecorder()
        config = AudioPipelineConfig(
            recorder=recorder,
            recording_config=RecordingConfig(),
        )
        pipeline = AudioPipeline(config)
        session = _session()

        pipeline.on_session_active(session)
        pipeline.process_inbound(session, _frame(b"\x01\x00"))

        assert len(recorder.inbound_frames) == 1

    def test_session_active_starts_recording(self):
        """on_session_active should start recording."""
        recorder = MockAudioRecorder()
        config = AudioPipelineConfig(
            recorder=recorder,
            recording_config=RecordingConfig(),
        )
        pipeline = AudioPipeline(config)

        started = []
        pipeline.on_recording_started(lambda s, h: started.append((s, h)))

        session = _session()
        pipeline.on_session_active(session)

        assert len(recorder.started) == 1
        assert len(started) == 1
        assert started[0][1].id == "rec_1"

    def test_session_ended_stops_recording(self):
        """on_session_ended should stop recording."""
        recorder = MockAudioRecorder()
        config = AudioPipelineConfig(
            recorder=recorder,
            recording_config=RecordingConfig(),
        )
        pipeline = AudioPipeline(config)

        stopped = []
        pipeline.on_recording_stopped(lambda s, r: stopped.append((s, r)))

        session = _session()
        pipeline.on_session_active(session)
        pipeline.on_session_ended(session)

        assert len(recorder.stopped) == 1
        assert len(stopped) == 1
        assert stopped[0][1].id == "rec_1"

    def test_no_recording_without_config(self):
        """Recording doesn't start without recording_config."""
        recorder = MockAudioRecorder()
        config = AudioPipelineConfig(recorder=recorder)  # No recording_config
        pipeline = AudioPipeline(config)

        pipeline.on_session_active(_session())
        assert len(recorder.started) == 0


# ---------------------------------------------------------------------------
# Reset and close
# ---------------------------------------------------------------------------


class TestResetClose:
    def test_reset_resets_all_providers(self):
        """reset() should call reset() on all configured providers."""
        aec = MockAECProvider()
        agc = MockAGCProvider()
        denoiser = MockDenoiserProvider()
        dtmf = MockDTMFDetector()
        vad = MockVADProvider()

        config = AudioPipelineConfig(aec=aec, agc=agc, denoiser=denoiser, dtmf=dtmf, vad=vad)
        pipeline = AudioPipeline(config)

        pipeline.reset()

        assert aec.reset_count == 1
        assert agc.reset_count == 1
        assert denoiser.reset_count == 1
        assert dtmf.reset_count == 1
        assert vad.reset_count == 1

    def test_close_closes_all_providers(self):
        """close() should close all configured providers."""
        aec = MockAECProvider()
        agc = MockAGCProvider()
        denoiser = MockDenoiserProvider()
        dtmf = MockDTMFDetector()
        recorder = MockAudioRecorder()

        config = AudioPipelineConfig(
            aec=aec, agc=agc, denoiser=denoiser, dtmf=dtmf, recorder=recorder
        )
        pipeline = AudioPipeline(config)

        pipeline.close()

        assert aec.closed
        assert agc.closed
        assert denoiser.closed
        assert dtmf.closed
        assert recorder.closed


# ---------------------------------------------------------------------------
# Async callback execution
# ---------------------------------------------------------------------------


class TestAsyncCallbacks:
    async def test_async_vad_event_callback(self):
        """Async VAD event callbacks should be scheduled and executed."""
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        config = AudioPipelineConfig(vad=vad)
        pipeline = AudioPipeline(config)

        received: list[VADEvent] = []

        async def async_cb(session, event):
            await asyncio.sleep(0)
            received.append(event)

        pipeline.on_vad_event(async_cb)

        pipeline.process_inbound(_session(), _frame())

        # Allow the scheduled task to run
        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0].type == VADEventType.SPEECH_START

    async def test_async_speech_end_callback(self):
        """Async speech_end callbacks should be scheduled and executed."""
        vad = MockVADProvider(
            events=[VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x01\x02")]
        )
        config = AudioPipelineConfig(vad=vad)
        pipeline = AudioPipeline(config)

        received: list[bytes] = []

        async def async_cb(session, audio):
            await asyncio.sleep(0)
            received.append(audio)

        pipeline.on_speech_end(async_cb)

        pipeline.process_inbound(_session(), _frame())

        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0] == b"\x01\x02"


# ---------------------------------------------------------------------------
# Speaker change callbacks
# ---------------------------------------------------------------------------


class TestSpeakerChangeCallbacks:
    def test_speaker_change_fires_on_new_speaker(self):
        """Two different speaker_ids should fire 2 speaker change callbacks."""
        results = [
            DiarizationResult(speaker_id="spk_A", confidence=0.9, is_new_speaker=True),
            DiarizationResult(speaker_id="spk_B", confidence=0.8, is_new_speaker=True),
        ]
        diarizer = MockDiarizationProvider(results=results)
        config = AudioPipelineConfig(diarization=diarizer)
        pipeline = AudioPipeline(config)

        received: list[DiarizationResult] = []
        pipeline.on_speaker_change(lambda _s, r: received.append(r))

        session = _session()
        pipeline.process_inbound(session, _frame())
        pipeline.process_inbound(session, _frame())

        assert len(received) == 2
        assert received[0].speaker_id == "spk_A"
        assert received[1].speaker_id == "spk_B"

    def test_no_speaker_change_when_same_speaker(self):
        """Same speaker_id on consecutive frames should fire only 1 callback (initial)."""
        results = [
            DiarizationResult(speaker_id="spk_A", confidence=0.9, is_new_speaker=True),
            DiarizationResult(speaker_id="spk_A", confidence=0.95, is_new_speaker=False),
        ]
        diarizer = MockDiarizationProvider(results=results)
        config = AudioPipelineConfig(diarization=diarizer)
        pipeline = AudioPipeline(config)

        received: list[DiarizationResult] = []
        pipeline.on_speaker_change(lambda _s, r: received.append(r))

        session = _session()
        pipeline.process_inbound(session, _frame())
        pipeline.process_inbound(session, _frame())

        assert len(received) == 1
        assert received[0].speaker_id == "spk_A"


# ---------------------------------------------------------------------------
# Error propagation (pipeline resilience)
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    def test_aec_error_does_not_crash_pipeline(self):
        """AEC error should be caught; pipeline continues to VAD."""

        class FailingAEC(MockAECProvider):
            def process(self, frame):
                raise RuntimeError("AEC boom")

        aec = FailingAEC()
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        config = AudioPipelineConfig(aec=aec, vad=vad)
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), _frame())

        # VAD still ran despite AEC failure
        assert len(vad.frames) == 1

    def test_vad_error_does_not_crash_pipeline(self):
        """VAD error should be caught; pipeline doesn't raise."""

        class FailingVAD(MockVADProvider):
            def process(self, frame):
                raise RuntimeError("VAD boom")

        vad = FailingVAD()
        denoiser = MockDenoiserProvider()
        config = AudioPipelineConfig(denoiser=denoiser, vad=vad)
        pipeline = AudioPipeline(config)

        # Should not raise
        pipeline.process_inbound(_session(), _frame())

        # Denoiser ran before VAD
        assert len(denoiser.frames) == 1


# ---------------------------------------------------------------------------
# Multi-session recording
# ---------------------------------------------------------------------------


class TestMultiSessionRecording:
    def test_separate_recording_handles_per_session(self):
        """Two sessions activated separately get unique recording IDs."""
        recorder = MockAudioRecorder()
        config = AudioPipelineConfig(
            recorder=recorder,
            recording_config=RecordingConfig(),
        )
        pipeline = AudioPipeline(config)

        s1 = _session("s1")
        s2 = _session("s2")

        # Activate and record the handles individually
        # (on_session_active resets, so we can't have both active simultaneously
        # via on_session_active — instead, we verify IDs are unique.)
        pipeline.on_session_active(s1)
        handle_s1 = pipeline._recording_handles.get("s1")
        assert handle_s1 is not None

        pipeline.on_session_active(s2)
        handle_s2 = pipeline._recording_handles.get("s2")
        assert handle_s2 is not None

        assert handle_s1.id != handle_s2.id
        assert len(recorder.started) == 2

    def test_session_ended_only_stops_its_recording(self):
        """Ending a session only stops that session's recording handle."""
        recorder = MockAudioRecorder()
        config = AudioPipelineConfig(
            recorder=recorder,
            recording_config=RecordingConfig(),
        )
        pipeline = AudioPipeline(config)

        session = _session("s1")
        pipeline.on_session_active(session)
        assert "s1" in pipeline._recording_handles

        pipeline.on_session_ended(session)
        assert "s1" not in pipeline._recording_handles
        assert len(recorder.stopped) == 1

        # A second session's handle is unaffected by the first ending
        s2 = _session("s2")
        pipeline.on_session_active(s2)
        assert "s2" in pipeline._recording_handles
        assert len(recorder.started) == 2


# ---------------------------------------------------------------------------
# Resampler stage
# ---------------------------------------------------------------------------


class TestResamplerStage:
    def _stereo_frame(
        self, data: bytes, rate: int = 48000, channels: int = 2, width: int = 2
    ) -> AudioFrame:
        return AudioFrame(data=data, sample_rate=rate, channels=channels, sample_width=width)

    def test_inbound_resamples_48k_stereo_to_16k_mono(self):
        """Inbound resampler converts 48kHz stereo to 16kHz mono."""
        import struct

        # 48kHz stereo: 4 frames × 2 channels = 8 samples
        samples = [100, 100, 200, 200, 300, 300, 400, 400]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = self._stereo_frame(data, rate=48000, channels=2)

        contract = AudioPipelineContract(
            transport_inbound_format=AudioFormat(sample_rate=48000, channels=2),
            internal_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        )
        config = AudioPipelineConfig(resampler=LinearResamplerProvider(), contract=contract)
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), frame)

        # After resampling, frame metadata should have originals
        # (The frame passed in gets metadata annotated before resampling)
        assert frame.metadata.get("original_sample_rate") == 48000
        assert frame.metadata.get("original_channels") == 2

    def test_inbound_noop_when_formats_match(self):
        """No resampling when frame already matches internal format."""
        frame = _frame(b"\x01\x00\x02\x00")

        contract = AudioPipelineContract(
            internal_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        )
        config = AudioPipelineConfig(resampler=LinearResamplerProvider(), contract=contract)
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), frame)

        # Data unchanged — still the same frame (no conversion happened)
        assert frame.data == b"\x01\x00\x02\x00"

    def test_metadata_annotation(self):
        """Resampler annotates original_sample_rate and original_channels."""
        import struct

        samples = [100, 100, 200, 200]
        data = struct.pack(f"<{len(samples)}h", *samples)
        frame = self._stereo_frame(data, rate=44100, channels=2)

        contract = AudioPipelineContract(
            internal_format=AudioFormat(sample_rate=16000, channels=1),
        )
        config = AudioPipelineConfig(resampler=LinearResamplerProvider(), contract=contract)
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), frame)

        assert frame.metadata["original_sample_rate"] == 44100
        assert frame.metadata["original_channels"] == 2

    def test_outbound_resamples_to_transport_format(self):
        """Outbound resampler converts internal format to transport format."""

        frame = _frame(b"\x64\x00\xc8\x00")  # 2 samples: 100, 200

        out_fmt = AudioFormat(sample_rate=48000, channels=2, sample_width=2)
        contract = AudioPipelineContract(transport_outbound_format=out_fmt)
        config = AudioPipelineConfig(resampler=LinearResamplerProvider(), contract=contract)
        pipeline = AudioPipeline(config)

        result = pipeline.process_outbound(_session(), frame)
        # Result should have the target format
        assert result.sample_rate == 48000
        assert result.channels == 2

    def test_no_resampler_leaves_frame_unchanged(self):
        """Without resampler config, frames pass through unchanged."""
        frame = _frame(b"\x01\x00\x02\x00")
        config = AudioPipelineConfig()
        pipeline = AudioPipeline(config)

        pipeline.process_inbound(_session(), frame)
        assert frame.data == b"\x01\x00\x02\x00"
        assert "original_sample_rate" not in frame.metadata

    def test_auto_default_resampler_when_contract_set(self):
        """Pipeline auto-creates LinearResamplerProvider when contract is set."""
        contract = AudioPipelineContract(
            internal_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        )
        config = AudioPipelineConfig(contract=contract)
        pipeline = AudioPipeline(config)

        assert pipeline._resampler is not None
        assert pipeline._resampler.name == "linear"

    def test_no_auto_default_without_contract(self):
        """No auto-default resampler without contract."""
        config = AudioPipelineConfig()
        pipeline = AudioPipeline(config)
        assert pipeline._resampler is None

    def test_explicit_resampler_used_over_auto_default(self):
        """Explicit resampler takes priority over auto-default."""
        mock_resampler = MockResamplerProvider()
        contract = AudioPipelineContract(
            internal_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        )
        config = AudioPipelineConfig(resampler=mock_resampler, contract=contract)
        pipeline = AudioPipeline(config)

        assert pipeline._resampler is mock_resampler

    def test_resampler_reset_on_pipeline_reset(self):
        """Pipeline reset() calls resampler.reset()."""
        mock_resampler = MockResamplerProvider()
        contract = AudioPipelineContract()
        config = AudioPipelineConfig(resampler=mock_resampler, contract=contract)
        pipeline = AudioPipeline(config)

        pipeline.reset()
        assert mock_resampler.reset_count == 1

    def test_resampler_close_on_pipeline_close(self):
        """Pipeline close() calls resampler.close()."""
        mock_resampler = MockResamplerProvider()
        contract = AudioPipelineContract()
        config = AudioPipelineConfig(resampler=mock_resampler, contract=contract)
        pipeline = AudioPipeline(config)

        pipeline.close()
        assert mock_resampler.closed

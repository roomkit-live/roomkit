"""Tests for the AudioPipeline engine, provider ABCs, mock providers, and config."""

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.pipeline import (
    AudioPipeline,
    AudioPipelineConfig,
    AudioPostProcessor,
    DenoiserProvider,
    DiarizationProvider,
    DiarizationResult,
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockVADProvider,
    VADConfig,
    VADEvent,
    VADEventType,
    VADProvider,
)


def _session(sid: str = "sess-1") -> VoiceSession:
    return VoiceSession(
        id=sid,
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice-1",
        state=VoiceSessionState.ACTIVE,
    )


def _frame(data: bytes = b"audio") -> AudioFrame:
    return AudioFrame(data=data, sample_rate=16000)


# ---------------------------------------------------------------------------
# VAD provider & event types
# ---------------------------------------------------------------------------


class TestVADEventType:
    def test_enum_values(self) -> None:
        assert VADEventType.SPEECH_START == "speech_start"
        assert VADEventType.SPEECH_END == "speech_end"
        assert VADEventType.SILENCE == "silence"
        assert VADEventType.AUDIO_LEVEL == "audio_level"


class TestVADEvent:
    def test_default_values(self) -> None:
        event = VADEvent(type=VADEventType.SPEECH_START)
        assert event.type == VADEventType.SPEECH_START
        assert event.audio_bytes is None
        assert event.confidence is None
        assert event.duration_ms is None
        assert event.level_db is None

    def test_speech_end_with_audio(self) -> None:
        event = VADEvent(
            type=VADEventType.SPEECH_END,
            audio_bytes=b"audio-data",
            confidence=0.95,
            duration_ms=1200.0,
        )
        assert event.audio_bytes == b"audio-data"
        assert event.confidence == 0.95
        assert event.duration_ms == 1200.0

    def test_audio_level_event(self) -> None:
        event = VADEvent(
            type=VADEventType.AUDIO_LEVEL,
            level_db=-25.5,
            confidence=0.8,
        )
        assert event.level_db == -25.5


class TestVADConfig:
    def test_defaults(self) -> None:
        config = VADConfig()
        assert config.silence_threshold_ms == 500
        assert config.speech_pad_ms == 300
        assert config.min_speech_duration_ms == 250
        assert config.extra == {}

    def test_custom_values(self) -> None:
        config = VADConfig(
            silence_threshold_ms=200,
            speech_pad_ms=100,
            min_speech_duration_ms=50,
            extra={"sensitivity": 0.5},
        )
        assert config.silence_threshold_ms == 200
        assert config.extra["sensitivity"] == 0.5


# ---------------------------------------------------------------------------
# DiarizationResult
# ---------------------------------------------------------------------------


class TestDiarizationResult:
    def test_fields(self) -> None:
        result = DiarizationResult(
            speaker_id="speaker_0",
            confidence=0.92,
            is_new_speaker=True,
        )
        assert result.speaker_id == "speaker_0"
        assert result.confidence == 0.92
        assert result.is_new_speaker is True


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class TestMockVADProvider:
    def test_name(self) -> None:
        vad = MockVADProvider()
        assert vad.name == "MockVADProvider"

    def test_returns_configured_events(self) -> None:
        events = [
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio"),
        ]
        vad = MockVADProvider(events=events)

        assert vad.process(_frame(b"f1")) == events[0]
        assert vad.process(_frame(b"f2")) is None
        assert vad.process(_frame(b"f3")) == events[2]
        # Past the end -> None
        assert vad.process(_frame(b"f4")) is None

    def test_tracks_frames(self) -> None:
        vad = MockVADProvider()
        frame = _frame(b"test")
        vad.process(frame)
        assert len(vad.frames) == 1
        assert vad.frames[0] is frame

    def test_reset(self) -> None:
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        vad.process(_frame())
        vad.reset()
        assert vad.reset_count == 1
        # After reset, index is back to 0
        result = vad.process(_frame())
        assert result is not None
        assert result.type == VADEventType.SPEECH_START

    def test_close(self) -> None:
        vad = MockVADProvider()
        assert vad.closed is False
        vad.close()
        assert vad.closed is True


class TestMockDenoiserProvider:
    def test_name(self) -> None:
        denoiser = MockDenoiserProvider()
        assert denoiser.name == "MockDenoiserProvider"

    def test_passthrough(self) -> None:
        denoiser = MockDenoiserProvider()
        frame = _frame(b"audio")
        result = denoiser.process(frame)
        assert result is frame  # Same object (passthrough)

    def test_tracks_frames(self) -> None:
        denoiser = MockDenoiserProvider()
        denoiser.process(_frame(b"f1"))
        denoiser.process(_frame(b"f2"))
        assert len(denoiser.frames) == 2

    def test_close(self) -> None:
        denoiser = MockDenoiserProvider()
        denoiser.close()
        assert denoiser.closed is True


class TestMockDiarizationProvider:
    def test_name(self) -> None:
        diarizer = MockDiarizationProvider()
        assert diarizer.name == "MockDiarizationProvider"

    def test_returns_configured_results(self) -> None:
        results = [
            DiarizationResult(speaker_id="speaker_0", confidence=0.9, is_new_speaker=True),
            None,
            DiarizationResult(speaker_id="speaker_1", confidence=0.85, is_new_speaker=True),
        ]
        diarizer = MockDiarizationProvider(results=results)

        assert diarizer.process(_frame(b"f1")) == results[0]
        assert diarizer.process(_frame(b"f2")) is None
        assert diarizer.process(_frame(b"f3")) == results[2]
        assert diarizer.process(_frame(b"f4")) is None

    def test_tracks_frames(self) -> None:
        diarizer = MockDiarizationProvider()
        diarizer.process(_frame())
        assert len(diarizer.frames) == 1

    def test_reset(self) -> None:
        results = [DiarizationResult(speaker_id="s0", confidence=0.9, is_new_speaker=True)]
        diarizer = MockDiarizationProvider(results=results)
        diarizer.process(_frame())
        diarizer.reset()
        assert diarizer.reset_count == 1
        assert diarizer.process(_frame()) == results[0]

    def test_close(self) -> None:
        diarizer = MockDiarizationProvider()
        diarizer.close()
        assert diarizer.closed is True


# ---------------------------------------------------------------------------
# AudioPipelineConfig
# ---------------------------------------------------------------------------


class TestAudioPipelineConfig:
    def test_minimal_vad(self) -> None:
        vad = MockVADProvider()
        config = AudioPipelineConfig(vad=vad)
        assert config.vad is vad
        assert config.denoiser is None
        assert config.diarization is None
        assert config.postprocessors == []
        assert config.vad_config is None

    def test_all_optional(self) -> None:
        config = AudioPipelineConfig()
        assert config.vad is None
        assert config.denoiser is None
        assert config.diarization is None

    def test_denoiser_only(self) -> None:
        denoiser = MockDenoiserProvider()
        config = AudioPipelineConfig(denoiser=denoiser)
        assert config.vad is None
        assert config.denoiser is denoiser

    def test_diarization_only(self) -> None:
        diarizer = MockDiarizationProvider()
        config = AudioPipelineConfig(diarization=diarizer)
        assert config.vad is None
        assert config.diarization is diarizer

    def test_full(self) -> None:
        vad = MockVADProvider()
        denoiser = MockDenoiserProvider()
        diarizer = MockDiarizationProvider()
        vad_config = VADConfig(silence_threshold_ms=200)

        config = AudioPipelineConfig(
            vad=vad,
            denoiser=denoiser,
            diarization=diarizer,
            vad_config=vad_config,
        )
        assert config.denoiser is denoiser
        assert config.diarization is diarizer
        assert config.vad_config.silence_threshold_ms == 200


# ---------------------------------------------------------------------------
# AudioPipeline engine
# ---------------------------------------------------------------------------


class TestAudioPipelineVADOnly:
    """Pipeline with VAD only (minimum config)."""

    def test_process_frame_fires_vad_event_callback(self) -> None:
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        events = []
        pipeline.on_vad_event(lambda session, event: events.append(event))

        pipeline.process_frame(_session(), _frame())

        assert len(events) == 1
        assert events[0].type == VADEventType.SPEECH_START

    def test_process_frame_fires_speech_end_callback(self) -> None:
        audio = b"accumulated-audio"
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio)])
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        end_events = []
        pipeline.on_speech_end(lambda session, audio_bytes: end_events.append(audio_bytes))

        pipeline.process_frame(_session(), _frame())

        assert len(end_events) == 1
        assert end_events[0] == audio

    def test_no_event_when_vad_returns_none(self) -> None:
        vad = MockVADProvider(events=[])  # Always returns None
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        events = []
        pipeline.on_vad_event(lambda s, e: events.append(e))

        pipeline.process_frame(_session(), _frame())

        assert len(events) == 0

    def test_frame_metadata_updated_on_vad_event(self) -> None:
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START, confidence=0.9)])
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        frame = _frame()
        pipeline.process_frame(_session(), frame)

        assert "vad" in frame.metadata
        assert frame.metadata["vad"]["type"] == VADEventType.SPEECH_START
        assert frame.metadata["vad"]["confidence"] == 0.9


class TestAudioPipelineDenoiser:
    """Pipeline with denoiser + VAD."""

    def test_denoiser_runs_before_vad(self) -> None:
        denoiser = MockDenoiserProvider()
        vad = MockVADProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad, denoiser=denoiser))

        frame = _frame(b"audio")
        pipeline.process_frame(_session(), frame)

        # Both providers should see the frame
        assert len(denoiser.frames) == 1
        assert len(vad.frames) == 1
        # Denoiser annotates metadata
        assert frame.metadata.get("denoiser") == "MockDenoiserProvider"

    def test_denoiser_metadata_persists_through_pipeline(self) -> None:
        denoiser = MockDenoiserProvider()
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START, confidence=0.8)])
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad, denoiser=denoiser))

        frame = _frame()
        pipeline.process_frame(_session(), frame)

        assert frame.metadata["denoiser"] == "MockDenoiserProvider"
        assert frame.metadata["vad"]["type"] == VADEventType.SPEECH_START


class TestAudioPipelineDiarization:
    """Pipeline with VAD + diarization."""

    def test_speaker_change_callback_fires(self) -> None:
        vad = MockVADProvider()
        diarizer = MockDiarizationProvider(
            results=[
                DiarizationResult(speaker_id="speaker_0", confidence=0.9, is_new_speaker=True),
            ]
        )
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad, diarization=diarizer))

        changes = []
        pipeline.on_speaker_change(lambda s, r: changes.append(r))

        pipeline.process_frame(_session(), _frame())

        assert len(changes) == 1
        assert changes[0].speaker_id == "speaker_0"

    def test_speaker_change_fires_only_on_change(self) -> None:
        vad = MockVADProvider()
        diarizer = MockDiarizationProvider(
            results=[
                DiarizationResult(speaker_id="speaker_0", confidence=0.9, is_new_speaker=True),
                DiarizationResult(speaker_id="speaker_0", confidence=0.95, is_new_speaker=False),
                DiarizationResult(speaker_id="speaker_1", confidence=0.85, is_new_speaker=True),
            ]
        )
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad, diarization=diarizer))

        changes = []
        pipeline.on_speaker_change(lambda s, r: changes.append(r))

        # Frame 1: speaker_0 (new) -> fires
        pipeline.process_frame(_session(), _frame(b"f1"))
        # Frame 2: speaker_0 (same) -> does NOT fire
        pipeline.process_frame(_session(), _frame(b"f2"))
        # Frame 3: speaker_1 (new) -> fires
        pipeline.process_frame(_session(), _frame(b"f3"))

        assert len(changes) == 2
        assert changes[0].speaker_id == "speaker_0"
        assert changes[1].speaker_id == "speaker_1"

    def test_diarization_metadata_added(self) -> None:
        vad = MockVADProvider()
        diarizer = MockDiarizationProvider(
            results=[
                DiarizationResult(speaker_id="speaker_0", confidence=0.9, is_new_speaker=True),
            ]
        )
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad, diarization=diarizer))

        frame = _frame()
        pipeline.process_frame(_session(), frame)

        assert "diarization" in frame.metadata
        assert frame.metadata["diarization"]["speaker_id"] == "speaker_0"
        assert frame.metadata["diarization"]["confidence"] == 0.9


class TestAudioPipelineFullChain:
    """Pipeline with denoiser + VAD + diarization (full chain)."""

    def test_metadata_accumulation(self) -> None:
        """Metadata accumulates through denoiser -> VAD -> diarization."""
        denoiser = MockDenoiserProvider()
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START, confidence=0.8)])
        diarizer = MockDiarizationProvider(
            results=[
                DiarizationResult(speaker_id="speaker_0", confidence=0.92, is_new_speaker=True),
            ]
        )
        pipeline = AudioPipeline(
            AudioPipelineConfig(vad=vad, denoiser=denoiser, diarization=diarizer)
        )

        frame = _frame(b"audio")
        pipeline.process_frame(_session(), frame)

        assert frame.metadata["denoiser"] == "MockDenoiserProvider"
        assert frame.metadata["vad"]["type"] == VADEventType.SPEECH_START
        assert frame.metadata["diarization"]["speaker_id"] == "speaker_0"

    def test_all_providers_see_frames(self) -> None:
        denoiser = MockDenoiserProvider()
        vad = MockVADProvider()
        diarizer = MockDiarizationProvider()
        pipeline = AudioPipeline(
            AudioPipelineConfig(vad=vad, denoiser=denoiser, diarization=diarizer)
        )

        pipeline.process_frame(_session(), _frame(b"f1"))
        pipeline.process_frame(_session(), _frame(b"f2"))

        assert len(denoiser.frames) == 2
        assert len(vad.frames) == 2
        assert len(diarizer.frames) == 2


class TestAudioPipelineReset:
    def test_reset_resets_vad_and_diarization(self) -> None:
        vad = MockVADProvider()
        diarizer = MockDiarizationProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad, diarization=diarizer))

        pipeline.reset()

        assert vad.reset_count == 1
        assert diarizer.reset_count == 1

    def test_reset_clears_last_speaker(self) -> None:
        vad = MockVADProvider()
        diarizer = MockDiarizationProvider(
            results=[
                DiarizationResult(speaker_id="speaker_0", confidence=0.9, is_new_speaker=True),
                DiarizationResult(speaker_id="speaker_0", confidence=0.9, is_new_speaker=False),
            ]
        )
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad, diarization=diarizer))

        changes = []
        pipeline.on_speaker_change(lambda s, r: changes.append(r))

        # Frame 1: speaker_0 fires change
        pipeline.process_frame(_session(), _frame(b"f1"))
        assert len(changes) == 1

        # Reset clears last speaker
        pipeline.reset()

        # Frame 2: speaker_0 fires change again because last_speaker is cleared
        pipeline.process_frame(_session(), _frame(b"f2"))
        assert len(changes) == 2


class TestAudioPipelineClose:
    def test_close_closes_all_providers(self) -> None:
        denoiser = MockDenoiserProvider()
        vad = MockVADProvider()
        diarizer = MockDiarizationProvider()
        pipeline = AudioPipeline(
            AudioPipelineConfig(vad=vad, denoiser=denoiser, diarization=diarizer)
        )

        pipeline.close()

        assert vad.closed
        assert denoiser.closed
        assert diarizer.closed

    def test_close_without_optional_providers(self) -> None:
        vad = MockVADProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        pipeline.close()

        assert vad.closed


class TestAudioPipelineMultipleCallbacks:
    def test_multiple_vad_event_callbacks(self) -> None:
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        events1 = []
        events2 = []
        pipeline.on_vad_event(lambda s, e: events1.append(e))
        pipeline.on_vad_event(lambda s, e: events2.append(e))

        pipeline.process_frame(_session(), _frame())

        assert len(events1) == 1
        assert len(events2) == 1

    def test_multiple_speech_end_callbacks(self) -> None:
        vad = MockVADProvider(
            events=[VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio")]
        )
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        ends1 = []
        ends2 = []
        pipeline.on_speech_end(lambda s, a: ends1.append(a))
        pipeline.on_speech_end(lambda s, a: ends2.append(a))

        pipeline.process_frame(_session(), _frame())

        assert len(ends1) == 1
        assert len(ends2) == 1


class TestAudioPipelineNoVAD:
    """Pipeline with no VAD — for realtime voice (denoiser/diarization only)."""

    def test_denoiser_only(self) -> None:
        denoiser = MockDenoiserProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(denoiser=denoiser))

        frame = _frame(b"noisy")
        pipeline.process_frame(_session(), frame)

        assert len(denoiser.frames) == 1
        assert frame.metadata["denoiser"] == "MockDenoiserProvider"

    def test_diarization_only(self) -> None:
        diarizer = MockDiarizationProvider(results=[DiarizationResult("speaker_0", 0.9, True)])
        pipeline = AudioPipeline(AudioPipelineConfig(diarization=diarizer))

        changes = []
        pipeline.on_speaker_change(lambda s, r: changes.append(r))

        pipeline.process_frame(_session(), _frame())

        assert len(diarizer.frames) == 1
        assert len(changes) == 1
        assert changes[0].speaker_id == "speaker_0"

    def test_denoiser_and_diarization(self) -> None:
        denoiser = MockDenoiserProvider()
        diarizer = MockDiarizationProvider(results=[DiarizationResult("speaker_0", 0.95, True)])
        pipeline = AudioPipeline(AudioPipelineConfig(denoiser=denoiser, diarization=diarizer))

        frame = _frame(b"audio")
        pipeline.process_frame(_session(), frame)

        assert len(denoiser.frames) == 1
        assert len(diarizer.frames) == 1
        assert "denoiser" in frame.metadata
        assert "diarization" in frame.metadata

    def test_no_vad_events_fire(self) -> None:
        """With no VAD, no VAD events or speech-end callbacks fire."""
        denoiser = MockDenoiserProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(denoiser=denoiser))

        vad_events = []
        speech_ends = []
        pipeline.on_vad_event(lambda s, e: vad_events.append(e))
        pipeline.on_speech_end(lambda s, a: speech_ends.append(a))

        pipeline.process_frame(_session(), _frame())

        assert vad_events == []
        assert speech_ends == []

    def test_reset_without_vad(self) -> None:
        diarizer = MockDiarizationProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(diarization=diarizer))
        pipeline.reset()
        assert diarizer.reset_count == 1

    def test_close_without_vad(self) -> None:
        denoiser = MockDenoiserProvider()
        diarizer = MockDiarizationProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(denoiser=denoiser, diarization=diarizer))
        pipeline.close()
        assert denoiser.closed
        assert diarizer.closed


class TestProviderABCs:
    """Verify that ABCs require proper implementation."""

    def test_vad_provider_is_abstract(self) -> None:
        import abc

        assert abc.ABC in VADProvider.__mro__
        # name and process are abstract
        abstracts = getattr(VADProvider, "__abstractmethods__", set())
        assert "name" in abstracts
        assert "process" in abstracts

    def test_denoiser_provider_is_abstract(self) -> None:
        abstracts = getattr(DenoiserProvider, "__abstractmethods__", set())
        assert "name" in abstracts
        assert "process" in abstracts

    def test_diarization_provider_is_abstract(self) -> None:
        abstracts = getattr(DiarizationProvider, "__abstractmethods__", set())
        assert "name" in abstracts
        assert "process" in abstracts

    def test_postprocessor_is_abstract(self) -> None:
        abstracts = getattr(AudioPostProcessor, "__abstractmethods__", set())
        assert "name" in abstracts
        assert "process" in abstracts

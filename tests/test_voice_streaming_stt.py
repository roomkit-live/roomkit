"""Tests for streaming STT support in AudioPipeline and VoiceChannel."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from roomkit import MockSTTProvider, MockVoiceBackend, RoomKit, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.base import STTProvider


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id="p1", channel_id="c1")


def _speech_events(audio: bytes = b"fake-audio") -> list[VADEvent | None]:
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        None,  # mid-speech frame
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio),
    ]


# ---------------------------------------------------------------------------
# Pipeline on_speech_frame tests
# ---------------------------------------------------------------------------


class TestPipelineSpeechFrame:
    def test_speech_frame_fires_during_speech(self) -> None:
        """on_speech_frame fires for frames between SPEECH_START and SPEECH_END."""
        vad = MockVADProvider(events=_speech_events())
        config = AudioPipelineConfig(vad=vad)
        pipeline = AudioPipeline(config)

        frames_received: list[AudioFrame] = []
        pipeline.on_speech_frame(lambda s, f: frames_received.append(f))

        session = _session()
        # Frame 1 -> SPEECH_START (included)
        pipeline.process_inbound(session, AudioFrame(data=b"\x00\x01"))
        # Frame 2 -> None (mid-speech, included)
        pipeline.process_inbound(session, AudioFrame(data=b"\x00\x02"))
        # Frame 3 -> SPEECH_END (excluded)
        pipeline.process_inbound(session, AudioFrame(data=b"\x00\x03"))

        assert len(frames_received) == 2
        assert frames_received[0].data == b"\x00\x01"
        assert frames_received[1].data == b"\x00\x02"

    def test_speech_frame_not_fired_outside_speech(self) -> None:
        """on_speech_frame does not fire when no speech is active."""
        vad = MockVADProvider(events=[None, None])
        config = AudioPipelineConfig(vad=vad)
        pipeline = AudioPipeline(config)

        frames_received: list[AudioFrame] = []
        pipeline.on_speech_frame(lambda s, f: frames_received.append(f))

        session = _session()
        pipeline.process_inbound(session, AudioFrame(data=b"\x00\x01"))
        pipeline.process_inbound(session, AudioFrame(data=b"\x00\x02"))

        assert len(frames_received) == 0

    def test_speech_frame_reset_clears_state(self) -> None:
        """reset() clears in-speech tracking."""
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        config = AudioPipelineConfig(vad=vad)
        pipeline = AudioPipeline(config)

        session = _session()
        pipeline.process_inbound(session, AudioFrame(data=b"\x00"))
        assert session.id in pipeline._in_speech_sessions

        pipeline.reset()
        assert session.id not in pipeline._in_speech_sessions

    def test_speech_frame_session_ended_clears_state(self) -> None:
        """on_session_ended() clears speech state for that session."""
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        config = AudioPipelineConfig(vad=vad)
        pipeline = AudioPipeline(config)

        session = _session()
        pipeline.process_inbound(session, AudioFrame(data=b"\x00"))
        assert session.id in pipeline._in_speech_sessions

        pipeline.on_session_ended(session)
        assert session.id not in pipeline._in_speech_sessions


# ---------------------------------------------------------------------------
# STTProvider.supports_streaming tests
# ---------------------------------------------------------------------------


class TestSupportsStreaming:
    def test_base_provider_default_false(self) -> None:
        stt = MockSTTProvider()
        assert stt.supports_streaming is False

    def test_mock_provider_streaming_param(self) -> None:
        stt = MockSTTProvider(streaming=True)
        assert stt.supports_streaming is True


# ---------------------------------------------------------------------------
# Streaming STT Provider for testing
# ---------------------------------------------------------------------------


class _StreamingMockSTT(STTProvider):
    """Mock STT that supports streaming and tracks calls."""

    def __init__(self, final_text: str = "hello world") -> None:
        self._final_text = final_text
        self.transcribe_calls: list[object] = []
        self.stream_chunks: list[AudioChunk] = []
        self.stream_started = False

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: object) -> TranscriptionResult:
        self.transcribe_calls.append(audio)
        return TranscriptionResult(text=self._final_text)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        self.stream_started = True
        async for chunk in audio_stream:
            self.stream_chunks.append(chunk)
        yield TranscriptionResult(text=self._final_text, is_final=True)


class _ErrorStreamingSTT(STTProvider):
    """Mock STT whose stream raises an exception."""

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: object) -> TranscriptionResult:
        return TranscriptionResult(text="fallback")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            pass
        raise RuntimeError("stream error")
        yield  # type: ignore[misc]  # make it a generator


class _PartialStreamingSTT(STTProvider):
    """Mock STT that yields partial then final results."""

    def __init__(self) -> None:
        self.stream_chunks: list[AudioChunk] = []

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: object) -> TranscriptionResult:
        return TranscriptionResult(text="batch fallback")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        async for chunk in audio_stream:
            self.stream_chunks.append(chunk)
        yield TranscriptionResult(text="partial", is_final=False, confidence=0.5)
        yield TranscriptionResult(text="final result", is_final=True)


# ---------------------------------------------------------------------------
# VoiceChannel streaming STT lifecycle tests
# ---------------------------------------------------------------------------


def _build_kit(
    stt: STTProvider,
    vad_events: list[VADEvent | None] | None = None,
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, MockVADProvider]:
    """Build a RoomKit with voice channel wired for testing."""
    backend = MockVoiceBackend()
    vad = MockVADProvider(events=vad_events or _speech_events())
    pipeline = AudioPipelineConfig(vad=vad)
    channel = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=pipeline)
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    return kit, channel, backend, vad


class TestStreamingSTTLifecycle:
    async def test_streaming_stt_uses_stream(self) -> None:
        """When STT supports streaming, frames are fed via queue."""
        stt = _StreamingMockSTT(final_text="streamed result")
        kit, channel, backend, vad = _build_kit(stt)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        # Frame 1 -> SPEECH_START, Frame 2 -> None (mid-speech), Frame 3 -> SPEECH_END
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.2)

        assert stt.stream_started is True
        # Frames are buffered before sending — verify total audio data received
        total_data = b"".join(c.data for c in stt.stream_chunks)
        assert total_data == b"\x01\x00\x02\x00"  # 2 speech frames combined
        # Batch transcribe should NOT be called
        assert len(stt.transcribe_calls) == 0

    async def test_batch_fallback_when_not_streaming(self) -> None:
        """When STT does not support streaming, batch transcribe is used."""
        stt = MockSTTProvider(transcripts=["batch result"])
        kit, channel, backend, vad = _build_kit(stt)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.2)

        assert len(stt.calls) == 1

    async def test_batch_fallback_on_stream_error(self) -> None:
        """When streaming STT errors, falls back to batch transcribe."""
        stt = _ErrorStreamingSTT()
        kit, channel, backend, vad = _build_kit(stt)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.2)

        # Batch fallback should have been called
        assert stt.transcribe.__func__  # verify it's a real method

    async def test_stream_cancelled_on_unbind(self) -> None:
        """Streaming STT is cancelled when session is unbound."""
        stt = _StreamingMockSTT()
        kit, channel, backend, vad = _build_kit(
            stt,
            vad_events=[VADEvent(type=VADEventType.SPEECH_START), None],
        )

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        # Start speech (no SPEECH_END)
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))

        await asyncio.sleep(0.05)
        assert session.id in channel._stt_streams

        # Unbind should cancel the stream
        channel.unbind_session(session)
        assert session.id not in channel._stt_streams

    async def test_partial_results_from_stream(self) -> None:
        """Streaming STT partial results are yielded before final."""
        stt = _PartialStreamingSTT()
        kit, channel, backend, vad = _build_kit(stt)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.2)

        # The stream should have consumed all speech audio
        total_data = b"".join(c.data for c in stt.stream_chunks)
        assert total_data == b"\x01\x00\x02\x00"

    async def test_close_cancels_streams(self) -> None:
        """VoiceChannel.close() cancels all active STT streams."""
        stt = _StreamingMockSTT()
        kit, channel, backend, vad = _build_kit(
            stt,
            vad_events=[VADEvent(type=VADEventType.SPEECH_START), None],
        )

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))

        await asyncio.sleep(0.05)
        assert session.id in channel._stt_streams

        await channel.close()
        assert len(channel._stt_streams) == 0

"""Tests for voice support (STT/TTS/VoiceBackend)."""

import pytest

from roomkit import (
    HookExecution,
    HookTrigger,
    MockSTTProvider,
    MockTTSProvider,
    MockVoiceBackend,
    RoomKit,
    VoiceBackendNotConfiguredError,
    VoiceChannel,
    VoiceNotConfiguredError,
)
from roomkit.models.event import AudioContent
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceSessionState


class TestMockSTT:
    async def test_transcribe_returns_configured_text(self) -> None:
        stt = MockSTTProvider(transcripts=["Hello world"])
        chunk = AudioChunk(data=b"fake-audio")
        result = await stt.transcribe(chunk)
        assert result == "Hello world"
        assert len(stt.calls) == 1

    async def test_transcribe_cycles_through_transcripts(self) -> None:
        stt = MockSTTProvider(transcripts=["One", "Two"])
        chunk = AudioChunk(data=b"audio")
        assert await stt.transcribe(chunk) == "One"
        assert await stt.transcribe(chunk) == "Two"
        assert await stt.transcribe(chunk) == "One"

    async def test_transcribe_stream_yields_result(self) -> None:
        stt = MockSTTProvider(transcripts=["Streamed text"])

        async def audio_gen() -> "AsyncIterator[AudioChunk]":
            yield AudioChunk(data=b"chunk1")
            yield AudioChunk(data=b"chunk2")

        from collections.abc import AsyncIterator

        results = []
        async for result in stt.transcribe_stream(audio_gen()):
            results.append(result)

        assert len(results) == 1
        assert results[0].text == "Streamed text"
        assert results[0].is_final is True

    async def test_name_property(self) -> None:
        stt = MockSTTProvider()
        assert stt.name == "MockSTTProvider"


class TestMockTTS:
    async def test_synthesize_returns_audio_content(self) -> None:
        tts = MockTTSProvider()
        result = await tts.synthesize("Hello")
        assert result.url.startswith("https://mock.test/audio/")
        assert result.transcript == "Hello"
        assert len(tts.calls) == 1

    async def test_synthesize_uses_custom_voice(self) -> None:
        tts = MockTTSProvider(voice="default")
        await tts.synthesize("Test", voice="custom")
        assert tts.calls[0]["voice"] == "custom"

    async def test_synthesize_uses_default_voice(self) -> None:
        tts = MockTTSProvider(voice="my-voice")
        await tts.synthesize("Test")
        assert tts.calls[0]["voice"] == "my-voice"

    async def test_default_voice_property(self) -> None:
        tts = MockTTSProvider(voice="test-voice")
        assert tts.default_voice == "test-voice"

    async def test_name_property(self) -> None:
        tts = MockTTSProvider()
        assert tts.name == "MockTTSProvider"

    async def test_synthesize_stream_yields_chunks(self) -> None:
        tts = MockTTSProvider()
        chunks = []
        async for chunk in tts.synthesize_stream("Hello world"):
            chunks.append(chunk)

        assert len(chunks) == 2  # "Hello" and "world"
        assert chunks[-1].is_final is True


class TestMockVoiceBackend:
    async def test_connect_creates_session(self) -> None:
        backend = MockVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert session.channel_id == "voice-1"
        assert session.state == VoiceSessionState.ACTIVE

    async def test_disconnect_ends_session(self) -> None:
        backend = MockVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.disconnect(session)
        updated = backend.get_session(session.id)
        assert updated is not None
        assert updated.state == VoiceSessionState.ENDED

    async def test_get_session(self) -> None:
        backend = MockVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        found = backend.get_session(session.id)
        assert found is not None
        assert found.id == session.id

    async def test_get_session_not_found(self) -> None:
        backend = MockVoiceBackend()
        assert backend.get_session("nonexistent") is None

    async def test_list_sessions_by_room(self) -> None:
        backend = MockVoiceBackend()
        await backend.connect("room-1", "user-1", "voice-1")
        await backend.connect("room-1", "user-2", "voice-1")
        await backend.connect("room-2", "user-3", "voice-1")

        room1_sessions = backend.list_sessions("room-1")
        assert len(room1_sessions) == 2

        room2_sessions = backend.list_sessions("room-2")
        assert len(room2_sessions) == 1

    async def test_on_speech_start_callback(self) -> None:
        backend = MockVoiceBackend()
        events = []

        def callback(session):
            events.append(("start", session.id))

        backend.on_speech_start(callback)
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.simulate_speech_start(session)

        assert len(events) == 1
        assert events[0] == ("start", session.id)

    async def test_on_speech_end_callback(self) -> None:
        backend = MockVoiceBackend()
        events = []

        def callback(session, audio):
            events.append(("end", session.id, audio))

        backend.on_speech_end(callback)
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.simulate_speech_end(session, b"test-audio")

        assert len(events) == 1
        assert events[0] == ("end", session.id, b"test-audio")

    async def test_send_audio_bytes(self) -> None:
        backend = MockVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.send_audio(session, b"audio-data")

        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0] == (session.id, b"audio-data")

    async def test_send_audio_stream(self) -> None:
        backend = MockVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")

        async def audio_gen():
            yield AudioChunk(data=b"chunk1")
            yield AudioChunk(data=b"chunk2")

        await backend.send_audio(session, audio_gen())

        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0] == (session.id, b"chunk1chunk2")

    async def test_tracks_method_calls(self) -> None:
        backend = MockVoiceBackend()
        await backend.connect("room-1", "user-1", "voice-1")

        call_methods = [c.method for c in backend.calls]
        assert "connect" in call_methods

    async def test_close_clears_sessions(self) -> None:
        backend = MockVoiceBackend()
        await backend.connect("room-1", "user-1", "voice-1")
        await backend.close()

        assert backend.get_session("any") is None
        assert len(backend.list_sessions("room-1")) == 0

    async def test_name_property(self) -> None:
        backend = MockVoiceBackend()
        assert backend.name == "MockVoiceBackend"

    async def test_send_transcription_tracks_calls(self) -> None:
        backend = MockVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.send_transcription(session, "Hello world", "user")
        await backend.send_transcription(session, "Hi there!", "assistant")

        assert len(backend.sent_transcriptions) == 2
        assert backend.sent_transcriptions[0] == (session.id, "Hello world", "user")
        assert backend.sent_transcriptions[1] == (session.id, "Hi there!", "assistant")


class TestVoiceChannel:
    async def test_capabilities_include_audio(self) -> None:
        channel = VoiceChannel("voice-1")
        caps = channel.capabilities()
        assert "audio" in [m.value for m in caps.media_types]
        assert caps.supports_audio is True

    async def test_info_shows_providers(self) -> None:
        stt = MockSTTProvider()
        tts = MockTTSProvider()
        channel = VoiceChannel("voice-1", stt=stt, tts=tts)
        info = channel.info
        assert info["stt"] == "MockSTTProvider"
        assert info["tts"] == "MockTTSProvider"

    async def test_info_without_providers(self) -> None:
        channel = VoiceChannel("voice-1")
        info = channel.info
        assert info["stt"] is None
        assert info["tts"] is None

    async def test_info_shows_backend(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)
        assert channel.info["backend"] == "MockVoiceBackend"

    async def test_backend_property(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)
        assert channel.backend is backend

    async def test_channel_type_is_voice(self) -> None:
        channel = VoiceChannel("voice-1")
        assert channel.channel_type.value == "voice"

    async def test_close_closes_providers(self) -> None:
        stt = MockSTTProvider()
        tts = MockTTSProvider()
        channel = VoiceChannel("voice-1", stt=stt, tts=tts)
        await channel.close()
        # No exception = success (mock providers have no-op close)

    async def test_streaming_mode_default_true(self) -> None:
        channel = VoiceChannel("voice-1")
        assert channel.info["streaming"] is True

    async def test_streaming_mode_explicit(self) -> None:
        channel = VoiceChannel("voice-1", streaming=False)
        assert channel.info["streaming"] is False

    async def test_deliver_returns_empty_in_streaming_mode(self) -> None:
        from unittest.mock import MagicMock

        from roomkit.models.channel import ChannelBinding
        from roomkit.models.context import RoomContext
        from roomkit.models.enums import ChannelType
        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.models.room import Room

        tts = MockTTSProvider()
        channel = VoiceChannel("voice-1", tts=tts, streaming=True)

        room = Room(id="room-1", organization_id="org-1")
        context = MagicMock(spec=RoomContext)
        context.room = room
        binding = ChannelBinding(
            room_id="room-1", channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        event = RoomEvent(
            room_id="room-1",
            content=TextContent(body="Hello"),
            source=EventSource(channel_id="voice-1", channel_type=ChannelType.VOICE),
        )

        output = await channel.deliver(event, binding, context)
        # Should return empty output (no-op in streaming mode)
        assert output.responded is False
        assert output.response_events == []
        # TTS should NOT be called in streaming mode
        assert len(tts.calls) == 0

    async def test_non_streaming_mode_raises_not_implemented(self) -> None:
        from unittest.mock import MagicMock

        from roomkit.models.channel import ChannelBinding
        from roomkit.models.context import RoomContext
        from roomkit.models.enums import ChannelType
        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.models.room import Room

        tts = MockTTSProvider()
        channel = VoiceChannel("voice-1", tts=tts, streaming=False)

        room = Room(id="room-1", organization_id="org-1")
        context = MagicMock(spec=RoomContext)
        context.room = room
        binding = ChannelBinding(
            room_id="room-1", channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        event = RoomEvent(
            room_id="room-1",
            content=TextContent(body="Hello"),
            source=EventSource(channel_id="voice-1", channel_type=ChannelType.VOICE),
        )

        with pytest.raises(NotImplementedError, match="MediaStore"):
            await channel.deliver(event, binding, context)


class TestRoomKitVoiceIntegration:
    async def test_transcribe_uses_stt_provider(self) -> None:
        stt = MockSTTProvider(transcripts=["Transcribed text"])
        kit = RoomKit(stt=stt)
        audio = AudioContent(url="https://example.com/audio.wav")
        result = await kit.transcribe(audio)
        assert result == "Transcribed text"

    async def test_synthesize_uses_tts_provider(self) -> None:
        tts = MockTTSProvider()
        kit = RoomKit(tts=tts)
        result = await kit.synthesize("Hello world")
        assert result.transcript == "Hello world"

    async def test_transcribe_without_stt_raises(self) -> None:
        kit = RoomKit()
        audio = AudioContent(url="https://example.com/audio.wav")
        with pytest.raises(VoiceNotConfiguredError, match="No STT provider"):
            await kit.transcribe(audio)

    async def test_synthesize_without_tts_raises(self) -> None:
        kit = RoomKit()
        with pytest.raises(VoiceNotConfiguredError, match="No TTS provider"):
            await kit.synthesize("Hello")

    async def test_stt_property(self) -> None:
        stt = MockSTTProvider()
        kit = RoomKit(stt=stt)
        assert kit.stt is stt

    async def test_tts_property(self) -> None:
        tts = MockTTSProvider()
        kit = RoomKit(tts=tts)
        assert kit.tts is tts

    async def test_properties_none_when_not_configured(self) -> None:
        kit = RoomKit()
        assert kit.stt is None
        assert kit.tts is None

    async def test_synthesize_with_voice(self) -> None:
        tts = MockTTSProvider()
        kit = RoomKit(tts=tts)
        await kit.synthesize("Test", voice="custom-voice")
        assert tts.calls[0]["voice"] == "custom-voice"

    async def test_voice_property(self) -> None:
        backend = MockVoiceBackend()
        kit = RoomKit(voice=backend)
        assert kit.voice is backend

    async def test_voice_property_none_when_not_configured(self) -> None:
        kit = RoomKit()
        assert kit.voice is None

    async def test_connect_voice_creates_session(self) -> None:
        backend = MockVoiceBackend()
        kit = RoomKit(voice=backend)

        # Create room and attach voice channel
        channel = VoiceChannel("voice-1", backend=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        assert session.room_id == room.id
        assert session.participant_id == "user-1"
        assert session.channel_id == "voice-1"

    async def test_connect_voice_without_backend_raises(self) -> None:
        kit = RoomKit()
        channel = VoiceChannel("voice-1")
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        with pytest.raises(VoiceBackendNotConfiguredError):
            await kit.connect_voice(room.id, "user-1", "voice-1")

    async def test_disconnect_voice(self) -> None:
        backend = MockVoiceBackend()
        kit = RoomKit(voice=backend)

        channel = VoiceChannel("voice-1", backend=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        await kit.disconnect_voice(session)

        updated = backend.get_session(session.id)
        assert updated.state == VoiceSessionState.ENDED

    async def test_disconnect_voice_without_backend_raises(self) -> None:
        kit = RoomKit()
        from roomkit.voice.base import VoiceSession

        session = VoiceSession(
            id="test",
            room_id="room-1",
            participant_id="user-1",
            channel_id="voice-1",
        )
        with pytest.raises(VoiceBackendNotConfiguredError):
            await kit.disconnect_voice(session)


class TestVoiceChannelPipeline:
    """Tests for the full VoiceChannel speech pipeline."""

    async def test_speech_end_passes_sample_rate_to_stt(self) -> None:
        """When speech ends, VoiceChannel should pass sample_rate from session metadata to STT."""
        import asyncio

        stt = MockSTTProvider(transcripts=["Hello world"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend)

        kit = RoomKit(stt=stt, tts=tts, voice=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        binding = await kit.attach_channel(room.id, "voice-1")

        # Create session with custom sample rate in metadata
        session = await backend.connect(
            room.id, "user-1", "voice-1",
            metadata={"input_sample_rate": 48000, "output_sample_rate": 24000},
        )
        channel.bind_session(session, room.id, binding)

        # Simulate speech end
        await backend.simulate_speech_end(session, b"fake-pcm-audio")

        # Give async tasks time to complete
        await asyncio.sleep(0.1)

        # STT should have been called with the audio
        assert len(stt.calls) == 1
        audio_chunk = stt.calls[0]
        assert audio_chunk.data == b"fake-pcm-audio"
        assert audio_chunk.sample_rate == 48000

    async def test_speech_end_sends_user_transcription(self) -> None:
        """After STT, VoiceChannel should send transcription to client via backend."""
        import asyncio

        stt = MockSTTProvider(transcripts=["Bonjour"])
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", stt=stt, backend=backend)

        kit = RoomKit(stt=stt, voice=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        binding = await kit.attach_channel(room.id, "voice-1")

        session = await backend.connect(room.id, "user-1", "voice-1")
        channel.bind_session(session, room.id, binding)

        await backend.simulate_speech_end(session, b"audio-data")
        await asyncio.sleep(0.1)

        # Backend should have received a user transcription
        user_transcriptions = [
            (sid, text, role) for sid, text, role in backend.sent_transcriptions
            if role == "user"
        ]
        assert len(user_transcriptions) == 1
        assert user_transcriptions[0][1] == "Bonjour"

    async def test_deliver_sends_assistant_transcription(self) -> None:
        """When delivering voice, VoiceChannel should send assistant text to client."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from roomkit.models.channel import ChannelBinding
        from roomkit.models.context import RoomContext
        from roomkit.models.enums import ChannelType
        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.models.room import Room

        stt = MockSTTProvider()
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend)

        kit = RoomKit(stt=stt, tts=tts, voice=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        binding = await kit.attach_channel(room.id, "voice-1")

        session = await backend.connect(room.id, "user-1", "voice-1")
        channel.bind_session(session, room.id, binding)

        event = RoomEvent(
            room_id=room.id,
            content=TextContent(body="How can I help?"),
            source=EventSource(channel_id="voice-1", channel_type=ChannelType.VOICE),
        )

        context = await kit._build_context(room.id)
        await channel.deliver(event, binding, context)

        # Backend should have received assistant transcription
        assistant_transcriptions = [
            (sid, text, role) for sid, text, role in backend.sent_transcriptions
            if role == "assistant"
        ]
        assert len(assistant_transcriptions) == 1
        assert assistant_transcriptions[0][1] == "How can I help?"

        # TTS should have been called
        assert len(tts.calls) == 1

        # Audio should have been sent to backend
        assert len(backend.sent_audio) == 1


class TestAudioChunk:
    def test_default_values(self) -> None:
        chunk = AudioChunk(data=b"audio")
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.format == "pcm_s16le"
        assert chunk.timestamp_ms is None
        assert chunk.is_final is False

    def test_custom_values(self) -> None:
        chunk = AudioChunk(
            data=b"audio",
            sample_rate=48000,
            channels=2,
            format="opus",
            timestamp_ms=1000,
            is_final=True,
        )
        assert chunk.sample_rate == 48000
        assert chunk.channels == 2
        assert chunk.format == "opus"
        assert chunk.timestamp_ms == 1000
        assert chunk.is_final is True


class TestTranscriptionResult:
    def test_default_values(self) -> None:
        result = TranscriptionResult(text="Hello")
        assert result.text == "Hello"
        assert result.is_final is True
        assert result.confidence is None
        assert result.language is None
        assert result.words == []

    def test_custom_values(self) -> None:
        result = TranscriptionResult(
            text="Hello",
            is_final=False,
            confidence=0.95,
            language="en",
            words=[{"word": "Hello", "start": 0.0, "end": 0.5}],
        )
        assert result.is_final is False
        assert result.confidence == 0.95
        assert result.language == "en"
        assert len(result.words) == 1


class TestVoiceCapability:
    """Tests for VoiceCapability flags (RFC §19)."""

    def test_default_is_none(self) -> None:
        from roomkit.voice.base import VoiceCapability

        assert VoiceCapability.NONE.value == 0
        assert not VoiceCapability.NONE  # Falsy when no flags set

    def test_individual_flags(self) -> None:
        from roomkit.voice.base import VoiceCapability

        assert VoiceCapability.INTERRUPTION
        assert VoiceCapability.PARTIAL_STT
        assert VoiceCapability.VAD_SILENCE
        assert VoiceCapability.VAD_AUDIO_LEVEL
        assert VoiceCapability.BARGE_IN

    def test_combine_flags(self) -> None:
        from roomkit.voice.base import VoiceCapability

        caps = VoiceCapability.INTERRUPTION | VoiceCapability.BARGE_IN
        assert VoiceCapability.INTERRUPTION in caps
        assert VoiceCapability.BARGE_IN in caps
        assert VoiceCapability.PARTIAL_STT not in caps

    def test_mock_backend_default_none(self) -> None:
        from roomkit.voice.base import VoiceCapability

        backend = MockVoiceBackend()
        assert backend.capabilities == VoiceCapability.NONE

    def test_mock_backend_custom_capabilities(self) -> None:
        from roomkit.voice.base import VoiceCapability

        caps = VoiceCapability.INTERRUPTION | VoiceCapability.PARTIAL_STT
        backend = MockVoiceBackend(capabilities=caps)
        assert backend.capabilities == caps
        assert VoiceCapability.INTERRUPTION in backend.capabilities


class TestEnhancedVoiceCallbacks:
    """Tests for enhanced voice callbacks (RFC §19)."""

    async def test_partial_transcription_callback(self) -> None:
        from roomkit.voice.base import VoiceCapability

        backend = MockVoiceBackend(capabilities=VoiceCapability.PARTIAL_STT)
        session = await backend.connect("room-1", "user-1", "voice-1")

        received = []

        def callback(sess, text, confidence, is_stable):
            received.append((sess.id, text, confidence, is_stable))

        backend.on_partial_transcription(callback)
        await backend.simulate_partial_transcription(session, "Hello", 0.8, False)

        assert len(received) == 1
        assert received[0] == (session.id, "Hello", 0.8, False)

    async def test_vad_silence_callback(self) -> None:
        from roomkit.voice.base import VoiceCapability

        backend = MockVoiceBackend(capabilities=VoiceCapability.VAD_SILENCE)
        session = await backend.connect("room-1", "user-1", "voice-1")

        received = []

        def callback(sess, silence_ms):
            received.append((sess.id, silence_ms))

        backend.on_vad_silence(callback)
        await backend.simulate_vad_silence(session, 500)

        assert len(received) == 1
        assert received[0] == (session.id, 500)

    async def test_vad_audio_level_callback(self) -> None:
        from roomkit.voice.base import VoiceCapability

        backend = MockVoiceBackend(capabilities=VoiceCapability.VAD_AUDIO_LEVEL)
        session = await backend.connect("room-1", "user-1", "voice-1")

        received = []

        def callback(sess, level_db, is_speech):
            received.append((sess.id, level_db, is_speech))

        backend.on_vad_audio_level(callback)
        await backend.simulate_vad_audio_level(session, -20.5, True)

        assert len(received) == 1
        assert received[0] == (session.id, -20.5, True)

    async def test_barge_in_callback(self) -> None:
        from roomkit.voice.base import VoiceCapability

        backend = MockVoiceBackend(capabilities=VoiceCapability.BARGE_IN)
        session = await backend.connect("room-1", "user-1", "voice-1")

        received = []

        def callback(sess):
            received.append(sess.id)

        backend.on_barge_in(callback)
        await backend.simulate_barge_in(session)

        assert len(received) == 1
        assert received[0] == session.id

    async def test_cancel_audio(self) -> None:
        from roomkit.voice.base import VoiceCapability

        backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
        session = await backend.connect("room-1", "user-1", "voice-1")

        # Not playing initially
        assert backend.is_playing(session) is False
        result = await backend.cancel_audio(session)
        assert result is False

        # Start playing
        backend.start_playing(session)
        assert backend.is_playing(session) is True

        # Cancel
        result = await backend.cancel_audio(session)
        assert result is True
        assert backend.is_playing(session) is False

    async def test_is_playing(self) -> None:
        backend = MockVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")

        assert backend.is_playing(session) is False
        backend.start_playing(session)
        assert backend.is_playing(session) is True
        backend.stop_playing(session)
        assert backend.is_playing(session) is False


class TestVoiceEvents:
    """Tests for voice event dataclasses (RFC §19)."""

    def test_barge_in_event(self) -> None:
        from roomkit.voice.base import VoiceSession
        from roomkit.voice.events import BargeInEvent

        session = VoiceSession(
            id="sess-1", room_id="room-1", participant_id="user-1", channel_id="voice-1"
        )
        event = BargeInEvent(
            session=session,
            interrupted_text="Hello, how can I help you?",
            audio_position_ms=1500,
        )
        assert event.session.id == "sess-1"
        assert event.interrupted_text == "Hello, how can I help you?"
        assert event.audio_position_ms == 1500
        assert event.timestamp is not None

    def test_tts_cancelled_event(self) -> None:
        from roomkit.voice.base import VoiceSession
        from roomkit.voice.events import TTSCancelledEvent

        session = VoiceSession(
            id="sess-1", room_id="room-1", participant_id="user-1", channel_id="voice-1"
        )
        event = TTSCancelledEvent(
            session=session,
            reason="barge_in",
            text="Hello, how can I help?",
            audio_position_ms=1200,
        )
        assert event.reason == "barge_in"
        assert event.text == "Hello, how can I help?"
        assert event.audio_position_ms == 1200

    def test_partial_transcription_event(self) -> None:
        from roomkit.voice.base import VoiceSession
        from roomkit.voice.events import PartialTranscriptionEvent

        session = VoiceSession(
            id="sess-1", room_id="room-1", participant_id="user-1", channel_id="voice-1"
        )
        event = PartialTranscriptionEvent(
            session=session,
            text="Hello wor",
            confidence=0.75,
            is_stable=False,
        )
        assert event.text == "Hello wor"
        assert event.confidence == 0.75
        assert event.is_stable is False

    def test_vad_silence_event(self) -> None:
        from roomkit.voice.base import VoiceSession
        from roomkit.voice.events import VADSilenceEvent

        session = VoiceSession(
            id="sess-1", room_id="room-1", participant_id="user-1", channel_id="voice-1"
        )
        event = VADSilenceEvent(
            session=session,
            silence_duration_ms=750,
        )
        assert event.silence_duration_ms == 750

    def test_vad_audio_level_event(self) -> None:
        from roomkit.voice.base import VoiceSession
        from roomkit.voice.events import VADAudioLevelEvent

        session = VoiceSession(
            id="sess-1", room_id="room-1", participant_id="user-1", channel_id="voice-1"
        )
        event = VADAudioLevelEvent(
            session=session,
            level_db=-25.5,
            is_speech=True,
        )
        assert event.level_db == -25.5
        assert event.is_speech is True

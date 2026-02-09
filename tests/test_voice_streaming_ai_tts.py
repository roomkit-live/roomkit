"""Tests for streaming AI → TTS pipeline (framework-native architecture)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from roomkit import MockSTTProvider, MockTTSProvider, MockVoiceBackend, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.sentence_splitter import split_sentences

# ---------------------------------------------------------------------------
# Sentence splitter tests (unchanged — unit tests, no architecture dependency)
# ---------------------------------------------------------------------------


async def _aiter(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


class TestSentenceSplitter:
    async def test_splits_at_sentence_boundary(self) -> None:
        tokens = ["Hello, ", "how are ", "you today? ", "I am fine."]
        result = [s async for s in split_sentences(_aiter(tokens), min_chunk_chars=10)]
        assert result == ["Hello, how are you today?", "I am fine."]

    async def test_min_chunk_chars_respected(self) -> None:
        tokens = ["Hi. ", "Ok. ", "This is a longer sentence that should flush."]
        result = [s async for s in split_sentences(_aiter(tokens), min_chunk_chars=20)]
        assert len(result) >= 1
        combined = " ".join(result)
        assert "Hi." in combined
        assert "longer sentence" in combined

    async def test_flushes_remaining_on_stream_end(self) -> None:
        tokens = ["This has no sentence ending"]
        result = [s async for s in split_sentences(_aiter(tokens), min_chunk_chars=5)]
        assert result == ["This has no sentence ending"]

    async def test_empty_stream(self) -> None:
        result = [s async for s in split_sentences(_aiter([]), min_chunk_chars=5)]
        assert result == []

    async def test_multiple_sentences(self) -> None:
        tokens = [
            "First sentence. ",
            "Second one! ",
            "Third? ",
            "Partial end",
        ]
        result = [s async for s in split_sentences(_aiter(tokens), min_chunk_chars=5)]
        assert "First sentence." in result
        assert "Second one!" in result
        combined = " ".join(result)
        assert "Third?" in combined
        assert "Partial end" in combined


# ---------------------------------------------------------------------------
# Mock providers for streaming AI → TTS tests
# ---------------------------------------------------------------------------


class _StreamingAIProvider(AIProvider):
    """Mock AI provider that supports streaming."""

    def __init__(self, response_tokens: list[str]) -> None:
        self._tokens = response_tokens
        self.generate_stream_calls: list[AIContext] = []

    @property
    def model_name(self) -> str:
        return "mock-streaming"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="".join(self._tokens))

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        self.generate_stream_calls.append(context)
        for token in self._tokens:
            yield token


class _NonStreamingAIProvider(AIProvider):
    """Mock AI provider that does NOT support streaming."""

    def __init__(self, response: str = "non-streaming response") -> None:
        self._response = response
        self.generate_calls: list[AIContext] = []

    @property
    def model_name(self) -> str:
        return "mock-non-streaming"

    async def generate(self, context: AIContext) -> AIResponse:
        self.generate_calls.append(context)
        return AIResponse(content=self._response)


class _StreamingInputTTS(TTSProvider):
    """Mock TTS that supports streaming text input."""

    def __init__(self) -> None:
        self.synthesize_stream_input_calls: list[list[str]] = []
        self.synthesize_stream_calls: list[str] = []

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.synthesize_stream_calls.append(text)
        yield AudioChunk(data=b"mock-full-audio", sample_rate=16000, is_final=True)

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        chunks: list[str] = []
        async for text in text_stream:
            chunks.append(text)
        self.synthesize_stream_input_calls.append(chunks)
        for i, chunk in enumerate(chunks):
            yield AudioChunk(
                data=f"audio-{chunk}".encode(),
                sample_rate=16000,
                is_final=(i == len(chunks) - 1),
            )


class _ErrorStreamingAIProvider(AIProvider):
    """Mock AI provider whose streaming raises an error."""

    @property
    def model_name(self) -> str:
        return "mock-error-streaming"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="fallback")

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        raise RuntimeError("stream error")
        yield  # type: ignore[misc]  # pragma: no cover


# ---------------------------------------------------------------------------
# Helper to build a wired RoomKit for testing
# ---------------------------------------------------------------------------


def _speech_events(audio: bytes = b"fake-audio") -> list[VADEvent | None]:
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        None,
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio),
    ]


def _build_kit(
    ai_provider: AIProvider,
    tts: TTSProvider | None = None,
    vad_events: list[VADEvent | None] | None = None,
    ai_system_prompt: str = "Be helpful.",
    ai_temperature: float = 0.5,
    ai_max_tokens: int = 128,
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, MockVADProvider, AIChannel]:
    stt = MockSTTProvider(transcripts=["Hello world"])
    backend = MockVoiceBackend()
    vad = MockVADProvider(events=vad_events or _speech_events())
    pipeline = AudioPipelineConfig(vad=vad)
    tts_provider = tts or _StreamingInputTTS()

    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        tts=tts_provider,
        backend=backend,
        pipeline=pipeline,
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)

    ai_channel = AIChannel(
        "ai-1",
        provider=ai_provider,
        system_prompt=ai_system_prompt,
        temperature=ai_temperature,
        max_tokens=ai_max_tokens,
    )
    kit.register_channel(ai_channel)
    return kit, channel, backend, vad, ai_channel


# ---------------------------------------------------------------------------
# supports_streaming_delivery property tests
# ---------------------------------------------------------------------------


class TestSupportsStreamingDelivery:
    def test_true_when_tts_and_backend(self) -> None:
        tts = _StreamingInputTTS()
        backend = MockVoiceBackend()
        channel = VoiceChannel("v", tts=tts, backend=backend)
        assert channel.supports_streaming_delivery is True

    def test_false_when_tts_not_streaming_input(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("v", tts=tts, backend=backend)
        assert channel.supports_streaming_delivery is False

    def test_false_when_no_tts(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("v", backend=backend)
        assert channel.supports_streaming_delivery is False

    def test_false_when_no_backend(self) -> None:
        tts = _StreamingInputTTS()
        channel = VoiceChannel("v", tts=tts)
        assert channel.supports_streaming_delivery is False


# ---------------------------------------------------------------------------
# AIChannel streaming response tests
# ---------------------------------------------------------------------------


class TestAIChannelStreamingResponse:
    async def test_returns_stream_when_provider_supports_it(self) -> None:
        """AIChannel returns response_stream when provider supports streaming."""
        ai = _StreamingAIProvider(["hello ", "world"])
        channel = AIChannel("ai-1", provider=ai, system_prompt="Test")

        from roomkit.models.event import EventSource, RoomEvent, TextContent

        event = RoomEvent(
            room_id="room-1",
            source=EventSource(channel_id="voice-1", channel_type=ChannelType.VOICE),
            content=TextContent(body="Hi"),
        )
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="ai-1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        from roomkit.models.context import RoomContext
        from roomkit.models.room import Room

        context = RoomContext(
            room=Room(id="room-1"),
            bindings=[binding],
        )

        output = await channel.on_event(event, binding, context)
        assert output.responded is True
        assert output.response_stream is not None
        assert output.response_events == []

        # Consume the stream
        tokens = []
        async for delta in output.response_stream:
            tokens.append(delta)
        assert tokens == ["hello ", "world"]

    async def test_falls_back_when_tools_configured(self) -> None:
        """AIChannel falls back to non-streaming when tools are in binding metadata."""
        ai = _StreamingAIProvider(["hello"])
        channel = AIChannel("ai-1", provider=ai, system_prompt="Test")

        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.models.room import Room

        event = RoomEvent(
            room_id="room-1",
            source=EventSource(channel_id="voice-1", channel_type=ChannelType.VOICE),
            content=TextContent(body="Hi"),
        )
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="ai-1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"tools": [{"name": "search", "description": "Search"}]},
        )
        context = RoomContext(room=Room(id="room-1"), bindings=[binding])

        output = await channel.on_event(event, binding, context)
        assert output.responded is True
        assert output.response_stream is None
        assert len(output.response_events) == 1

    async def test_falls_back_when_provider_not_streaming(self) -> None:
        """AIChannel uses generate() when provider doesn't support streaming."""
        ai = _NonStreamingAIProvider("fallback response")
        channel = AIChannel("ai-1", provider=ai, system_prompt="Test")

        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.models.room import Room

        event = RoomEvent(
            room_id="room-1",
            source=EventSource(channel_id="voice-1", channel_type=ChannelType.VOICE),
            content=TextContent(body="Hi"),
        )
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="ai-1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        context = RoomContext(room=Room(id="room-1"), bindings=[binding])

        output = await channel.on_event(event, binding, context)
        assert output.responded is True
        assert output.response_stream is None
        assert len(output.response_events) == 1


# ---------------------------------------------------------------------------
# Framework-native streaming integration tests
# ---------------------------------------------------------------------------


class TestStreamingAiToTts:
    async def test_streaming_pipeline_end_to_end(self) -> None:
        """Full flow: speech → STT → framework routes streaming AI → TTS → audio."""
        ai = _StreamingAIProvider(["I'm ", "doing ", "great!"])
        tts = _StreamingInputTTS()
        kit, channel, backend, vad, _ = _build_kit(ai, tts=tts)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        # Simulate speech: START, mid, END
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.5)

        # AI streaming was called (via framework)
        assert len(ai.generate_stream_calls) == 1
        # TTS streaming input was called (via deliver_stream)
        assert len(tts.synthesize_stream_input_calls) == 1
        # Audio was sent to backend
        assert len(backend.sent_audio) >= 1
        # Assistant transcription was sent
        assistant_transcriptions = [
            (sid, text, role)
            for sid, text, role in backend.sent_transcriptions
            if role == "assistant"
        ]
        assert len(assistant_transcriptions) >= 1
        assert assistant_transcriptions[0][1] == "I'm doing great!"

        await kit.close()

    async def test_events_stored_after_streaming(self) -> None:
        """User and AI events are stored in the conversation store."""
        ai = _StreamingAIProvider(["Response text."])
        tts = _StreamingInputTTS()
        kit, channel, backend, vad, _ = _build_kit(ai, tts=tts)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.5)

        events = await kit._store.list_events(room.id, offset=0, limit=50)
        texts = [e.content.body for e in events if hasattr(e.content, "body")]
        assert "Hello world" in texts  # user event
        assert "Response text." in texts  # AI event (stored by framework)

        await kit.close()

    async def test_non_streaming_provider_uses_normal_routing(self) -> None:
        """When AI doesn't support streaming, normal generate() path is used."""
        ai = _NonStreamingAIProvider("non-streaming response")
        tts = _StreamingInputTTS()
        kit, channel, backend, vad, _ = _build_kit(ai, tts=tts)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.5)

        # TTS streaming input should NOT be called (normal deliver() path)
        assert len(tts.synthesize_stream_input_calls) == 0
        # AI generate() WAS called (non-streaming)
        assert len(ai.generate_calls) >= 1

        await kit.close()

    async def test_error_during_streaming_handled(self) -> None:
        """When streaming AI errors, it's handled gracefully."""
        ai = _ErrorStreamingAIProvider()
        tts = _StreamingInputTTS()
        kit, channel, backend, vad, _ = _build_kit(ai, tts=tts)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.5)

        # Should not crash — error logged by framework
        await kit.close()

    async def test_ai_context_uses_channel_config(self) -> None:
        """AI context built by AIChannel uses its system prompt and params."""
        ai = _StreamingAIProvider(["Ok."])
        tts = _StreamingInputTTS()
        kit, channel, backend, vad, _ = _build_kit(
            ai,
            tts=tts,
            ai_system_prompt="Be helpful.",
            ai_temperature=0.5,
            ai_max_tokens=128,
        )

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.5)

        assert len(ai.generate_stream_calls) == 1
        ctx = ai.generate_stream_calls[0]
        assert ctx.system_prompt == "Be helpful."
        assert ctx.temperature == 0.5
        assert ctx.max_tokens == 128
        assert ctx.messages[-1].role == "user"
        assert ctx.messages[-1].content == "Hello world"

        await kit.close()

    async def test_deliver_stream_calls_synthesize_stream_input(self) -> None:
        """VoiceChannel.deliver_stream() pipes text to TTS streaming input."""
        tts = _StreamingInputTTS()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)

        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.models.room import Room

        kit = RoomKit(voice=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.connect_voice(room.id, "user-1", "voice-1")

        event = RoomEvent(
            room_id=room.id,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body=""),
        )
        binding = ChannelBinding(
            room_id=room.id,
            channel_id="voice-1",
            channel_type=ChannelType.VOICE,
        )
        context = RoomContext(room=Room(id=room.id), bindings=[binding])

        async def text_gen() -> AsyncIterator[str]:
            yield "Hello "
            yield "world!"

        await channel.deliver_stream(text_gen(), event, binding, context)

        assert len(tts.synthesize_stream_input_calls) == 1
        assert len(backend.sent_audio) >= 1
        # Transcription sent
        assistant_transcriptions = [
            (sid, text, role)
            for sid, text, role in backend.sent_transcriptions
            if role == "assistant"
        ]
        assert len(assistant_transcriptions) >= 1
        assert assistant_transcriptions[0][1] == "Hello world!"

        await kit.close()

"""Tests for streaming AI → TTS pipeline in VoiceChannel."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from roomkit import MockSTTProvider, MockVoiceBackend, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.sentence_splitter import split_sentences

# ---------------------------------------------------------------------------
# Sentence splitter tests
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
        # "Hi. Ok. " is only 8 chars, not enough to split
        # Everything accumulates until the longer text triggers a split or flush
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
        # "Third?" and "Partial end" may be separate or combined depending on min_chunk
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
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, MockVADProvider]:
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
        ai_provider=ai_provider,
        ai_channel_id="ai-1",
        ai_system_prompt="Be helpful.",
        ai_temperature=0.5,
        ai_max_tokens=128,
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    # Register AI channel for fallback routing
    ai_channel = AIChannel("ai-1", provider=ai_provider, system_prompt="Be helpful.")
    kit.register_channel(ai_channel)
    return kit, channel, backend, vad


# ---------------------------------------------------------------------------
# _can_stream_ai_tts property tests
# ---------------------------------------------------------------------------


class TestCanStreamAiTts:
    def test_true_when_both_support_streaming(self) -> None:
        ai = _StreamingAIProvider(["hello"])
        tts = _StreamingInputTTS()
        channel = VoiceChannel("v", ai_provider=ai, tts=tts)
        assert channel._can_stream_ai_tts is True

    def test_false_when_ai_not_streaming(self) -> None:
        ai = _NonStreamingAIProvider()
        tts = _StreamingInputTTS()
        channel = VoiceChannel("v", ai_provider=ai, tts=tts)
        assert channel._can_stream_ai_tts is False

    def test_false_when_tts_not_streaming_input(self) -> None:
        from roomkit import MockTTSProvider

        ai = _StreamingAIProvider(["hello"])
        tts = MockTTSProvider()
        channel = VoiceChannel("v", ai_provider=ai, tts=tts)
        assert channel._can_stream_ai_tts is False

    def test_false_when_no_ai_provider(self) -> None:
        tts = _StreamingInputTTS()
        channel = VoiceChannel("v", tts=tts)
        assert channel._can_stream_ai_tts is False

    def test_false_when_no_tts(self) -> None:
        ai = _StreamingAIProvider(["hello"])
        channel = VoiceChannel("v", ai_provider=ai)
        assert channel._can_stream_ai_tts is False


# ---------------------------------------------------------------------------
# Streaming AI → TTS integration tests
# ---------------------------------------------------------------------------


class TestStreamingAiToTts:
    async def test_streaming_pipeline_end_to_end(self) -> None:
        """Full flow: speech → STT → streaming AI → TTS → audio output."""
        ai = _StreamingAIProvider(["I'm ", "doing ", "great!"])
        tts = _StreamingInputTTS()
        kit, channel, backend, vad = _build_kit(ai, tts=tts)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        # Simulate speech: START, mid, END
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.5)

        # AI streaming was called
        assert len(ai.generate_stream_calls) == 1
        # TTS streaming input was called
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
        kit, channel, backend, vad = _build_kit(ai, tts=tts)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.5)

        # Check events stored
        events = await kit._store.list_events(room.id, offset=0, limit=50)
        texts = [e.content.body for e in events if hasattr(e.content, "body")]
        assert "Hello world" in texts  # user event
        assert "Response text." in texts  # AI event

        await kit.close()

    async def test_fallback_to_route_text_when_not_streaming(self) -> None:
        """When AI doesn't support streaming, falls back to normal routing."""
        ai = _NonStreamingAIProvider("non-streaming response")
        tts = _StreamingInputTTS()
        kit, channel, backend, vad = _build_kit(ai, tts=tts)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        assert channel._can_stream_ai_tts is False

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.5)

        # TTS streaming input should NOT be called (normal path instead)
        assert len(tts.synthesize_stream_input_calls) == 0
        # But AI generate() WAS called via normal route
        assert len(ai.generate_calls) >= 1

        await kit.close()

    async def test_error_during_streaming_falls_back(self) -> None:
        """When streaming AI errors before audio starts, falls back."""
        ai = _ErrorStreamingAIProvider()
        tts = _StreamingInputTTS()
        kit, channel, backend, vad = _build_kit(ai, tts=tts)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        await asyncio.sleep(0.5)

        # Should not crash — error logged and fallback attempted
        await kit.close()

    async def test_ai_context_includes_system_prompt(self) -> None:
        """AI context built for streaming includes system prompt and params."""
        ai = _StreamingAIProvider(["Ok."])
        tts = _StreamingInputTTS()
        kit, channel, backend, vad = _build_kit(ai, tts=tts)

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
        # Last message should be the user's text
        assert ctx.messages[-1].role == "user"
        assert ctx.messages[-1].content == "Hello world"

        await kit.close()

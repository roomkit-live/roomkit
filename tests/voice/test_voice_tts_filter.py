"""Integration tests for tts_filter on VoiceChannel."""

from __future__ import annotations

from collections.abc import AsyncIterator

from roomkit import (
    ChannelType,
    MockTTSProvider,
    MockVoiceBackend,
    StripBrackets,
    StripInternalTags,
    VoiceChannel,
)
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.tts.base import TTSProvider


class _StreamingMockTTS(TTSProvider):
    """Mock TTS that supports synthesize_stream_input for streaming tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []
        self.stream_input_texts: list[list[str]] = []

    @property
    def name(self) -> str:
        return "StreamingMockTTS"

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None):  # type: ignore[override]
        from roomkit.models.event import AudioContent as AC

        self.calls.append({"text": text, "voice": voice})
        return AC(url="mock://audio", mime_type="audio/wav", transcript=text)

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        chunks: list[str] = []
        async for text in text_stream:
            chunks.append(text)
        self.stream_input_texts.append(chunks)
        combined = "".join(chunks)
        self.calls.append({"text": combined, "voice": voice})
        raw = f"audio-{combined}".encode()
        if len(raw) % 2 != 0:
            raw += b"\x00"
        yield AudioChunk(data=raw, sample_rate=16000, is_final=True)

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.calls.append({"text": text, "voice": voice})
        raw = f"audio-{text}".encode()
        if len(raw) % 2 != 0:
            raw += b"\x00"
        yield AudioChunk(data=raw, sample_rate=16000, is_final=True)


def _make_session(
    session_id: str = "sess-1",
    room_id: str = "room-1",
    channel_id: str = "voice-1",
) -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id=room_id,
        participant_id="user-1",
        channel_id=channel_id,
    )


# ---------------------------------------------------------------------------
# say() + tts_filter
# ---------------------------------------------------------------------------


class TestSayWithFilter:
    async def test_say_strips_internal_tags(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend, tts_filter=StripInternalTags())
        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.say(session, "[internal]Think in French[/internal] Bonjour!")

        assert len(tts.calls) == 1
        assert tts.calls[0]["text"] == "Bonjour!"

    async def test_say_strips_brackets(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend, tts_filter=StripBrackets())
        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.say(session, "[laughs] That was funny!")

        assert len(tts.calls) == 1
        assert tts.calls[0]["text"] == "That was funny!"

    async def test_say_empty_after_filter_skips_tts(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend, tts_filter=StripInternalTags())
        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.say(session, "[internal]all internal[/internal]")

        # TTS should not have been called
        assert len(tts.calls) == 0
        assert len(backend.sent_audio) == 0

    async def test_say_no_filter_unchanged(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)
        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.say(session, "[internal]visible[/internal] text")

        # Without filter, the raw text goes to TTS
        assert len(tts.calls) == 1
        assert tts.calls[0]["text"] == "[internal]visible[/internal] text"

    async def test_say_with_plain_callable(self) -> None:
        """A plain function (not TTSStreamFilter) works as tts_filter."""
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel(
            "voice-1",
            tts=tts,
            backend=backend,
            tts_filter=lambda t: t.upper(),
        )
        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.say(session, "hello world")

        assert tts.calls[0]["text"] == "HELLO WORLD"


# ---------------------------------------------------------------------------
# _deliver_voice (non-streaming) + tts_filter
# ---------------------------------------------------------------------------


class TestDeliverVoiceWithFilter:
    async def test_deliver_voice_strips_tags(self) -> None:
        """Non-streaming delivery path applies filter before TTS."""
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend, tts_filter=StripInternalTags())
        session = await backend.connect("room-1", "user-1", "voice-1")

        # Wire up minimal framework for _deliver_voice
        from roomkit import AIChannel, MockAIProvider, RoomKit

        kit = RoomKit()
        kit.register_channel(channel)
        ai = AIChannel(
            "ai-1",
            provider=MockAIProvider(
                responses=["[internal]Respond warmly[/internal] Hello there!"]
            ),
        )
        kit.register_channel(ai)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")

        # Bind the session manually
        from roomkit.models.channel import ChannelBinding

        binding = ChannelBinding(
            channel_id="voice-1",
            room_id=room.id,
            channel_type=ChannelType.VOICE,
        )
        channel._session_bindings[session.id] = (room.id, binding)

        # Create event + context
        from roomkit.models.enums import EventType
        from roomkit.models.event import EventSource, RoomEvent, TextContent

        event = RoomEvent(
            id="evt-1",
            room_id=room.id,
            type=EventType.MESSAGE,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body="[internal]Respond warmly[/internal] Hello there!"),
        )
        context = await kit._build_context(room.id)

        await channel._deliver_voice(event, binding, context)

        # TTS should receive cleaned text
        assert len(tts.calls) == 1
        assert tts.calls[0]["text"] == "Hello there!"

    async def test_deliver_voice_empty_after_filter_skips(self) -> None:
        """If filter removes all text, TTS is skipped."""
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend, tts_filter=StripInternalTags())
        session = await backend.connect("room-1", "user-1", "voice-1")

        from roomkit import AIChannel, MockAIProvider, RoomKit

        kit = RoomKit()
        kit.register_channel(channel)
        ai = AIChannel(
            "ai-1",
            provider=MockAIProvider(responses=["[internal]all hidden[/internal]"]),
        )
        kit.register_channel(ai)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")

        from roomkit.models.channel import ChannelBinding

        binding = ChannelBinding(
            channel_id="voice-1",
            room_id=room.id,
            channel_type=ChannelType.VOICE,
        )
        channel._session_bindings[session.id] = (room.id, binding)

        from roomkit.models.enums import EventType
        from roomkit.models.event import EventSource, RoomEvent, TextContent

        event = RoomEvent(
            id="evt-1",
            room_id=room.id,
            type=EventType.MESSAGE,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body="[internal]all hidden[/internal]"),
        )
        context = await kit._build_context(room.id)

        await channel._deliver_voice(event, binding, context)

        assert len(tts.calls) == 0


# ---------------------------------------------------------------------------
# deliver_stream + tts_filter
# ---------------------------------------------------------------------------


class TestDeliverStreamWithFilter:
    async def test_streaming_strips_internal_tags(self) -> None:
        """Streaming delivery wraps token stream through the filter."""
        from roomkit import RoomKit
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import EventType
        from roomkit.models.event import EventSource, RoomEvent, TextContent

        tts = _StreamingMockTTS()
        backend = MockVoiceBackend()
        filt = StripInternalTags()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend, tts_filter=filt)

        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await backend.connect(room.id, "user-1", "voice-1")

        binding = ChannelBinding(
            channel_id="voice-1",
            room_id=room.id,
            channel_type=ChannelType.VOICE,
        )
        channel._session_bindings[session.id] = (room.id, binding)

        event = RoomEvent(
            id="evt-1",
            room_id=room.id,
            type=EventType.MESSAGE,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body="streaming"),
        )
        context = await kit._build_context(room.id)

        async def token_stream() -> AsyncIterator[str]:
            yield "[internal]"
            yield "think"
            yield "[/internal]"
            yield " Hello"
            yield " world!"

        await channel.deliver_stream(token_stream(), event, binding, context)

        # TTS should have received clean text (no "[internal]think[/internal]")
        assert len(tts.calls) >= 1
        synthesized_text = " ".join(c["text"] for c in tts.calls)
        assert "think" not in synthesized_text
        assert "internal" not in synthesized_text.lower()

    async def test_streaming_transcription_is_filtered(self) -> None:
        """The final transcription sent to backend is also filtered."""
        from roomkit import RoomKit
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import EventType
        from roomkit.models.event import EventSource, RoomEvent, TextContent

        tts = _StreamingMockTTS()
        backend = MockVoiceBackend()
        filt = StripInternalTags()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend, tts_filter=filt)

        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await backend.connect(room.id, "user-1", "voice-1")

        binding = ChannelBinding(
            channel_id="voice-1",
            room_id=room.id,
            channel_type=ChannelType.VOICE,
        )
        channel._session_bindings[session.id] = (room.id, binding)

        event = RoomEvent(
            id="evt-1",
            room_id=room.id,
            type=EventType.MESSAGE,
            source=EventSource(channel_id="ai-1", channel_type=ChannelType.AI),
            content=TextContent(body="streaming"),
        )
        context = await kit._build_context(room.id)

        async def token_stream() -> AsyncIterator[str]:
            yield "[internal]secret[/internal]"
            yield " Visible text."

        await channel.deliver_stream(token_stream(), event, binding, context)

        # Check the "assistant" transcription (the final full text)
        assistant_transcriptions = [
            (sid, text, role)
            for sid, text, role in backend.sent_transcriptions
            if role == "assistant"
        ]
        for _, text, _ in assistant_transcriptions:
            assert "secret" not in text

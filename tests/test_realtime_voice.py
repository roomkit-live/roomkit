"""Unit tests for RealtimeVoiceChannel using mocks."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomKit,
    TextContent,
)
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent
from roomkit.voice.realtime.base import RealtimeSessionState
from roomkit.voice.realtime.events import RealtimeTranscriptionEvent
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


@pytest.fixture
def provider() -> MockRealtimeProvider:
    return MockRealtimeProvider()


@pytest.fixture
def transport() -> MockRealtimeTransport:
    return MockRealtimeTransport()


@pytest.fixture
def channel(
    provider: MockRealtimeProvider, transport: MockRealtimeTransport
) -> RealtimeVoiceChannel:
    return RealtimeVoiceChannel(
        "rt-voice-1",
        provider=provider,
        transport=transport,
        system_prompt="You are a test agent.",
        voice="alloy",
    )


@pytest.fixture
async def kit(channel: RealtimeVoiceChannel) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(channel)
    return kit


@pytest.fixture
async def room_id(kit: RoomKit) -> str:
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt-voice-1")
    return room.id


class TestSessionLifecycle:
    async def test_start_session(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws-connection")

        assert session.state == RealtimeSessionState.ACTIVE
        assert session.room_id == room_id
        assert session.participant_id == "user-1"
        assert session.channel_id == "rt-voice-1"
        assert session.provider_session_id is not None

        # Verify provider was connected
        connect_calls = [c for c in provider.calls if c.method == "connect"]
        assert len(connect_calls) == 1
        assert connect_calls[0].args["system_prompt"] == "You are a test agent."
        assert connect_calls[0].args["voice"] == "alloy"

        # Verify transport accepted the connection
        accept_calls = [c for c in transport.calls if c.method == "accept"]
        assert len(accept_calls) == 1

    async def test_end_session(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await channel.end_session(session)

        assert session.state == RealtimeSessionState.ENDED

        # Verify provider and transport were disconnected
        disconnect_provider = [c for c in provider.calls if c.method == "disconnect"]
        disconnect_transport = [c for c in transport.calls if c.method == "disconnect"]
        assert len(disconnect_provider) == 1
        assert len(disconnect_transport) == 1


class TestAudioForwarding:
    async def test_client_audio_to_provider(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Simulate client sending audio
        await transport.simulate_client_audio(session, b"client-audio-data")
        await asyncio.sleep(0.05)

        # Verify provider received the audio
        assert len(provider.sent_audio) == 1
        assert provider.sent_audio[0] == (session.id, b"client-audio-data")

    async def test_provider_audio_to_client(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Simulate provider producing audio
        await provider.simulate_audio(session, b"provider-audio-data")
        await asyncio.sleep(0.05)

        # Verify transport sent audio to client
        assert len(transport.sent_audio) == 1
        assert transport.sent_audio[0] == (session.id, b"provider-audio-data")


class TestTranscriptions:
    async def test_transcription_emitted_as_room_event(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Simulate final transcription from provider
        await provider.simulate_transcription(session, "Hello world", "user", True)
        await asyncio.sleep(0.1)

        # Verify a RoomEvent was emitted
        events = await kit.get_timeline(room_id)
        text_events = [
            e
            for e in events
            if isinstance(e.content, TextContent) and e.content.body == "Hello world"
        ]
        assert len(text_events) == 1
        assert text_events[0].metadata.get("role") == "user"
        assert text_events[0].metadata.get("source") == "realtime_voice"

    async def test_non_final_transcription_not_emitted(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Simulate non-final transcription
        await provider.simulate_transcription(session, "Hel", "user", False)
        await asyncio.sleep(0.1)

        # No RoomEvent should be stored (only sent to client UI)
        events = await kit.get_timeline(room_id)
        text_events = [
            e for e in events if isinstance(e.content, TextContent) and e.content.body == "Hel"
        ]
        assert len(text_events) == 0

        # But the client should have received the transcription UI message
        transcription_msgs = [
            m for _, m in transport.sent_messages if m.get("type") == "transcription"
        ]
        assert len(transcription_msgs) == 1


class TestTextInjection:
    async def test_text_injection_from_other_channel(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        _session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Simulate event from another channel
        event = RoomEvent(
            room_id=room_id,
            source=EventSource(
                channel_id="supervisor-ws",
                channel_type=ChannelType.WEBSOCKET,
            ),
            content=TextContent(body="Offer 20% discount"),
        )
        from roomkit.models.channel import ChannelBinding

        binding = ChannelBinding(
            channel_id="rt-voice-1",
            room_id=room_id,
            channel_type=ChannelType.REALTIME_VOICE,
        )
        context = await kit._build_context(room_id)

        _output = await channel.on_event(event, binding, context)

        # Verify text was injected into provider
        assert len(provider.injected_texts) == 1
        assert provider.injected_texts[0][1] == "Offer 20% discount"
        assert provider.injected_texts[0][2] == "system"  # Default role

    async def test_text_injection_with_custom_role(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        _session = await channel.start_session(room_id, "user-1", "fake-ws")

        event = RoomEvent(
            room_id=room_id,
            source=EventSource(
                channel_id="other-ch",
                channel_type=ChannelType.WEBSOCKET,
            ),
            content=TextContent(body="I need help with returns"),
            metadata={"inject_role": "user"},
        )
        from roomkit.models.channel import ChannelBinding

        binding = ChannelBinding(
            channel_id="rt-voice-1",
            room_id=room_id,
            channel_type=ChannelType.REALTIME_VOICE,
        )
        context = await kit._build_context(room_id)

        await channel.on_event(event, binding, context)

        assert provider.injected_texts[0][2] == "user"


class TestToolCalls:
    async def test_tool_call_handled_via_hook(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Register a tool call hook
        @kit.hook(
            HookTrigger.ON_REALTIME_TOOL_CALL,
            execution=HookExecution.SYNC,
            name="handle_tool",
        )
        async def handle_tool(event: object, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        # Simulate tool call from provider
        await provider.simulate_tool_call(session, "call-123", "get_weather", {"city": "NYC"})
        await asyncio.sleep(0.1)

        # Verify tool result was submitted back to provider
        assert len(provider.tool_results) == 1
        assert provider.tool_results[0][1] == "call-123"


class TestSpeakingIndicators:
    async def test_response_start_sends_speaking_indicator(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await provider.simulate_response_start(session)
        await asyncio.sleep(0.05)

        speaking_msgs = [
            m
            for _, m in transport.sent_messages
            if m.get("type") == "speaking" and m.get("who") == "assistant"
        ]
        assert len(speaking_msgs) >= 1
        assert speaking_msgs[0]["speaking"] is True

    async def test_response_end_clears_speaking_indicator(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await provider.simulate_response_end(session)
        await asyncio.sleep(0.05)

        speaking_msgs = [
            m
            for _, m in transport.sent_messages
            if m.get("type") == "speaking" and m.get("who") == "assistant"
        ]
        assert len(speaking_msgs) >= 1
        assert speaking_msgs[-1]["speaking"] is False


class TestTranscriptionHooks:
    async def test_transcription_hook_can_block_selectively(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        """Hook allows some transcriptions and blocks others."""
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Register a hook that only allows transcriptions containing "allowed"
        @kit.hook(
            HookTrigger.ON_TRANSCRIPTION,
            execution=HookExecution.SYNC,
            name="selective_hook",
        )
        async def selective_hook(event: object, ctx: RoomContext) -> HookResult:
            if isinstance(event, RealtimeTranscriptionEvent):
                if "allowed" in event.text:
                    return HookResult.allow()
                return HookResult.block("Not allowed")
            return HookResult.allow()

        # This one should be blocked
        await provider.simulate_transcription(session, "blocked text", "user", True)
        await asyncio.sleep(0.1)

        # This one should pass
        await provider.simulate_transcription(session, "allowed text", "user", True)
        await asyncio.sleep(0.1)

        events = await kit.get_timeline(room_id)
        text_events = [e for e in events if isinstance(e.content, TextContent)]
        assert len(text_events) == 1
        assert text_events[0].content.body == "allowed text"

    async def test_transcription_hook_can_block(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Register a hook that blocks transcriptions
        @kit.hook(
            HookTrigger.ON_TRANSCRIPTION,
            execution=HookExecution.SYNC,
            name="block_transcription",
        )
        async def block_hook(event: object, ctx: RoomContext) -> HookResult:
            return HookResult.block("Blocked for testing")

        await provider.simulate_transcription(session, "Should be blocked", "user", True)
        await asyncio.sleep(0.1)

        events = await kit.get_timeline(room_id)
        text_events = [
            e
            for e in events
            if isinstance(e.content, TextContent) and e.content.body == "Should be blocked"
        ]
        assert len(text_events) == 0


class TestSelfLoopPrevention:
    async def test_on_event_skips_own_channel_events(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        await channel.start_session(room_id, "user-1", "fake-ws")

        # Simulate event from own channel
        event = RoomEvent(
            room_id=room_id,
            source=EventSource(
                channel_id="rt-voice-1",  # Same channel ID
                channel_type=ChannelType.REALTIME_VOICE,
            ),
            content=TextContent(body="Own transcription"),
        )
        from roomkit.models.channel import ChannelBinding

        binding = ChannelBinding(
            channel_id="rt-voice-1",
            room_id=room_id,
            channel_type=ChannelType.REALTIME_VOICE,
        )
        context = await kit._build_context(room_id)

        await channel.on_event(event, binding, context)

        # Verify no text was injected (self-loop prevented)
        assert len(provider.injected_texts) == 0


class TestPerRoomConfig:
    async def test_per_room_config_overrides(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        _session = await channel.start_session(
            room_id,
            "user-1",
            "fake-ws",
            metadata={
                "system_prompt": "You are a sales agent.",
                "voice": "echo",
                "temperature": 0.5,
            },
        )

        connect_calls = [c for c in provider.calls if c.method == "connect"]
        assert connect_calls[0].args["system_prompt"] == "You are a sales agent."
        assert connect_calls[0].args["voice"] == "echo"
        assert connect_calls[0].args["temperature"] == 0.5


class TestDeliverIsNoop:
    async def test_deliver_returns_empty_output(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        room_id: str,
    ) -> None:
        event = RoomEvent(
            room_id=room_id,
            source=EventSource(
                channel_id="other-ch",
                channel_type=ChannelType.WEBSOCKET,
            ),
            content=TextContent(body="Hello"),
        )
        from roomkit.models.channel import ChannelBinding

        binding = ChannelBinding(
            channel_id="rt-voice-1",
            room_id=room_id,
            channel_type=ChannelType.REALTIME_VOICE,
        )
        context = await kit._build_context(room_id)

        output = await channel.deliver(event, binding, context)

        assert output.responded is False
        assert output.response_events == []


class TestCloseCleanup:
    async def test_close_cleans_up(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await channel.close()

        # Verify provider and transport were closed
        close_provider = [c for c in provider.calls if c.method == "close"]
        close_transport = [c for c in transport.calls if c.method == "close"]
        assert len(close_provider) == 1
        assert len(close_transport) == 1

        # Session should be ended
        assert session.state == RealtimeSessionState.ENDED

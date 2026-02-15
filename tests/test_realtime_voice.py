"""Unit tests for RealtimeVoiceChannel using mocks."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
from roomkit.voice.realtime.base import RealtimeSession, RealtimeSessionState
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

    async def test_tool_result_truncated_when_exceeding_max_length(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Tool handler returning a huge string gets truncated before submission."""
        max_len = 500

        async def big_handler(session: object, name: str, arguments: dict[str, Any]) -> str:
            return "x" * 100_000

        ch = RealtimeVoiceChannel(
            "rt-trunc",
            provider=provider,
            transport=transport,
            tool_handler=big_handler,
            tool_result_max_length=max_len,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-trunc")

        session = await ch.start_session(room.id, "user-1", "fake-ws")

        await provider.simulate_tool_call(session, "call-big", "big_tool", {})
        await asyncio.sleep(0.1)

        assert len(provider.tool_results) == 1
        _session_id, _call_id, submitted = provider.tool_results[0]
        # Total should equal the max length (truncated content + notice)
        assert len(submitted) == max_len
        assert "truncated" in submitted
        assert "100000 chars" in submitted
        assert "delivered to the client" in submitted

    async def test_tool_result_under_limit_not_truncated(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Normal-sized tool results pass through unchanged."""
        small_result = '{"status": "ok", "data": "hello"}'

        async def small_handler(session: object, name: str, arguments: dict[str, Any]) -> str:
            return small_result

        ch = RealtimeVoiceChannel(
            "rt-small",
            provider=provider,
            transport=transport,
            tool_handler=small_handler,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-small")

        session = await ch.start_session(room.id, "user-1", "fake-ws")

        await provider.simulate_tool_call(session, "call-sm", "small_tool", {})
        await asyncio.sleep(0.1)

        assert len(provider.tool_results) == 1
        _session_id, _call_id, submitted = provider.tool_results[0]
        assert submitted == small_result


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


# ---------------------------------------------------------------------------
# Resampling (transport_sample_rate)
# ---------------------------------------------------------------------------


@pytest.fixture
def resample_channel(
    provider: MockRealtimeProvider, transport: MockRealtimeTransport
) -> RealtimeVoiceChannel:
    """Channel with transport_sample_rate set for resampling tests."""
    return RealtimeVoiceChannel(
        "rt-resample",
        provider=provider,
        transport=transport,
        input_sample_rate=16000,
        output_sample_rate=24000,
        transport_sample_rate=8000,
    )


@pytest.fixture
async def resample_kit(resample_channel: RealtimeVoiceChannel) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(resample_channel)
    return kit


@pytest.fixture
async def resample_room_id(resample_kit: RoomKit) -> str:
    room = await resample_kit.create_room()
    await resample_kit.attach_channel(room.id, "rt-resample")
    return room.id


class TestTransportSampleRateNone:
    """transport_sample_rate=None (default) disables resampling."""

    async def test_no_resamplers_created(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")
        assert session.id not in channel._session_resamplers

    async def test_audio_passes_through(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Client → Provider: no resampling
        audio = b"\x01\x00" * 80
        await transport.simulate_client_audio(session, audio)
        await asyncio.sleep(0.05)
        assert len(provider.sent_audio) == 1
        assert provider.sent_audio[0] == (session.id, audio)

        # Provider → Client: no resampling
        await provider.simulate_audio(session, audio)
        await asyncio.sleep(0.05)
        assert len(transport.sent_audio) == 1
        assert transport.sent_audio[0] == (session.id, audio)


class TestResamplingEnabled:
    """transport_sample_rate != provider rates enables resampling."""

    async def test_resamplers_created(
        self,
        resample_kit: RoomKit,
        resample_channel: RealtimeVoiceChannel,
        resample_room_id: str,
    ) -> None:
        session = await resample_channel.start_session(resample_room_id, "user-1", "fake-ws")
        assert session.id in resample_channel._session_resamplers
        inbound, outbound = resample_channel._session_resamplers[session.id]
        assert inbound is not None
        assert outbound is not None

    async def test_inbound_audio_resampled(
        self,
        resample_kit: RoomKit,
        resample_channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        resample_room_id: str,
    ) -> None:
        session = await resample_channel.start_session(resample_room_id, "user-1", "fake-ws")

        # Send 10ms of 8kHz audio (80 samples) as client audio
        import struct

        audio_8k = struct.pack("<80h", *([500] * 80))
        await transport.simulate_client_audio(session, audio_8k)
        await asyncio.sleep(0.05)

        # Provider should receive resampled audio (different size)
        assert len(provider.sent_audio) == 1
        received = provider.sent_audio[0][1]
        # 8kHz → 16kHz: should be roughly 160 samples (320 bytes)
        # SincResampler has a one-frame delay on first frame, so first call
        # produces output for 0 or ~80 samples depending on implementation.
        # Just verify it's different from input (was resampled).
        assert received != audio_8k

    async def test_outbound_audio_resampled(
        self,
        resample_kit: RoomKit,
        resample_channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        resample_room_id: str,
    ) -> None:
        session = await resample_channel.start_session(resample_room_id, "user-1", "fake-ws")

        # Send two 10ms chunks of 24kHz audio (240 samples each).
        # The sinc resampler has a one-frame delay: first chunk is buffered,
        # second chunk triggers output for the first.
        import struct

        audio_24k = struct.pack("<240h", *([500] * 240))
        await provider.simulate_audio(session, audio_24k)
        await provider.simulate_audio(session, audio_24k)
        # Allow time for the queue-based sender's pre-buffer timeout
        await asyncio.sleep(0.2)

        # Transport should receive resampled audio (different size)
        assert len(transport.sent_audio) >= 1
        received = transport.sent_audio[0][1]
        # 24kHz → 8kHz: resampled output should differ from input
        assert received != audio_24k

    async def test_session_cleanup_removes_resamplers(
        self,
        resample_kit: RoomKit,
        resample_channel: RealtimeVoiceChannel,
        resample_room_id: str,
    ) -> None:
        session = await resample_channel.start_session(resample_room_id, "user-1", "fake-ws")
        assert session.id in resample_channel._session_resamplers

        await resample_channel.end_session(session)
        assert session.id not in resample_channel._session_resamplers


class TestInterruptionFlush:
    """Speech start (interrupt) discards stale audio and resets resampler."""

    async def test_interrupt_discards_pending_audio(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        """Audio pushed before speech_start should NOT arrive at transport."""
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Push audio from provider (creates tasks in event loop)
        await provider.simulate_audio(session, b"\x01\x00" * 80)
        await provider.simulate_audio(session, b"\x01\x00" * 80)

        # Speech start fires BEFORE the tasks above run — should discard them
        await provider.simulate_speech_start(session)

        # Let pending tasks run
        await asyncio.sleep(0.05)

        # No audio should have been sent (tasks were stale)
        assert len(transport.sent_audio) == 0

    async def test_interrupt_resets_outbound_resampler(
        self,
        resample_kit: RoomKit,
        resample_channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        resample_room_id: str,
    ) -> None:
        """Interrupt resets the outbound resampler so stale buffered audio
        doesn't leak into the next response."""
        import struct

        session = await resample_channel.start_session(resample_room_id, "user-1", "fake-ws")

        # Push one chunk (sinc resampler buffers first frame)
        audio_24k = struct.pack("<240h", *([500] * 240))
        await provider.simulate_audio(session, audio_24k)
        await asyncio.sleep(0.05)

        # The first frame is still in the resampler buffer — interrupt should discard it
        await provider.simulate_speech_start(session)
        await asyncio.sleep(0.05)

        # Verify the resampler state was cleared
        resamplers = resample_channel._session_resamplers.get(session.id)
        assert resamplers is not None
        # Outbound resampler should have no pending state
        assert len(resamplers[1]._state) == 0


class TestResamplingMatchingRates:
    """No resamplers when transport_sample_rate matches provider rates."""

    async def test_no_resamplers_when_rates_match(self) -> None:
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        ch = RealtimeVoiceChannel(
            "rt-match",
            provider=provider,
            transport=transport,
            input_sample_rate=16000,
            output_sample_rate=16000,
            transport_sample_rate=16000,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-match")

        session = await ch.start_session(room.id, "user-1", "fake-ws")
        assert session.id not in ch._session_resamplers


# ---------------------------------------------------------------------------
# GeminiLiveProvider: audio buffering during reconnection
# ---------------------------------------------------------------------------

genai = pytest.importorskip("google.genai", reason="google-genai not installed")


def _make_gemini_provider() -> Any:
    """Create a GeminiLiveProvider with mocked client."""
    from roomkit.providers.gemini.realtime import GeminiLiveProvider

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "roomkit.providers.gemini.realtime.GeminiLiveProvider.__init__",
            lambda self, **kw: None,
        )
        p = GeminiLiveProvider.__new__(GeminiLiveProvider)

    # Initialize fields normally set in __init__
    p._live_sessions = {}
    p._live_ctxmgrs = {}
    p._live_configs = {}
    p._receive_tasks = {}
    p._sessions = {}
    p._transcription_buffers = {}
    p._audio_chunk_count = {}
    p._audio_buffers = {}
    p._send_audio_count = {}
    p._error_suppressed = set()
    p._session_telemetry = {}
    p._resumption_handles = {}
    p._audio_callbacks = []
    p._transcription_callbacks = []
    p._speech_start_callbacks = []
    p._speech_end_callbacks = []
    p._tool_call_callbacks = []
    p._response_start_callbacks = []
    p._response_end_callbacks = []
    p._error_callbacks = []
    return p


def _make_session(session_id: str = "sess-1") -> RealtimeSession:
    return RealtimeSession(
        id=session_id,
        room_id="room-1",
        participant_id="user-1",
        channel_id="rt-1",
        state=RealtimeSessionState.ACTIVE,
    )


class TestAudioBufferingDuringReconnect:
    """Audio chunks sent while state==CONNECTING are buffered, not dropped."""

    async def test_audio_buffered_when_connecting(self) -> None:
        provider = _make_gemini_provider()
        session = _make_session()
        session.state = RealtimeSessionState.CONNECTING

        # Set up a mock live session and buffer
        mock_live = MagicMock()
        provider._live_sessions[session.id] = mock_live
        provider._audio_buffers[session.id] = deque(maxlen=100)

        await provider.send_audio(session, b"chunk-1")
        await provider.send_audio(session, b"chunk-2")
        await provider.send_audio(session, b"chunk-3")

        # Audio should be in the buffer, not sent to the live session
        mock_live.send_realtime_input.assert_not_called()
        assert list(provider._audio_buffers[session.id]) == [
            b"chunk-1",
            b"chunk-2",
            b"chunk-3",
        ]

    async def test_buffer_bounded_at_100_frames(self) -> None:
        provider = _make_gemini_provider()
        session = _make_session()
        session.state = RealtimeSessionState.CONNECTING

        mock_live = MagicMock()
        provider._live_sessions[session.id] = mock_live
        provider._audio_buffers[session.id] = deque(maxlen=100)

        # Send 110 chunks — oldest 10 should be evicted
        for i in range(110):
            await provider.send_audio(session, f"chunk-{i}".encode())

        buf = provider._audio_buffers[session.id]
        assert len(buf) == 100
        assert buf[0] == b"chunk-10"  # oldest surviving
        assert buf[-1] == b"chunk-109"  # newest

    async def test_audio_sent_normally_when_active(self) -> None:
        provider = _make_gemini_provider()
        session = _make_session()
        session.state = RealtimeSessionState.ACTIVE

        mock_live = AsyncMock()
        provider._live_sessions[session.id] = mock_live
        provider._audio_buffers[session.id] = deque(maxlen=100)

        await provider.send_audio(session, b"chunk-1")

        mock_live.send_realtime_input.assert_called_once()
        assert len(provider._audio_buffers[session.id]) == 0


class TestErrorDeduplication:
    """Only one send_audio_failed error fires per reconnection cycle."""

    async def test_first_error_fires_callback(self) -> None:
        provider = _make_gemini_provider()
        session = _make_session()
        session.state = RealtimeSessionState.ACTIVE

        mock_live = AsyncMock()
        mock_live.send_realtime_input.side_effect = ConnectionError("ws closed")
        provider._live_sessions[session.id] = mock_live
        provider._audio_buffers[session.id] = deque(maxlen=100)

        errors: list[tuple[str, str]] = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))

        await provider.send_audio(session, b"chunk-1")

        assert len(errors) == 1
        assert errors[0][0] == "send_audio_failed"

    async def test_subsequent_errors_suppressed(self) -> None:
        provider = _make_gemini_provider()
        session = _make_session()
        session.state = RealtimeSessionState.ACTIVE

        mock_live = AsyncMock()
        mock_live.send_realtime_input.side_effect = ConnectionError("ws closed")
        provider._live_sessions[session.id] = mock_live
        provider._audio_buffers[session.id] = deque(maxlen=100)

        errors: list[tuple[str, str]] = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))

        # First call fires the error and sets state to CONNECTING
        await provider.send_audio(session, b"chunk-1")
        assert len(errors) == 1

        # Subsequent calls while CONNECTING go to buffer, no more errors
        await provider.send_audio(session, b"chunk-2")
        await provider.send_audio(session, b"chunk-3")
        assert len(errors) == 1  # still just 1

    async def test_error_suppression_cleared_after_reconnect_cycle(self) -> None:
        provider = _make_gemini_provider()
        session = _make_session()

        # Simulate a completed reconnect cycle
        provider._error_suppressed.add(session.id)
        provider._error_suppressed.discard(session.id)

        # Now a new error should fire
        session.state = RealtimeSessionState.ACTIVE
        mock_live = AsyncMock()
        mock_live.send_realtime_input.side_effect = ConnectionError("ws closed again")
        provider._live_sessions[session.id] = mock_live
        provider._audio_buffers[session.id] = deque(maxlen=100)

        errors: list[tuple[str, str]] = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))

        await provider.send_audio(session, b"chunk-1")
        assert len(errors) == 1

    async def test_disconnect_cleans_up_suppression(self) -> None:
        provider = _make_gemini_provider()
        session = _make_session()

        provider._sessions[session.id] = session
        provider._audio_buffers[session.id] = deque(maxlen=100)
        provider._error_suppressed.add(session.id)
        provider._session_telemetry[session.id] = {
            "started_at": 0,
            "turn_count": 0,
            "tool_result_bytes": 0,
        }

        await provider.disconnect(session)

        assert session.id not in provider._error_suppressed
        assert session.id not in provider._audio_buffers
        assert session.id not in provider._session_telemetry

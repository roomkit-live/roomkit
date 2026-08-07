"""Unit tests for RealtimeVoiceChannel using mocks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.interruption import InterruptionConfig
from roomkit.voice.pipeline.config import AudioPipelineConfig
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
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_sample_rate": 0},
            {"output_sample_rate": -1},
            {"transport_sample_rate": True},
            {"tool_result_max_length": 0},
            {"tool_search_threshold": 0},
        ],
    )
    def test_invalid_numeric_configuration_is_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            RealtimeVoiceChannel(
                "rt-invalid",
                provider=MockRealtimeProvider(),
                transport=MockRealtimeTransport(),
                **kwargs,
            )

    async def test_start_session(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws-connection")

        assert session.state == VoiceSessionState.ACTIVE
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
        channel._playback_started_at[session.id] = 1.0
        channel._playback_position_ms[session.id] = 250.0

        await channel.end_session(session)

        assert session.state == VoiceSessionState.ENDED
        assert session.id not in channel._playback_started_at
        assert session.id not in channel._playback_position_ms

        # Verify provider and transport were disconnected
        disconnect_provider = [c for c in provider.calls if c.method == "disconnect"]
        disconnect_transport = [c for c in transport.calls if c.method == "disconnect"]
        assert len(disconnect_provider) == 1
        assert len(disconnect_transport) == 1

    async def test_end_session_cleans_up_when_client_notification_fails(self) -> None:
        class BrokenNotificationTransport(MockRealtimeTransport):
            async def send_message(self, session: VoiceSession, message: dict[str, Any]) -> None:
                if message.get("type") == "session_ended":
                    raise ConnectionError("client is gone")
                await super().send_message(session, message)

        provider = MockRealtimeProvider()
        transport = BrokenNotificationTransport()
        channel = RealtimeVoiceChannel("rt-dead-client", provider=provider, transport=transport)
        session = await channel.start_session("room-1", "user-1", "fake-ws")

        await channel.end_session(session)

        assert session.state == VoiceSessionState.ENDED
        assert channel._sessions == {}
        assert channel._session_spans == {}
        assert [call.method for call in provider.calls].count("disconnect") == 1
        assert [call.method for call in transport.calls].count("disconnect") == 1

    async def test_start_notification_failure_rolls_back_live_session(self) -> None:
        class BrokenStartNotificationTransport(MockRealtimeTransport):
            async def send_message(self, session: VoiceSession, message: dict[str, Any]) -> None:
                if message.get("type") == "session_started":
                    raise ConnectionError("cannot notify client")
                await super().send_message(session, message)

        provider = MockRealtimeProvider()
        transport = BrokenStartNotificationTransport()
        channel = RealtimeVoiceChannel(
            "rt-start-notify-fail", provider=provider, transport=transport
        )

        with pytest.raises(ConnectionError, match="cannot notify client"):
            await channel.start_session("room-1", "user-1", "fake-ws")

        assert channel._sessions == {}
        assert channel._session_spans == {}
        assert [call.method for call in provider.calls].count("disconnect") == 1
        assert [call.method for call in transport.calls].count("disconnect") == 1


class TestAudioForwarding:
    async def test_client_audio_to_provider(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
        advance,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Simulate client sending audio
        await transport.simulate_client_audio(session, b"client-audio-data")
        await advance()

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
        advance,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Simulate provider producing audio
        await provider.simulate_audio(session, b"provider-audio-data")
        await advance()

        # Verify transport sent audio to client
        assert len(transport.sent_audio) == 1
        assert transport.sent_audio[0] == (session.id, b"provider-audio-data")


class TestRealtimeBargeInGuard:
    """Provider VAD is shielded while AEC converges at playback onset."""

    @staticmethod
    def _make_guarded_channel(
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        *,
        guard_ms: int = 600,
    ) -> RealtimeVoiceChannel:
        return RealtimeVoiceChannel(
            "rt-guard",
            provider=provider,
            transport=transport,
            pipeline=AudioPipelineConfig(
                interruption=InterruptionConfig(allow_during_first_ms=guard_ms)
            ),
        )

    async def test_sends_timeline_preserving_silence_during_guard(self) -> None:
        provider = MockRealtimeProvider()
        channel = self._make_guarded_channel(provider, MockRealtimeTransport())
        session = _make_session()
        mic_audio = b"\x34\x12" * 240
        channel._framework = MagicMock()

        channel._on_transport_audio_played(
            session,
            AudioFrame(data=b"\x01\x00" * 240, sample_rate=24000),
        )
        await channel._forward_pipeline_frame(
            session,
            mic_audio,
            None,
            None,
            ("mic-track", "room-1"),
        )

        assert provider.sent_audio == [(session.id, b"\x00" * len(mic_audio))]
        recording_call = channel._framework._room_recorder_mgr.on_data.call_args
        assert recording_call.args[0:3] == ("room-1", "mic-track", mic_audio)

    async def test_forwards_real_audio_after_guard_expires(self) -> None:
        provider = MockRealtimeProvider()
        channel = self._make_guarded_channel(provider, MockRealtimeTransport())
        session = _make_session()
        mic_audio = b"\x34\x12" * 240

        channel._on_transport_audio_played(
            session,
            AudioFrame(data=b"\x01\x00" * 240, sample_rate=24000),
        )
        channel._playback_started_at[session.id] -= 1.0
        await channel._forward_pipeline_frame(session, mic_audio, None, None, None)

        assert provider.sent_audio == [(session.id, mic_audio)]

    async def test_silence_does_not_start_guard_and_playback_end_clears_it(self) -> None:
        provider = MockRealtimeProvider()
        channel = self._make_guarded_channel(provider, MockRealtimeTransport())
        session = _make_session()

        channel._on_transport_audio_played(
            session,
            AudioFrame(data=b"\x00" * 480, sample_rate=24000),
        )
        assert session.id not in channel._playback_started_at

        channel._on_transport_audio_played(
            session,
            AudioFrame(data=b"\x01\x00" * 240, sample_rate=24000),
        )
        assert session.id in channel._playback_started_at
        assert channel._playback_position_ms[session.id] == 10.0

        channel._on_transport_audio_played(
            session,
            AudioFrame(
                data=b"\x00" * 480,
                sample_rate=24000,
                metadata={"playback_ended": True},
            ),
        )
        assert session.id not in channel._playback_started_at
        assert session.id not in channel._playback_position_ms

    async def test_zero_guard_never_replaces_input(self) -> None:
        provider = MockRealtimeProvider()
        channel = self._make_guarded_channel(
            provider,
            MockRealtimeTransport(),
            guard_ms=0,
        )
        session = _make_session()
        mic_audio = b"\x34\x12" * 240

        channel._on_transport_audio_played(
            session,
            AudioFrame(data=b"\x01\x00" * 240, sample_rate=24000),
        )
        await channel._forward_pipeline_frame(session, mic_audio, None, None, None)

        assert provider.sent_audio == [(session.id, mic_audio)]


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

    async def test_final_transcription_after_detach_dropped_quietly(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
        caplog,
    ) -> None:
        """A final transcript that finalizes after the channel is detached must
        be dropped without an ERROR traceback.

        Reproduces the teardown race where the assistant's closing utterance is
        the one that ends the session: its transcript is finalized by the
        provider after ``end_session`` has already detached the channel, so the
        trailing RoomEvent has nowhere to land. That is an expected condition,
        not a failure.
        """
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Simulate teardown detaching the channel binding out from under the
        # in-flight transcription (a host's end_session calls detach_channel).
        await kit.detach_channel(room_id, "rt-voice-1")

        with caplog.at_level(logging.DEBUG, logger="roomkit.channels.realtime_voice"):
            await provider.simulate_transcription(
                session, "D'accord, je termine la session.", "assistant", True
            )
            await asyncio.sleep(0.1)

        # No RoomEvent could be emitted — the channel is gone.
        events = await kit.get_timeline(room_id)
        assert not [e for e in events if isinstance(e.content, TextContent)]

        # The race is logged at DEBUG, never ERROR (no traceback noise).
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("Channel detached during teardown" in r.getMessage() for r in caplog.records)


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
            HookTrigger.ON_TOOL_CALL,
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

        async def big_handler(name: str, arguments: dict[str, Any]) -> str:
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

        async def small_handler(name: str, arguments: dict[str, Any]) -> str:
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

    async def test_dict_tool_result_serialized(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """A non-str handler result is JSON-serialized before submission."""

        async def dict_handler(name: str, arguments: dict[str, Any]) -> Any:
            return {"status": "ok", "value": 42}

        ch = RealtimeVoiceChannel(
            "rt-dict",
            provider=provider,
            transport=transport,
            tool_handler=dict_handler,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-dict")
        session = await ch.start_session(room.id, "user-1", "fake-ws")

        await provider.simulate_tool_call(session, "call-d", "dict_tool", {})
        await asyncio.sleep(0.1)

        assert len(provider.tool_results) == 1
        assert provider.tool_results[0][2] == '{"status": "ok", "value": 42}'

    async def test_slow_result_serialization_warns(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Serialization past the loop-segment budget logs an actionable warning."""
        monkeypatch.setattr("roomkit.channels._realtime_tools._LOOP_SEGMENT_BUDGET_S", -1.0)

        async def dict_handler(name: str, arguments: dict[str, Any]) -> Any:
            return {"big": "payload"}

        ch = RealtimeVoiceChannel(
            "rt-slowser",
            provider=provider,
            transport=transport,
            tool_handler=dict_handler,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-slowser")
        session = await ch.start_session(room.id, "user-1", "fake-ws")

        with caplog.at_level(logging.WARNING, logger="roomkit.channels.realtime_voice"):
            await provider.simulate_tool_call(session, "call-w", "big_tool", {})
            await asyncio.sleep(0.1)

        assert any("held the event loop" in r.message for r in caplog.records)
        assert len(provider.tool_results) == 1  # result still submitted


class TestSpeakingIndicators:
    async def test_response_start_sends_speaking_indicator(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
        advance,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await provider.simulate_response_start(session)
        await advance()

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
        advance,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await provider.simulate_response_end(session)
        await advance()

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


class TestHookContextSkip:
    """Hot paths skip _build_context entirely when no hooks are registered.

    Partial transcriptions stream continuously while the AI speaks and
    speech events fire on every turn boundary — building a RoomContext
    (full recent-events load) for a no-op hook dispatch starves the
    event loop on long sessions.
    """

    async def test_partial_transcription_skips_context_without_hooks(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        with patch.object(kit, "_build_context", wraps=kit._build_context) as spy:
            await provider.simulate_transcription(session, "Hel", "user", False)
            await asyncio.sleep(0.1)

        spy.assert_not_called()

    async def test_partial_transcription_fires_hook_when_registered(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        seen: list[str] = []

        @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, HookExecution.ASYNC)
        async def on_partial(event: object, ctx: RoomContext) -> None:
            seen.append(event.text)  # type: ignore[attr-defined]

        session = await channel.start_session(room_id, "user-1", "fake-ws")
        await provider.simulate_transcription(session, "Hel", "user", False)
        await asyncio.sleep(0.1)

        assert seen == ["Hel"]

    async def test_partials_never_land_after_their_final(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        """A final must not overtake the partial that preceded it on the wire.

        Gemini delivers a short utterance as one transcript chunk plus its
        final in the same server message.  Each event runs in its own task
        and the partial path awaits one hop more than the final path, so
        unserialised processing let the final's hook fire first — the late
        partial then resurrected the utterance downstream (duplicate chat
        bubbles carrying identical text).
        """
        order: list[tuple[str, bool]] = []

        @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, HookExecution.ASYNC)
        async def on_partial(event: object, ctx: RoomContext) -> None:
            order.append((event.text, False))  # type: ignore[attr-defined]

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.SYNC)
        async def on_final(event: object, ctx: RoomContext) -> None:
            order.append((event.text, True))  # type: ignore[attr-defined]

        session = await channel.start_session(room_id, "user-1", "fake-ws")
        await provider.simulate_transcription(session, "Salut !", "user", False)
        await provider.simulate_transcription(session, "Salut !", "user", True)
        await asyncio.sleep(0.2)

        assert order == [("Salut !", False), ("Salut !", True)]

    async def test_speech_events_skip_context_without_hooks(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        with patch.object(kit, "_build_context", wraps=kit._build_context) as spy:
            await provider.simulate_speech_start(session)
            await provider.simulate_speech_end(session)
            await asyncio.sleep(0.1)

        spy.assert_not_called()

    async def test_speech_hooks_fire_when_registered(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
    ) -> None:
        order: list[str] = []

        @kit.hook(HookTrigger.ON_SPEECH_START, HookExecution.ASYNC)
        async def on_start(event: object, ctx: RoomContext) -> None:
            order.append("start")

        @kit.hook(HookTrigger.ON_SPEECH_END, HookExecution.ASYNC)
        async def on_end(event: object, ctx: RoomContext) -> None:
            order.append("end")

        session = await channel.start_session(room_id, "user-1", "fake-ws")
        await provider.simulate_speech_start(session)
        await asyncio.sleep(0.05)
        await provider.simulate_speech_end(session)
        await asyncio.sleep(0.1)

        assert order == ["start", "end"]


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
        assert session.state == VoiceSessionState.ENDED


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
        advance,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Client → Provider: no resampling
        audio = b"\x01\x00" * 80
        await transport.simulate_client_audio(session, audio)
        await advance()
        assert len(provider.sent_audio) == 1
        assert provider.sent_audio[0] == (session.id, audio)

        # Provider → Client: no resampling
        await provider.simulate_audio(session, audio)
        await advance()
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

        # Send two 10ms chunks of 8kHz audio (80 samples each).
        # The sinc resampler has a one-frame delay: the first chunk is buffered,
        # the second triggers output for the first.
        import struct

        audio_8k = struct.pack("<80h", *([500] * 80))
        await transport.simulate_client_audio(session, audio_8k)
        await transport.simulate_client_audio(session, audio_8k)

        # Resampling runs in a thread pool that other tests share, so zero-delay
        # event-loop yields are a racy barrier under load. Poll with real time so
        # the executor thread gets wall-clock to deliver the frame (bounded).
        for _ in range(200):
            if provider.sent_audio:
                break
            await asyncio.sleep(0.01)

        # Provider should receive resampled audio (different size)
        assert len(provider.sent_audio) >= 1
        received = provider.sent_audio[0][1]
        # 8kHz → 16kHz: resampled output should differ from input
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

    async def test_physical_playback_barge_in_truncates_after_response_done(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
        advance,
    ) -> None:
        """Generation may be done while buffered assistant audio is still audible."""
        session = await channel.start_session(room_id, "user-1", "fake-ws")
        channel._on_transport_audio_played(
            session,
            AudioFrame(data=b"\x01\x00" * 240, sample_rate=24000),
        )
        channel._playback_started_at[session.id] -= 0.75
        channel._playback_position_ms[session.id] = 750.0
        # response.done bookkeeping removes this count before physical drain.
        channel._audio_forward_count.pop(session.id, None)

        await provider.simulate_speech_start(session)
        await advance()

        truncates = [call for call in provider.calls if call.method == "truncate_audio"]
        assert len(truncates) == 1
        assert 700 <= truncates[0].args["audio_end_ms"] <= 850

    async def test_local_vad_barge_in_cancels_active_provider_and_truncates(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
        advance,
    ) -> None:
        """Local VAD must replace both parts of provider-managed interruption."""
        session = await channel.start_session(room_id, "user-1", "fake-ws")
        channel._provider_idle[session.id] = False
        channel._playback_started_at[session.id] = 1.0
        channel._playback_position_ms[session.id] = 320.0

        channel._on_pipeline_speech_start(session)
        await advance()

        calls = [call for call in provider.calls if call.method in {"interrupt", "truncate_audio"}]
        assert [call.method for call in calls] == ["interrupt", "truncate_audio"]
        assert calls[1].args["audio_end_ms"] == 320

    async def test_interrupt_discards_pending_audio(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
        advance,
    ) -> None:
        """Audio pushed before speech_start should NOT arrive at transport."""
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Push audio from provider (creates tasks in event loop)
        await provider.simulate_audio(session, b"\x01\x00" * 80)
        await provider.simulate_audio(session, b"\x01\x00" * 80)

        # Speech start fires BEFORE the tasks above run — should discard them
        await provider.simulate_speech_start(session)

        # Let pending tasks run
        await advance()

        # No audio should have been sent (tasks were stale)
        assert len(transport.sent_audio) == 0

    async def test_interrupt_resets_outbound_resampler(
        self,
        resample_kit: RoomKit,
        resample_channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        resample_room_id: str,
        advance,
    ) -> None:
        """Interrupt resets the outbound resampler so stale buffered audio
        doesn't leak into the next response."""
        import struct

        session = await resample_channel.start_session(resample_room_id, "user-1", "fake-ws")

        # Push one chunk (sinc resampler buffers first frame)
        audio_24k = struct.pack("<240h", *([500] * 240))
        await provider.simulate_audio(session, audio_24k)
        await advance()

        # The first frame is still in the resampler buffer — interrupt should discard it
        await provider.simulate_speech_start(session)
        await advance()

        # Verify the resampler state was cleared
        resamplers = resample_channel._session_resamplers.get(session.id)
        assert resamplers is not None
        # Outbound resampler should have no pending state. The Sinc provider
        # keeps a look-ahead buffer in ``_state`` (must be empty after
        # reset); the Numpy provider is stateless so the invariant is
        # vacuously true — skip the attribute check in that case.
        state = getattr(resamplers[1], "_state", None)
        if state is not None:
            assert len(state) == 0


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
    from roomkit.providers.gemini.realtime import _TranscriptionBuffer

    p._sessions = {}
    p._transcription_buffer = _TranscriptionBuffer()
    p._audio_callbacks = []
    p._transcription_callbacks = []
    p._speech_start_callbacks = []
    p._speech_end_callbacks = []
    p._tool_call_callbacks = []
    p._response_start_callbacks = []
    p._response_end_callbacks = []
    p._error_callbacks = []
    p._blob_cls = None
    p._mime_cache = {}
    return p


def _make_session(session_id: str = "sess-1") -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id="room-1",
        participant_id="user-1",
        channel_id="rt-1",
        state=VoiceSessionState.ACTIVE,
    )


class TestAudioBufferingDuringReconnect:
    """Audio chunks sent while state==CONNECTING are buffered, not dropped."""

    async def test_audio_buffered_when_connecting(self) -> None:
        from roomkit.providers.gemini.realtime import _GeminiSessionState

        provider = _make_gemini_provider()
        session = _make_session()
        session.state = VoiceSessionState.CONNECTING

        # Set up a mock live session via consolidated state
        mock_live = MagicMock()
        state = _GeminiSessionState(session=session, live_session=mock_live)
        provider._sessions[session.id] = state

        await provider.send_audio(session, b"chunk-1")
        await provider.send_audio(session, b"chunk-2")
        await provider.send_audio(session, b"chunk-3")

        # Audio should be in the buffer, not sent to the live session
        mock_live.send_realtime_input.assert_not_called()
        assert list(state.audio_buffer) == [
            b"chunk-1",
            b"chunk-2",
            b"chunk-3",
        ]

    async def test_buffer_bounded_at_100_frames(self) -> None:
        from roomkit.providers.gemini.realtime import _GeminiSessionState

        provider = _make_gemini_provider()
        session = _make_session()
        session.state = VoiceSessionState.CONNECTING

        mock_live = MagicMock()
        state = _GeminiSessionState(session=session, live_session=mock_live)
        provider._sessions[session.id] = state

        # Send 110 chunks — oldest 10 should be evicted
        for i in range(110):
            await provider.send_audio(session, f"chunk-{i}".encode())

        assert len(state.audio_buffer) == 100
        assert state.audio_buffer[0] == b"chunk-10"  # oldest surviving
        assert state.audio_buffer[-1] == b"chunk-109"  # newest

    async def test_audio_sent_normally_when_active(self) -> None:
        from roomkit.providers.gemini.realtime import _GeminiSessionState

        provider = _make_gemini_provider()
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        mock_live = AsyncMock()
        state = _GeminiSessionState(session=session, live_session=mock_live)
        provider._sessions[session.id] = state

        await provider.send_audio(session, b"chunk-1")

        mock_live.send_realtime_input.assert_called_once()
        assert len(state.audio_buffer) == 0


class TestErrorDeduplication:
    """Only one send_audio_failed error fires per reconnection cycle."""

    async def test_first_error_fires_callback(self) -> None:
        from roomkit.providers.gemini.realtime import _GeminiSessionState

        provider = _make_gemini_provider()
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        mock_live = AsyncMock()
        mock_live.send_realtime_input.side_effect = ConnectionError("ws closed")
        state = _GeminiSessionState(session=session, live_session=mock_live)
        provider._sessions[session.id] = state

        errors: list[tuple[str, str]] = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))

        await provider.send_audio(session, b"chunk-1")

        assert len(errors) == 1
        assert errors[0][0] == "send_audio_failed"

    async def test_subsequent_errors_suppressed(self) -> None:
        from roomkit.providers.gemini.realtime import _GeminiSessionState

        provider = _make_gemini_provider()
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        mock_live = AsyncMock()
        mock_live.send_realtime_input.side_effect = ConnectionError("ws closed")
        state = _GeminiSessionState(session=session, live_session=mock_live)
        provider._sessions[session.id] = state

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
        from roomkit.providers.gemini.realtime import _GeminiSessionState

        provider = _make_gemini_provider()
        session = _make_session()

        # Simulate a completed reconnect cycle: suppression was set then cleared
        state = _GeminiSessionState(
            session=session,
            error_suppressed=False,
        )

        # Now a new error should fire
        session.state = VoiceSessionState.ACTIVE
        mock_live = AsyncMock()
        mock_live.send_realtime_input.side_effect = ConnectionError("ws closed again")
        state.live_session = mock_live
        provider._sessions[session.id] = state

        errors: list[tuple[str, str]] = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))

        await provider.send_audio(session, b"chunk-1")
        assert len(errors) == 1

    async def test_disconnect_cleans_up_suppression(self) -> None:
        from roomkit.providers.gemini.realtime import _GeminiSessionState

        provider = _make_gemini_provider()
        session = _make_session()

        state = _GeminiSessionState(
            session=session,
            error_suppressed=True,
            started_at=0.0,
        )
        provider._sessions[session.id] = state

        await provider.disconnect(session)

        assert session.id not in provider._sessions


class TestEndOfResponseOrdering:
    """Verify end_of_response reaches transport AFTER all audio chunks."""

    async def test_end_of_response_after_audio_tasks(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        """end_of_response must arrive at the transport after all audio.

        _on_provider_audio schedules send tasks via loop.create_task().
        _on_provider_response_end must schedule end_of_response as a task
        too, so asyncio's FIFO ordering ensures it runs after all pending
        audio tasks — not synchronously before them.
        """
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        # Track the order of operations on the transport
        call_log: list[str] = []
        orig_send_audio = transport.send_audio
        orig_end_of_response = transport.end_of_response

        async def tracking_send_audio(s: VoiceSession, audio: bytes | AsyncIterator[Any]) -> None:
            call_log.append("audio")
            await orig_send_audio(s, audio)

        def tracking_end_of_response(s: VoiceSession) -> None:
            call_log.append("end_of_response")
            orig_end_of_response(s)

        transport.send_audio = tracking_send_audio  # type: ignore[assignment]
        transport.end_of_response = tracking_end_of_response  # type: ignore[assignment]

        # Simulate: provider sends multiple audio chunks then response_end
        # (all in one synchronous callback burst — no event loop yields)
        await provider.simulate_response_start(session)
        for i in range(5):
            await provider.simulate_audio(session, f"chunk-{i}".encode())
        await provider.simulate_response_end(session)

        # Let all tasks complete
        await asyncio.sleep(0.1)

        # All audio calls must appear before end_of_response
        assert "end_of_response" in call_log
        eor_index = call_log.index("end_of_response")
        audio_indices = [i for i, v in enumerate(call_log) if v == "audio"]
        assert len(audio_indices) == 5
        assert all(ai < eor_index for ai in audio_indices), (
            f"Audio after end_of_response: {call_log}"
        )

    async def test_end_of_response_deactivates_aec_without_playback_callback(self) -> None:
        """Queued transports release AEC after their ordered response marker."""
        from roomkit.voice.pipeline.aec.mock import MockAECProvider
        from roomkit.voice.pipeline.config import AudioPipelineConfig

        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        aec = MockAECProvider()
        channel = RealtimeVoiceChannel(
            "rt-aec",
            provider=provider,
            transport=transport,
            pipeline=AudioPipelineConfig(aec=aec),
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, channel.channel_id)
        session = await channel.start_session(room.id, "user-1", "fake-ws")
        aec.reset_streams.clear()  # Ignore the normal session-start cleanup.

        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"\x01\x00" * 160)
        await provider.simulate_response_end(session)
        await asyncio.sleep(0.1)

        assert aec.active_changes == [(session.id, True), (session.id, False)]
        assert aec.reset_streams == []

        await kit.close()


class TestInjectImageGracefulHandling:
    """inject_image should not crash when the provider doesn't support it."""

    async def test_inject_image_unsupported_provider_no_exception(
        self,
        channel: RealtimeVoiceChannel,
    ) -> None:
        session = _make_session()

        # MockRealtimeProvider does not override inject_image → NotImplementedError
        # Channel should catch it gracefully, not propagate
        await channel.inject_image(session, b"\x89PNG\r\n", "image/png")


class TestStartSessionCancellation:
    """Verify start_session cleans up on CancelledError without false-positive logs."""

    async def test_cancel_during_provider_connect_runs_cleanup(
        self,
        transport: MockRealtimeTransport,
    ) -> None:
        """Cancelling provider.connect mid-flight must run cleanup + raise CancelledError."""
        connect_started = asyncio.Event()

        class HangingProvider(MockRealtimeProvider):
            async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
                connect_started.set()
                await asyncio.sleep(60)  # hang until cancelled

        provider = HangingProvider()
        channel = RealtimeVoiceChannel(
            "rt-cancel",
            provider=provider,
            transport=transport,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-cancel")

        task = asyncio.create_task(channel.start_session(room.id, "user-1", "fake-ws"))
        await connect_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Cleanup ran: transport.disconnect was invoked (best-effort, suppressed
        # if it would raise — but the call MUST have been attempted).
        disconnect_calls = [c for c in transport.calls if c.method == "disconnect"]
        assert len(disconnect_calls) == 1

    async def test_cancel_logs_info_not_error(
        self,
        transport: MockRealtimeTransport,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Intentional cancellation logs INFO, not ERROR with a stack trace."""
        connect_started = asyncio.Event()

        class HangingProvider(MockRealtimeProvider):
            async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
                connect_started.set()
                await asyncio.sleep(60)

        provider = HangingProvider()
        channel = RealtimeVoiceChannel(
            "rt-cancel-log",
            provider=provider,
            transport=transport,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-cancel-log")

        import logging

        caplog.set_level(logging.INFO, logger="roomkit.channels.realtime_voice")

        task = asyncio.create_task(channel.start_session(room.id, "user-1", "fake-ws"))
        await connect_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        msgs = [
            (r.levelname, r.getMessage())
            for r in caplog.records
            if "provider.connect" in r.getMessage()
        ]
        assert any(level == "INFO" and "cancelled" in msg for level, msg in msgs), msgs
        # No ERROR-level dump for an intentional cancel.
        assert not any(
            level == "ERROR" and "provider.connect failed" in msg for level, msg in msgs
        )

    async def test_real_failure_still_logs_error_with_stack(
        self,
        transport: MockRealtimeTransport,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Real exceptions still get the ERROR log + stack trace."""

        class FailingProvider(MockRealtimeProvider):
            async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
                raise RuntimeError("provider blew up")

        provider = FailingProvider()
        channel = RealtimeVoiceChannel(
            "rt-fail",
            provider=provider,
            transport=transport,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-fail")

        import logging

        caplog.set_level(logging.INFO, logger="roomkit.channels.realtime_voice")

        with pytest.raises(RuntimeError, match="provider blew up"):
            await channel.start_session(room.id, "user-1", "fake-ws")

        # ERROR log present; transport cleanup attempted.
        error_msgs = [
            r
            for r in caplog.records
            if r.levelname == "ERROR" and "provider.connect failed" in r.getMessage()
        ]
        assert len(error_msgs) == 1
        disconnect_calls = [c for c in transport.calls if c.method == "disconnect"]
        assert len(disconnect_calls) == 1

    async def test_transport_accept_failure_cleans_every_partial_session_map(
        self,
    ) -> None:
        class FailingTransport(MockRealtimeTransport):
            async def accept(self, session: VoiceSession, connection: Any) -> None:
                await super().accept(session, connection)
                raise RuntimeError("transport blew up")

        channel = RealtimeVoiceChannel(
            "rt-accept-fail",
            provider=MockRealtimeProvider(),
            transport=FailingTransport(),
            pipeline=AudioPipelineConfig(),
        )

        with pytest.raises(RuntimeError, match="transport blew up"):
            await channel.start_session("room-1", "user-1", "fake-ws")

        assert channel._session_spans == {}
        assert channel._session_tools == {}
        assert channel._has_pipeline_vad == {}
        assert channel._preconnect_audio == {}
        assert channel._idle_events == {}
        assert [c.method for c in channel.transport.calls].count("disconnect") == 1

    async def test_provider_failure_releases_pipeline_and_handshake_state(
        self,
    ) -> None:
        class FailingProvider(MockRealtimeProvider):
            async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
                raise RuntimeError("provider blew up")

        channel = RealtimeVoiceChannel(
            "rt-provider-fail",
            provider=FailingProvider(),
            transport=MockRealtimeTransport(),
            pipeline=AudioPipelineConfig(),
            transport_sample_rate=8000,
        )

        with pytest.raises(RuntimeError, match="provider blew up"):
            await channel.start_session("room-1", "user-1", "fake-ws")

        assert channel._session_spans == {}
        assert channel._session_resamplers == {}
        assert channel._session_transport_rates == {}
        assert channel._preconnect_audio == {}
        assert channel._pipeline is not None
        assert channel._pipeline._outbound_locks == {}

    async def test_invalid_negotiated_rate_rolls_back_accepted_transport(self) -> None:
        class InvalidRateTransport(MockRealtimeTransport):
            async def accept(self, session: VoiceSession, connection: Any) -> None:
                await super().accept(session, connection)
                session.metadata["transport_sample_rate"] = "8000"

        provider = MockRealtimeProvider()
        transport = InvalidRateTransport()
        channel = RealtimeVoiceChannel(
            "rt-invalid-rate",
            provider=provider,
            transport=transport,
            pipeline=AudioPipelineConfig(),
        )

        with pytest.raises(ValueError, match="negotiated transport_sample_rate"):
            await channel.start_session("room-1", "user-1", "fake-ws")

        assert [call.method for call in provider.calls].count("connect") == 0
        assert [call.method for call in transport.calls].count("disconnect") == 1
        assert channel._session_spans == {}
        assert channel._session_transport_rates == {}


class TestHandshakeAudioLifecycle:
    async def test_caller_audio_during_provider_handshake_is_flushed_in_order(self) -> None:
        connecting = asyncio.Event()
        release = asyncio.Event()
        captured: list[VoiceSession] = []

        class SlowProvider(MockRealtimeProvider):
            async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
                captured.append(session)
                connecting.set()
                await release.wait()
                await super().connect(session, **kwargs)

        provider = SlowProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel("rt-handshake", provider=provider, transport=transport)

        start = asyncio.create_task(channel.start_session("room-1", "user-1", "fake-ws"))
        await connecting.wait()
        await transport.simulate_client_audio(captured[0], b"first")
        await transport.simulate_client_audio(captured[0], b"second")
        assert provider.sent_audio == []

        release.set()
        session = await start

        assert provider.sent_audio == [
            (session.id, b"first"),
            (session.id, b"second"),
        ]
        assert session.id not in channel._preconnect_audio
        await channel.end_session(session)

    async def test_pipeline_transport_callback_is_registered_only_once(self) -> None:
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel(
            "rt-shared-pipeline",
            provider=provider,
            transport=transport,
            pipeline=AudioPipelineConfig(),
        )

        first = await channel.start_session("room-1", "user-1", "first")
        second = await channel.start_session("room-1", "user-2", "second")

        assert len(transport._audio_callbacks) == 1
        await transport.simulate_client_audio(second, b"\x01\x02" * 80)
        await asyncio.sleep(0)
        assert provider.sent_audio == [(second.id, b"\x01\x02" * 80)]

        await channel.end_session(first)
        await channel.end_session(second)

    async def test_pipeline_audio_during_handshake_is_flushed_after_processing(
        self,
    ) -> None:
        connecting = asyncio.Event()
        release = asyncio.Event()
        captured: list[VoiceSession] = []

        class SlowProvider(MockRealtimeProvider):
            async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
                captured.append(session)
                connecting.set()
                await release.wait()
                await super().connect(session, **kwargs)

        provider = SlowProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel(
            "rt-pipeline-handshake",
            provider=provider,
            transport=transport,
            pipeline=AudioPipelineConfig(),
        )
        start = asyncio.create_task(channel.start_session("room-1", "user-1", "fake-ws"))
        await connecting.wait()

        await transport.simulate_client_audio(captured[0], b"\x01\x02" * 80)
        for _ in range(100):
            pending = channel._preconnect_audio.get(captured[0].id, [])
            if pending:
                break
            await asyncio.sleep(0.001)
        assert pending and pending[0][0] is True
        assert provider.sent_audio == []

        release.set()
        session = await start

        assert provider.sent_audio == [(session.id, b"\x01\x02" * 80)]
        assert session.id not in channel._preconnect_audio
        await channel.end_session(session)

    async def test_preconnect_audio_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connecting = asyncio.Event()
        release = asyncio.Event()
        captured: list[VoiceSession] = []

        class SlowProvider(MockRealtimeProvider):
            async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
                captured.append(session)
                connecting.set()
                await release.wait()
                await super().connect(session, **kwargs)

        monkeypatch.setattr(
            "roomkit.channels._realtime_audio._MAX_PRECONNECT_AUDIO_BYTES",
            3,
        )
        provider = SlowProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel("rt-handshake", provider=provider, transport=transport)
        start = asyncio.create_task(channel.start_session("room-1", "user-1", "fake-ws"))
        await connecting.wait()

        await transport.simulate_client_audio(captured[0], b"four")
        release.set()
        session = await start

        assert provider.sent_audio == []
        assert channel._preconnect_audio == {}
        await channel.end_session(session)


class TestFatalProviderCleanup:
    async def test_fatal_provider_error_ends_channel_and_transport(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
        advance,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")
        session.state = VoiceSessionState.ENDED

        await provider.simulate_error(session, "connection_closed", "peer closed")
        await advance()

        assert channel.get_room_sessions(room_id) == []
        assert [c.method for c in transport.calls].count("disconnect") == 1

    async def test_recoverable_provider_error_keeps_active_session(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        room_id: str,
        advance,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await provider.simulate_error(session, "warning", "recoverable")
        await advance()

        assert channel.get_room_sessions(room_id) == [session]
        await channel.end_session(session)


class TestSendWorkerOrdering:
    """The per-session send worker preserves audio→EOR order structurally."""

    async def test_eor_after_audio_with_yielding_transport(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        """Ordering must hold even when the transport yields mid-send.

        Adversarial transport: parks on the loop before recording each
        send. With one task per chunk, a later-created end-of-response
        task could overtake parked audio tasks; the single send worker
        makes the order structural.
        """
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        call_log: list[str] = []
        orig_send_audio = transport.send_audio
        orig_end_of_response = transport.end_of_response

        async def yielding_send_audio(s: VoiceSession, audio: Any) -> None:
            await asyncio.sleep(0)
            call_log.append("audio")
            await orig_send_audio(s, audio)

        def tracking_end_of_response(s: VoiceSession) -> None:
            call_log.append("end_of_response")
            orig_end_of_response(s)

        transport.send_audio = yielding_send_audio  # type: ignore[assignment]
        transport.end_of_response = tracking_end_of_response  # type: ignore[assignment]

        await provider.simulate_response_start(session)
        for i in range(5):
            await provider.simulate_audio(session, f"chunk-{i}".encode())
        await provider.simulate_response_end(session)
        await asyncio.sleep(0.1)

        assert call_log.count("audio") == 5
        eor_index = call_log.index("end_of_response")
        assert all(i < eor_index for i, v in enumerate(call_log) if v == "audio"), (
            f"Audio after end_of_response: {call_log}"
        )

    async def test_one_worker_per_session(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        """A burst of chunks reuses one resident worker, not a task each."""
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await provider.simulate_response_start(session)
        for i in range(20):
            await provider.simulate_audio(session, f"chunk-{i}".encode())
        await asyncio.sleep(0.05)

        assert len(channel._audio_send_queues) == 1
        assert len(channel._audio_send_workers) == 1
        worker = channel._audio_send_workers[session.id]
        assert not worker.done()

        # Teardown releases the worker via the sentinel
        await channel.end_session(session)
        await asyncio.sleep(0.05)
        assert session.id not in channel._audio_send_queues
        assert worker.done()

    async def test_barge_in_drains_queued_audio(
        self,
        kit: RoomKit,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        room_id: str,
    ) -> None:
        """User speech drops queued (stale) audio without processing it."""
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        gate = asyncio.Event()
        orig_send_audio = transport.send_audio
        sent: list[bytes] = []

        async def gated_send_audio(s: VoiceSession, audio: Any) -> None:
            await gate.wait()
            sent.append(audio)
            await orig_send_audio(s, audio)

        transport.send_audio = gated_send_audio  # type: ignore[assignment]

        await provider.simulate_response_start(session)
        for i in range(10):
            await provider.simulate_audio(session, f"chunk-{i}".encode())
        await asyncio.sleep(0.02)  # worker parks on chunk-0 behind the gate

        await provider.simulate_speech_start(session)  # barge-in: drain queue
        gate.set()
        await asyncio.sleep(0.05)

        # chunk-0 was in flight behind the gate; the other 9 were drained.
        # Its generation is stale by the time the gate opens, so at most
        # the in-flight chunk reaches the transport.
        assert len(sent) <= 1

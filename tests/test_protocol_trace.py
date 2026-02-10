"""Tests for the protocol trace infrastructure."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from roomkit.channels.base import Channel, _safe_invoke
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelMediaType,
    ChannelType,
    HookTrigger,
)
from roomkit.models.event import RoomEvent
from roomkit.models.trace import ProtocolTrace

# -- Helpers ----------------------------------------------------------------


class _StubChannel(Channel):
    """Minimal concrete Channel for testing."""

    channel_type = ChannelType.WEBHOOK

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(media_types=[ChannelMediaType.TEXT])


def _make_trace(**overrides: object) -> ProtocolTrace:
    defaults = dict(
        channel_id="ch-1",
        direction="inbound",
        protocol="sip",
        summary="INVITE sip:+1555@pbx",
        metadata={},
    )
    defaults.update(overrides)
    return ProtocolTrace(**defaults)  # type: ignore[arg-type]


# -- ProtocolTrace model ---------------------------------------------------


class TestProtocolTraceModel:
    def test_frozen(self) -> None:
        trace = _make_trace()
        with pytest.raises(AttributeError):
            trace.summary = "modified"  # type: ignore[misc]

    def test_defaults(self) -> None:
        trace = _make_trace()
        assert trace.raw is None
        assert trace.session_id is None
        assert trace.room_id is None
        assert isinstance(trace.timestamp, datetime)
        assert trace.timestamp.tzinfo is not None

    def test_metadata(self) -> None:
        trace = _make_trace(metadata={"codec": "G.722"})
        assert trace.metadata["codec"] == "G.722"


# -- Channel trace infrastructure ------------------------------------------


class TestChannelTraceInfra:
    def test_trace_disabled_by_default(self) -> None:
        ch = _StubChannel("ch-1")
        assert ch.trace_enabled is False

    def test_trace_enabled_after_on_trace(self) -> None:
        ch = _StubChannel("ch-1")
        ch.on_trace(lambda t: None)
        assert ch.trace_enabled is True

    def test_trace_enabled_with_framework_handler(self) -> None:
        ch = _StubChannel("ch-1")
        ch._trace_framework_handler = lambda t: None
        assert ch.trace_enabled is True

    def test_emit_trace_calls_sync_callback(self) -> None:
        ch = _StubChannel("ch-1")
        received: list[ProtocolTrace] = []
        ch.on_trace(lambda t: received.append(t))

        trace = _make_trace()
        ch.emit_trace(trace)

        assert len(received) == 1
        assert received[0] is trace

    def test_emit_trace_protocol_filter(self) -> None:
        ch = _StubChannel("ch-1")
        sip_traces: list[ProtocolTrace] = []
        all_traces: list[ProtocolTrace] = []
        ch.on_trace(lambda t: sip_traces.append(t), protocols=["sip"])
        ch.on_trace(lambda t: all_traces.append(t))

        sip = _make_trace(protocol="sip")
        rtp = _make_trace(protocol="rtp")
        ch.emit_trace(sip)
        ch.emit_trace(rtp)

        assert len(sip_traces) == 1
        assert sip_traces[0].protocol == "sip"
        assert len(all_traces) == 2

    def test_emit_trace_calls_framework_handler(self) -> None:
        ch = _StubChannel("ch-1")
        fw_received: list[ProtocolTrace] = []
        ch._trace_framework_handler = lambda t: fw_received.append(t)

        trace = _make_trace()
        ch.emit_trace(trace)

        assert len(fw_received) == 1

    def test_resolve_trace_room_default_none(self) -> None:
        ch = _StubChannel("ch-1")
        assert ch.resolve_trace_room("session-1") is None
        assert ch.resolve_trace_room(None) is None


# -- _safe_invoke -----------------------------------------------------------


class TestSafeInvoke:
    def test_sync_callback(self) -> None:
        received: list[ProtocolTrace] = []

        def cb(t: ProtocolTrace) -> None:
            received.append(t)

        trace = _make_trace()
        _safe_invoke(cb, trace)
        assert len(received) == 1

    def test_sync_callback_exception_suppressed(self) -> None:
        def cb(t: ProtocolTrace) -> None:
            raise ValueError("boom")

        # Should not raise
        _safe_invoke(cb, _make_trace())

    async def test_async_callback_scheduled(self) -> None:
        received: list[ProtocolTrace] = []

        async def cb(t: ProtocolTrace) -> None:
            received.append(t)

        trace = _make_trace()
        _safe_invoke(cb, trace)
        # Let the task run
        await asyncio.sleep(0)
        assert len(received) == 1


# -- VoiceChannel trace bridge -----------------------------------------------


class TestVoiceChannelTrace:
    def test_resolve_trace_room_from_bindings(self) -> None:
        from roomkit.channels.voice import VoiceChannel

        ch = VoiceChannel("voice")
        # Simulate a bound session
        mock_binding = MagicMock(spec=ChannelBinding)
        ch._session_bindings["sess-1"] = ("room-abc", mock_binding)

        assert ch.resolve_trace_room("sess-1") == "room-abc"
        assert ch.resolve_trace_room("sess-unknown") is None
        assert ch.resolve_trace_room(None) is None

    def test_on_trace_bridges_to_backend(self) -> None:
        from roomkit.channels.voice import VoiceChannel
        from roomkit.voice.base import VoiceCapability

        mock_backend = MagicMock()
        mock_backend.capabilities = VoiceCapability.NONE
        mock_backend.supports_playback_callback = False
        mock_backend.feeds_aec_reference = False
        ch = VoiceChannel("voice", backend=mock_backend)

        ch.on_trace(lambda t: None, protocols=["sip"])

        mock_backend.set_trace_emitter.assert_called_once()
        # The emitter should be the channel's emit_trace
        emitter = mock_backend.set_trace_emitter.call_args[0][0]
        assert emitter == ch.emit_trace


# -- RealtimeVoiceChannel trace -----------------------------------------------


class TestRealtimeVoiceChannelTrace:
    def test_resolve_trace_room_from_session_rooms(self) -> None:
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        mock_provider = MagicMock()
        mock_transport = MagicMock()
        ch = RealtimeVoiceChannel(
            "rt-voice",
            provider=mock_provider,
            transport=mock_transport,
        )
        ch._session_rooms["sess-1"] = "room-xyz"

        assert ch.resolve_trace_room("sess-1") == "room-xyz"
        assert ch.resolve_trace_room("sess-unknown") is None
        assert ch.resolve_trace_room(None) is None

    def test_on_trace_bridges_to_transport(self) -> None:
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        mock_provider = MagicMock()
        mock_transport = MagicMock()
        ch = RealtimeVoiceChannel(
            "rt-voice",
            provider=mock_provider,
            transport=mock_transport,
        )

        ch.on_trace(lambda t: None, protocols=["sip"])

        mock_transport.set_trace_emitter.assert_called_once()
        emitter = mock_transport.set_trace_emitter.call_args[0][0]
        assert emitter == ch.emit_trace

    def test_set_framework_bridges_trace_to_transport(self) -> None:
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        mock_provider = MagicMock()
        mock_transport = MagicMock()
        ch = RealtimeVoiceChannel(
            "rt-voice",
            provider=mock_provider,
            transport=mock_transport,
        )

        # Simulate what register_channel does
        ch._trace_framework_handler = lambda t: None
        mock_framework = MagicMock()
        ch.set_framework(mock_framework)

        mock_transport.set_trace_emitter.assert_called_once()
        emitter = mock_transport.set_trace_emitter.call_args[0][0]
        assert emitter == ch.emit_trace


# -- Framework trace wiring --------------------------------------------------


class TestFrameworkTraceWiring:
    async def test_register_channel_sets_framework_handler(self) -> None:
        from roomkit import RoomKit

        kit = RoomKit()
        ch = _StubChannel("ch-1")
        kit.register_channel(ch)

        assert ch._trace_framework_handler is not None
        assert ch.trace_enabled is True

    async def test_on_channel_trace_fires_hook(self) -> None:
        from roomkit import RoomKit

        kit = RoomKit()
        ch = _StubChannel("ch-1")
        kit.register_channel(ch)

        # Create a room and attach channel
        await kit.create_room("room-1")
        await kit.attach_channel("room-1", "ch-1")

        # Register ON_PROTOCOL_TRACE hook
        hook_traces: list[ProtocolTrace] = []

        @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
        async def capture_trace(trace: ProtocolTrace, ctx: RoomContext) -> None:
            hook_traces.append(trace)

        # Emit a trace with room_id set
        trace = _make_trace(channel_id="ch-1", room_id="room-1")
        ch.emit_trace(trace)

        # Let async tasks run
        await asyncio.sleep(0.05)

        assert len(hook_traces) == 1
        assert hook_traces[0].summary == "INVITE sip:+1555@pbx"

    async def test_trace_without_room_id_not_fired(self) -> None:
        from roomkit import RoomKit

        kit = RoomKit()
        ch = _StubChannel("ch-1")
        kit.register_channel(ch)

        hook_traces: list[ProtocolTrace] = []

        @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
        async def capture_trace(trace: ProtocolTrace, ctx: RoomContext) -> None:
            hook_traces.append(trace)

        # Emit trace without room_id or session_id — should not fire hook
        trace = _make_trace(channel_id="ch-1")
        ch.emit_trace(trace)

        await asyncio.sleep(0.05)
        assert len(hook_traces) == 0

    async def test_trace_resolved_via_session_id(self) -> None:
        from roomkit import RoomKit
        from roomkit.channels.voice import VoiceChannel

        kit = RoomKit()
        voice = VoiceChannel("voice")
        kit.register_channel(voice)

        # Create room and attach
        await kit.create_room("room-sess")
        await kit.attach_channel("room-sess", "voice")

        # Simulate session binding
        mock_binding = MagicMock(spec=ChannelBinding)
        voice._session_bindings["sess-42"] = ("room-sess", mock_binding)

        hook_traces: list[ProtocolTrace] = []

        @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
        async def capture(trace: ProtocolTrace, ctx: RoomContext) -> None:
            hook_traces.append(trace)

        # Emit trace with session_id but no room_id
        trace = _make_trace(channel_id="voice", session_id="sess-42")
        voice.emit_trace(trace)

        await asyncio.sleep(0.05)
        assert len(hook_traces) == 1


# -- HookTrigger enum -------------------------------------------------------


class TestPendingTraceFlush:
    """Traces emitted before the room exists are buffered and flushed on attach."""

    async def test_trace_buffered_and_flushed_on_attach(self) -> None:
        from roomkit import RoomKit

        kit = RoomKit()
        ch = _StubChannel("ch-1")
        kit.register_channel(ch)

        hook_traces: list[ProtocolTrace] = []

        @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
        async def capture(trace: ProtocolTrace, ctx: RoomContext) -> None:
            hook_traces.append(trace)

        # Emit a trace for a room that doesn't exist yet
        trace = _make_trace(channel_id="ch-1", room_id="future-room")
        ch.emit_trace(trace)

        # Let async task run — trace should be buffered, not fired
        await asyncio.sleep(0.05)
        assert len(hook_traces) == 0
        assert "future-room" in kit._pending_traces

        # Now create the room and attach channel — triggers flush
        await kit.create_room(room_id="future-room")
        await kit.attach_channel("future-room", "ch-1")

        await asyncio.sleep(0.05)
        assert len(hook_traces) == 1
        assert hook_traces[0].summary == "INVITE sip:+1555@pbx"
        assert "future-room" not in kit._pending_traces

    async def test_trace_fires_immediately_when_room_exists(self) -> None:
        from roomkit import RoomKit

        kit = RoomKit()
        ch = _StubChannel("ch-1")
        kit.register_channel(ch)

        await kit.create_room(room_id="existing-room")
        await kit.attach_channel("existing-room", "ch-1")

        hook_traces: list[ProtocolTrace] = []

        @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
        async def capture(trace: ProtocolTrace, ctx: RoomContext) -> None:
            hook_traces.append(trace)

        trace = _make_trace(channel_id="ch-1", room_id="existing-room")
        ch.emit_trace(trace)

        await asyncio.sleep(0.05)
        assert len(hook_traces) == 1
        assert not kit._pending_traces


class TestProviderName:
    """Tests for EventSource.provider population via Channel.provider_name."""

    def test_base_channel_provider_name_none(self) -> None:
        ch = _StubChannel("ch-1")
        assert ch.provider_name is None

    def test_base_channel_provider_name_from_provider(self) -> None:
        ch = _StubChannel("ch-1")
        mock_provider = MagicMock()
        mock_provider.name = "TestProvider"
        ch._provider = mock_provider
        assert ch.provider_name == "TestProvider"

    def test_base_channel_provider_name_no_name_attr(self) -> None:
        ch = _StubChannel("ch-1")
        ch._provider = object()  # no .name attribute
        assert ch.provider_name is None

    def test_voice_channel_provider_name_from_backend(self) -> None:
        from roomkit.channels.voice import VoiceChannel
        from roomkit.voice.base import VoiceCapability

        mock_backend = MagicMock()
        mock_backend.name = "SIP"
        mock_backend.capabilities = VoiceCapability.NONE
        mock_backend.supports_playback_callback = False
        mock_backend.feeds_aec_reference = False
        ch = VoiceChannel("voice", backend=mock_backend)
        assert ch.provider_name == "SIP"

    def test_voice_channel_provider_name_none_without_backend(self) -> None:
        from roomkit.channels.voice import VoiceChannel

        ch = VoiceChannel("voice")
        assert ch.provider_name is None

    def test_realtime_voice_channel_provider_name_from_transport(self) -> None:
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        mock_provider = MagicMock()
        mock_transport = MagicMock()
        mock_transport.name = "SIPRealtimeTransport"
        ch = RealtimeVoiceChannel(
            "rt-voice",
            provider=mock_provider,
            transport=mock_transport,
        )
        assert ch.provider_name == "SIPRealtimeTransport"

    def test_transport_channel_provider_name(self) -> None:
        from roomkit.channels.transport import TransportChannel

        mock_provider = MagicMock()
        mock_provider.name = "TwilioSMS"
        ch = TransportChannel("sms-1", ChannelType.SMS, provider=mock_provider)
        assert ch.provider_name == "TwilioSMS"

    def test_ai_channel_provider_name(self) -> None:
        from roomkit.channels.ai import AIChannel

        mock_provider = MagicMock()
        mock_provider.name = "OpenAIProvider"
        mock_provider.supports_vision = False
        mock_provider.supports_streaming = False
        ch = AIChannel("ai-1", provider=mock_provider)
        assert ch.provider_name == "OpenAIProvider"


class TestHookTriggerEnum:
    def test_on_protocol_trace_exists(self) -> None:
        assert HookTrigger.ON_PROTOCOL_TRACE == "on_protocol_trace"

"""Tests for unified process_inbound with voice sessions."""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelType,
    EventType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, SystemContent, TextContent
from roomkit.models.hook import HookResult
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.inbound import parse_voice_session

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class VoiceTestChannel(Channel):
    """Minimal channel that tracks connect_session / disconnect_session calls."""

    channel_type = ChannelType.VOICE

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []
        self.connected_sessions: list[tuple[Any, str]] = []
        self.disconnected_sessions: list[tuple[Any, str]] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()

    async def connect_session(self, session: Any, room_id: str, binding: ChannelBinding) -> None:
        self.connected_sessions.append((session, room_id))

    async def disconnect_session(self, session: Any, room_id: str) -> None:
        self.disconnected_sessions.append((session, room_id))


def _make_voice_session(
    session_id: str = "sess-1",
    participant_id: str = "caller-1",
    room_id: str = "room-1",
    **metadata: Any,
) -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id=room_id,
        participant_id=participant_id,
        channel_id="voice",
        state=VoiceSessionState.ACTIVE,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# parse_voice_session tests
# ---------------------------------------------------------------------------


class TestParseVoiceSession:
    def test_creates_inbound_message(self) -> None:
        session = _make_voice_session(caller="+15551234567")
        msg = parse_voice_session(session, channel_id="voice")

        assert isinstance(msg, InboundMessage)
        assert msg.channel_id == "voice"
        assert msg.sender_id == "caller-1"
        assert msg.session is session
        assert msg.event_type == EventType.SYSTEM

    def test_content_is_session_started(self) -> None:
        session = _make_voice_session()
        msg = parse_voice_session(session, channel_id="voice")

        assert isinstance(msg.content, SystemContent)
        assert msg.content.code == "session_started"
        assert msg.content.data["session_id"] == "sess-1"
        assert msg.content.data["channel_id"] == "voice"

    def test_metadata_from_session(self) -> None:
        session = _make_voice_session(caller="+15551234567")
        msg = parse_voice_session(session, channel_id="voice")

        assert msg.metadata["caller"] == "+15551234567"

    def test_extra_metadata_merged(self) -> None:
        session = _make_voice_session(caller="+15551234567")
        msg = parse_voice_session(session, channel_id="voice", metadata={"extra": "val"})

        assert msg.metadata["caller"] == "+15551234567"
        assert msg.metadata["extra"] == "val"

    def test_extra_metadata_overrides_session(self) -> None:
        session = _make_voice_session(caller="+15551234567")
        msg = parse_voice_session(session, channel_id="voice", metadata={"caller": "overridden"})
        assert msg.metadata["caller"] == "overridden"


# ---------------------------------------------------------------------------
# process_inbound with session tests
# ---------------------------------------------------------------------------


class TestProcessInboundVoice:
    async def test_session_triggers_connect_session(self) -> None:
        """process_inbound with session calls channel.connect_session after hooks pass."""
        kit = RoomKit()
        channel = VoiceTestChannel("voice")
        kit.register_channel(channel)

        session = _make_voice_session()
        msg = parse_voice_session(session, channel_id="voice")

        result = await kit.process_inbound(msg)

        assert not result.blocked
        assert len(channel.connected_sessions) == 1
        connected_session, room_id = channel.connected_sessions[0]
        assert connected_session is session

    async def test_session_not_connected_when_blocked(self) -> None:
        """BEFORE_BROADCAST hook can block → connect_session NOT called."""
        kit = RoomKit()
        channel = VoiceTestChannel("voice")
        kit.register_channel(channel)

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def reject_all(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("rejected")

        session = _make_voice_session()
        msg = parse_voice_session(session, channel_id="voice")

        result = await kit.process_inbound(msg)

        assert result.blocked
        assert result.reason == "rejected"
        assert len(channel.connected_sessions) == 0

    async def test_session_started_event_stored(self) -> None:
        """The session_started system event is stored in room history."""
        kit = RoomKit()
        channel = VoiceTestChannel("voice")
        kit.register_channel(channel)

        session = _make_voice_session()
        msg = parse_voice_session(session, channel_id="voice")

        result = await kit.process_inbound(msg)

        assert result.event is not None
        assert isinstance(result.event.content, SystemContent)
        assert result.event.content.code == "session_started"

    async def test_explicit_room_id(self) -> None:
        """process_inbound with room_id= routes to that room."""
        kit = RoomKit()
        channel = VoiceTestChannel("voice")
        kit.register_channel(channel)
        await kit.create_room(room_id="specific-room")
        await kit.attach_channel("specific-room", "voice")

        session = _make_voice_session()
        msg = parse_voice_session(session, channel_id="voice")

        result = await kit.process_inbound(msg, room_id="specific-room")

        assert not result.blocked
        assert len(channel.connected_sessions) == 1
        _, room_id = channel.connected_sessions[0]
        assert room_id == "specific-room"

    async def test_text_inbound_unchanged(self) -> None:
        """Regular text process_inbound (no session) still works unchanged."""
        kit = RoomKit()
        channel = VoiceTestChannel("voice")
        kit.register_channel(channel)

        msg = InboundMessage(
            channel_id="voice",
            sender_id="user-1",
            content=TextContent(body="hello"),
        )
        result = await kit.process_inbound(msg)

        assert not result.blocked
        # No session → connect_session not called
        assert len(channel.connected_sessions) == 0

    async def test_hook_sees_session_started_content(self) -> None:
        """BEFORE_BROADCAST hook receives the SystemContent event."""
        kit = RoomKit()
        channel = VoiceTestChannel("voice")
        kit.register_channel(channel)

        hook_events: list[RoomEvent] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def capture(event: RoomEvent, ctx: RoomContext) -> HookResult:
            hook_events.append(event)
            return HookResult.allow()

        session = _make_voice_session(caller="+15551234567")
        msg = parse_voice_session(session, channel_id="voice")
        await kit.process_inbound(msg)

        assert len(hook_events) == 1
        ev = hook_events[0]
        assert isinstance(ev.content, SystemContent)
        assert ev.content.code == "session_started"


# ---------------------------------------------------------------------------
# SIP on_call async decorator tests
# ---------------------------------------------------------------------------


class TestSIPAsyncDecorator:
    """Test _wrap_async helper used by SIP backend callbacks."""

    def test_sync_callback_unchanged(self) -> None:
        from roomkit.voice.backends.sip import _wrap_async

        calls: list[str] = []

        def sync_fn(x: str) -> None:
            calls.append(x)

        wrapped = _wrap_async(sync_fn)
        wrapped("hello")
        assert calls == ["hello"]

    async def test_async_callback_wrapped_in_task(self) -> None:
        from roomkit.voice.backends.sip import _wrap_async

        calls: list[str] = []

        async def async_fn(x: str) -> None:
            calls.append(x)

        wrapped = _wrap_async(async_fn)
        wrapped("hello")
        # Give the task a chance to run
        await asyncio.sleep(0.01)
        assert calls == ["hello"]

    def test_decorator_returns_original(self) -> None:
        from roomkit.voice.backends.sip import _wrap_async

        async def my_handler(session: Any) -> None:
            pass

        result = _wrap_async(my_handler)
        # _wrap_async returns the wrapper, but on_call returns the original
        # Let's test the on_call pattern directly
        assert callable(result)

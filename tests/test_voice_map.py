"""Tests for VoiceChannel voice_map and system event filtering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from roomkit import MockTTSProvider, MockVoiceBackend, VoiceChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room

# -- Helpers ------------------------------------------------------------------


def _make_event(
    room_id: str = "room-1",
    channel_id: str = "agent-a",
    body: str = "Hello",
    event_type: EventType = EventType.MESSAGE,
    visibility: str = "all",
) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        type=event_type,
        source=EventSource(channel_id=channel_id, channel_type=ChannelType.AI),
        content=TextContent(body=body),
        visibility=visibility,
    )


def _binding(channel_id: str = "voice", room_id: str = "room-1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=ChannelType.VOICE,
        category=ChannelCategory.TRANSPORT,
    )


def _ctx(room_id: str = "room-1") -> RoomContext:
    return RoomContext(room=Room(id=room_id), bindings=[_binding(room_id=room_id)])


# -- voice_map resolution ----------------------------------------------------


class TestResolveVoice:
    def test_voice_map_resolves_agent_voice(self) -> None:
        channel = VoiceChannel(
            "voice",
            voice_map={"agent-a": "voice-alice", "agent-b": "voice-bob"},
        )
        assert channel._resolve_voice("agent-a") == "voice-alice"
        assert channel._resolve_voice("agent-b") == "voice-bob"

    def test_voice_map_returns_none_for_unknown(self) -> None:
        channel = VoiceChannel(
            "voice",
            voice_map={"agent-a": "voice-alice"},
        )
        assert channel._resolve_voice("agent-unknown") is None

    def test_voice_map_empty_returns_none(self) -> None:
        channel = VoiceChannel("voice")
        assert channel._resolve_voice("agent-a") is None

    def test_voice_map_none_returns_none(self) -> None:
        channel = VoiceChannel("voice", voice_map=None)
        assert channel._resolve_voice("anything") is None


# -- System event filter in deliver() ----------------------------------------


class TestSystemEventFilter:
    async def test_system_event_not_delivered_via_tts(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice", tts=tts, backend=backend)

        event = _make_event(event_type=EventType.SYSTEM, body="[Handoff: a -> b]")
        result = await channel.deliver(event, _binding(), _ctx())

        assert isinstance(result, ChannelOutput)
        assert tts.calls == []  # TTS should NOT be called

    async def test_internal_visibility_not_delivered_via_tts(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice", tts=tts, backend=backend)

        event = _make_event(visibility="internal", body="internal message")
        result = await channel.deliver(event, _binding(), _ctx())

        assert isinstance(result, ChannelOutput)
        assert tts.calls == []

    async def test_normal_message_still_delivered(self) -> None:
        """MESSAGE type + visibility='all' should reach TTS."""
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice", tts=tts, backend=backend)

        # Need a framework for the full deliver path to work
        kit = MagicMock()
        kit.hook_engine = MagicMock()
        # run_sync_hooks returns an "allowed" result
        sync_result = MagicMock()
        sync_result.allowed = True
        sync_result.event = "Hello"
        kit.hook_engine.run_sync_hooks = AsyncMock(return_value=sync_result)
        kit.hook_engine.run_async_hooks = AsyncMock()
        kit._build_context = AsyncMock(return_value=_ctx())
        channel.set_framework(kit)

        session = await backend.connect("room-1", "user-1", "voice")
        channel.bind_session(session, "room-1", _binding())

        event = _make_event(body="Hello")
        await channel.deliver(event, _binding(), _ctx())

        # TTS was called (send_audio on backend)
        assert len(backend.sent_audio) == 1


# -- deliver passes voice from voice_map -------------------------------------


class TestDeliverPassesVoice:
    async def test_deliver_voice_passes_voice_from_map(self) -> None:
        """_deliver_voice should pass voice= from voice_map to _send_tts."""
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel(
            "voice",
            tts=tts,
            backend=backend,
            voice_map={"agent-a": "voice-alice"},
        )

        kit = MagicMock()
        kit.hook_engine = MagicMock()
        sync_result = MagicMock()
        sync_result.allowed = True
        sync_result.event = "Hello from agent A"
        kit.hook_engine.run_sync_hooks = AsyncMock(return_value=sync_result)
        kit.hook_engine.run_async_hooks = AsyncMock()
        kit._build_context = AsyncMock(return_value=_ctx())
        kit._emit_framework_event = AsyncMock()
        channel.set_framework(kit)

        session = await backend.connect("room-1", "user-1", "voice")
        channel.bind_session(session, "room-1", _binding())

        event = _make_event(channel_id="agent-a", body="Hello from agent A")

        with patch.object(channel, "_send_tts", new_callable=AsyncMock) as mock_send:
            await channel._deliver_voice(event, _binding(), _ctx())
            mock_send.assert_called_once_with(session, "Hello from agent A", voice="voice-alice")

    async def test_deliver_voice_passes_none_for_unknown_agent(self) -> None:
        """Unknown agent should pass voice=None."""
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel(
            "voice",
            tts=tts,
            backend=backend,
            voice_map={"agent-a": "voice-alice"},
        )

        kit = MagicMock()
        kit.hook_engine = MagicMock()
        sync_result = MagicMock()
        sync_result.allowed = True
        sync_result.event = "Hi"
        kit.hook_engine.run_sync_hooks = AsyncMock(return_value=sync_result)
        kit.hook_engine.run_async_hooks = AsyncMock()
        kit._build_context = AsyncMock(return_value=_ctx())
        kit._emit_framework_event = AsyncMock()
        channel.set_framework(kit)

        session = await backend.connect("room-1", "user-1", "voice")
        channel.bind_session(session, "room-1", _binding())

        event = _make_event(channel_id="agent-unknown", body="Hi")

        with patch.object(channel, "_send_tts", new_callable=AsyncMock) as mock_send:
            await channel._deliver_voice(event, _binding(), _ctx())
            mock_send.assert_called_once_with(session, "Hi", voice=None)

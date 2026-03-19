"""Tests for auto_connect: VoiceChannel auto-starts on attach_channel."""

from __future__ import annotations

from roomkit import (
    RoomKit,
    VoiceChannel,
)
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSessionState
from roomkit.voice.pipeline import AudioPipelineConfig


class TestAutoConnect:
    async def test_auto_connect_backend_starts_session(self) -> None:
        """attach_channel auto-starts a session for auto_connect backends."""
        backend = MockVoiceBackend()
        assert not backend.auto_connect  # MockVoiceBackend defaults to False

        # Subclass to enable auto_connect
        class LocalMockBackend(MockVoiceBackend):
            @property
            def auto_connect(self) -> bool:
                return True

        backend = LocalMockBackend()
        kit = RoomKit()
        voice = VoiceChannel("voice", backend=backend, pipeline=AudioPipelineConfig())
        kit.register_channel(voice)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "voice")

        # Session should have been auto-created and is active
        sessions = backend.list_sessions("room-1")
        assert len(sessions) == 1
        assert sessions[0].state == VoiceSessionState.ACTIVE

        await kit.close()

    async def test_non_auto_connect_does_not_start(self) -> None:
        """attach_channel does NOT auto-start for auto_connect=False backends."""
        backend = MockVoiceBackend()
        kit = RoomKit()
        voice = VoiceChannel("voice", backend=backend, pipeline=AudioPipelineConfig())
        kit.register_channel(voice)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "voice")

        # No session should exist
        sessions = backend.list_sessions("room-1")
        assert len(sessions) == 0

        await kit.close()

    async def test_hooks_fire_on_auto_connect(self) -> None:
        """Hooks registered BEFORE attach_channel fire during auto-connect."""
        from roomkit import HookExecution, HookTrigger

        class LocalMockBackend(MockVoiceBackend):
            @property
            def auto_connect(self) -> bool:
                return True

        backend = LocalMockBackend()
        kit = RoomKit()
        voice = VoiceChannel("voice", backend=backend, pipeline=AudioPipelineConfig())
        kit.register_channel(voice)

        await kit.create_room(room_id="room-1")

        hook_fired = []

        @kit.hook(HookTrigger.ON_CHANNEL_ATTACHED, execution=HookExecution.ASYNC)
        async def on_attached(event, ctx):
            hook_fired.append("attached")

        await kit.attach_channel("room-1", "voice")

        assert "attached" in hook_fired
        assert len(backend.list_sessions("room-1")) == 1

        await kit.close()

    async def test_auto_connect_no_backend_is_noop(self) -> None:
        """attach_channel on VoiceChannel without backend does nothing."""
        kit = RoomKit()
        voice = VoiceChannel("voice")
        kit.register_channel(voice)

        await kit.create_room(room_id="room-1")
        # Should not raise
        await kit.attach_channel("room-1", "voice")

        await kit.close()

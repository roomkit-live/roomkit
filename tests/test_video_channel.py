"""Integration tests for VideoChannel — end-to-end video in a room."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    HookExecution,
    HookTrigger,
    MockVideoBackend,
    MockVisionProvider,
    RoomKit,
    VideoChannel,
    VideoFrame,
)
from roomkit.models.enums import ChannelType
from roomkit.models.session_event import SessionStartedEvent


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


# ---------------------------------------------------------------------------
# Channel registration and room wiring
# ---------------------------------------------------------------------------


class TestVideoChannelRegistration:
    async def test_register_video_channel(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        assert "video-1" in kit._channels

    async def test_attach_to_room(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        binding = await kit.attach_channel("r1", "video-1")
        assert binding.channel_id == "video-1"
        assert binding.channel_type == ChannelType.VIDEO

    async def test_channel_info(self) -> None:
        backend = MockVideoBackend()
        vision = MockVisionProvider()
        ch = VideoChannel("video-1", backend=backend, vision=vision, vision_interval_ms=3000)
        info = ch.info
        assert info["backend"] == "MockVideoBackend"
        assert info["vision"] == "MockVisionProvider"
        assert info["vision_interval_ms"] == 3000

    async def test_capabilities(self) -> None:
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        caps = ch.capabilities()
        assert caps.supports_video is True


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestVideoSessionLifecycle:
    async def test_connect_video(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        assert session.room_id == "r1"
        assert session.participant_id == "user-1"
        # Session is bound
        assert session.id in ch._session_bindings

    async def test_disconnect_video(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        await kit.disconnect_video(session)
        assert session.id not in ch._session_bindings

    async def test_connect_unregistered_raises(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        with pytest.raises(Exception, match="not a registered VideoChannel"):
            await kit.connect_video("r1", "user-1", "nonexistent")

    async def test_connect_unattached_raises(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        # Not attached
        with pytest.raises(Exception, match="not attached"):
            await kit.connect_video("r1", "user-1", "video-1")

    async def test_backend_disconnect_unbinds(self, kit: RoomKit) -> None:
        """When backend signals client disconnected, session is unbound."""
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        assert session.id in ch._session_bindings

        await backend.simulate_client_disconnected(session)
        assert session.id not in ch._session_bindings


# ---------------------------------------------------------------------------
# Hook triggers
# ---------------------------------------------------------------------------


class TestVideoHookTriggers:
    async def test_session_started_hook_fires(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        started_events: list[SessionStartedEvent] = []

        @kit.hook(HookTrigger.ON_VIDEO_SESSION_STARTED, HookExecution.ASYNC)
        async def on_started(event: SessionStartedEvent, context: object) -> None:
            started_events.append(event)

        await kit.connect_video("r1", "user-1", "video-1")
        await asyncio.sleep(0.1)

        assert len(started_events) == 1
        assert started_events[0].room_id == "r1"
        assert started_events[0].channel_type == ChannelType.VIDEO
        assert started_events[0].participant_id == "user-1"

    async def test_session_ended_hook_fires(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        ended_data: list[dict] = []

        @kit.hook(HookTrigger.ON_VIDEO_SESSION_ENDED, HookExecution.ASYNC)
        async def on_ended(event: object, context: object) -> None:
            ended_data.append({"event": event})

        session = await kit.connect_video("r1", "user-1", "video-1")
        await kit.disconnect_video(session)
        await asyncio.sleep(0.1)

        assert len(ended_data) == 1


# ---------------------------------------------------------------------------
# Vision analysis
# ---------------------------------------------------------------------------


class TestVideoVision:
    async def test_frame_triggers_vision(self, kit: RoomKit) -> None:
        """Frames received by backend trigger vision analysis."""
        backend = MockVideoBackend()
        vision = MockVisionProvider(descriptions=["A person waving"])
        ch = VideoChannel("video-1", backend=backend, vision=vision, vision_interval_ms=0)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        frame = VideoFrame(data=b"\x00" * 100, codec="h264", timestamp_ms=0.0)
        await backend.simulate_video_received(session, frame)
        await asyncio.sleep(0.1)

        # Vision provider was called
        assert len(vision.calls) == 1
        # Last result is cached
        result = ch.get_last_vision_result(session.id)
        assert result is not None
        assert result.description == "A person waving"

    async def test_vision_interval_skips_frames(self, kit: RoomKit) -> None:
        """Frames arriving before interval are skipped."""
        backend = MockVideoBackend()
        vision = MockVisionProvider()
        ch = VideoChannel("video-1", backend=backend, vision=vision, vision_interval_ms=2000)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        # Send 3 frames: ts=0, ts=500, ts=2000
        for ts in [0.0, 500.0, 2000.0]:
            frame = VideoFrame(data=b"\x00", codec="h264", timestamp_ms=ts)
            await backend.simulate_video_received(session, frame)
        await asyncio.sleep(0.1)

        # Only ts=0 and ts=2000 should be analyzed (interval=2000ms)
        assert len(vision.calls) == 2

    async def test_no_vision_no_analysis(self, kit: RoomKit) -> None:
        """Without vision provider, frames are received but not analyzed."""
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        frame = VideoFrame(data=b"\x00", codec="h264")
        await backend.simulate_video_received(session, frame)
        await asyncio.sleep(0.05)

        assert ch.get_last_vision_result(session.id) is None

    async def test_vision_emits_framework_event(self, kit: RoomKit) -> None:
        """Vision results are emitted as framework events."""
        backend = MockVideoBackend()
        vision = MockVisionProvider(descriptions=["A cat on a desk"])
        ch = VideoChannel("video-1", backend=backend, vision=vision, vision_interval_ms=0)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        vision_events: list[dict] = []

        @kit.on("video_vision_result")
        async def on_vision(event: object) -> None:
            vision_events.append({"event": event})

        session = await kit.connect_video("r1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 50, codec="h264", timestamp_ms=0.0)
        await backend.simulate_video_received(session, frame)
        await asyncio.sleep(0.1)

        assert len(vision_events) == 1


# ---------------------------------------------------------------------------
# Channel close
# ---------------------------------------------------------------------------


class TestVideoChannelClose:
    async def test_close_cleans_up(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        vision = MockVisionProvider()
        ch = VideoChannel("video-1", backend=backend, vision=vision)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        await kit.connect_video("r1", "user-1", "video-1")
        await ch.close()

        assert ch._session_bindings == {}
        # Backend was closed
        assert backend.calls[-1].method == "close"


# ---------------------------------------------------------------------------
# Multi-session
# ---------------------------------------------------------------------------


class TestMultiSession:
    async def test_multiple_sessions_in_room(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        vision = MockVisionProvider(descriptions=["Frame A", "Frame B"])
        ch = VideoChannel("video-1", backend=backend, vision=vision, vision_interval_ms=0)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        s1 = await kit.connect_video("r1", "user-1", "video-1")
        s2 = await kit.connect_video("r1", "user-2", "video-1")

        frame = VideoFrame(data=b"\x00", codec="h264", timestamp_ms=0.0)
        await backend.simulate_video_received(s1, frame)
        await backend.simulate_video_received(s2, frame)
        await asyncio.sleep(0.1)

        # Each session has its own vision result
        r1 = ch.get_last_vision_result(s1.id)
        r2 = ch.get_last_vision_result(s2.id)
        assert r1 is not None
        assert r2 is not None

    async def test_resolve_trace_room(self) -> None:
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        assert ch.resolve_trace_room(None) is None
        assert ch.resolve_trace_room("nonexistent") is None


# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------


class TestVideoChannelExport:
    def test_importable_from_roomkit(self) -> None:
        import roomkit

        assert hasattr(roomkit, "VideoChannel")

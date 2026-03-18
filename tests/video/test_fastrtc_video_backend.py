"""Tests for FastRTCVideoBackend (combined audio+video)."""

from __future__ import annotations

import numpy as np

from roomkit.video.backends.fastrtc import FastRTCVideoBackend
from roomkit.video.base import VideoSessionState


class TestFastRTCVideoBackendLifecycle:
    """Session lifecycle: connect, disconnect, close."""

    async def test_connect_creates_both_sessions(self):
        backend = FastRTCVideoBackend()
        session = await backend.connect("room-1", "user-1", "voice")

        # Voice session exists
        assert backend.get_session(session.id) is not None

        # Video session exists with same ID
        video_session = backend.get_video_session(session.id)
        assert video_session is not None
        assert video_session.id == session.id
        assert video_session.room_id == "room-1"
        assert video_session.state == VideoSessionState.ACTIVE

    async def test_disconnect_cleans_both_sessions(self):
        backend = FastRTCVideoBackend()
        session = await backend.connect("room-1", "user-1", "voice")

        await backend.disconnect(session)

        assert backend.get_session(session.id) is None
        assert backend.get_video_session(session.id) is None

    async def test_close_clears_all(self):
        backend = FastRTCVideoBackend()
        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        await backend.close()

        assert backend.get_session(s1.id) is None
        assert backend.get_video_session(s1.id) is None
        assert backend.get_session(s2.id) is None
        assert backend.get_video_session(s2.id) is None

    async def test_list_video_sessions(self):
        backend = FastRTCVideoBackend()
        await backend.connect("room-1", "user-1", "voice")
        await backend.connect("room-1", "user-2", "voice")
        await backend.connect("room-2", "user-3", "voice")

        room1 = backend.list_video_sessions("room-1")
        assert len(room1) == 2

        room2 = backend.list_video_sessions("room-2")
        assert len(room2) == 1

    async def test_name_property(self):
        backend = FastRTCVideoBackend()
        assert backend.name == "FastRTC-AV"


class TestFastRTCVideoBackendCallbacks:
    """Callback registration and video frame dispatch."""

    async def test_on_video_received_callback(self):
        backend = FastRTCVideoBackend()
        session = await backend.connect("room-1", "user-1", "voice")
        session.metadata["websocket_id"] = "test-ws-id"

        received = []
        backend.on_video_received(lambda s, f: received.append((s, f)))

        # Simulate video frame
        video_data = np.zeros((480, 640, 3), dtype=np.uint8)
        backend._handle_video_frame("test-ws-id", video_data, 640, 480)

        assert len(received) == 1
        video_session, frame = received[0]
        assert video_session.id == session.id
        assert frame.codec == "raw_rgb24"
        assert frame.width == 640
        assert frame.height == 480
        assert frame.sequence == 0

    async def test_video_tap(self):
        backend = FastRTCVideoBackend()
        session = await backend.connect("room-1", "user-1", "voice")
        session.metadata["websocket_id"] = "test-ws-id"

        tapped = []
        backend.add_video_tap(lambda s, f: tapped.append((s, f)))

        video_data = np.zeros((480, 640, 3), dtype=np.uint8)
        backend._handle_video_frame("test-ws-id", video_data, 640, 480)

        assert len(tapped) == 1

    async def test_frame_sequence_increments(self):
        backend = FastRTCVideoBackend()
        session = await backend.connect("room-1", "user-1", "voice")
        session.metadata["websocket_id"] = "test-ws-id"

        received = []
        backend.on_video_received(lambda s, f: received.append(f))

        video_data = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            backend._handle_video_frame("test-ws-id", video_data, 640, 480)

        assert [f.sequence for f in received] == [0, 1, 2]
        # First frame is keyframe
        assert received[0].keyframe is True
        assert received[1].keyframe is False

    async def test_no_callback_no_crash(self):
        backend = FastRTCVideoBackend()
        session = await backend.connect("room-1", "user-1", "voice")
        session.metadata["websocket_id"] = "test-ws-id"

        # No callbacks registered — should not crash
        video_data = np.zeros((480, 640, 3), dtype=np.uint8)
        backend._handle_video_frame("test-ws-id", video_data, 640, 480)

    async def test_session_ready_callback_on_connect(self):
        backend = FastRTCVideoBackend()
        ready_sessions = []
        backend.on_session_ready(lambda s: ready_sessions.append(s))

        await backend.connect("room-1", "user-1", "voice")

        assert len(ready_sessions) == 1
        assert ready_sessions[0].room_id == "room-1"

    async def test_disconnect_callback(self):
        backend = FastRTCVideoBackend()
        disconnected = []
        backend.on_client_disconnected(lambda s: disconnected.append(s))

        session = await backend.connect("room-1", "user-1", "voice")
        await backend.disconnect(session)

        assert len(disconnected) == 1
        assert disconnected[0].state == VideoSessionState.ENDED


class TestFastRTCVideoBackendWebRTC:
    """WebRTC-specific registration and emit queues."""

    async def test_register_webrtc_creates_video_queue(self):
        backend = FastRTCVideoBackend()
        session = await backend.connect("room-1", "user-1", "voice")

        backend._register_webrtc("webrtc-id-1", session.id)

        assert "webrtc-id-1" in backend._emit_queues
        assert "webrtc-id-1" in backend._video_emit_queues

    async def test_disconnect_clears_video_queue(self):
        backend = FastRTCVideoBackend()
        session = await backend.connect("room-1", "user-1", "voice")
        backend._register_webrtc("webrtc-id-1", session.id)

        await backend.disconnect(session)

        assert "webrtc-id-1" not in backend._video_emit_queues

    async def test_send_video_queues_data(self):
        backend = FastRTCVideoBackend(video_width=4, video_height=2)
        session = await backend.connect("room-1", "user-1", "voice")
        backend._register_webrtc("webrtc-id-1", session.id)

        video_session = backend.get_video_session(session.id)
        assert video_session is not None

        # 4x2 RGB = 24 bytes
        await backend.send_video(video_session, b"\x00" * 24)

        queue = backend._video_emit_queues["webrtc-id-1"]
        assert not queue.empty()
        frame = queue.get_nowait()
        assert frame.shape == (2, 4, 3)

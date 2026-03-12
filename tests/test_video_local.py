"""Tests for LocalVideoBackend (requires opencv-python-headless)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")

from roomkit.video.backends.local import LocalVideoBackend  # noqa: E402
from roomkit.video.base import VideoSessionState  # noqa: E402
from roomkit.video.video_frame import VideoFrame  # noqa: E402


class TestLocalVideoBackendConstruction:
    def test_creates_with_defaults(self) -> None:
        backend = LocalVideoBackend()
        assert backend.name == "LocalVideoBackend"
        assert backend._fps == 30
        assert backend._width == 640
        assert backend._height == 480
        assert backend._device == 0

    def test_custom_params(self) -> None:
        backend = LocalVideoBackend(device=1, fps=15, width=1280, height=720)
        assert backend._device == 1
        assert backend._fps == 15
        assert backend._width == 1280
        assert backend._height == 720


class TestLocalVideoBackendSession:
    async def test_connect_creates_session(self) -> None:
        backend = LocalVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")
        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert session.state == VideoSessionState.ACTIVE
        assert session.metadata["device"] == 0
        assert session.metadata["fps"] == 30

    async def test_get_session(self) -> None:
        backend = LocalVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")
        assert backend.get_session(session.id) is session
        assert backend.get_session("nonexistent") is None

    async def test_list_sessions(self) -> None:
        backend = LocalVideoBackend()
        await backend.connect("room-1", "user-1", "video-1")
        await backend.connect("room-1", "user-2", "video-1")
        await backend.connect("room-2", "user-3", "video-1")
        assert len(backend.list_sessions("room-1")) == 2
        assert len(backend.list_sessions("room-2")) == 1

    async def test_disconnect_sets_ended(self) -> None:
        backend = LocalVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.disconnect(session)
        stored = backend.get_session(session.id)
        assert stored is not None
        assert stored.state == VideoSessionState.ENDED

    async def test_close_clears_sessions(self) -> None:
        backend = LocalVideoBackend()
        await backend.connect("room-1", "user-1", "video-1")
        await backend.close()
        assert backend.list_sessions("room-1") == []


class TestLocalVideoBackendCapture:
    async def test_start_capture_with_mock_camera(self) -> None:
        """Test capture with a mocked cv2.VideoCapture."""
        import numpy as np

        backend = LocalVideoBackend(fps=10, width=320, height=240)
        session = await backend.connect("room-1", "user-1", "video-1")

        received_frames: list[VideoFrame] = []

        def on_frame(sess, frame):
            received_frames.append(frame)

        backend.on_video_received(on_frame)

        # Mock VideoCapture
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 320,
            cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 10.0,
        }.get(prop, 0)
        mock_cap.read.return_value = (True, fake_frame)

        with (
            patch.object(backend._cv2, "VideoCapture", return_value=mock_cap),
            patch.object(backend._cv2, "cvtColor", return_value=fake_frame),
        ):
            await backend.start_capture(session)
            # Wait for a few frames
            await asyncio.sleep(0.3)
            await backend.stop_capture(session)

        assert len(received_frames) > 0
        frame = received_frames[0]
        assert frame.codec == "raw_rgb24"
        assert frame.width == 320
        assert frame.height == 240  # actual from mock get()
        assert frame.sequence == 0
        assert frame.keyframe is True

    async def test_stop_capture_releases_camera(self) -> None:
        backend = LocalVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (False, None)  # Immediately fail reads

        with patch.object(backend._cv2, "VideoCapture", return_value=mock_cap):
            await backend.start_capture(session)
            await asyncio.sleep(0.1)
            await backend.stop_capture(session)

        mock_cap.release.assert_called_once()

    async def test_session_ready_callback_fires(self) -> None:
        backend = LocalVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")

        ready_sessions = []
        backend.on_session_ready(lambda s: ready_sessions.append(s))

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (False, None)

        with patch.object(backend._cv2, "VideoCapture", return_value=mock_cap):
            await backend.start_capture(session)
            await asyncio.sleep(0.05)
            await backend.stop_capture(session)

        assert len(ready_sessions) == 1
        assert ready_sessions[0].id == session.id

    async def test_camera_not_opened_raises(self) -> None:
        backend = LocalVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with (
            patch.object(backend._cv2, "VideoCapture", return_value=mock_cap),
            pytest.raises(RuntimeError, match="Cannot open camera"),
        ):
            await backend.start_capture(session)

    async def test_double_start_is_noop(self) -> None:
        backend = LocalVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (False, None)

        with patch.object(backend._cv2, "VideoCapture", return_value=mock_cap):
            await backend.start_capture(session)
            await backend.start_capture(session)  # no-op
            await backend.stop_capture(session)


class TestLocalVideoBackendExport:
    def test_lazy_getter(self) -> None:
        from roomkit.video import get_local_video_backend

        cls = get_local_video_backend()
        assert cls is LocalVideoBackend

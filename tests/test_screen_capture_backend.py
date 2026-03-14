"""Tests for ScreenCaptureBackend."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from roomkit.video.base import VideoCapability, VideoSession, VideoSessionState
from roomkit.video.video_frame import VideoFrame

# ---------------------------------------------------------------------------
# Mock mss module
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 1920, 1080
PIXEL_COUNT = WIDTH * HEIGHT
RGB_BYTES = bytes([0xFF, 0x00, 0x00] * PIXEL_COUNT)  # All red


def _make_screenshot(rgb: bytes = RGB_BYTES, w: int = WIDTH, h: int = HEIGHT) -> MagicMock:
    shot = MagicMock()
    shot.rgb = rgb
    shot.width = w
    shot.height = h
    return shot


def _mock_mss_module(screenshot: MagicMock | None = None) -> MagicMock:
    """Build a mock mss module."""
    mock = MagicMock()
    sct = MagicMock()
    sct.monitors = [
        {"left": 0, "top": 0, "width": 3840, "height": 2160},  # 0 = all
        {"left": 0, "top": 0, "width": WIDTH, "height": HEIGHT},  # 1 = primary
        {"left": 1920, "top": 0, "width": 1920, "height": 1080},  # 2 = secondary
    ]
    sct.grab.return_value = screenshot or _make_screenshot()
    mock.mss.return_value.__enter__ = MagicMock(return_value=sct)
    mock.mss.return_value.__exit__ = MagicMock(return_value=False)
    return mock


@pytest.fixture
def mock_mss():
    return _mock_mss_module()


@pytest.fixture
def backend(mock_mss):
    with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
        from roomkit.video.backends.screen import ScreenCaptureBackend

        return ScreenCaptureBackend(monitor=1, fps=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScreenCaptureBackendProperties:
    def test_name(self, backend):
        assert backend.name == "ScreenCaptureBackend"

    def test_capabilities(self, backend):
        assert backend.capabilities == VideoCapability.SCREEN_SHARE


class TestScreenCaptureBackendConfig:
    def test_invalid_fps(self, mock_mss):
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            with pytest.raises(ValueError, match="fps must be >= 1"):
                ScreenCaptureBackend(fps=0)

    def test_invalid_scale_zero(self, mock_mss):
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            with pytest.raises(ValueError, match="scale must be in"):
                ScreenCaptureBackend(scale=0.0)

    def test_invalid_scale_over(self, mock_mss):
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            with pytest.raises(ValueError, match="scale must be in"):
                ScreenCaptureBackend(scale=1.5)


class TestScreenCaptureBackendConnect:
    async def test_connect_creates_session(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")

        assert isinstance(session, VideoSession)
        assert session.state == VideoSessionState.ACTIVE
        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert session.metadata["monitor"] == 1
        assert session.metadata["backend"] == "ScreenCaptureBackend"

    async def test_connect_with_metadata(self, backend):
        session = await backend.connect(
            "room-1", "user-1", "video-1", metadata={"custom": "value"}
        )
        assert session.metadata["custom"] == "value"
        assert session.metadata["monitor"] == 1

    async def test_get_session(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")
        assert backend.get_session(session.id) is session
        assert backend.get_session("nonexistent") is None

    async def test_list_sessions(self, backend):
        await backend.connect("room-1", "user-1", "video-1")
        await backend.connect("room-2", "user-2", "video-2")

        assert len(backend.list_sessions("room-1")) == 1
        assert len(backend.list_sessions("room-2")) == 1
        assert len(backend.list_sessions("room-3")) == 0


class TestScreenCaptureBackendDisconnect:
    async def test_disconnect_updates_state(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.disconnect(session)

        stored = backend.get_session(session.id)
        assert stored is not None
        assert stored.state == VideoSessionState.ENDED

    async def test_disconnect_fires_callback(self, backend):
        disconnected: list[VideoSession] = []
        backend.on_client_disconnected(lambda s: disconnected.append(s))

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.disconnect(session)

        assert len(disconnected) == 1

    async def test_close_cleans_all(self, backend):
        await backend.connect("room-1", "user-1", "video-1")
        await backend.connect("room-2", "user-2", "video-2")

        await backend.close()

        assert len(backend.list_sessions("room-1")) == 0
        assert len(backend.list_sessions("room-2")) == 0


class TestScreenCaptureBackendSendVideo:
    async def test_send_video_noop(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.send_video(session, b"\x00")  # no error


class TestScreenCaptureBackendCallbacks:
    async def test_session_ready_callback(self, backend):
        ready: list[VideoSession] = []
        backend.on_session_ready(lambda s: ready.append(s))

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)

        # Let capture thread start
        await asyncio.sleep(0.05)
        await backend.stop_capture(session)

        assert len(ready) == 1
        assert ready[0].id == session.id

    async def test_video_received_callback(self, backend):
        """Capture thread delivers frames via on_video_received."""
        received: list[VideoFrame] = []
        backend.on_video_received(lambda _s, f: received.append(f))

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)

        # Let a few frames arrive
        await asyncio.sleep(0.3)
        await backend.stop_capture(session)

        assert len(received) >= 1
        frame = received[0]
        assert frame.codec == "raw_rgb24"
        assert frame.width == WIDTH
        assert frame.height == HEIGHT
        assert frame.keyframe is True
        assert frame.sequence == 0

    async def test_duplicate_start_capture_warns(self, backend):
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await backend.start_capture(session)  # should warn, not crash
        await backend.stop_capture(session)


class TestScreenCaptureBackendRegion:
    async def test_region_override(self, mock_mss):
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(region=(100, 200, 800, 600), fps=5)

        received: list[VideoFrame] = []
        backend.on_video_received(lambda _s, f: received.append(f))

        # Update mock screenshot dimensions for the region
        sct = mock_mss.mss.return_value.__enter__.return_value
        region_rgb = bytes([0x00, 0xFF, 0x00] * (800 * 600))
        sct.grab.return_value = _make_screenshot(region_rgb, 800, 600)

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await asyncio.sleep(0.3)
        await backend.stop_capture(session)

        assert len(received) >= 1
        # _resolve_monitor should have built the region dict
        grab_arg = sct.grab.call_args[0][0]
        assert grab_arg == {"left": 100, "top": 200, "width": 800, "height": 600}


class TestScreenCaptureBackendDiffSkipping:
    async def test_identical_frames_skipped(self, mock_mss):
        """With diff_threshold > 0, identical frames after the first are skipped."""
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(fps=30, diff_threshold=0.01)

        received: list[VideoFrame] = []
        backend.on_video_received(lambda _s, f: received.append(f))

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await asyncio.sleep(0.2)
        await backend.stop_capture(session)

        # With identical screenshots, only the first frame should be delivered
        assert len(received) == 1

    async def test_different_frames_not_skipped(self, mock_mss):
        """When screenshots change significantly, frames are not skipped."""
        call_count = 0

        def alternating_grab(_monitor):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return _make_screenshot(bytes([0x00, 0x00, 0xFF] * PIXEL_COUNT))
            return _make_screenshot(RGB_BYTES)

        sct = mock_mss.mss.return_value.__enter__.return_value
        sct.grab.side_effect = alternating_grab

        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(fps=30, diff_threshold=0.01)

        received: list[VideoFrame] = []
        backend.on_video_received(lambda _s, f: received.append(f))

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await asyncio.sleep(0.2)
        await backend.stop_capture(session)

        # Alternating frames should all be delivered
        assert len(received) >= 2

    async def test_no_diff_threshold_delivers_all(self, mock_mss):
        """With diff_threshold=0 (default), all frames are delivered."""
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(fps=30, diff_threshold=0.0)

        received: list[VideoFrame] = []
        backend.on_video_received(lambda _s, f: received.append(f))

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await asyncio.sleep(0.2)
        await backend.stop_capture(session)

        # Even identical frames should all be delivered
        assert len(received) >= 2


class TestScreenCaptureBackendMonitorValidation:
    async def test_invalid_monitor_index_delivers_no_frames(self, mock_mss):
        """Invalid monitor index logs error and delivers no frames."""
        with patch("roomkit.video.backends.screen._import_mss", return_value=mock_mss):
            from roomkit.video.backends.screen import ScreenCaptureBackend

            backend = ScreenCaptureBackend(monitor=99, fps=5)

        received: list[VideoFrame] = []
        backend.on_video_received(lambda _s, f: received.append(f))

        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.start_capture(session)
        await asyncio.sleep(0.1)
        await backend.stop_capture(session)

        # Thread exits gracefully, no frames delivered
        assert len(received) == 0

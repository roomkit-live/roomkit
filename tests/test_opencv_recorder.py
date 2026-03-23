"""Tests for OpenCVVideoRecorder (video/recorder/opencv.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from roomkit.video.recorder.base import (
    VideoRecordingConfig,
    VideoRecordingHandle,
)
from roomkit.video.video_frame import VideoFrame

# -- Fake cv2 module ----------------------------------------------------------


class _FakeVideoWriter:
    """Fake cv2.VideoWriter for testing."""

    def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames: list[Any] = []
        self._opened = True

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def write(self, frame: Any) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self._opened = False


class _FakeCV2:
    """Fake cv2 module with essential constants and classes."""

    COLOR_RGB2BGR = 4

    def __init__(self, *, writer_opens: bool = True) -> None:
        self._writer_opens = writer_opens
        self.last_writer: _FakeVideoWriter | None = None

    def VideoWriter_fourcc(self, *args: str) -> int:  # noqa: N802
        return 0x7634706D  # 'mp4v'

    def VideoWriter(  # noqa: N802
        self,
        path: str,
        fourcc: int,
        fps: float,
        size: tuple[int, int],
    ) -> _FakeVideoWriter:
        writer = _FakeVideoWriter(path, fourcc, fps, size)
        writer._opened = self._writer_opens
        self.last_writer = writer
        return writer

    def cvtColor(self, src: Any, code: int) -> Any:  # noqa: N802
        return src  # Pass through for testing


# -- Helpers ------------------------------------------------------------------


def _make_session() -> Any:
    """Create a minimal VideoSession-like object."""

    @dataclass
    class FakeVideoSession:
        id: str = "session-abc123456789"
        room_id: str = "room-1"
        participant_id: str = "p-1"
        channel_id: str = "ch-1"

    return FakeVideoSession()


def _make_raw_frame(
    width: int = 4,
    height: int = 3,
    codec: str = "raw_rgb24",
) -> VideoFrame:
    """Create a minimal raw RGB VideoFrame."""
    data = bytes(width * height * 3)
    return VideoFrame(data=data, codec=codec, width=width, height=height)


# -- Tests: Construction and name -------------------------------------------


class TestOpenCVVideoRecorderInit:
    def test_name(self) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            assert recorder.name == "OpenCVVideoRecorder"

    def test_import_error_raised(self) -> None:
        """_import_cv2 should raise clear ImportError when cv2 is missing."""
        from roomkit.video.recorder.opencv import _import_cv2

        with (
            patch.dict("sys.modules", {"cv2": None}),
            patch("builtins.__import__", side_effect=ImportError("No cv2")),
            pytest.raises(ImportError, match="opencv-python-headless"),
        ):
            _import_cv2()


# -- Tests: Start recording --------------------------------------------------


class TestStartRecording:
    def test_start_returns_handle(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path), fps=30.0)

            handle = recorder.start(session, config)
            assert handle.session_id == session.id
            assert handle.path.endswith(".mp4")
            assert handle.id  # Non-empty

    def test_start_with_auto_codec_uses_mp4v(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path), codec="auto")

            handle = recorder.start(session, config)
            active = recorder._active[handle.id]
            assert active.codec == "mp4v"

    def test_start_with_custom_codec(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path), codec="XVID")

            handle = recorder.start(session, config)
            active = recorder._active[handle.id]
            assert active.codec == "XVID"

    def test_start_creates_lazy_writer(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path))

            handle = recorder.start(session, config)
            active = recorder._active[handle.id]
            # Writer should be None until first frame
            assert active.writer is None


# -- Tests: Stop recording ---------------------------------------------------


class TestStopRecording:
    def test_stop_unknown_handle_returns_empty_result(self) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            handle = VideoRecordingHandle(id="nonexistent", session_id="s1")

            result = recorder.stop(handle)
            assert result.id == "nonexistent"
            assert handle.state == "stopped"

    def test_stop_releases_writer(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path))

            handle = recorder.start(session, config)
            # Write a frame to create the writer
            recorder.tap_frame(handle, _make_raw_frame())

            writer = recorder._active[handle.id].writer
            result = recorder.stop(handle)

            assert handle.state == "stopped"
            assert writer is not None
            assert not writer._opened  # released
            assert result.frame_count == 1
            assert result.duration_seconds >= 0.0

    def test_stop_with_no_writer(self, tmp_path: Any) -> None:
        """Stop when writer was never created (no frames written)."""
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path))

            handle = recorder.start(session, config)
            result = recorder.stop(handle)

            assert result.frame_count == 0
            assert handle.state == "stopped"

    def test_stop_returns_correct_format(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path), format="avi")

            handle = recorder.start(session, config)
            result = recorder.stop(handle)
            assert result.format == "avi"


# -- Tests: Write frame -------------------------------------------------------


class TestTapFrame:
    def test_tap_frame_unknown_handle_does_nothing(self) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            handle = VideoRecordingHandle(id="nonexistent", session_id="s1")
            frame = _make_raw_frame()
            # Should not raise
            recorder.tap_frame(handle, frame)

    def test_tap_frame_creates_writer_on_first_frame(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path), fps=15.0)

            handle = recorder.start(session, config)
            assert recorder._active[handle.id].writer is None

            recorder.tap_frame(handle, _make_raw_frame())
            assert recorder._active[handle.id].writer is not None
            assert recorder._active[handle.id].frame_count == 1

    def test_tap_frame_rgb_converts_and_writes(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path))

            handle = recorder.start(session, config)
            recorder.tap_frame(handle, _make_raw_frame(codec="raw_rgb24"))
            recorder.tap_frame(handle, _make_raw_frame(codec="raw_rgb24"))

            assert recorder._active[handle.id].frame_count == 2

    def test_tap_frame_bgr_writes_directly(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path))

            handle = recorder.start(session, config)
            recorder.tap_frame(handle, _make_raw_frame(codec="raw_bgr24"))

            assert recorder._active[handle.id].frame_count == 1

    def test_tap_frame_encoded_codec_skipped(self, tmp_path: Any) -> None:
        """Encoded frames (h264, vp8, etc.) cannot be written by VideoWriter."""
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path))

            handle = recorder.start(session, config)
            # First write a raw frame to create the writer
            recorder.tap_frame(handle, _make_raw_frame(codec="raw_rgb24"))
            # Then try an encoded frame
            encoded = VideoFrame(data=b"\x00" * 100, codec="h264", width=4, height=3)
            recorder.tap_frame(handle, encoded)

            # Only the raw frame should be counted
            assert recorder._active[handle.id].frame_count == 1

    def test_tap_frame_writer_open_fails(self, tmp_path: Any) -> None:
        """When VideoWriter.isOpened() returns False, writer is set to None."""
        fake_cv2 = _FakeCV2(writer_opens=False)
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path))

            handle = recorder.start(session, config)
            recorder.tap_frame(handle, _make_raw_frame())

            # Writer should be set back to None since isOpened() returned False
            assert recorder._active[handle.id].writer is None
            assert recorder._active[handle.id].frame_count == 0


# -- Tests: Close -------------------------------------------------------------


class TestClose:
    def test_close_releases_all_writers(self, tmp_path: Any) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path))

            handle1 = recorder.start(session, config)
            recorder.tap_frame(handle1, _make_raw_frame())

            handle2 = recorder.start(session, config)
            recorder.tap_frame(handle2, _make_raw_frame())

            writers = [
                recorder._active[handle1.id].writer,
                recorder._active[handle2.id].writer,
            ]

            recorder.close()

            # All active recordings should be cleared
            assert len(recorder._active) == 0
            for w in writers:
                assert not w._opened  # released

    def test_close_with_no_active_recordings(self) -> None:
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            recorder.close()  # Should not raise
            assert len(recorder._active) == 0

    def test_close_with_lazy_writer_none(self, tmp_path: Any) -> None:
        """Close when a recording was started but no frame was written."""
        fake_cv2 = _FakeCV2()
        with patch("roomkit.video.recorder.opencv._import_cv2", return_value=fake_cv2):
            from roomkit.video.recorder.opencv import OpenCVVideoRecorder

            recorder = OpenCVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=str(tmp_path))

            recorder.start(session, config)
            recorder.close()  # Should not raise
            assert len(recorder._active) == 0

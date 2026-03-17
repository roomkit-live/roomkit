"""Tests for PyAV video recorder — H.264 encoding via FFmpeg."""

from __future__ import annotations

import os
import tempfile

import pytest

av = pytest.importorskip("av", reason="av (PyAV) not installed")
np = pytest.importorskip("numpy", reason="numpy not installed")

from roomkit.video.base import VideoSession  # noqa: E402
from roomkit.video.recorder.base import VideoRecordingConfig, safe_filename  # noqa: E402
from roomkit.video.recorder.pyav import PyAVVideoRecorder, _pick_codec  # noqa: E402
from roomkit.video.video_frame import VideoFrame  # noqa: E402


def _make_session(sid: str = "sess-1") -> VideoSession:
    return VideoSession(id=sid, room_id="r1", participant_id="u1", channel_id="v1")


def _make_raw_frame(
    width: int = 320,
    height: int = 240,
    codec: str = "raw_rgb24",
) -> VideoFrame:
    """Create a raw RGB/BGR frame with random pixel data."""
    data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()
    return VideoFrame(data=data, codec=codec, width=width, height=height)


class TestSafeFilename:
    def test_alphanumeric_unchanged(self) -> None:
        assert safe_filename("abc123") == "abc123"

    def test_special_chars_replaced(self) -> None:
        assert safe_filename("a/b:c") == "a_b_c"

    def test_hyphens_preserved(self) -> None:
        assert safe_filename("my-session") == "my-session"


class TestPickCodec:
    def test_explicit_codec(self) -> None:
        config = VideoRecordingConfig(codec="libx264")
        assert _pick_codec(config, av) == "libx264"

    def test_auto_resolves(self) -> None:
        config = VideoRecordingConfig(codec="auto")
        result = _pick_codec(config, av)
        assert result in ("libx264", "h264_nvenc")


class TestPyAVVideoRecorder:
    def test_name(self) -> None:
        recorder = PyAVVideoRecorder()
        assert recorder.name == "PyAVVideoRecorder"

    def test_start_creates_handle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            real_tmpdir = os.path.realpath(tmpdir)
            recorder = PyAVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=tmpdir, codec="libx264")

            handle = recorder.start(session, config)
            assert handle.state == "recording"
            assert handle.session_id == "sess-1"
            assert handle.path.startswith(real_tmpdir)
            assert handle.path.endswith(".mp4")

            recorder.close()

    def test_start_stop_empty(self) -> None:
        """Start and stop with no frames produces a valid result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = PyAVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=tmpdir, codec="libx264")

            handle = recorder.start(session, config)
            result = recorder.stop(handle)

            assert handle.state == "stopped"
            assert result.id == handle.id
            assert result.frame_count == 0

    def test_record_rgb_frames(self) -> None:
        """Record raw RGB frames to H.264 MP4."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = PyAVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(
                storage=tmpdir,
                codec="libx264",
                fps=10.0,
            )

            handle = recorder.start(session, config)

            for _ in range(10):
                recorder.tap_frame(handle, _make_raw_frame(codec="raw_rgb24"))

            result = recorder.stop(handle)

            assert result.frame_count == 10
            assert result.format == "mp4"
            assert result.size_bytes > 0
            assert os.path.exists(result.url)

    def test_record_bgr_frames(self) -> None:
        """Record raw BGR frames to H.264 MP4."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = PyAVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=tmpdir, codec="libx264")

            handle = recorder.start(session, config)
            for _ in range(5):
                recorder.tap_frame(handle, _make_raw_frame(codec="raw_bgr24"))
            result = recorder.stop(handle)

            assert result.frame_count == 5
            assert result.size_bytes > 0

    def test_encoded_frames_skipped(self) -> None:
        """Encoded frames (h264) are skipped — only raw frames are written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = PyAVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(storage=tmpdir, codec="libx264")

            handle = recorder.start(session, config)
            encoded = VideoFrame(data=b"\x00" * 100, codec="h264")
            recorder.tap_frame(handle, encoded)
            result = recorder.stop(handle)

            assert result.frame_count == 0

    def test_tap_frame_unknown_handle(self) -> None:
        """tap_frame with unknown handle is a no-op."""
        from roomkit.video.recorder.base import VideoRecordingHandle

        recorder = PyAVVideoRecorder()
        handle = VideoRecordingHandle(id="nonexistent", session_id="x")
        recorder.tap_frame(handle, _make_raw_frame())
        assert recorder._active == {}

    def test_tap_frame_after_stop_is_noop(self) -> None:
        """Late-arriving frame after stop must not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = PyAVVideoRecorder()
            config = VideoRecordingConfig(storage=tmpdir, codec="libx264")
            handle = recorder.start(_make_session(), config)
            result = recorder.stop(handle)
            assert result.frame_count == 0
            # Frame arrives after stop — should be silently ignored
            recorder.tap_frame(handle, _make_raw_frame())
            assert recorder._active == {}

    def test_close_flushes_active(self) -> None:
        """close() flushes and finalizes all active recordings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = PyAVVideoRecorder()
            config = VideoRecordingConfig(storage=tmpdir, codec="libx264")

            h1 = recorder.start(_make_session("s1"), config)
            h2 = recorder.start(_make_session("s2"), config)

            recorder.tap_frame(h1, _make_raw_frame())
            recorder.tap_frame(h2, _make_raw_frame())

            recorder.close()

            assert os.path.exists(h1.path)
            assert os.path.exists(h2.path)
            assert os.path.getsize(h1.path) > 0
            assert os.path.getsize(h2.path) > 0

    def test_path_traversal_rejected(self) -> None:
        """Storage path with '..' components is rejected."""
        recorder = PyAVVideoRecorder()
        session = _make_session()
        # A path where normpath still contains '..' (relative traversal)
        config = VideoRecordingConfig(storage="recordings/../../etc")
        with pytest.raises(ValueError, match="must not contain"):
            recorder.start(session, config)

    def test_mp4v_format_normalized(self) -> None:
        """Legacy 'mp4v' format is normalized to 'mp4'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = PyAVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(
                storage=tmpdir,
                format="mp4v",
                codec="libx264",
            )

            handle = recorder.start(session, config)
            assert handle.path.endswith(".mp4")
            recorder.close()

    def test_default_storage_uses_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no storage path, defaults to ./recordings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_tmpdir = os.path.realpath(tmpdir)
            monkeypatch.chdir(tmpdir)
            recorder = PyAVVideoRecorder()
            session = _make_session()
            config = VideoRecordingConfig(codec="libx264")

            handle = recorder.start(session, config)
            expected_dir = os.path.join(real_tmpdir, "recordings")
            assert handle.path.startswith(expected_dir)
            recorder.close()

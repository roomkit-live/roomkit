"""Video recording providers."""

from __future__ import annotations

from roomkit.video.recorder.base import (
    VideoRecorder,
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoRecordingResult,
)
from roomkit.video.recorder.mock import MockVideoRecorder


def get_opencv_recorder() -> type:
    """Lazy loader for OpenCVVideoRecorder (requires opencv-python-headless)."""
    from roomkit.video.recorder.opencv import OpenCVVideoRecorder

    return OpenCVVideoRecorder


def get_pyav_recorder() -> type:
    """Lazy loader for PyAVVideoRecorder (requires av)."""
    from roomkit.video.recorder.pyav import PyAVVideoRecorder

    return PyAVVideoRecorder


__all__ = [
    "MockVideoRecorder",
    "VideoRecorder",
    "VideoRecordingConfig",
    "VideoRecordingHandle",
    "VideoRecordingResult",
    "get_opencv_recorder",
    "get_pyav_recorder",
]

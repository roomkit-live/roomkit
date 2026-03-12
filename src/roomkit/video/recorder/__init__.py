"""Video recording providers."""

from __future__ import annotations

from roomkit.video.recorder.base import (
    VideoRecorder,
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoRecordingResult,
)
from roomkit.video.recorder.mock import MockVideoRecorder

__all__ = [
    "MockVideoRecorder",
    "VideoRecorder",
    "VideoRecordingConfig",
    "VideoRecordingHandle",
    "VideoRecordingResult",
]

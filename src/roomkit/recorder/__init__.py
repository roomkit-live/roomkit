"""Room-level media recording package."""

from roomkit.recorder.base import (
    ChannelRecordingConfig,
    MediaRecorder,
    MediaRecordingConfig,
    MediaRecordingHandle,
    MediaRecordingResult,
    RecordingTrack,
    RoomRecorderBinding,
)
from roomkit.recorder.mock import MockMediaRecorder

__all__ = [
    "ChannelRecordingConfig",
    "MediaRecorder",
    "MediaRecordingConfig",
    "MediaRecordingHandle",
    "MediaRecordingResult",
    "MockMediaRecorder",
    "RecordingTrack",
    "RoomRecorderBinding",
    "get_pyav_media_recorder",
]


def get_pyav_media_recorder() -> type:
    """Lazy-load ``PyAVMediaRecorder`` to avoid hard PyAV dependency."""
    from roomkit.recorder.pyav import PyAVMediaRecorder

    return PyAVMediaRecorder

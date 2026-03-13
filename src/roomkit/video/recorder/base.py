"""Video recorder ABC and related data types."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame


def safe_filename(session_id: str) -> str:
    """Sanitize session ID for use in filenames."""
    return re.sub(r"[^\w\-]", "_", session_id)


def validate_storage_path(storage: str) -> str:
    """Validate and resolve a storage directory path.

    Rejects paths containing '..' components to prevent traversal.
    Creates the directory if it doesn't exist. Returns the resolved path.
    """
    # Check raw parts before normpath collapses them
    import pathlib

    if ".." in pathlib.PurePath(storage).parts:
        raise ValueError(f"Storage path must not contain '..': {storage}")
    resolved = os.path.normpath(storage)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def build_recording_path(
    session: VideoSession,
    config: VideoRecordingConfig,
    *,
    format_override: str | None = None,
) -> str:
    """Build and validate the output file path for a recording.

    Returns the full resolved path to the recording file.
    """
    safe_id = safe_filename(session.id[:16])
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    fmt = format_override or config.format
    storage = config.storage or os.path.join(os.getcwd(), "recordings")
    resolved = validate_storage_path(storage)
    return os.path.join(resolved, f"{safe_id}_{ts}.{fmt}")


@dataclass
class VideoRecordingConfig:
    """Configuration for video recording.

    Attributes:
        storage: Directory path for recording files.
        format: Output container format (``mp4``, ``avi``, ``mkv``).
        codec: Video codec. For PyAV: ``auto`` (NVENC if available,
            else libx264), ``libx264``, ``h264_nvenc``, ``libx265``.
            For OpenCV: ``mp4v``, ``XVID``.
        fps: Output frame rate.
        metadata: Provider-specific extra configuration.
    """

    storage: str = ""
    format: str = "mp4"
    codec: str = "auto"
    fps: float = 15.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class VideoRecordingHandle:
    """Handle to an active video recording."""

    id: str
    session_id: str
    state: str = "recording"
    started_at: datetime | None = None
    path: str = ""


@dataclass
class VideoRecordingResult:
    """Result returned when a video recording is stopped."""

    id: str
    url: str = ""
    duration_seconds: float = 0.0
    frame_count: int = 0
    format: str = "mp4"
    size_bytes: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


class VideoRecorder(ABC):
    """Abstract base class for video recording providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    def start(self, session: VideoSession, config: VideoRecordingConfig) -> VideoRecordingHandle:
        """Start recording a video session."""
        ...

    @abstractmethod
    def stop(self, handle: VideoRecordingHandle) -> VideoRecordingResult:
        """Stop an active recording and finalize the file."""
        ...

    @abstractmethod
    def tap_frame(self, handle: VideoRecordingHandle, frame: VideoFrame) -> None:
        """Feed a video frame to the recorder."""
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""

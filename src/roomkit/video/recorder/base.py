"""Video recorder ABC and related data types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame


@dataclass
class VideoRecordingConfig:
    """Configuration for video recording.

    Attributes:
        storage: Directory path for recording files.
        format: Output container format (``mp4``, ``avi``, ``mkv``).
        codec: Video codec (``mp4v``, ``XVID``, ``avc1``).
        fps: Output frame rate. If 0, uses the frame timestamps.
        metadata: Provider-specific extra configuration.
    """

    storage: str = ""
    format: str = "mp4"
    codec: str = "mp4v"
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

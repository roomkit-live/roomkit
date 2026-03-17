"""Media recorder ABC and related data types for room-level recording."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


def safe_filename(value: str) -> str:
    """Sanitize a string for use in filenames."""
    return re.sub(r"[^\w\-]", "_", value)


def validate_storage_path(storage: str) -> str:
    """Validate and resolve a storage directory path.

    Rejects paths containing '..' components to prevent traversal.
    Creates the directory if it doesn't exist. Returns the resolved path.
    """
    import pathlib

    if ".." in pathlib.PurePath(storage).parts:
        raise ValueError(f"Storage path must not contain '..': {storage}")
    resolved = str(pathlib.Path(storage).resolve())
    os.makedirs(resolved, exist_ok=True)
    return resolved


@dataclass
class RecordingTrack:
    """Describes a single media track within a room recording."""

    id: str
    kind: str  # "audio", "video", "screen_share"
    channel_id: str
    participant_id: str | None = None
    codec: str = ""
    sample_rate: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class ChannelRecordingConfig:
    """Per-channel recording preferences for room-level media recording.

    Controls which media types from a channel are fed to the room's
    :class:`MediaRecorder` instances.
    """

    audio: bool = False
    video: bool = False
    screen_share: bool = False
    per_participant: bool = True


@dataclass
class MediaRecordingConfig:
    """Configuration for a room-level media recording session."""

    storage: str = ""
    video_codec: str = "libx264"
    video_fps: int = 30
    audio_codec: str = "aac"
    audio_sample_rate: int = 16000
    format: str = "mp4"
    min_tracks: int = 1
    """Minimum tracks before encoding starts. Set to the expected
    total (e.g. 3 for 1 video + 2 audio) so encoding waits for all
    channels to connect. Required because MP4 cannot add streams
    after the first packet is muxed."""


@dataclass
class MediaRecordingHandle:
    """Handle to an active room-level recording."""

    id: str
    room_id: str
    state: str = "recording"
    started_at: datetime | None = None
    path: str = ""


@dataclass
class MediaRecordingResult:
    """Result returned when a room recording is stopped."""

    id: str
    url: str = ""
    duration_seconds: float = 0.0
    tracks: list[RecordingTrack] = field(default_factory=list)
    format: str = "mp4"
    size_bytes: int = 0


@dataclass
class RoomRecorderBinding:
    """Binds a :class:`MediaRecorder` to a room with configuration."""

    recorder: MediaRecorder
    config: MediaRecordingConfig
    enabled: bool = True
    name: str = ""


class MediaRecorder(ABC):
    """Abstract base class for room-level media recorders.

    A MediaRecorder receives audio and video data from one or more
    channels in a room and muxes them into a single output file.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        """Start a new recording session."""
        ...

    @abstractmethod
    def on_recording_stop(self, handle: MediaRecordingHandle) -> MediaRecordingResult:
        """Stop an active recording and finalize output."""
        ...

    @abstractmethod
    def on_track_added(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        """Register a new media track in the recording."""
        ...

    @abstractmethod
    def on_track_removed(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        """Remove a media track from the recording (flush encoder)."""
        ...

    @abstractmethod
    def on_data(
        self,
        handle: MediaRecordingHandle,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        """Feed media data for a specific track."""
        ...

    def close(self) -> None:  # noqa: B027
        """Release resources."""

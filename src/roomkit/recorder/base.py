"""Media recorder ABC and related data types for room-level recording."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

PCM_CODECS = {1: "pcm_s8", 2: "pcm_s16le", 4: "pcm_s32le"}
"""Codec name for a PCM sample width in bytes, as :class:`RecordingTrack` names it.

Signed little-endian throughout, which is the framework's own convention rather
than a reading of the bytes: the resamplers map widths to int8/int16/int32 and
:class:`~roomkit.voice.audio_frame.AudioFrame` carries no format field to say
otherwise. Four bytes is therefore ``pcm_s32le`` and never ``pcm_f32le`` — the
ambiguity is real, and it is settled here so that it is settled in one place. A
backend holding float samples converts them before handing them to the
framework, as the pipeline stages already do.
"""


def pcm_codec(sample_width: int) -> str:
    """Name the PCM codec of a sample width, for a track's declaration.

    Raises for a width the framework does not carry, rather than guessing: a
    recording declared in the wrong format is a file that opens and is wrong,
    which is the failure this mapping exists to prevent.
    """
    try:
        return PCM_CODECS[sample_width]
    except KeyError:
        raise ValueError(f"sample_width must be 1, 2 or 4 bytes, got {sample_width}") from None


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
    """Describes a single media track within a room recording.

    The description is how a recorder learns to interpret the bytes it is then
    handed: ``on_data`` carries none of it, and one PCM format is
    indistinguishable from another by inspection. A caller therefore declares
    what it will actually deliver — not what the framework normally carries —
    and does not then deliver something else (RFC section 12.11).
    """

    id: str
    kind: str  # "audio", "video", "screen_share"
    channel_id: str
    participant_id: str | None = None
    codec: str = ""
    sample_rate: int | None = None
    channels: int | None = None
    """Audio channel count. ``None`` leaves it to the recorder, which reads mono."""

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
    metadata: dict[str, Any] = field(default_factory=dict)
    """Caller-supplied metadata, carried to the recorder verbatim.

    What a caller wants filed with the recording — a matter id, a retention
    class — and the recorder's to interpret; the framework never reads it.
    """


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

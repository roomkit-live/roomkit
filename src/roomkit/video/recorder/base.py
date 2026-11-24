"""Video recorder ABC and related data types."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from roomkit.recorder.base import (
    safe_filename,
    validate_recording_format,
    validate_storage_path,
)
from roomkit.voice.pipeline.recorder.base import encrypt_finished_recording

if TYPE_CHECKING:
    from roomkit.video.base import VideoSession
    from roomkit.video.video_frame import VideoFrame
    from roomkit.voice.pipeline.recorder.base import RecordingEncryption


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
    fmt = validate_recording_format(format_override or config.format)
    storage = config.storage or os.path.join(os.getcwd(), "recordings")
    resolved = validate_storage_path(storage)
    return os.path.join(resolved, f"{safe_id}_{ts}.{fmt}")


def store_finished_recording(
    config: VideoRecordingConfig, path: str
) -> tuple[str, int, dict[str, object]]:
    """Store a finished recording file at rest as configured (RFC §17.6).

    Returns the stored file's path, its size and the result metadata. With
    ``encryption`` the file goes through the cipher: the path is the
    encrypted artifact's, and ``metadata["encryption"]`` the cipher's name. A
    file the cipher could not encrypt is deleted, and no path is returned.
    With ``storage_encrypted_at_rest`` and no ``encryption``, the file stays
    as written and the metadata says ``"storage"``. With neither, it stays as
    written, in the clear, and the metadata claims nothing. A file never created is reported
    at its path, with size 0.
    """
    if not os.path.exists(path):
        return path, 0, {}
    if config.encryption is None:
        claim: dict[str, object] = (
            {"encryption": "storage"} if config.storage_encrypted_at_rest else {}
        )
        return path, os.path.getsize(path), claim
    stored = encrypt_finished_recording(config.encryption, path)
    if stored is None:
        return "", 0, {}
    size = os.path.getsize(stored) if os.path.exists(stored) else 0
    return stored, size, {"encryption": config.encryption.name}


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
        encryption: Encryption applied to each finished recording file
            (RFC §17.6), the same
            :class:`~roomkit.voice.pipeline.recorder.base.RecordingEncryption`
            the audio and room recorders take. The result then names the
            encrypted artifact; a file the cipher cannot encrypt is deleted.
        storage_encrypted_at_rest: Whether the configured storage encrypts
            every byte at rest: an explicit deployment assertion, not
            encryption performed by RoomKit. Without ``encryption``, the
            file is stored as written and the result says ``"encryption":
            "storage"``; ``encryption``, when also set, still applies. With
            neither, the file is stored as written, unencrypted.
    """

    storage: str = ""
    format: str = "mp4"
    codec: str = "auto"
    fps: float = 15.0
    metadata: dict[str, object] = field(default_factory=dict)
    encryption: RecordingEncryption | None = None
    storage_encrypted_at_rest: bool = False


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

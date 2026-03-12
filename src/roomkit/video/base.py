"""Base models for video support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Flag, StrEnum, auto, unique
from typing import Any

from roomkit.video.video_frame import ENCODED_CODECS


@unique
class VideoSessionState(StrEnum):
    """State of a video session."""

    CONNECTING = "connecting"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class VideoCapability(Flag):
    """Capabilities a VideoBackend can support.

    Backends declare their capabilities via the ``capabilities`` property.
    This allows RoomKit to know which features are available and enables
    integrators to choose backends based on their needs.

    Example::

        class MyBackend(VideoBackend):
            @property
            def capabilities(self) -> VideoCapability:
                return (
                    VideoCapability.SIMULCAST
                    | VideoCapability.SCREEN_SHARE
                )
    """

    NONE = 0
    """No optional capabilities (default)."""

    SIMULCAST = auto()
    """Backend supports multiple resolution streams."""

    SVC = auto()
    """Backend supports Scalable Video Coding layers."""

    SCREEN_SHARE = auto()
    """Backend supports a separate screen-share track."""

    RECORDING = auto()
    """Backend supports server-side recording."""

    BANDWIDTH_ESTIMATION = auto()
    """Backend supports bandwidth estimation and adaptation."""


VALID_OUTBOUND_CODECS = ENCODED_CODECS


@dataclass
class VideoChunk:
    """A chunk of encoded video data for outbound streaming."""

    data: bytes
    codec: str = "h264"
    width: int = 640
    height: int = 480
    timestamp_ms: int | None = None
    keyframe: bool = False
    is_final: bool = False

    def __post_init__(self) -> None:
        if self.codec not in VALID_OUTBOUND_CODECS:
            raise ValueError(
                f"codec must be one of {sorted(VALID_OUTBOUND_CODECS)}, got {self.codec!r}"
            )


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass
class VideoSession:
    """Active video connection for a participant."""

    id: str
    room_id: str
    participant_id: str
    channel_id: str
    state: VideoSessionState = VideoSessionState.CONNECTING
    provider_session_id: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


# Type aliases for video callbacks
VideoReceivedCallback = Callable[["VideoSession", Any], Any]
"""Callback for raw video frames from the transport: (session, frame)."""

VideoSessionReadyCallback = Callable[["VideoSession"], Any]
"""Callback for when a session's video path becomes live."""

VideoDisconnectCallback = Callable[["VideoSession"], Any]
"""Callback for client disconnection."""

"""Session lifecycle events (channel-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import ChannelType

if TYPE_CHECKING:
    pass


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class SessionStartedEvent:
    """A session has started on any channel type.

    For voice channels this fires when the audio path is live and ready
    to send/receive (same dual-signal timing as the former
    ``ON_VOICE_SESSION_READY``).  For text-based transport channels it
    fires when the inbound pipeline auto-creates a new room for a first
    message.

    This is the safe point to send greetings or start telemetry.
    """

    room_id: str
    """The room this session belongs to."""

    channel_id: str
    """The channel that triggered the session."""

    channel_type: ChannelType
    """The type of channel (VOICE, SMS, WEBSOCKET, etc.)."""

    participant_id: str
    """The participant whose session started."""

    session: Any | None = None
    """The VoiceSession for voice channels, None for text channels."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the session started."""

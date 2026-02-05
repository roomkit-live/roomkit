"""Base models for realtime voice support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum, unique
from typing import Any


@unique
class RealtimeSessionState(StrEnum):
    """State of a realtime voice session."""

    CONNECTING = "connecting"
    ACTIVE = "active"
    ENDED = "ended"


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass
class RealtimeSession:
    """Active realtime voice session for a participant."""

    id: str
    room_id: str
    participant_id: str
    channel_id: str
    state: RealtimeSessionState = RealtimeSessionState.CONNECTING
    provider_session_id: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

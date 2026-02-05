"""Room model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.enums import RoomStatus


class RoomTimers(BaseModel):
    """Timer configuration for a room."""

    inactive_after_seconds: int | None = Field(default=None, ge=0)
    closed_after_seconds: int | None = Field(default=None, ge=0)
    last_activity_at: datetime | None = None


class Room(BaseModel):
    """A conversation room."""

    id: str
    organization_id: str | None = None
    status: RoomStatus = RoomStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None
    timers: RoomTimers = Field(default_factory=RoomTimers)
    metadata: dict[str, Any] = Field(default_factory=dict)
    event_count: int = Field(default=0, ge=0)
    latest_index: int = Field(default=0, ge=0)

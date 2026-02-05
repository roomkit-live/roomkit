"""Framework-level event model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class FrameworkEvent(BaseModel):
    """An event emitted by the framework for observability."""

    type: str
    room_id: str | None = None
    channel_id: str | None = None
    event_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(default_factory=dict)

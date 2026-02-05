"""Task and observation models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.enums import TaskStatus


class Task(BaseModel):
    """A task assigned within a room."""

    id: str
    room_id: str
    title: str
    description: str | None = None
    assigned_to: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """An observation produced by an intelligence channel."""

    id: str
    room_id: str
    channel_id: str
    content: str
    category: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

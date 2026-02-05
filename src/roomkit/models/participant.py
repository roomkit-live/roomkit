"""Participant model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.enums import (
    IdentificationStatus,
    ParticipantRole,
    ParticipantStatus,
)


class Participant(BaseModel):
    """A participant in a room conversation."""

    id: str
    room_id: str
    channel_id: str
    display_name: str | None = None
    role: ParticipantRole = ParticipantRole.MEMBER
    status: ParticipantStatus = ParticipantStatus.ACTIVE
    identification: IdentificationStatus = IdentificationStatus.PENDING
    identity_id: str | None = None
    candidates: list[str] | None = None
    connected_via: list[str] = Field(default_factory=list)
    external_id: str | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    joined_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

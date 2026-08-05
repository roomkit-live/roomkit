"""Participant model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

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
    # Every channel the room has reached this participant through, ``channel_id``
    # included and first (RFC §5.5). A participant is one record per (room, id),
    # so a second channel lands here rather than forking a second record; what
    # `channel_id` holds is the *primary* channel, which only a deliberate join
    # moves. Order is order of first sight.
    connected_via: list[str] = Field(default_factory=list)
    external_id: str | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    joined_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _include_primary_channel(self) -> Participant:
        """Keep the RFC §5.5 primary-channel invariant true at the boundary."""
        self.connected_via = list(dict.fromkeys([self.channel_id, *self.connected_via]))
        return self

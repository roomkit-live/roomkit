"""Models for human-in-the-loop pending input requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum, unique
from typing import Any

from roomkit.models.enums import ChannelType


def _utcnow() -> datetime:
    return datetime.now(UTC)


@unique
class PendingInputStatus(StrEnum):
    """Status of a pending human input request."""

    PENDING = "pending"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


@dataclass
class PendingInput:
    """A pending human input request.

    Mutable — transitions from ``PENDING`` to ``RESOLVED``/``REJECTED``/``TIMED_OUT``
    when the application calls :meth:`HumanInputHandler.resolve` or
    :meth:`HumanInputHandler.reject`.
    """

    pending_id: str
    tool_name: str
    arguments: dict[str, Any]
    room_id: str
    tool_call_id: str
    channel_id: str
    status: PendingInputStatus = PendingInputStatus.PENDING
    result: str | None = None
    reject_reason: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


@dataclass(frozen=True)
class PendingInputEvent:
    """Event fired through ON_USER_INPUT_REQUIRED hooks.

    Carries the pending request details so notification layers
    (WebSocket, REST, etc.) can inform the user.
    """

    pending_id: str
    """Handler-generated ID for resolving this request."""

    tool_name: str
    """Name of the tool that requires human input."""

    arguments: dict[str, Any]
    """Tool arguments (e.g. questions, options)."""

    room_id: str
    """Room where the tool call originated."""

    tool_call_id: str
    """Provider-assigned tool call ID."""

    channel_id: str
    """Channel that triggered the tool call."""

    channel_type: ChannelType
    """Type of the originating channel."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the pending request was created."""

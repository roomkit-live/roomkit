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
    detached: bool = False
    """No one will call ``wait()`` on this request — its creator frees it with
    :meth:`HumanInputHandler.release`. ``wait()`` owns the cleanup of every
    other request."""
    created_at: datetime = field(default_factory=_utcnow)
    # Belongs with the origin fields above; appended so adding it leaves the
    # positional order of every existing field untouched.
    actor_id: str | None = None
    """Participant whose turn raised this request, when the tool loop knew one.

    A request that names nobody is a request a notification layer has to
    broadcast, and an answer it cannot attribute. ``None`` when the turn had no
    author (a system injection, a webhook, a scheduled run) or when the creator
    runs its own tool loop and did not supply one. It names the turn without
    authenticating it — resolve it against the room's roster before treating it
    as a principal, as ``current_tool_actor_id()`` documents."""
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

    actor_id: str | None = None
    """Participant whose turn raised the request, when the tool loop knew one.

    What lets a notification layer ask *the person who asked* rather than
    everyone in the room. Appended rather than grouped with the origin fields
    above so existing positional construction keeps working. ``None`` when the
    turn had no author."""

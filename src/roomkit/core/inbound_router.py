"""Inbound room router — determines which room an inbound message belongs to."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from roomkit.models.enums import ChannelType, RoomStatus
from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


class InboundRoomRouter(ABC):
    """Route an inbound message to a room (or ``None`` for auto-create)."""

    @abstractmethod
    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
    ) -> str | None:
        """Return room_id for the message, or ``None`` to create a new room."""
        ...


class DefaultInboundRoomRouter(InboundRoomRouter):
    """Default router (RFC §10.4): by participant, then by a *single* binding.

    Returns ``None`` rather than choosing when the message could belong to
    more than one room. A new room is recoverable; a message delivered into
    someone else's conversation is not — it is stored there, broadcast to that
    room's channels, and read back as context by that room's agent.
    """

    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
    ) -> str | None:
        # Strategy 1 (RFC §10.4): the sender's own latest room on this channel
        # type. Tried first because it identifies the conversation, where a
        # binding only identifies the pipe.
        if participant_id:
            room = await self._store.find_latest_room(
                participant_id=participant_id,
                channel_type=str(channel_type),
                status=str(RoomStatus.ACTIVE),
            )
            if room:
                return room.id

        # Strategy 2: a channel dedicated to one conversation, before anyone
        # has spoken in it. Only when the channel is bound to exactly one
        # active room — a channel shared across rooms (delegation, or a room
        # re-created after its predecessor closed) makes this ambiguous, and
        # the framework creating a fresh room is the safe answer.
        candidates = await self._store.find_room_ids_by_channel(
            channel_id, status=str(RoomStatus.ACTIVE), limit=2
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.warning(
                "Channel %s is bound to %d active rooms — refusing to guess which one "
                "this message belongs to. Pass room_id explicitly, or install a custom "
                "InboundRoomRouter.",
                channel_id,
                len(candidates),
            )

        return None

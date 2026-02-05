"""Inbound room router — determines which room an inbound message belongs to."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from roomkit.models.enums import ChannelType, RoomStatus
from roomkit.store.base import ConversationStore


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
    """Default router: find room by channel binding, then by participant."""

    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
    ) -> str | None:
        # Strategy 1: Find room by channel binding (current behavior)
        room_id = await self._store.find_room_id_by_channel(
            channel_id, status=str(RoomStatus.ACTIVE)
        )
        if room_id is not None:
            return room_id

        # Strategy 2: Find room by participant if available
        if participant_id:
            room = await self._store.find_latest_room(
                participant_id=participant_id,
                channel_type=str(channel_type),
                status=str(RoomStatus.ACTIVE),
            )
            if room:
                return room.id

        return None

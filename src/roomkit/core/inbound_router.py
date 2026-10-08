"""Inbound room router — determines which room an inbound message belongs to."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from roomkit.models.enums import ChannelType, RoomStatus
from roomkit.models.store_filter import EventFilter
from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")

# Page size of the scan for another sender's message (RFC §10.4 step 3). The
# scan stops at the first one, so a room with any history answers on its
# first page; only a room holding nothing but the sender's own words is read
# to the end.
_HISTORY_PAGE = 200


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
        # Strategy 1 (RFC §10.4): the sender's own room. Tried first because it
        # identifies the conversation, where a binding of the channel to some
        # room only identifies the pipe.
        if participant_id:
            room_id = await self._senders_room(channel_id, channel_type, participant_id)
            if room_id is not None:
                return room_id

        # Strategy 2: a channel dedicated to one conversation. Only when the
        # channel is bound to exactly one active room — a channel shared across
        # rooms (delegation, or a room re-created after its predecessor closed)
        # makes this ambiguous, and the framework creating a fresh room is the
        # safe answer — and only when that room is open to this sender: on a
        # number shared by many correspondents, the one room bound to it is
        # the first correspondent's conversation.
        candidates = await self._store.find_room_ids_by_channel(
            channel_id, status=str(RoomStatus.ACTIVE), limit=2
        )
        if len(candidates) == 1:
            if await self._admits(candidates[0], channel_id, participant_id or None):
                return candidates[0]
            return None
        if len(candidates) > 1:
            logger.warning(
                "Channel %s is bound to %d active rooms — refusing to guess which one "
                "this message belongs to. Pass room_id explicitly, or install a custom "
                "InboundRoomRouter.",
                channel_id,
                len(candidates),
            )

        return None

    async def _senders_room(
        self, channel_id: str, channel_type: ChannelType, sender: str
    ) -> str | None:
        """The latest active room that is *sender*'s own (RFC §10.4 step 1).

        First the room whose binding of this very channel names the sender: a
        binding speaks for its own channel, so a correspondent of one number
        is not taken to that room when they write to another. Then the room
        where the sender is a participant, by channel type. ``find_latest_room``
        also matches a binding of another channel of that type, which is the
        cross-number move refused here, so a room it finds only that way is
        not the sender's.
        """
        active = str(RoomStatus.ACTIVE)
        room_id = await self._store.find_room_id_by_binding(channel_id, sender, status=active)
        if room_id is not None:
            return room_id
        room = await self._store.find_latest_room(
            participant_id=sender, channel_type=str(channel_type), status=active
        )
        if room is None or await self._store.get_participant(room.id, sender) is None:
            return None
        return room.id

    async def _admits(self, room_id: str, channel_id: str, sender: str | None) -> bool:
        """Whether the one room bound to *channel_id* is open to *sender* (RFC §10.4).

        A group binding is open to everyone. Otherwise the room is another
        correspondent's conversation as soon as anything names one: the
        binding, a participant who joined through the channel, or a message
        the room received on it.
        """
        binding = await self._store.get_binding(room_id, channel_id)
        if binding is None:
            return False
        if binding.group:
            return True
        if binding.participant_id is not None and binding.participant_id != sender:
            return False
        for participant in await self._store.list_participants(room_id):
            if participant.id != sender and channel_id in participant.connected_via:
                return False
        return not await _heard_from_other(self._store, room_id, channel_id, sender)


async def _heard_from_other(
    store: ConversationStore, room_id: str, channel_id: str, sender: str | None
) -> bool:
    """Whether the room received a message on *channel_id* naming another sender.

    Received means stored and not ``BLOCKED`` (the default page skips those).
    An event whose source names no participant is not a correspondent's: a
    channel that does not stamp its senders cannot tell one from another, and
    the binding and the participants speak for the room then.
    """
    received = EventFilter(source_channel_id=channel_id)
    offset = 0
    while True:
        page = await store.list_events(
            room_id, offset=offset, limit=_HISTORY_PAGE, event_filter=received
        )
        if any(e.source.participant_id not in (None, sender) for e in page):
            return True
        if len(page) < _HISTORY_PAGE:
            return False
        offset += len(page)

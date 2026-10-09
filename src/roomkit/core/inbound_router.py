"""Inbound room router — determines which room an inbound message belongs to."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.delivery import SYSTEM_SENDER_ID
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
    """Default router (RFC §10.4): the sender's own room, then a channel bound
    to a single conversation that is open to the sender, else ``None``.

    Returns ``None`` rather than choosing when the message could belong to
    more than one room. A new room is recoverable; a message delivered into
    someone else's conversation is not — it is stored there, broadcast to that
    room's channels, and read back as context by that room's agent.
    """

    def __init__(
        self,
        store: ConversationStore,
        *,
        recipient_of: Callable[[ChannelBinding], str | None] | None = None,
    ) -> None:
        self._store = store
        # The address a binding delivers to, normalized (the kit passes its
        # channels' Channel.recipient_address); it names the correspondent.
        self._recipient_of = recipient_of

    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
    ) -> str | None:
        # Step 1 (RFC §10.4): the sender's own room. Tried first because it
        # identifies the conversation, where a binding of the channel to some
        # room only identifies the pipe.
        if participant_id:
            room_id = await self._senders_room(channel_id, participant_id)
            if room_id is not None:
                return room_id

        # Step 3: a channel dedicated to one conversation. Only when the
        # channel is bound to exactly one active room — a channel shared across
        # rooms (delegation, a room re-created after its predecessor closed, a
        # number shared by many correspondents) makes this ambiguous, and the
        # framework creating a fresh room is the safe answer — and only when
        # that room is open to this sender: on a shared number, the one room
        # bound to it is the first correspondent's conversation.
        candidates = await self._store.find_room_ids_by_channel(
            channel_id, status=str(RoomStatus.ACTIVE), limit=2
        )
        if len(candidates) == 1:
            if await self._admits(candidates[0], channel_id, channel_type, participant_id or None):
                return candidates[0]
            return None
        if len(candidates) > 1:
            # Debug, not a warning: on a number shared by many correspondents,
            # several rooms bound to the channel is the ordinary state, and
            # every new correspondent passes through here.
            logger.debug(
                "Channel %s is bound to %d active rooms — refusing to guess which one "
                "this message belongs to. Pass room_id explicitly, or install a custom "
                "InboundRoomRouter.",
                channel_id,
                len(candidates),
            )

        return None

    async def _senders_room(self, channel_id: str, sender: str) -> str | None:
        """The latest active room that is *sender*'s own (RFC §10.4 step 1).

        First the room whose binding of this very channel names the sender: a
        binding speaks for its own channel, so a correspondent of one number
        is not taken to that room when they write to another. Then the room
        where the sender is a participant; the stores record which channels a
        participant reached, not their types, so that half matches whatever
        the channel type.
        """
        active = str(RoomStatus.ACTIVE)
        room_id = await self._store.find_room_id_by_binding(channel_id, sender, status=active)
        if room_id is not None:
            return room_id
        return await self._store.find_room_id_by_participant(sender, status=active)

    async def _admits(
        self, room_id: str, channel_id: str, channel_type: ChannelType, sender: str | None
    ) -> bool:
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
        if sender == SYSTEM_SENDER_ID:
            # The framework names the room it writes to, and nothing it writes
            # counts as a correspondent's: a sender borrowing the name, on a
            # channel whose senders choose their id, gets a room of its own.
            return False
        own = await sender_known_as(self._store, channel_type, sender)
        if not binding_admits(binding, own):
            return False
        recipient = self._recipient_of(binding) if self._recipient_of is not None else None
        if recipient is not None and recipient not in own:
            # The room delivers to someone else: it is their conversation.
            return False
        for participant in await self._store.list_participants(room_id):
            if channel_id in participant.connected_via and not own & {
                participant.id,
                participant.identity_id,
            }:
                return False
        return not await _heard_from_other(self._store, room_id, channel_id, own)


async def sender_known_as(
    store: ConversationStore, channel_type: ChannelType, sender: str | None
) -> frozenset[str]:
    """The ids a room may know *sender* by: the address, and its identity.

    Identity resolution names a participant and stamps an event with the
    identity's id, not the address (RFC §11), so the identity the store
    resolves the address to is the sender's own too. The router is not given
    the caller's organization, so only an unscoped registration resolves
    here (§17.2).
    """
    if sender is None:
        return frozenset()
    identity = await store.resolve_identity(str(channel_type), sender)
    return frozenset({sender} if identity is None else {sender, identity.id})


def recordable_sender(sender: str | None) -> bool:
    """Whether *sender* can be a binding's correspondent (RFC §10.4).

    Not an empty address, and not the framework's own sender: what the host
    writes through ``deliver()`` names no correspondent.
    """
    return bool(sender) and sender != SYSTEM_SENDER_ID


def binding_names_no_one(binding: ChannelBinding) -> bool:
    """Whether *binding* carries one conversation and no correspondent yet."""
    return not binding.group and binding.participant_id is None


def binding_admits(binding: ChannelBinding, known_as: frozenset[str]) -> bool:
    """Whether *binding* lets a sender known by *known_as* in (RFC §10.4).

    A group binding lets everyone in; any other, the correspondent it names,
    or anyone while it names no one.
    """
    return binding.group or binding.participant_id is None or binding.participant_id in known_as


async def _heard_from_other(
    store: ConversationStore, room_id: str, channel_id: str, own: frozenset[str]
) -> bool:
    """Whether the room received a message on *channel_id* naming another sender.

    Received means stored and not ``BLOCKED`` (the default page skips those).
    An event whose source names no participant, or the framework's own sender
    (``SYSTEM_SENDER_ID``: the host speaking through ``deliver()``), is not a
    correspondent's; *own* holds the ids the sender is known by.
    """
    received = EventFilter(source_channel_id=channel_id)
    offset = 0
    while True:
        page = await store.list_events(
            room_id, offset=offset, limit=_HISTORY_PAGE, event_filter=received
        )
        for event in page:
            named = event.source.participant_id
            if named not in (None, SYSTEM_SENDER_ID) and named not in own:
                return True
        if len(page) < _HISTORY_PAGE:
            return False
        offset += len(page)

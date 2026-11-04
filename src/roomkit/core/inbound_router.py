"""Inbound room router — determines which room an inbound message belongs to."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roomkit.channels.addressing import UnreadableRecipient, read_recipient
from roomkit.models.channel import ChannelBinding
from roomkit.models.delivery import SYSTEM_SENDER_ID
from roomkit.models.enums import ChannelType, RoomStatus
from roomkit.models.identity import Identity
from roomkit.models.participant import own_record_id
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter, RoomFilter
from roomkit.store.base import ConversationStore

if TYPE_CHECKING:
    from roomkit.channels.base import Channel

logger = logging.getLogger("roomkit.framework")

# Page size of the scan for another sender's message (RFC §10.4 step 3). The
# scan stops at the first one, so a room with any history answers on its
# first page; only a room holding nothing but the sender's own words is read
# to the end.
_HISTORY_PAGE = 200

# The statuses that refuse new events (RFC §5.1): a room in one is no longer
# a routed message's.
_REFUSING = frozenset({RoomStatus.CLOSED, RoomStatus.ARCHIVED})

# The statuses of the rooms step 1 reads (RFC §10.4): a conversation paused
# for inactivity is the sender's as an active one is.
_OPEN = (RoomStatus.ACTIVE, RoomStatus.PAUSED)


class InboundRoomRouter(ABC):
    """Route an inbound message to a room (or ``None`` for auto-create)."""

    @abstractmethod
    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
        *,
        organization_id: str | None = None,
        identity: Identity | None = None,
    ) -> str | None:
        """Return room_id for the message, or ``None`` to create a new room.

        *participant_id* is the address naming the message's conversation
        (``Channel.conversation_address``): the sender's, or on a chat
        channel the chat it was posted in. *channel_data* is the message's
        metadata.

        *organization_id* is the organization the caller acts for (RFC
        §17.2), ``None`` when the call names none. *identity* is the one the
        store resolves *participant_id* to within that organization
        (``resolve_identity``), read by the framework once per message
        before the first route, or ``None``: the router and the check made
        when the sender is recorded name the sender alike (RFC §10.4).

        The answer is held to RFC §10.1 step 2's rules whoever gave it: a
        room of another organization than *organization_id* fails the
        message (``RoomNotFoundError``), and a room id the store lacks is
        created under that id. An override must accept both keywords; the
        framework passes them on every call.
        """
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
        channel_of: Callable[[str], Channel | None] | None = None,
    ) -> None:
        self._store = store
        # The kit's registered channel by id: its recipient key and the
        # address a binding delivers to (Channel.recipient_address), in its
        # form, name the correspondent (RFC §5.7). Without it no binding's
        # recipient names anyone.
        self._channel_of = channel_of

    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
        *,
        organization_id: str | None = None,
        identity: Identity | None = None,
    ) -> str | None:
        # The organization's rooms only, every room when the call names none
        # (RFC §10.4, §17.2): another tenant's room is no candidate, so it
        # neither wins nor makes the organization's one room ambiguous.
        own = known_as(participant_id, identity, channel_id)
        # Step 1 (RFC §10.4): the sender's own room. Tried first because it
        # identifies the conversation, where a binding of the channel to some
        # room only identifies the pipe. The framework's name is no one's, so
        # nothing it names is a sender's conversation.
        if participant_id and recordable_sender(participant_id):
            room_id = await self._senders_room(
                channel_id, participant_id, _identity_name(identity), organization_id
            )
            if room_id is not None:
                return room_id

        # Step 3: a channel dedicated to one conversation. Only when the
        # channel is bound to exactly one room step 3 reads — a channel shared
        # across rooms (delegation, a room re-created after its predecessor
        # closed, a number shared by many correspondents) makes this
        # ambiguous, and the framework creating a fresh room is the safe
        # answer — and only when that room is open to this sender: on a
        # shared number, the one room bound to it is the first
        # correspondent's conversation.
        candidates = await self._step_three_rooms(channel_id, organization_id)
        if len(candidates) == 1:
            if await self._admits(candidates[0], channel_id, participant_id or None, own):
                return candidates[0]
            return None
        if len(candidates) > 1:
            # Debug, not a warning: on a number shared by many correspondents,
            # several rooms bound to the channel is the ordinary state, and
            # every new correspondent passes through here.
            logger.debug(
                "Channel %s is bound to %d active or paused group rooms — refusing to guess "
                "which one this message belongs to. Pass room_id explicitly, or install a "
                "custom InboundRoomRouter.",
                channel_id,
                len(candidates),
            )

        return None

    async def _step_three_rooms(self, channel_id: str, organization_id: str | None) -> list[str]:
        """Up to two rooms bound to *channel_id* that step 3 reads (RFC
        §10.4): the ACTIVE rooms, and the PAUSED rooms whose binding of the
        channel is ``group``, a group's conversation being the channel's.
        Any other PAUSED room is no one's to be admitted into, and does not
        count."""
        active = await self._store.find_room_ids_by_channel(
            channel_id,
            RoomFilter(statuses=[RoomStatus.ACTIVE], organization_id=organization_id),
            limit=2,
        )
        if len(active) > 1:
            return active
        paused_groups = await self._store.find_room_ids_by_channel(
            channel_id,
            RoomFilter(statuses=[RoomStatus.PAUSED], organization_id=organization_id),
            limit=2,
            group=True,
        )
        return [*active, *paused_groups][:2]

    async def _senders_room(
        self,
        channel_id: str,
        sender: str,
        identity_id: str | None,
        organization_id: str | None,
    ) -> str | None:
        """The newest ACTIVE or PAUSED room of *organization_id* that is
        *sender*'s own (RFC §10.4 step 1), whatever its status: a room its
        timer paused for inactivity is the sender's conversation as an
        ACTIVE one is, and the message's delivered commit resumes it (§5.1).
        *identity_id* is the identity the framework resolved the address to.

        First the room whose binding of this very channel names the sender,
        as its correspondent or, naming none, as its recipient (RFC §5.7): a
        binding speaks for its own channel, so a correspondent of one number
        is not taken to that room when they write to another. Otherwise the
        room where the sender is a participant reached through this channel:
        an id names its participant only on a channel they are reached
        through (RFC §5.5), so a number linked to someone's identity, or a
        sender id equal to a participant's on another channel, does not land
        in a room they joined elsewhere, which would then carry that room's
        messages to this sender. Within each lookup the sender is named by
        their address or their identity (``link_address``), and the newest
        room either name finds wins: an older room found by the address does
        not hide a newer one found by the identity.
        """
        step_one = RoomFilter(statuses=list(_OPEN), organization_id=organization_id)
        channel = self._channel_of(channel_id) if self._channel_of is not None else None
        room_id = await self._store.find_room_id_by_binding(
            channel_id,
            sender,
            identity_id,
            step_one,
            recipient_key=channel.recipient_key if channel is not None else None,
        )
        if room_id is not None:
            return room_id
        return await self._store.find_room_id_by_participant(
            channel_id, sender, identity_id, step_one
        )

    async def _admits(
        self, room_id: str, channel_id: str, sender: str | None, own: frozenset[str]
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
        channel = self._channel_of(channel_id) if self._channel_of is not None else None
        recipient = routed_recipient(channel, binding)
        return await step_three_admits(self._store, binding, sender, own, recipient)


def routed_recipient(
    channel: Channel | None, binding: ChannelBinding
) -> str | UnreadableRecipient | None:
    """The recipient *binding* delivers to, as routing reads it (RFC §5.7, §10.4).

    Its address in the channel's form; ``None`` when it holds no value, on a
    ``group`` binding, whose recipient names no one, or on a channel whose
    recipient names no conversation (or no channel); an
    :class:`UnreadableRecipient` for a value under the channel's recipient
    key that is not a recipient. That one is not "no recipient": it is a
    recipient no sender is, so on a binding naming no participant it admits
    no one by step 3 and makes the room no longer a routed message's under
    the lock, and the record of a sender never writes over it. It is logged
    as a warning naming the binding and the value's type, never the value,
    which may be a person's number.
    """
    if channel is None or binding.group:
        return None
    key = channel.recipient_key
    held = read_recipient(binding.metadata, key) if key is not None else None
    if isinstance(held, UnreadableRecipient):
        logger.warning(
            "The binding of channel %r in room %r holds a %s under %r, which is not a "
            "recipient: routing reads it as someone else's address, so no sender the "
            "binding does not name is admitted through it, and deliveries to it are "
            "refused no_recipient (RFC §5.7)",
            binding.channel_id,
            binding.room_id,
            held.kind,
            key,
        )
        return held
    return channel.recipient_address(binding)


async def step_three_admits(
    store: ConversationStore,
    binding: ChannelBinding,
    sender: str | None,
    own: frozenset[str],
    recipient: str | UnreadableRecipient | None,
) -> bool:
    """Whether the room of *binding*, its binding of the channel, admits
    *sender* by RFC §10.4 step 3's conditions.

    *own* holds the ids the sender is known by (:func:`known_as`), and
    *recipient* the address the binding delivers to, in its channel's form,
    as :func:`routed_recipient` reads it: a value that is not a recipient
    is someone else's. The default router asks it of the one room bound to
    the channel, and the framework again under the room lock before it
    records the sender there (§10.1 step 2): one predicate, so the two never
    drift apart.
    """
    if binding.group:
        return True
    if not recordable_sender(sender):
        # No one's: the framework's name, and a message with no address, whose
        # words the next sender admitted here would read, what it wrote
        # counting as no one heard. It opens a room of its own (RFC §10.4).
        return False
    if not binding_admits(binding, own):
        return False
    if recipient is not None and (
        isinstance(recipient, UnreadableRecipient) or recipient not in own
    ):
        # The room delivers to someone else: it is their conversation. A
        # value that is not a recipient is no sender's (RFC §5.7).
        return False
    for participant in await store.list_participants(binding.room_id):
        if binding.channel_id in participant.connected_via and not own & {
            participant.id,
            participant.identity_id,
        }:
            return False
    return not await _heard_from_other(store, binding.room_id, binding.channel_id, own)


async def still_the_messages(
    store: ConversationStore,
    room: Room,
    channel_id: str,
    binding: ChannelBinding | None,
    sender: str | None,
    own: frozenset[str],
    recipient: str | UnreadableRecipient | None,
) -> bool:
    """Whether a routed message's *room*, read under the room lock, is still
    the message's (RFC §10.1 step 2, §10.4), whichever strategy chose it.

    *binding* is the room's binding of *channel_id*, *own* the ids the sender
    is known by and *recipient* the address the binding delivers to, as
    :func:`routed_recipient` reads it. Not the message's: a room whose
    status refuses events; a PAUSED room that is not the sender's
    conversation by step 1 (the binding names them, or a participant they
    are was reached through the channel) and whose binding is not
    ``group``; a binding, not ``group``, naming a correspondent other than
    the sender (its participant, else its recipient, a value that is not a
    recipient being no sender's, RFC §5.7); a binding naming no one that the
    room no longer opens to the sender by step 3's conditions, unless the
    sender is a participant reached through the channel; a message with no
    address is held to those conditions too, which it meets through no such
    binding, and is recorded on none. A message under the framework's own
    name is not routed at all: its caller refuses it first (RFC §10.1 step 2).
    """
    if room.status in _REFUSING:
        return False
    group = binding is not None and binding.group
    # An unreadable recipient is a correspondent no sender is: it is in no
    # one's names, so the room is not the message's (RFC §5.7).
    correspondent = None if binding is None or group else binding.participant_id or recipient
    if room.status != RoomStatus.PAUSED and (
        binding is None or group or (correspondent is not None and correspondent in own)
    ):
        return True
    members = [
        participant
        for participant in await store.list_participants(room.id)
        if channel_id in participant.connected_via
        and own & {participant.id, participant.identity_id}
    ]
    # A participant who is the sender names them by its id and its identity
    # too: a binding naming that participant names the sender.
    names = own | {name for member in members for name in (member.id, member.identity_id) if name}
    named = correspondent is not None and correspondent in names
    if room.status == RoomStatus.PAUSED and not (group or members or named):
        return False
    if binding is None or group:
        return True
    if correspondent is not None:
        return named
    if members:
        return True
    # Whoever chose the room, step 3's conditions hold under the lock (RFC
    # §10.4): someone heard or reached on the channel since the router read
    # the room makes it no longer the message's.
    return await step_three_admits(store, binding, sender or None, own, recipient)


def known_as(sender: str | None, identity: Identity | None, channel_id: str) -> frozenset[str]:
    """The ids a room may know *sender* by on *channel_id*: the address, its
    identity, and the record of their own the identity step left them
    pending under there (:func:`own_record_id`, RFC §11.2).

    Identity resolution names a participant and stamps an event with the
    identity's id, not the address (RFC §11), so the identity the store
    resolves the address to, within the caller's organization (§17.2), is
    the sender's own too. A message with no address and the framework's own
    name are known by nothing: no record is theirs (:func:`names_no_sender`).
    """
    if not sender or not recordable_sender(sender):
        return frozenset()
    names = {sender, own_record_id(channel_id, sender)}
    if (name := _identity_name(identity)) is not None:
        names.add(name)
    return frozenset(names)


def _identity_name(identity: Identity | None) -> str | None:
    """The id *identity* names its sender by; none for an identity under the
    framework's own name, which is no sender's (RFC §10.4)."""
    return identity.id if identity is not None and identity.id != SYSTEM_SENDER_ID else None


def recordable_sender(sender: str | None) -> bool:
    """Whether *sender* can be a binding's correspondent (RFC §10.4).

    Not an empty or blank address, which identifies no one, and not the
    framework's own sender: what the host writes through ``deliver()`` names
    no correspondent.
    """
    return bool(sender and sender.strip()) and sender != SYSTEM_SENDER_ID


def names_no_sender(sender: str | None) -> str | None:
    """What a binding step 2 makes for a routed message names when *sender*
    cannot be recorded, a message with no address (RFC §5.7, §10.4): the
    framework's own name, which no sender is, as :func:`known_as` and the
    refusal of a message routed under it make sure. Unnamed, the binding would
    pass step 3 for the next sender, who would read what the message left;
    ``None`` for a sender step 2 records itself.
    """
    return None if recordable_sender(sender) else SYSTEM_SENDER_ID


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

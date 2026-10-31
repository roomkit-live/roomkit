"""Abstract base class for conversation storage."""

from __future__ import annotations

import copy
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from roomkit.channels.addressing import written_recipient
from roomkit.core.exceptions import (
    AddressLinkedError,
    EventNotFoundError,
    IdempotencyKeyTakenError,
    IdentityNotFoundError,
    ParticipantExistsError,
    RoomAlreadyExistsError,
)
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import EventType, RoomStatus
from roomkit.models.event import DeleteContent, EditContent, RoomEvent, ThreadSummary
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter, RoomFilter
from roomkit.models.task import Observation, Task
from roomkit.models.verification import Verification, VerificationRequest
from roomkit.models.voice_delivery import VoiceDeliveryRecord

logger = logging.getLogger("roomkit.store")

# Page size of the fallback scan over rooms (find_room_id_by_participant).
_ROOM_PAGE = 100


def link_target(
    identity: Identity | None, identity_id: str, organization_id: str | None
) -> Identity:
    """The identity *organization_id* links an address to (RFC §5.6, §14.1).

    An unknown identity, or one of another organization, is refused as not
    found: an organization links its own identities only, and no
    organization stands for the identities of none. The empty string and
    ``None`` both name no organization, as the address key stores them.
    """
    if identity is None or (identity.organization_id or "") != (organization_id or ""):
        raise IdentityNotFoundError(f"Identity {identity_id} not found")
    return identity


def without_address(
    channel_addresses: dict[str, list[str]], channel_type: str, address: str
) -> dict[str, list[str]]:
    """*channel_addresses* without *address* under *channel_type*, a type left
    with no address dropped (``unlink_address``, RFC §5.6)."""
    kept = [held for held in channel_addresses.get(channel_type, []) if held != address]
    rest = {key: value for key, value in channel_addresses.items() if key != channel_type}
    return {**rest, channel_type: kept} if kept else rest


def sender_names(participant_id: str, identity_id: str | None) -> frozenset[str]:
    """The names a routing lookup knows the sender by (RFC §10.4 step 1): the
    address, and the identity the caller resolved it to."""
    return frozenset({participant_id} if identity_id is None else {participant_id, identity_id})


def reached_as(participant: Participant, channel_id: str, names: frozenset[str]) -> bool:
    """Whether *participant* is the sender known by *names*, reached through
    *channel_id* (RFC §5.5, §10.4 step 1): its id or its identity is one of
    the names, and its ``connected_via`` names the channel."""
    return channel_id in participant.connected_via and bool(
        names & {participant.id, participant.identity_id}
    )


def names_by_recipient(binding: ChannelBinding, address: str, recipient_key: str | None) -> bool:
    """Whether *binding* names *address* through its recipient (RFC §5.7,
    §10.4 step 1): a binding stored with a recipient and no correspondent,
    and not ``group``, is its recipient's conversation.

    The recipient is the value under *recipient_key*, the channel's, read by
    the rule a delivery reads it by (:func:`written_recipient`: a string
    that is not blank, or an integer), compared as stored: the framework
    writes a recipient in its channel's form (§5.7)."""
    if recipient_key is None or binding.participant_id is not None or binding.group:
        return False
    value = written_recipient(binding.metadata, recipient_key)
    return value is not None and str(value) == address


def _recency(room: Room) -> tuple[float, str]:
    """A room's place in creation order: created_at, then id. Timestamps,
    since a naive created_at beside an aware one would make the datetimes
    themselves refuse to compare."""
    return (room.created_at.timestamp(), room.id)


def deletes_edits(event: RoomEvent, target: RoomEvent | None) -> bool:
    """Whether committing *event* with *target* marks the target's edits
    deleted: *event* is a DELIVERED delete of it (RFC §10.3)."""
    return (
        target is not None
        and event.type == EventType.DELETE
        and isinstance(event.content, DeleteContent)
        and event.content.target_event_id == target.id
    )


def edits_target(event: RoomEvent, target_id: str) -> bool:
    """Whether *event* is an edit of *target_id*."""
    content = event.content
    return isinstance(content, EditContent) and content.target_event_id == target_id


def is_marked_deleted(event: RoomEvent) -> bool:
    """Whether *event* carries an accepted delete's mark (``metadata.deleted``)."""
    return event.metadata.get("deleted") is True


def marked_deleted(event: RoomEvent) -> RoomEvent:
    """*event* marked ``metadata.deleted``, its other metadata kept."""
    return event.model_copy(update={"metadata": {**event.metadata, "deleted": True}})


def room_taken(room_id: str) -> RoomAlreadyExistsError:
    """The refusal of a room created under an id the store holds (RFC §14.3)."""
    return RoomAlreadyExistsError(f"Room {room_id} already exists")


def target_gone(target: RoomEvent) -> EventNotFoundError:
    """The refusal of a write of *target*, which its room does not hold
    (RFC §14.3): a change's target (``commit_event``, the whole commit
    fails) or an update (``update_event``); nothing is written."""
    return EventNotFoundError(f"Event {target.id} is not in room {target.room_id}")


def participant_exists(participant: Participant) -> ParticipantExistsError:
    """The refusal of *participant*, whose id its room already has (RFC
    §5.5, §14.3): nothing is written."""
    return ParticipantExistsError(
        f"Participant {participant.id} is already in room {participant.room_id}"
    )


def key_taken(event: RoomEvent) -> IdempotencyKeyTakenError:
    """The refusal of *event*, whose key its source channel holds in its room
    for another event (RFC §13.4, §14.3): nothing is written."""
    return IdempotencyKeyTakenError(
        f"Idempotency key {event.idempotency_key!r} of channel {event.source.channel_id}"
        f" is taken in room {event.room_id}"
    )


def committed_state(room: Room, activity: bool) -> dict[str, Any]:
    """What a commit writes on *room* beside its counters, in the commit's
    transaction (RFC §5.1): an event the pipeline stores DELIVERED is the
    room's activity, which moves ``last_activity_at`` to the commit's
    instant and resumes a PAUSED room (ACTIVE, ``paused_at`` null); anything
    else writes nothing."""
    if not activity:
        return {}
    now = datetime.now(UTC)
    state: dict[str, Any] = {"timers": room.timers.model_copy(update={"last_activity_at": now})}
    if room.status is RoomStatus.PAUSED:
        state.update(status=RoomStatus.ACTIVE, paused_at=None, updated_at=now)
    return state


def room_holds(
    room: Room,
    status: RoomStatus,
    paused_at: datetime | None,
    last_activity_at: datetime | None,
) -> bool:
    """Whether *room* is still in *status* with *paused_at* and
    ``timers.last_activity_at`` *last_activity_at*: the state a timer
    transition was computed from (:meth:`ConversationStore.update_room_if`)."""
    return (
        room.status is status
        and room.paused_at == paused_at
        and room.timers.last_activity_at == last_activity_at
    )


def address_linked(channel_type: str) -> AddressLinkedError:
    """The refusal of an address another identity of the organization holds
    (RFC §5.6). The address is left out: the caller passed it."""
    return AddressLinkedError(
        f"This {channel_type} address is linked to another identity of the organization"
    )


def _with_recorded_results(target: RoomEvent, stored: RoomEvent) -> RoomEvent:
    """*target*, a change's target as its commit writes it, carrying the
    delivery results *stored*, the row as it stands, holds: only the record
    of a delivery set writes them (RFC §10.3, §14.3), and a copy read before
    the record would erase it over a store whose ``update_event`` writes the
    row as given."""
    if target.delivery_results == stored.delivery_results:
        return target
    return target.model_copy(update={"delivery_results": stored.delivery_results})


class ConversationStore(ABC):
    """Persistent storage for rooms, events, bindings, and participants.

    Implement this ABC to plug in any storage backend (SQL, Redis, etc.).
    The library ships with `InMemoryStore` for development and testing.

    What a store returns is its snapshot, and what it is written is its own
    (RFC §14.4): a room, binding, participant or event a read or a write
    returns is treated as frozen by its caller, and may be the same object
    for every reader or a new one per read; a write must keep the caller's
    later change to the object it passed out of the store (copy it in, or
    serialize it). A room's private state alone is the reader's copy.
    """

    # Capability used by RoomKit's lock-manager pairing guard. Persistent
    # backends are conservatively assumed shareable across processes; a truly
    # process-local implementation can opt out explicitly. This keeps the core
    # dependent on the store contract rather than a growing list of concrete
    # backend types.
    is_process_local: bool = False

    async def get_voice_delivery(self, room_id: str, key_hash: str) -> VoiceDeliveryRecord | None:
        """Read an injection reservation; unsupported stores fail before sending."""
        raise NotImplementedError("This store does not support voice delivery reservations")

    async def claim_voice_delivery(self, record: VoiceDeliveryRecord) -> VoiceDeliveryRecord:
        """Atomically reserve an injection and return its authoritative record.

        Insert if absent, or replace an explicitly retryable completed attempt
        with matching content. Otherwise return the existing record unchanged.
        Ownership requires a matching attempt_id. Retain records until room deletion.

        Raises ``ValueError`` when *record* carries an outcome, or when its room
        does not exist.
        """
        raise NotImplementedError("This store does not support voice delivery reservations")

    async def complete_voice_delivery(self, record: VoiceDeliveryRecord) -> bool:
        """Set the outcome only if this token still owns an unresolved attempt.

        Repeating the same completion is successful; replacing a terminal
        outcome or completing another owner's attempt is refused.
        """
        raise NotImplementedError("This store does not support voice delivery reservations")

    # Connection tenure

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[None, None]:
        """Serve the store calls in this block from one backend connection.

        A pooled backend MAY bind a single connection for the duration of the
        block, so a stretch of calls pays one checkout instead of one per call.
        That checkout is not free: asyncpg, for one, resets a connection on
        release (``pg_advisory_unlock_all(); CLOSE ALL; UNLISTEN *;
        RESET ALL;``), a full round trip whose cost rivals the reads it brackets.

        The default binds nothing and yields: a store with no connection pool
        has none to bind, and behaves exactly as it did without the block.

        **This is not a transaction.** No atomicity, no isolation, no rollback,
        no snapshot — a failure midway leaves the earlier calls applied, exactly
        as it would outside the block. It bounds *connection tenure*, nothing
        else. A caller needing atomicity asks for it explicitly (see
        :meth:`commit_event`).

        The block MUST contain store calls and nothing else, awaited
        **sequentially**: no hook, no provider call, no lock acquisition, no
        ``gather``/``create_task`` over store calls. Two reasons, both learned
        the hard way — holding a pooled connection across foreign code is how a
        fleet parks its backends behind someone else's HTTP request, and a child
        task inherits this context, so it would use the bound connection
        concurrently with its parent.

        Reentrant: a nested block joins the outer one rather than taking a
        second connection.

        Yields nothing on purpose. The bound connection is the backend's
        business; callers keep calling the store's ordinary methods.
        """
        yield

    # A room's private state (RFC §14.1): the keys the runtime keeps for
    # itself, the register of authors among them (§10.1 step 12). Not
    # abstract: a store written before it keeps them where they were.

    _private_state_warned: bool = False

    async def get_private_state(self, room_id: str, key: str) -> dict[str, Any] | None:
        """The value the room keeps for itself under *key*, apart from its
        metadata; ``None`` when the room does not exist or holds none. The
        caller's own copy (RFC §14.4).

        What a key holds is the runtime's, never a client's: the room's
        register of authors, whose digests name the room's senders to
        whoever tries every phone number or name, is one. A store keeps it
        apart from the room's metadata, which clients may read. This default
        reads it from the metadata, where it was kept before, and warns once:
        override this method and :meth:`set_private_state` to keep it
        private.

        The keys are the implementation's (``author_register``; a later one
        takes a ``_`` prefix), and each reserves the same key in the room's
        metadata: this default keeps the value there, and
        :meth:`set_private_state` removes the copy there, so an application
        must not keep its own metadata under one.
        """
        _warn_private_state_in_metadata(self)
        room = await self.get_room(room_id)
        value = None if room is None else room.metadata.get(key)
        return copy.deepcopy(value) if isinstance(value, dict) else None

    async def set_private_state(
        self, room_id: str, key: str, value: dict[str, Any] | None
    ) -> None:
        """Keep *value* under *key* for the room, apart from its metadata, or
        remove the key when *value* is ``None``; the copy the room's metadata
        may still hold under the key (a room stored before the key moved) is
        removed in the same write. Writes nothing for a room the store does
        not hold. The caller holds the room lock (RFC §13.5), and the room's
        deletion takes its private state.

        *key* reserves the same key in the room's metadata
        (:meth:`get_private_state`). This default writes the key in the
        room's metadata, where it was kept before.
        """
        _warn_private_state_in_metadata(self)
        if value is None:
            await self.patch_room_metadata(room_id, {}, unset=[key])
        else:
            await self.patch_room_metadata(room_id, {key: value})

    # Room operations

    @abstractmethod
    async def create_room(self, room: Room) -> Room:
        """Persist a new room.

        A room is created once (RFC §14.3): an id the store holds raises
        :class:`~roomkit.core.exceptions.RoomAlreadyExistsError`, nothing
        written, and the room it holds is never replaced.
        """
        ...

    @abstractmethod
    async def get_room(self, room_id: str) -> Room | None:
        """Get a room by ID, or ``None`` if it doesn't exist."""
        ...

    @abstractmethod
    async def update_room(self, room: Room) -> Room:
        """Update an existing room.

        ``Room.delivered_index`` (see :meth:`advance_delivered_index`),
        ``Room.event_count`` and ``Room.latest_index`` (moved by committing
        an event) are store-managed: implementations SHOULD NOT let a
        caller's stale copy rewind them. The shipped stores keep the stored
        values and return the room with them; a store that does so MUST
        override :meth:`advance_delivered_index` and :meth:`commit_event`,
        whose defaults write these fields through this method.
        ``Room.timers`` is the caller's, ``last_activity_at`` included:
        :meth:`~roomkit.RoomKit.set_room_timers` sets the idle clock through
        it.
        """
        ...

    async def update_room_if(
        self,
        room: Room,
        *,
        status: RoomStatus,
        paused_at: datetime | None,
        last_activity_at: datetime | None,
    ) -> Room | None:
        """Write *room* as :meth:`update_room` does, only if the stored room
        is still in *status* with *paused_at* and ``timers.last_activity_at``
        *last_activity_at*: a compare-and-set. Returns the room written, or
        ``None`` with nothing written when the stored room moved.

        A timer transition is written so (RFC §5.1): it is computed from a
        room read earlier, and a streamed row or a greeting commits without
        the room lock, so a commit that resumed the room or moved its
        activity in between must not be overwritten by a stale close or
        pause.

        The default is **NOT atomic**: it reads the room, compares, then
        writes through :meth:`update_room`. A store whose commits can run
        concurrently with it (shared across processes, or committing off the
        room lock) SHOULD override it with one transaction that holds the
        room's row from the comparison to the write, as the shipped stores
        do.
        """
        stored = await self.get_room(room.id)
        if stored is None or not room_holds(stored, status, paused_at, last_activity_at):
            return None
        return await self.update_room(room)

    async def room_exists(self, room_id: str) -> bool:
        """Whether a room exists, without materialising it.

        Inbound routing asks this on every message and discards everything
        else, so going through :meth:`get_room` selects every column, decodes
        the JSONB ones and validates a whole ``Room`` — to answer a yes/no.
        Backends SHOULD override with an existence query.
        """
        return await self.get_room(room_id) is not None

    async def binding_exists(self, room_id: str, channel_id: str) -> bool:
        """Whether a channel is attached to a room, without materialising it.

        Same reasoning as :meth:`room_exists`: routing only needs the yes/no,
        and the decision it feeds is re-taken under the room lock anyway.
        """
        return await self.get_binding(room_id, channel_id) is not None

    async def get_delivered_index(self, room_id: str) -> int:
        """Read a room's delivery cursor, and nothing else.

        The delivery lane consults this on every turn — twice per event on the
        Postgres path, once outside the claim and once under it. Going through
        :meth:`get_room` for it means selecting every column and rebuilding the
        whole ``Room`` model, JSONB fields decoded and validated, to reach one
        integer. Backends SHOULD override this with a single-column read.

        Returns ``-1`` for an unknown room, matching the cursor's initial value.
        """
        room = await self.get_room(room_id)
        return -1 if room is None else room.delivered_index

    @abstractmethod
    async def advance_delivered_index(
        self, room_id: str, index: int, *, force: bool = False
    ) -> bool:
        """Advance the room's delivered index to *index* (RFC §10.1 step 14).

        Compare-and-set: succeeds only from ``index - 1`` — the strict
        per-room delivery order — unless *force*, the delivery-gap skip
        policy, which advances from any lower value. Returns True if the
        cursor moved.

        A store implements it as one atomic conditional write (RFC §14.1):
        a read-check-write default is correct only under the room's delivery
        claim, and two processes would move the cursor past an index alike.
        """
        ...

    async def patch_room_metadata(
        self,
        room_id: str,
        patch: dict[str, Any],
        *,
        unset: Sequence[str] = (),
    ) -> Room | None:
        """Merge ``patch`` into a room's metadata, removing ``unset`` keys first.

        The targeted alternative to ``update_room`` for metadata-only changes.
        ``update_room`` rewrites the whole room row from an in-memory ``Room``
        (read-modify-write), so a caller holding a stale object silently
        clobbers concurrent metadata patches and regresses the counters
        (``event_count`` / ``latest_index`` / ``timers``) advanced by
        ``commit_event``. This method touches only the metadata keys it is
        given and stamps ``updated_at``.

        Returns the updated room, or ``None`` when *room_id* does not exist.

        The default implementation is **NOT atomic** — it reads, merges, then
        writes back via ``update_room``. Persistent backends that may be
        shared across processes MUST override this with a single storage-level
        partial update (e.g. a JSONB merge).
        """
        room = await self.get_room(room_id)
        if room is None:
            return None
        metadata = {k: v for k, v in room.metadata.items() if k not in set(unset)}
        metadata.update(patch)
        return await self.update_room(
            room.model_copy(update={"metadata": metadata, "updated_at": datetime.now(UTC)})
        )

    @abstractmethod
    async def delete_room(self, room_id: str) -> bool:
        """Delete a room. Returns ``True`` if the room existed."""
        ...

    @abstractmethod
    async def list_rooms(self, offset: int = 0, limit: int = 50) -> list[Room]:
        """List rooms with pagination, the oldest first: by ``created_at``,
        then by ``id``, so a page holds the same rooms in every store."""
        ...

    @abstractmethod
    async def find_rooms(
        self,
        organization_id: str | None = None,
        status: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Room]:
        """Find rooms matching the given filters, in the order of
        :meth:`list_rooms`.

        A room matches *metadata_filter* when each of its keys holds the given
        value. PostgresStore tests a nested value by containment (``@>``), the
        other stores by equality: ``{"a": {"b": 1}}`` matches
        ``{"a": {"b": 1, "c": 2}}`` on Postgres alone.
        """
        ...

    @abstractmethod
    async def find_latest_room(
        self,
        participant_id: str,
        channel_type: str | None = None,
        status: str | None = None,
    ) -> Room | None:
        """Find the most recent room for a participant.

        The most recent is the last created; of rooms created at the same
        instant, the one of highest id, in every store.
        """
        ...

    @abstractmethod
    async def find_room_id_by_channel(
        self, channel_id: str, status: str | None = None
    ) -> str | None:
        """Find a room ID that has a binding for the given channel_id."""
        ...

    async def find_room_ids_by_channel(
        self,
        channel_id: str,
        room_filter: RoomFilter | None = None,
        limit: int = 2,
        *,
        group: bool | None = None,
    ) -> list[str]:
        """Room IDs bound to *channel_id* among the rooms *room_filter* reads,
        oldest first: by creation time, the room id breaking a tie. *group*
        keeps the rooms whose binding of the channel is declared ``group``
        (``True``) or is not (``False``); ``None`` keeps both.

        Routing must be able to tell "exactly one room" from "several", because
        picking one of several delivers a message into a conversation it does
        not belong to (RFC §10.4). ``find_room_id_by_channel`` cannot express
        that — it returns one id either way — so this returns up to *limit*,
        and the default of 2 is enough to answer "is this ambiguous?".
        *room_filter* names the statuses and the organization (RFC §17.2);
        ``None`` reads every room.

        Implementations MUST order the result deterministically: the same
        stored state has to give the same answer whatever the backend.

        This is not abstract, so a store that does not override it keeps
        working: with a filter naming one status at most, no organization and
        no *group* it falls back to the single lookup and therefore cannot
        report ambiguity, otherwise to a scan of the rooms the filter reads.
        Override it.
        """
        room_filter = room_filter or RoomFilter()
        statuses = room_filter.statuses
        single = statuses is None or len(statuses) == 1
        if room_filter.organization_id is None and group is None and single:
            status = None if statuses is None else statuses[0].value
            room_id = await self.find_room_id_by_channel(channel_id, status=status)
            return [room_id] if room_id is not None else []
        matched: list[tuple[float, str]] = []
        for room in await self._rooms_read_by(room_filter):
            binding = await self.get_binding(room.id, channel_id)
            if binding is not None and (group is None or binding.group is group):
                matched.append(_recency(room))
        return [room_id for _created_at, room_id in sorted(matched)[:limit]]

    async def find_room_id_by_binding(
        self,
        channel_id: str,
        participant_id: str,
        identity_id: str | None = None,
        room_filter: RoomFilter | None = None,
        *,
        recipient_key: str | None = None,
    ) -> str | None:
        """The newest room whose binding of *channel_id* names the sender
        (RFC §10.4 step 1, §14.1): its correspondent is *participant_id*, the
        sender's address, or *identity_id*, the identity the caller resolved
        the address to (``None`` when none).

        A binding names its correspondent (RFC §5.7): the sender a room was
        created for, or the first one routed to it. One stored with a
        recipient and no correspondent, and not ``group``, names its
        recipient: its value under *recipient_key*, the channel's key, which
        a caller passes on a channel whose recipient is a conversation's
        address (:func:`names_by_recipient`). A binding speaks for its own
        channel only, so a correspondent of one number is not taken to that
        room when they write to another (§10.4 step 1). The newest room either
        name finds wins: the most recently created, the greatest id breaking
        a tie, in every store. *room_filter* names the statuses and the
        organization (§17.2); ``None`` reads every room.

        Not abstract, so a store that does not override it keeps working:
        the fallback reads the binding of every room bound to the channel.
        Override it with an indexed query.
        """
        names = sender_names(participant_id, identity_id)
        room_ids = await self.find_room_ids_by_channel(
            channel_id, room_filter or RoomFilter(), limit=sys.maxsize
        )
        for room_id in reversed(room_ids):
            binding = await self.get_binding(room_id, channel_id)
            if binding is not None and (
                binding.participant_id in names
                or names_by_recipient(binding, participant_id, recipient_key)
            ):
                return room_id
        return None

    async def find_room_id_by_participant(
        self,
        channel_id: str,
        participant_id: str,
        identity_id: str | None = None,
        room_filter: RoomFilter | None = None,
    ) -> str | None:
        """The newest room holding a participant who is the sender and was
        reached through *channel_id* (RFC §10.4 step 1, §14.1).

        The participant is named by *participant_id*, the sender's address,
        or by *identity_id*, the identity the caller resolved it to (``None``
        when none): its id or its identity is one of them. Reached through the
        channel is its ``connected_via`` naming it, whatever the participant's
        status (§5.5): an id names its participant only on a channel they are
        reached through. Participant records only, never a binding: a binding
        speaks for its own channel and is read by
        :meth:`find_room_id_by_binding`. The newest room either name finds
        wins, ordered as there; *room_filter* names the statuses and the
        organization (§17.2), ``None`` reading every room.

        Not abstract, so a store that does not override it keeps working:
        the fallback pages through the rooms. Override it with an indexed
        query.
        """
        names = sender_names(participant_id, identity_id)
        best: tuple[float, str] | None = None
        for room in await self._rooms_read_by(room_filter or RoomFilter()):
            if best is not None and _recency(room) <= best:
                continue
            if any(
                reached_as(participant, channel_id, names)
                for participant in await self.list_participants(room.id)
            ):
                best = _recency(room)
        return best[1] if best is not None else None

    async def _rooms_read_by(self, room_filter: RoomFilter) -> list[Room]:
        """Every room *room_filter* reads, paged through :meth:`find_rooms`."""
        statuses: list[str | None] = (
            [None] if room_filter.statuses is None else [s.value for s in room_filter.statuses]
        )
        found: list[Room] = []
        for status in statuses:
            offset = 0
            while True:
                rooms = await self.find_rooms(
                    organization_id=room_filter.organization_id,
                    status=status,
                    limit=_ROOM_PAGE,
                    offset=offset,
                )
                found.extend(rooms)
                if len(rooms) < _ROOM_PAGE:
                    break
                offset += len(rooms)
        return found

    # Event operations

    @abstractmethod
    async def add_event(self, event: RoomEvent) -> RoomEvent:
        """Store a new event."""
        ...

    @abstractmethod
    async def get_event(self, event_id: str) -> RoomEvent | None:
        """Get an event by ID."""
        ...

    @abstractmethod
    async def update_event(self, event: RoomEvent) -> RoomEvent:
        """Update an existing event (e.g., mark as edited or deleted).

        The event's ``delivery_results`` stay as stored, whatever *event*
        carries: only :meth:`record_delivery_results` writes them (RFC §10.3,
        §14.3), so a copy read before a delivery set was recorded never
        erases its outcome. A change's target that :meth:`commit_event` or
        :meth:`write_change_target` writes keeps them alike.
        """
        ...

    async def record_delivery_results(
        self, room_id: str, event_id: str, results: dict[str, Any]
    ) -> None:
        """Write *results*, the outcome of an event's delivery set, on the
        event as stored, and nothing else of it (RFC §10.3, §14.3).

        Written once the delivery set is done, off the room lock: rewriting
        the event as it stood at its commit would undo an edit or a delete
        committed since. An event *room_id* does not hold is left alone.

        It is the one write of ``delivery_results``: a rewrite of the row
        keeps them (:meth:`update_event`).

        The default reads the stored event and writes it back with the
        results through :meth:`update_event`, which is safe only under the
        room lock, where the framework calls it, and works for a store whose
        ``update_event`` writes the row as given. A store whose
        ``update_event`` keeps the stored results, as its contract says,
        overrides this method: the default reads the event back, and when
        the results were not written it raises :class:`NotImplementedError`
        saying so, rather than record nothing.
        """
        stored = await self.get_event(event_id)
        if stored is None or stored.room_id != room_id:
            return
        await self.update_event(stored.model_copy(update={"delivery_results": results}))
        written = await self.get_event(event_id)
        if written is None or written.delivery_results != results:
            raise NotImplementedError(
                f"{type(self).__name__} keeps delivery_results on update_event: "
                "override record_delivery_results to write them"
            )

    @abstractmethod
    async def delete_event(
        self, room_id: str, event_id: str, *, cascade_replies: bool = True
    ) -> list[RoomEvent]:
        """Hard-delete a persisted event, optionally cascading to its thread replies.

        ``parent_event_id`` has no DB-level FK, so a bare root delete would
        orphan its replies — ``cascade_replies`` (default) removes them in the
        same operation. Whatever it says, the edits and deletes of every
        event removed go with it (RFC §10.3): an edit holds its target's
        corrected content, and a change of an event that no longer exists
        changes nothing. One transaction, a removed event's idempotency key
        released with it (§13.4).

        Returns the removed events as they were stored: the root, its
        replies, then the changes, each group in index order; empty when
        *event_id* does not exist in *room_id*.
        """
        ...

    @abstractmethod
    async def list_events(
        self,
        room_id: str,
        offset: int = 0,
        limit: int = 50,
        visibility_filter: str | None = None,
        *,
        after_index: int | None = None,
        before_index: int | None = None,
        event_filter: EventFilter | None = None,
        newest_first: bool = False,
    ) -> list[RoomEvent]:
        """List events in a room with pagination and filtering.

        Supports two pagination modes:

        - **Offset-based** (default): ``offset`` + ``limit`` for simple page access.
        - **Cursor-based**: ``after_index`` or ``before_index`` for efficient
          keyset pagination on large rooms.  When either is set, ``offset`` is
          ignored.

        ``after_index`` and ``before_index`` are mutually exclusive.

        When *event_filter* is provided, its ``visibility`` field takes
        precedence over *visibility_filter*.

        The page holds what the room **received**: an event stored ``BLOCKED``
        (refused by a hook, sent by a source that could not write, stopped by
        a cap) is skipped unless ``event_filter.include_blocked`` asks for it,
        and skipped before the page is cut, so a page of ``limit`` events is
        full whatever was refused around them (RFC §14.1). A read by id
        (:meth:`get_event`) is never filtered.

        By default the offset-based mode returns the *oldest* ``limit`` events
        (the head of the room). Pass ``newest_first=True`` to return the most
        recent ``limit`` events instead — still in ascending chronological
        order, so a "give me the latest page" snapshot reads top-to-bottom.
        ``newest_first`` only applies to the offset-based mode; it is ignored
        when a cursor (*after_index* / *before_index*) is supplied.

        .. note::

            Cursor pagination relies on events having a valid ``index``
            assigned by :meth:`add_event_auto_index`.  Events stored via
            :meth:`add_event` keep the model default ``index=0`` and will
            all compare equal, producing incorrect cursor results.  Use
            ``add_event_auto_index`` for rooms that need cursor pagination.

        Args:
            room_id: Room to query.
            offset: Number of events to skip (offset-based mode), counted from
                the newest end when ``newest_first`` is set.
            limit: Maximum number of events to return.
            visibility_filter: Optional visibility value to filter by.
                Ignored when *event_filter* provides a visibility.
            after_index: Return events with ``index > after_index`` (ascending).
            before_index: Return the last ``limit`` events with
                ``index < before_index``, in ascending order.
            event_filter: Rich filter criteria (event types, source, time range,
                correlation ID). See :class:`EventFilter`.
            newest_first: In offset-based mode, return the most recent ``limit``
                events (ascending order) instead of the oldest. Ignored when a
                cursor is supplied.
        """
        ...

    @abstractmethod
    async def get_thread_summaries(
        self, room_id: str, root_event_ids: list[str]
    ) -> dict[str, ThreadSummary]:
        """Return reply aggregates for the given thread roots.

        For each root that has replies, the result maps its id to a
        :class:`ThreadSummary` (reply count + last-reply timestamp). Roots with
        no replies are absent from the mapping. Used to render a "N replies"
        affordance without fetching every reply. A reply the room refused
        (stored ``BLOCKED``) is not counted, as a default :meth:`list_events`
        read of the thread does not serve it (RFC §14.1): the affordance and
        the list agree.
        """
        ...

    @abstractmethod
    async def check_idempotency(self, room_id: str, key: str, *, channel_id: str | None) -> bool:
        """Whether *channel_id* already committed *key* in the room: a key is
        unique per room and per source channel (RFC §13.4). ``None`` asks
        whether any channel of the room committed it: a proactive delivery's
        key names one publication in its room, whatever transport carries it
        (§22.1). ``True`` if duplicate."""
        ...

    async def get_event_by_idempotency_key(
        self, room_id: str, key: str, *, channel_id: str | None
    ) -> RoomEvent | None:
        """The event *channel_id* committed with a previously-seen idempotency
        key in the room (RFC §13.4, §14.1), ``BLOCKED`` included. With
        *channel_id* ``None``, the event any channel of the room committed
        with it, for a proactive delivery's key (§22.1): the one of lowest
        index when several did.

        A redelivered message must observe what its first delivery did, not a
        refusal — a webhook that retries because it never saw the first
        response is asking what happened, and "duplicate" does not answer it.

        Not abstract, and ``None`` is a legitimate answer: a store that cannot
        resolve the key falls back to reporting the duplicate as blocked, which
        is what every store did before this existed.
        """
        return None

    @abstractmethod
    async def get_event_count(self, room_id: str, event_filter: EventFilter | None = None) -> int:
        """Count the events a room currently holds, exactly, on demand.

        This is the authoritative count and the only one that survives
        deletions. ``Room.event_count`` is a *maintained tally* incremented at
        each commit (RFC §10.1 step 12) and never decremented, so the two
        diverge once events are deleted — deliberately: an exact count is a
        scan of the room's whole timeline, and no commit should pay for one.

        Call this when the number has to be right. Read ``Room.event_count``
        when a cheap, monotonically-growing hint is enough.

        Both count every committed row, the refused ones included: a
        ``BLOCKED`` event consumed an index (RFC §8.3) and stays in the count
        while a default :meth:`list_events` page skips it (§14.1).

        ``event_filter`` counts a subset instead: exactly the rows a
        :meth:`list_events` page would serve under that filter, with no page
        (RFC §14.1). The filter's own ``include_blocked`` decides whether the
        refused rows are in, ``False`` by default as on a page, so
        ``EventFilter()`` counts the received rows where no filter counts them
        all. A host that needs the exact number of a subset (an agent's turns
        before an instant) asks here rather than measuring a page and
        inheriting its cap.
        """
        ...

    async def add_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Atomically assign the next index and store the event.

        The default implementation reads the count and writes in two steps.
        Backends should override this with an atomic implementation (e.g.
        a single SQL transaction) to prevent race conditions on the index.
        """
        count = await self.get_event_count(room_id)
        indexed = event.model_copy(update={"index": count})
        return await self.add_event(indexed)

    @abstractmethod
    async def commit_event(
        self,
        room_id: str,
        event: RoomEvent,
        *,
        target: RoomEvent | None = None,
        activity: bool = True,
    ) -> RoomEvent:
        """Store *event* and bump the room counters as ONE atomic commit.

        The commit point of RFC §10.1 (step 12) / §14.3: (re)assigning the
        authoritative index, persisting the event, and updating the room's
        ``event_count`` / ``latest_index`` / ``timers.last_activity_at`` form a
        **single logical transaction**. An observer MUST never see a stored
        event the room counters do not reflect.

        *activity* says whether the commit is the room's activity (RFC
        §5.1): an event stored DELIVERED through the pipeline. Only then
        does ``timers.last_activity_at`` move to the commit's instant, and a
        PAUSED room resume (ACTIVE, ``paused_at`` null), in the same
        transaction (:func:`committed_state`). The framework passes
        ``activity=False`` for what is no activity, a lifecycle record, a
        record stored BLOCKED and a record committed outside the pipeline,
        which leave both; a store MUST accept the keyword.

        *target* is the target of the DELIVERED edit or delete *event* is, as
        its commit writes it (RFC §10.3): the edited content and
        ``metadata.edited``, or ``metadata.deleted``. It is written in the
        same transaction, replacing the stored row, and a target *room_id*
        no longer holds fails the whole commit with
        :class:`~roomkit.core.exceptions.EventNotFoundError`, nothing
        written (§14.3). The framework passes it only with a change, so a
        store that does not accept the keyword still commits every other
        event. A delete also marks every edit of its target deleted
        (``metadata.deleted``), in the same transaction: an edit holds a
        version of the deleted text, withheld from a channel's history even
        where the history does not hold the message (:func:`deletes_edits`).

        An idempotency key is unique per room and per source channel (RFC
        §13.4, §14.3): an *event* whose key its channel holds in the room for
        another event raises
        :class:`~roomkit.core.exceptions.IdempotencyKeyTakenError`, nothing
        written, its target included (:func:`key_taken`). The framework
        checks the key under the room lock first, so only a direct caller
        meets it. An event under an id the store holds is refused by the
        database with its own error, which is not this one.

        The converse does not hold, and that is deliberate. ``event_count`` is
        the running tally RFC §10.1 step 12 describes — ``event_count += 1``,
        never a recount — and deletion adjusts no counter (the RFC prescribes
        none), so after events are deleted the tally overstates the timeline.
        Keeping it exact would mean scanning every event on every commit, which
        makes the hot path cost grow with a room's history. Callers who need
        the real number call :meth:`get_event_count`, which is exact and
        separate on purpose.

        The authoritative index is (re)computed inside the commit (RFC §8.1) so
        a persistent store shared across processes serializes concurrent
        writers on the store, not only on an in-process room lock.

        Returns the committed event — its ``index`` may differ from any
        provisional value the caller assigned before hooks ran, if the store
        serialized a concurrent writer.

        A store implements it as one storage transaction (RFC §14.3): a
        read-then-write default would let a concurrent writer, in this
        process or another, see the event without the counters, or index
        two events alike.
        """
        ...

    async def write_change_target(self, room_id: str, target: RoomEvent) -> None:
        """Write *target* as the edit or the delete the persistence policy
        keeps out of the timeline changed it (RFC §10.3): the target's row
        alone, no row of the change, and when the target is now deleted
        (``metadata.deleted``) every edit of it marked deleted too, in one
        transaction, as :meth:`commit_event` does with a change it stores.
        A target *room_id* no longer holds raises
        :class:`~roomkit.core.exceptions.EventNotFoundError`, nothing
        written.

        The default is **NOT atomic**: it writes through :meth:`update_event`
        and marks the edits through the store's own reads and writes. A store
        that can do it in one transaction SHOULD override it.
        """
        stored = await self.get_event(target.id)
        if stored is None or stored.room_id != room_id:
            raise target_gone(target)
        await self.update_event(_with_recorded_results(target, stored))
        if is_marked_deleted(target):
            await self._mark_edits_deleted(room_id, target.id)

    async def _mark_edits_deleted(self, room_id: str, target_id: str) -> None:
        """Mark every edit of *target_id* deleted, through the store's own
        reads and writes (the default :meth:`commit_event`'s, not atomic). An
        edit is told by its content's target, whatever its type, as the
        shipped stores tell it."""
        every_row = EventFilter(include_blocked=True)
        offset = 0
        while page := await self.list_events(
            room_id, offset=offset, limit=_ROOM_PAGE, event_filter=every_row
        ):
            for edit in page:
                if edits_target(edit, target_id):
                    await self.update_event(marked_deleted(edit))
            offset += len(page)

    async def get_conversation(
        self,
        room_id: str,
        *,
        limit: int = 50,
        after_index: int | None = None,
    ) -> list[RoomEvent]:
        """Return message events only — suitable for AI context rebuilding.

        Filters to ``MESSAGE`` events, excluding tool calls, lifecycle,
        and system noise. Tool call history is not included because AI
        providers track their own tool call context internally via the
        message history passed to each generation call.

        Use :meth:`get_timeline` to retrieve the full activity log
        including tool calls.

        Without a cursor this returns the **most recent** ``limit`` messages,
        in ascending order — the last element is the newest message in the
        room. That is what "the conversation" means for a context rebuild:
        the head of a room whose history outgrew ``limit`` is history no
        model should be answering from. Callers who do want the opening
        messages ask for them explicitly via :meth:`list_events` /
        :meth:`get_timeline`.

        With ``after_index`` it stays a forward cursor — the first ``limit``
        messages *after* that index, ascending — so keyset pagination
        ("give me what arrived since I last looked") reads in order and
        resumes where it stopped.

        The conversation is the host's whole one: a message stored
        ``BLOCKED`` (refused by a hook, or by a source that could not write)
        is in it, as the audit record it is, unlike a :meth:`list_events`
        page, which serves the received rows unless asked otherwise. This
        read fills ``RoomContext.recent_events``, and hooks read that whole
        (RFC §7.5 rule 8). What a *channel* may be handed as history is a
        per-reader question, and
        :func:`~roomkit.core.visibility.visible_events` answers it (same
        rule): it drops BLOCKED events along with what visibility withholds.

        Returned events are immutable snapshots (RFC §14.4): treat them
        as frozen. A store may share objects between reads or return
        fresh ones — rely on neither.
        """
        return await self.list_events(
            room_id,
            limit=limit,
            after_index=after_index,
            event_filter=EventFilter(event_types=[EventType.MESSAGE], include_blocked=True),
            # Ignored when a cursor is supplied; spelled out so the no-cursor
            # read is unambiguously the tail of the room, not its head.
            newest_first=after_index is None,
        )

    async def get_timeline(
        self,
        room_id: str,
        *,
        event_filter: EventFilter | None = None,
        limit: int = 50,
        after_index: int | None = None,
        newest_first: bool = False,
    ) -> list[RoomEvent]:
        """Return the full activity timeline for a room.

        Returns the events the room received, in order; a row stored
        ``BLOCKED`` is served only when *event_filter* asks for it
        (``include_blocked=True``), as for :meth:`list_events`. Use
        *event_filter* to narrow results further (e.g. only tool calls, only
        a specific correlation group).

        Without a cursor the default is the *oldest* ``limit`` events; pass
        ``newest_first=True`` for the most recent ``limit`` (still ascending) —
        the right shape for a reconnect snapshot that must show recent history,
        not the room's opening events.
        """
        return await self.list_events(
            room_id,
            limit=limit,
            after_index=after_index,
            event_filter=event_filter,
            newest_first=newest_first,
        )

    # Binding operations

    @abstractmethod
    async def add_binding(self, binding: ChannelBinding) -> ChannelBinding:
        """Attach a channel binding to a room."""
        ...

    @abstractmethod
    async def get_binding(self, room_id: str, channel_id: str) -> ChannelBinding | None:
        """Get a channel binding, or ``None`` if not attached."""
        ...

    @abstractmethod
    async def update_binding(self, binding: ChannelBinding) -> ChannelBinding:
        """Update an existing channel binding."""
        ...

    @abstractmethod
    async def remove_binding(self, room_id: str, channel_id: str) -> bool:
        """Detach a channel from a room. Returns ``True`` if it was attached."""
        ...

    @abstractmethod
    async def list_bindings(self, room_id: str) -> list[ChannelBinding]:
        """List all channel bindings for a room."""
        ...

    # Participant operations

    @abstractmethod
    async def add_participant(self, participant: Participant) -> Participant:
        """Add a participant to a room."""
        ...

    @abstractmethod
    async def get_participant(self, room_id: str, participant_id: str) -> Participant | None:
        """Get a participant by ID within a room."""
        ...

    @abstractmethod
    async def update_participant(self, participant: Participant) -> Participant:
        """Update a participant."""
        ...

    @abstractmethod
    async def delete_participant(self, room_id: str, participant_id: str) -> None:
        """Remove the room's participant *participant_id*; removing one the
        room does not have changes nothing (RFC §14.1)."""
        ...

    @abstractmethod
    async def list_participants(self, room_id: str) -> list[Participant]:
        """List all participants in a room."""
        ...

    async def load_room_context(
        self, room_id: str
    ) -> tuple[Room | None, list[ChannelBinding], list[Participant]]:
        """Read a room together with its bindings and participants.

        These three reads always travel together when a room's context is
        assembled, and that happens on every inbound message. The default
        implementation issues them as three separate calls, inside
        :meth:`connection` so a pooled backend that binds one there already
        pays a single checkout for all three; overriding this method too buys
        the queries themselves in one place, which is what ``PostgresStore``
        does.

        This is a convenience, NOT a consistent snapshot: the reads carry no
        more cross-read atomicity than calling the three methods in sequence
        would. A backend needing that must take it explicitly.

        Returns ``(None, [], [])`` for a room that does not exist, mirroring
        :meth:`get_room` rather than raising.
        """
        async with self.connection():
            room = await self.get_room(room_id)
            if room is None:
                return None, [], []
            return room, await self.list_bindings(room_id), await self.list_participants(room_id)

    # Identity operations

    @abstractmethod
    async def create_identity(self, identity: Identity) -> Identity:
        """Create a new identity record, its addresses filed under its organization.

        Raises :class:`AddressLinkedError`, writing nothing, when one of its
        ``channel_addresses`` belongs to another identity of its organization
        (RFC §5.6).
        """
        ...

    @abstractmethod
    async def get_identity(self, identity_id: str) -> Identity | None:
        """Get an identity by ID."""
        ...

    @abstractmethod
    async def resolve_identity(
        self, channel_type: str, address: str, organization_id: str | None = None
    ) -> Identity | None:
        """Look up an identity by channel type and address (RFC §14.1).

        *organization_id* scopes the lookup. An address is unique **within** an
        organization, not globally: without scoping, a phone number registered
        by one tenant would resolve to that tenant's identity for every other
        tenant too (RFC §17.2). Leave it unset in a single-tenant deployment —
        those registrations live in their own unscoped space, and an unscoped
        lookup resolves them only, never an organization's.
        """
        ...

    @abstractmethod
    async def link_address(
        self,
        identity_id: str,
        channel_type: str,
        address: str,
        organization_id: str | None = None,
    ) -> None:
        """Link a channel address to an identity of *organization_id* (RFC §5.6, §14.1).

        The same address may be linked to a different identity in each
        organization: a number belongs to one person at one tenant and to
        someone else at another. Within one organization it belongs to one
        identity: an address another identity of the organization holds is
        refused with :class:`AddressLinkedError`, and moving it is an
        :meth:`unlink_address` from its holder, then a link. An unknown
        identity, or one of another organization
        (``None`` stands for the identities of no organization), is refused
        with :class:`IdentityNotFoundError`. Linking an address the identity
        holds changes nothing. A refusal writes nothing.

        Addresses compare as stored: link them in their channel's form (E.164
        on a channel addressed by phone number, RFC §5.7), the form the
        framework resolves a sender's address in.
        """
        ...

    async def unlink_address(
        self,
        identity_id: str,
        channel_type: str,
        address: str,
        organization_id: str | None = None,
    ) -> None:
        """Remove a channel address from an identity of *organization_id* (RFC §5.6, §14.1).

        The address then resolves to no identity of the organization and can
        be linked to another: moving an address is an unlink then a link.
        An address the identity does not hold changes nothing, another
        identity's included. An unknown identity, or one of another
        organization, is refused with :class:`IdentityNotFoundError`, as
        :meth:`link_address` refuses it. An unlink changes which identity a
        later message resolves to, never a stored conversation: the rooms,
        bindings and participants that named the address stay as they are.

        Not abstract, so a store written before it keeps working, and
        refuses it by name.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not unlink identity addresses (RFC §14.1)"
        )

    # Verification operations (RFC §11.7, Level 2). Not abstract: a store
    # written before them keeps working, and refuses them by name.

    async def get_verification(
        self, identity_id: str, organization_id: str | None = None
    ) -> Verification | None:
        """The identity's verification, within an organization; None if it has none."""
        raise _verification_unsupported(self)

    async def put_verification(self, verification: Verification) -> Verification:
        """Create or replace the verification of ``verification.identity_id``."""
        raise _verification_unsupported(self)

    async def list_verifications_in_force(self) -> list[Verification]:
        """Every VERIFIED verification with an expiry, across organizations.

        Read when the framework starts, to re-arm the timers that end them when
        they expire (§11.7.5).
        """
        raise _verification_unsupported(self)

    async def create_verification_request(
        self, request: VerificationRequest
    ) -> VerificationRequest:
        """Store a new verification request."""
        raise _verification_unsupported(self)

    async def get_verification_request(self, request_id: str) -> VerificationRequest | None:
        """A verification request by id; None if there is none."""
        raise _verification_unsupported(self)

    async def update_verification_request(
        self, request: VerificationRequest
    ) -> VerificationRequest:
        """Replace a stored verification request."""
        raise _verification_unsupported(self)

    # Task operations

    @abstractmethod
    async def add_task(self, task: Task) -> Task:
        """Store a new task."""
        ...

    @abstractmethod
    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        ...

    @abstractmethod
    async def list_tasks(self, room_id: str, status: str | None = None) -> list[Task]:
        """List tasks for a room, optionally filtered by status."""
        ...

    @abstractmethod
    async def update_task(self, task: Task) -> Task:
        """Update a task."""
        ...

    # Observation operations

    @abstractmethod
    async def add_observation(self, observation: Observation) -> Observation:
        """Store a new observation."""
        ...

    @abstractmethod
    async def list_observations(self, room_id: str) -> list[Observation]:
        """List all observations for a room."""
        ...

    # Read tracking

    @abstractmethod
    async def mark_read(self, room_id: str, channel_id: str, event_id: str) -> None:
        """Mark an event as read for a channel.

        The channel's marker is a high-water mark on event ``index``: it moves
        to the event's index when that is higher, never back. An event that
        is not in *room_id* (another room's, or none) marks nothing.
        """
        ...

    @abstractmethod
    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        """Mark all events as read for a channel.

        Moves the marker to the room's highest event index, never back. A room
        with no event gets no marker: its first event, index 0, arrives unread.
        """
        ...

    @abstractmethod
    async def get_unread_count(self, room_id: str, channel_id: str) -> int:
        """Return the number of unread events for a channel in a room.

        The events whose ``index`` is above the channel's marker; every event
        when the channel has none.
        """
        ...

    @abstractmethod
    async def list_read_markers(self, room_id: str) -> dict[str, int]:
        """Return every channel's read high-water-mark in a room.

        Maps ``channel_id`` -> the highest read event ``index``. Channels with
        no marker are absent. With one channel per member, this is the raw
        material for aggregating per-member "seen by" receipts.
        """
        ...

    async def close(self) -> None:
        """Release any resources held by the store (e.g. a connection pool).

        Called by ``RoomKit.close()``. The default is a no-op — override it in
        backends that own external resources. Implementations MUST be
        idempotent and MUST NOT close resources they do not own.
        """
        return None


def _warn_private_state_in_metadata(store: ConversationStore) -> None:
    """Say once per store that it keeps the rooms' private state in their
    metadata, where the register of authors' digests name the senders to
    whoever reads them."""
    if store._private_state_warned:
        return
    store._private_state_warned = True
    logger.warning(
        "%s keeps each room's private state, the register of authors among it, in "
        "the room's metadata, which clients may read: the register's digests name "
        "the room's senders to whoever tries every phone number or name. Implement "
        "get_private_state() and set_private_state() to keep it apart.",
        type(store).__name__,
    )


def _verification_unsupported(store: ConversationStore) -> NotImplementedError:
    """The refusal of a store that does not keep verifications (RFC §11.7, Level 2)."""
    return NotImplementedError(
        f"{type(store).__name__} does not store identity verifications (RFC §11.7)"
    )

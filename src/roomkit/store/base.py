"""Abstract base class for conversation storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent, ThreadSummary
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.models.task import Observation, Task


class ConversationStore(ABC):
    """Persistent storage for rooms, events, bindings, and participants.

    Implement this ABC to plug in any storage backend (SQL, Redis, etc.).
    The library ships with `InMemoryStore` for development and testing.
    """

    # Room operations

    @abstractmethod
    async def create_room(self, room: Room) -> Room:
        """Persist a new room."""
        ...

    @abstractmethod
    async def get_room(self, room_id: str) -> Room | None:
        """Get a room by ID, or ``None`` if it doesn't exist."""
        ...

    @abstractmethod
    async def update_room(self, room: Room) -> Room:
        """Update an existing room.

        ``Room.delivered_index`` is store-managed (see
        :meth:`advance_delivered_index`): implementations SHOULD NOT let a
        caller's stale copy rewind it. The shipped stores exclude it from
        this write path.
        """
        ...

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

    async def advance_delivered_index(
        self, room_id: str, index: int, *, force: bool = False
    ) -> bool:
        """Advance the room's delivered index to *index* (RFC §10.1 step 14).

        Compare-and-set: succeeds only from ``index - 1`` — the strict
        per-room delivery order — unless *force*, the delivery-gap skip
        policy, which advances from any lower value. Returns True if the
        cursor moved.

        This default is read-check-write through :meth:`get_room` /
        :meth:`update_room`; it is only correct under the room's delivery
        claim, and only for stores whose ``update_room`` persists
        ``delivered_index`` as given. A store that shields the field there
        (as the shipped stores do) MUST override this with an atomic
        conditional write.
        """
        room = await self.get_room(room_id)
        if room is None:
            return False
        if room.delivered_index >= index:
            return False
        if not force and room.delivered_index != index - 1:
            return False
        await self.update_room(room.model_copy(update={"delivered_index": index}))
        return True

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
        """List rooms with pagination."""
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
        """Find rooms matching the given filters."""
        ...

    @abstractmethod
    async def find_latest_room(
        self,
        participant_id: str,
        channel_type: str | None = None,
        status: str | None = None,
    ) -> Room | None:
        """Find the most recent room for a participant."""
        ...

    @abstractmethod
    async def find_room_id_by_channel(
        self, channel_id: str, status: str | None = None
    ) -> str | None:
        """Find a room ID that has a binding for the given channel_id."""
        ...

    async def find_room_ids_by_channel(
        self, channel_id: str, status: str | None = None, limit: int = 2
    ) -> list[str]:
        """Room IDs bound to *channel_id*, deterministically ordered.

        Routing must be able to tell "exactly one room" from "several", because
        picking one of several delivers a message into a conversation it does
        not belong to (RFC §10.4). ``find_room_id_by_channel`` cannot express
        that — it returns one id either way — so this returns up to *limit*,
        and the default of 2 is enough to answer "is this ambiguous?".

        Implementations MUST order the result deterministically: the same
        stored state has to give the same answer whatever the backend.

        This is not abstract, so a store written before it existed keeps
        working — it falls back to the single lookup and therefore cannot
        report ambiguity. Override it.
        """
        room_id = await self.find_room_id_by_channel(channel_id, status=status)
        return [room_id] if room_id is not None else []

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
        """Update an existing event (e.g., mark as edited or deleted)."""
        ...

    @abstractmethod
    async def delete_event(
        self, room_id: str, event_id: str, *, cascade_replies: bool = True
    ) -> list[str]:
        """Hard-delete a persisted event, optionally cascading to its thread replies.

        ``parent_event_id`` has no DB-level FK, so a bare root delete would
        orphan its replies — ``cascade_replies`` (default) removes them in the
        same operation. Returns the deleted event IDs (root first, then
        replies); empty when *event_id* does not exist in *room_id*.
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
            offset: Number of events to skip (offset-based mode).
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
        affordance without fetching every reply.
        """
        ...

    @abstractmethod
    async def check_idempotency(self, room_id: str, key: str) -> bool:
        """Check if an idempotency key has been seen. Returns ``True`` if duplicate."""
        ...

    @abstractmethod
    async def get_event_count(self, room_id: str) -> int:
        """Count the events a room currently holds, exactly, on demand.

        This is the authoritative count and the only one that survives
        deletions. ``Room.event_count`` is a *maintained tally* incremented at
        each commit (RFC §10.1 step 12) and never decremented, so the two
        diverge once events are deleted — deliberately: an exact count is a
        scan of the room's whole timeline, and no commit should pay for one.

        Call this when the number has to be right. Read ``Room.event_count``
        when a cheap, monotonically-growing hint is enough.
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

    async def commit_event(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Store *event* and bump the room counters as ONE atomic commit.

        The commit point of RFC §10.1 (step 12) / §14.3: (re)assigning the
        authoritative index, persisting the event, and updating the room's
        ``event_count`` / ``latest_index`` / ``timers.last_activity_at`` form a
        **single logical transaction**. An observer MUST never see a stored
        event the room counters do not reflect.

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

        The default implementation is **NOT atomic** — it indexes, inserts,
        then updates the room in separate steps. Persistent backends that may be
        shared across processes MUST override this with a single storage
        transaction.
        """
        indexed = await self.add_event_auto_index(room_id, event)
        room = await self.get_room(room_id)
        if room is not None:
            timers = room.timers.model_copy(update={"last_activity_at": datetime.now(UTC)})
            await self.update_room(
                room.model_copy(
                    update={
                        # The tally, incremented — same contract as the shipped
                        # stores, so a custom backend inheriting this default
                        # does not silently behave differently after deletions.
                        "event_count": room.event_count + 1,
                        "latest_index": indexed.index,
                        "timers": timers,
                    }
                )
            )
        return indexed

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

        Returned events are immutable snapshots (RFC §14.4): treat them
        as frozen. A store may share objects between reads or return
        fresh ones — rely on neither.
        """
        return await self.list_events(
            room_id,
            limit=limit,
            after_index=after_index,
            event_filter=EventFilter(event_types=[EventType.MESSAGE]),
            # Ignored when a cursor is supplied; spelled out so the no-cursor
            # read is unambiguously the tail of the room, not its head.
            newest_first=after_index is None,
        )

    async def get_timeline(
        self,
        room_id: str,
        *,
        event_filter: EventFilter | None = None,
        limit: int = 100,
        after_index: int | None = None,
        newest_first: bool = False,
    ) -> list[RoomEvent]:
        """Return the full activity timeline for a room.

        Returns all persisted events in order. Use *event_filter* to narrow
        results (e.g. only tool calls, only a specific correlation group).

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
    async def list_participants(self, room_id: str) -> list[Participant]:
        """List all participants in a room."""
        ...

    async def load_room_context(
        self, room_id: str
    ) -> tuple[Room | None, list[ChannelBinding], list[Participant]]:
        """Read a room together with its bindings and participants.

        These three reads always travel together when a room's context is
        assembled, and that happens on every inbound message. The default
        implementation issues them as three separate calls; a pooled backend
        SHOULD override it to serve them over a single connection, because
        there each separate call also pays a connection checkout whose cost
        rivals the queries themselves.

        This is a convenience, NOT a consistent snapshot: the reads carry no
        more cross-read atomicity than calling the three methods in sequence
        would. A backend needing that must take it explicitly.

        Returns ``(None, [], [])`` for a room that does not exist, mirroring
        :meth:`get_room` rather than raising.
        """
        room = await self.get_room(room_id)
        if room is None:
            return None, [], []
        return room, await self.list_bindings(room_id), await self.list_participants(room_id)

    # Identity operations

    @abstractmethod
    async def create_identity(self, identity: Identity) -> Identity:
        """Create a new identity record."""
        ...

    @abstractmethod
    async def get_identity(self, identity_id: str) -> Identity | None:
        """Get an identity by ID."""
        ...

    @abstractmethod
    async def resolve_identity(self, channel_type: str, address: str) -> Identity | None:
        """Look up an identity by channel type and address."""
        ...

    @abstractmethod
    async def link_address(self, identity_id: str, channel_type: str, address: str) -> None:
        """Link a channel address to an identity."""
        ...

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
        """Mark an event as read for a channel."""
        ...

    @abstractmethod
    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        """Mark all events as read for a channel."""
        ...

    @abstractmethod
    async def get_unread_count(self, room_id: str, channel_id: str) -> int:
        """Return the number of unread events for a channel in a room."""
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

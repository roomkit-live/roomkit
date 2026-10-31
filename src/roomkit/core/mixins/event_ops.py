"""EventOpsMixin — direct, host-owned writes to persisted events (RFC §10.3, §10.5)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core._authors import with_committed_record
from roomkit.core._publication_keys import refuse_reserved
from roomkit.core._reentrance import charge_direct_change
from roomkit.core.exceptions import RoomClosedError, RoomNotFoundError
from roomkit.core.mixins.helpers import HelpersMixin, _refuses_writes
from roomkit.models.enums import HookTrigger
from roomkit.models.event import EventContent, EventSource, RoomEvent, refuse_too_deep

logger = logging.getLogger("roomkit.framework")

if TYPE_CHECKING:
    from roomkit.core._closing import DoorRuns
    from roomkit.core._reentrance import RoomWork
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.room import Room
    from roomkit.store.base import ConversationStore


@runtime_checkable
class EventOpsHost(Protocol):
    """Contract: capabilities a host class must provide for EventOpsMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation store holding the events.
        _hook_engine: Hook engine for the mutation triggers.
        _lock_manager: Per-room lock for serialised mutation.

    Methods:
        get_room: From :class:`RoomLifecycleMixin`.
    """

    _store: ConversationStore
    _hook_engine: HookEngine
    _lock_manager: RoomLockManager


class EventOpsMixin(HelpersMixin):
    """Direct update / delete of persisted events.

    The host application owns authorization on this path — unlike inbound
    EDIT/DELETE events (RFC §10.3), no author/identity check runs here. Both
    operations serialise against inbound processing via the room lock and fire
    their mutation trigger (ON_EVENT_UPDATED / ON_EVENT_DELETED) after the
    lock is released (RFC §10.1), so observers such as denormalized-projection
    maintainers see every stored-state change regardless of origin.

    Host contract: :class:`EventOpsHost`.
    """

    _store: ConversationStore
    _hook_engine: HookEngine
    _lock_manager: RoomLockManager
    _room_work: RoomWork
    _door_runs: DoorRuns
    get_room: Any  # see EventOpsHost / RoomLifecycleMixin

    async def commit_event(
        self, room_id: str, event: RoomEvent, *, organization_id: str | None = None
    ) -> RoomEvent:
        """Commit a record no member receives to the room's timeline (RFC §10.5).

        For a trace, a display snapshot, the copy a branched conversation starts
        from. The record takes the room's next index, which its delivery lane
        passes at its turn with nothing to run, so the room's next event waits
        on it no longer than on the events before it; no delivery set, no hook
        and no broadcast follow. It is no activity (RFC §5.1): the room's
        ``last_activity_at`` stays, and a PAUSED room stays paused. It is stored as
        given, whatever the persistence policy, but for its thread parent,
        rooted as every committed event's is (a reply named is replaced by its
        thread's root, an event the room does not hold dropped, RFC §10.1
        step 8), and the record of its author, which the room fixes when the
        event names one (``metadata["author"]``, RFC §10.1 step 12, any value
        given dropped);
        the source write rule (RFC §7.5 rule 2) does not apply to it. An event
        the members must receive goes through :meth:`send_event`.

        The room is read under its lock, which a hook already holding it
        re-enters, scoped to *organization_id* (RFC §17.2).

        Raises:
            ValueError: *event* names another room, carries an
                ``idempotency_key`` in the framework's publication namespace
                (``roomkit:publication:``), or holds more than five composite
                levels (RFC §5.3); nothing is read or written.
            RoomNotFoundError: the room is missing, or another organization's.
            RoomClosedError: the room's status refuses new events (RFC §5.1);
                nothing is written.
            IdempotencyKeyTakenError: the room holds an event of the record's
                source channel under its idempotency key: a record is never a
                replay (RFC §10.5, §14.3); nothing is written.
        """
        if event.room_id != room_id:
            # The stores file the row under the event's room: the room checked
            # must be the room written (RFC §10.5).
            raise ValueError(f"commit_event into {room_id} got an event of room {event.room_id}")
        # A publication's key is the delivery path's alone (RFC §22.1): a
        # record carrying one would stand in for the publication.
        refuse_reserved(event.idempotency_key)
        # No event holding a sixth composite level is committed (RFC §5.3).
        refuse_too_deep(event.content)
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id, organization_id=organization_id)
            room = await self._room_timers.apply_due(room)
            if _refuses_writes(room):
                raise RoomClosedError(f"Room {room_id} does not accept new events")
            record = await self._with_thread_root(room_id, event)
            # Read back if the kit's close cuts it as it lands (RFC §12.10.4).
            self._door_runs.committing(record)
            return await self._commit_indexed(room_id, record)

    async def update_event(
        self,
        room_id: str,
        event_id: str,
        *,
        content: EventContent | None = None,
        source: EventSource | None = None,
        metadata: dict[str, Any] | None = None,
        organization_id: str | None = None,
    ) -> RoomEvent | None:
        """Update a persisted event's content, source, and/or metadata.

        Provided fields replace the stored values wholesale (``metadata`` is
        not merged — pass the full mapping). Unlike the inbound EDIT path,
        no ``edited`` marker is added; the caller controls metadata, but for
        the record of its author the event was committed with
        (``metadata["author"]``, RFC §10.1 step 12), which the room fixes:
        the update keeps it, and writes none on an event committed with none,
        whatever *metadata* holds under the key. The record holds while the
        source stays the one recorded; otherwise a reader ranks the event's
        author against the room's register.

        Fires ON_EVENT_UPDATED with the updated event, and returns without
        waiting for its hooks (RFC §9.3). Returns the updated event, or
        ``None`` when *event_id* does not exist in *room_id* (no hook fires).
        An update that changes nothing, no field or the values the event
        holds, writes nothing, fires nothing and returns the event as stored
        (RFC §10.3). Scoped to *organization_id* (RFC §17.2), the room of
        another organization is a missing room: ``None``, nothing written, its
        due timers included; unset, the call is unscoped, as the kit's other
        calls.

        Made from the body of an ASYNC hook a cascade fires once an event's
        delivery set has run, an update that changes something draws one of
        that cascade's hook calls before it writes (RFC §9.5.1), and its
        ON_EVENT_UPDATED hooks run on the budget of the work it is made in,
        one of its own outside the room's work: a hook that updates its
        event on every update stops at the cap.

        Raises:
            ValueError: *content* holds more than five composite levels (RFC
                §5.3); nothing is read or written.
            ReentryLoopCapError: made from an ASYNC hook's body past the
                budget that body runs on; nothing is written.
        """
        refuse_too_deep(content)
        change = self._room_work.direct_change(room_id)
        async with self._lock_manager.locked(room_id):
            # At any status, once the room's due timer transitions are
            # applied (RFC §10.3, §5.1).
            if await self._room_in_scope(room_id, organization_id) is None:
                return None
            target = await self._store.get_event(event_id)
            if target is None or target.room_id != room_id:
                return None
            update: dict[str, Any] = {}
            if content is not None:
                update["content"] = content
            if source is not None:
                update["source"] = source
            if metadata is not None:
                update["metadata"] = with_committed_record(metadata, target)
            changed = target.model_copy(update=update)
            if changed == target:
                return target
            charge_direct_change(change, room_id, "update_event")
            # A context is store reads: only for a hook that will read it.
            hooked = self._hook_engine.has_hooks(HookTrigger.ON_EVENT_UPDATED)
            held = await self._build_context(room_id) if hooked else None
            updated = await self._store.update_event(changed)
        if held is not None:
            recent = [updated if e.id == updated.id else e for e in held.recent_events]
            context = await self._context_after_change(
                room_id, held.model_copy(update={"recent_events": recent})
            )
            self._hook_engine._start_async_hooks(
                room_id, HookTrigger.ON_EVENT_UPDATED, updated, context, budget=change.budget
            )
        return updated

    async def delete_event(
        self,
        room_id: str,
        event_id: str,
        *,
        cascade_replies: bool = True,
        organization_id: str | None = None,
    ) -> list[str]:
        """Hard-delete a persisted event, cascading to thread replies by default.

        Whatever *cascade_replies* says, the edits and deletes of every event
        removed go with it (RFC §10.3): an edit holds its target's corrected
        content. Fires ON_EVENT_DELETED once per removed event, each with the
        event as it was stored, once the room lock is released, and returns
        without waiting for its hooks (RFC §9.3). Returns the removed event
        IDs (the root, its replies, then the changes), or an empty list when
        *event_id* does not exist in *room_id* (no hook fires). Scoped to
        *organization_id* (RFC §17.2), the room of another organization is a
        missing room: an empty list, nothing written, its due timers
        included; unset, the call is unscoped.

        Made from the body of an ASYNC hook a cascade fires once an event's
        delivery set has run, a delete of an event the room holds draws one
        of that cascade's hook calls before it writes (RFC §9.5.1), and its
        ON_EVENT_DELETED hooks run on the budget of the work it is made in,
        one of its own outside the room's work.

        Raises:
            ReentryLoopCapError: made from an ASYNC hook's body past the
                budget that body runs on; nothing is written.
        """
        change = self._room_work.direct_change(room_id)
        async with self._lock_manager.locked(room_id):
            if await self._room_in_scope(room_id, organization_id) is None:
                return []
            target = await self._store.get_event(event_id)
            if target is None or target.room_id != room_id:
                return []
            charge_direct_change(change, room_id, "delete_event")
            # A context is store reads: only for a hook that will read it.
            hooked = self._hook_engine.has_hooks(HookTrigger.ON_EVENT_DELETED)
            held = await self._build_context(room_id) if hooked else None
            removed = await self._store.delete_event(
                room_id, event_id, cascade_replies=cascade_replies
            )
        if removed and held is not None:
            gone = {event.id for event in removed}
            recent = [e for e in held.recent_events if e.id not in gone]
            context = await self._context_after_change(
                room_id, held.model_copy(update={"recent_events": recent})
            )
            for event in removed:
                self._hook_engine._start_async_hooks(
                    room_id, HookTrigger.ON_EVENT_DELETED, event, context, budget=change.budget
                )
        return [event.id for event in removed]

    async def _context_after_change(self, room_id: str, held: RoomContext) -> RoomContext:
        """The room's context for a direct change's triggers, read once the
        room lock is released (RFC §10.3); *held*, the context the call held
        under the lock as the change left it, when that read fails: the change
        is stored and stands (§13.6), and the failure is logged."""
        try:
            return await self._build_context(room_id)
        except Exception:
            logger.warning(
                "Room %s could not be read once the change's lock was released; its "
                "triggers fire with the room as the change left it",
                room_id,
                exc_info=True,
                extra={"room_id": room_id},
            )
            return held

    async def _room_in_scope(self, room_id: str, organization_id: str | None) -> Room | None:
        """The room, its due timer transitions applied (RFC §5.1), or ``None``
        when the store holds none or it is another organization's than
        *organization_id* (RFC §17.2), which is read before any transition
        is written. Call under the room lock."""
        try:
            return await self._room_timers.room_due(room_id, organization_id=organization_id)
        except RoomNotFoundError:
            return None

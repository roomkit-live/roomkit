"""A room's timer transitions, applied when they fall due (RFC §5.1).

Every locked section that reads a room's status (the gate of §10.1 step 6,
an attach or a binding change, a direct mutation, a record's commit) first
applies the transitions due at its instant (:meth:`TimerTransitions.apply_due`),
so an event that arrives after its room fell due is refused as if the timer
had run on time. A room nobody writes to transitions when the host checks its
timers (:meth:`TimerTransitions.check`, :meth:`TimerTransitions.check_all`).

Each transition takes the instant it fell due, all of them in one write; each
emits its framework event once stored, before anything the section decides
next, and fires its hook once the section has released the room lock.

The kit builds one :class:`TimerTransitions` and hands it what it needs: the
store, the lock manager, and the kit's own writers of what a transition
reports. Not part of the public API.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit.core._room_timers import DueTransition, due_transitions, transitioned
from roomkit.core.locks import _has_room_lock, unlocked_context
from roomkit.core.task_utils import await_unless_cut, run_to_its_end, start_named
from roomkit.models.enums import EventType, HookTrigger, RoomStatus

if TYPE_CHECKING:
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.room import Room
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")

# Writes a transition tries before it leaves the room to the next section,
# each one beaten by a commit off the room lock (a streamed row, a greeting).
_ATTEMPTS = 3

RoomRead = Callable[..., Awaitable["Room"]]
"""A scoped room read: ``(room_id, *, organization_id)``, raising
``RoomNotFoundError`` for a room missing or another organization's."""


def _report(room_id: str, transition: DueTransition) -> tuple[HookTrigger, str, str]:
    """The trigger, the code and the message of *transition*."""
    if transition.status is RoomStatus.PAUSED:
        trigger, verb = HookTrigger.ON_ROOM_PAUSED, "paused"
    else:
        trigger, verb = HookTrigger.ON_ROOM_CLOSED, "closed"
    return trigger, f"room_{verb}_by_timer", f"Room {room_id} {verb} by timer"


def _data(transition: DueTransition) -> dict[str, Any]:
    """What a transition's record and hook event carry: the threshold that
    elapsed and the instant it fell due."""
    return {"threshold": transition.threshold, "at": transition.at.isoformat()}


def _log_failed_announcement(task: asyncio.Task[None]) -> None:
    if task.cancelled() or (error := task.exception()) is None:
        return
    logger.error("A room's timer hooks failed", exc_info=error)


class TimerTransitions:
    """Applies a room's due timer transitions (RFC §5.1): the host's check
    and every locked section's.

    *read_room* is the kit's scoped read (``get_room``); *count_closed*
    counts a room stored in a status that refuses events, for the streams
    that read the status once (RFC §5.1); *stop_recordings*
    stops a closing room's recordings; *write_record* writes a lifecycle
    record (``_emit_system_event``); *emit* a framework event; *fire_hook*
    a lifecycle hook in a given context; *build_context* reads a room's
    context. *background* is the set of the kit's background tasks, which
    its close cancels and waits for; *writes* the set of its writes past a
    decision, which its close waits for and never cancels.
    """

    def __init__(
        self,
        *,
        store: ConversationStore,
        lock_manager: RoomLockManager,
        read_room: RoomRead,
        count_closed: Callable[[], None],
        stop_recordings: Callable[[str], Awaitable[Any]],
        write_record: Callable[..., Awaitable[None]],
        emit: Callable[..., Awaitable[None]],
        fire_hook: Callable[..., Awaitable[None]],
        build_context: Callable[[str], Awaitable[RoomContext]],
        background: set[asyncio.Task[Any]],
        writes: set[asyncio.Task[Any]],
    ) -> None:
        self._store = store
        self._lock_manager = lock_manager
        self._read_room = read_room
        self._count_closed = count_closed
        self._stop_recordings = stop_recordings
        self._write_record = write_record
        self._emit = emit
        self._fire_hook = fire_hook
        self._build_context = build_context
        self._background = background
        self._writes = writes

    async def check(self, room_id: str, *, organization_id: str | None = None) -> Room:
        """Apply the room's due transitions under its lock, for a room nobody
        writes to, and return the room as stored once they are; their hooks
        have started when this returns (outside a lock held from before), not
        ended (RFC §9.3)."""
        hooks: list[asyncio.Task[None]] = []
        async with self._lock_manager.locked(room_id):
            room = await self._read_room(room_id, organization_id=organization_id)
            room = await self.apply_due(room, hooks=hooks)
        await self.settle(room_id, hooks)
        return room

    async def check_all(self, *, organization_id: str | None = None) -> list[Room]:
        """:meth:`check` every ACTIVE and PAUSED room, one tenant's with
        *organization_id*, and return the rooms that transitioned."""
        rooms = [
            room
            for status in (RoomStatus.ACTIVE, RoomStatus.PAUSED)
            for room in await self._store.find_rooms(
                organization_id=organization_id, status=status.value
            )
        ]
        changed: list[Room] = []
        for room in rooms:
            checked = await self.check(room.id, organization_id=organization_id)
            if checked.status != room.status:
                changed.append(checked)
        return changed

    async def apply_due(
        self, room: Room, *, hooks: list[asyncio.Task[None]] | None = None
    ) -> Room:
        """*room* with the transitions due now applied, as stored once they
        are, their records counted; call under its lock, before the section
        reads its status. *room* itself when nothing is due, which costs no
        I/O.

        The transitions are written in one write, each at the instant it fell
        due (:meth:`_write_due`); a close then stops the room's recordings.
        Each then has its lifecycle record and its framework event, before
        anything the section decides next; their hooks fire once the lock
        is released, from a task the kit holds, which *hooks* collects for a
        caller that waits for it (:meth:`settle`). Once due, they are written
        and reported whole whatever becomes of the caller (RFC §13.6): no
        later section would find them due again.
        """
        if not due_transitions(room, datetime.now(UTC)):
            return room
        return await run_to_its_end(
            self._apply(room, hooks), held=self._writes, name=f"roomkit-decided-write-{room.id}"
        )

    async def _apply(self, room: Room, hooks: list[asyncio.Task[None]] | None) -> Room:
        """:meth:`apply_due`'s writes and reports, for a room whose
        transitions fell due."""
        room, due = await self._write_due(room)
        if not due:
            return room
        if room.status is RoomStatus.CLOSED:
            self._count_closed()
            await self._stop_recordings(room.id)
        for transition in due:
            await self._announce(room.id, transition)
        # Read once the records are written: the room as stored, which the
        # hooks receive and the caller is handed.
        context = await self._build_context(room.id)
        task = self._fire_hooks(room.id, due, context)
        if hooks is not None:
            hooks.append(task)
        return context.room

    async def _write_due(self, room: Room) -> tuple[Room, list[DueTransition]]:
        """Write the transitions due on *room*, a compare-and-set on the
        status, ``paused_at`` and ``last_activity_at`` they were computed
        from (``update_room_if``): a commit off the room lock (a streamed
        row, a greeting) that resumed the room or moved its activity in
        between is never overwritten, and the transitions are those due on
        the room it left. The room written and its transitions, or the room
        as last read and none."""
        for _ in range(_ATTEMPTS):
            now = datetime.now(UTC)
            due = due_transitions(room, now)
            if not due:
                return room, []
            written = await self._store.update_room_if(
                transitioned(room, due, now),
                status=room.status,
                paused_at=room.paused_at,
                last_activity_at=room.timers.last_activity_at,
            )
            if written is not None:
                return written, due
            fresh = await self._store.get_room(room.id)
            if fresh is None:
                return room, []
            room = fresh
        return room, []

    async def room_due(self, room_id: str, *, organization_id: str | None = None) -> Room | None:
        """The room, read under its lock with its due transitions applied, or
        ``None`` when the store holds none. Scoped to *organization_id*, the
        read raises as the kit's ``get_room`` does."""
        if organization_id is not None:
            room: Room | None = await self._read_room(room_id, organization_id=organization_id)
        else:
            room = await self._store.get_room(room_id)
        return None if room is None else await self.apply_due(room)

    async def due_context(
        self, context: RoomContext, rebuild: Callable[[], Awaitable[RoomContext]]
    ) -> RoomContext:
        """*context*, read under the room lock, once the room's due
        transitions are applied: rebuilt by *rebuild* when one was, since
        they wrote the room and its records."""
        if not due_transitions(context.room, datetime.now(UTC)):
            return context
        await self.apply_due(context.room)
        return await rebuild()

    async def settle(self, room_id: str, hooks: list[asyncio.Task[None]]) -> None:
        """Wait, once the caller's section released the lock, for the task
        :meth:`apply_due` started to start their hooks; not under a lock held
        from before it, which that task waits for. The kit's close cutting it
        ends the wait quietly: the caller was not cancelled."""
        if hooks and not _has_room_lock(room_id, self._lock_manager):
            await await_unless_cut(asyncio.shield(asyncio.gather(*hooks)))

    async def _announce(self, room_id: str, transition: DueTransition) -> None:
        """*transition*'s lifecycle record and its framework event, once the
        room is stored (RFC §5.1, §8.2)."""
        _, code, message = _report(room_id, transition)
        logger.info(message, extra={"room_id": room_id})
        await self._write_record(
            room_id,
            EventType.SYSTEM,
            code=code,
            message=message,
            data=_data(transition),
            # The record of the transition itself, written once the status
            # has flipped: the one write a CLOSED room still owes its
            # timeline (RFC §5.1).
            records_transition=True,
        )
        name = "room_paused" if transition.status is RoomStatus.PAUSED else "room_closed"
        await self._emit(name, room_id=room_id, data={"reason": "timer"})

    def _fire_hooks(
        self, room_id: str, due: list[DueTransition], context: RoomContext
    ) -> asyncio.Task[None]:
        """Start *due*'s hooks, in order, with *context*, the room as the
        transitions left it, once the section that applied them has released
        the room lock (RFC §5.1, §9.3).

        A task started in a context that holds no lock, which asks for the
        lock as any later entry and waits for the section, and any ahead of
        it, to release it; held with the kit's background work.
        """

        async def fire() -> None:
            async with self._lock_manager.locked(room_id):
                pass
            for transition in due:
                trigger, code, message = _report(room_id, transition)
                await self._fire_hook(
                    room_id,
                    trigger,
                    EventType.SYSTEM,
                    code=code,
                    message=message,
                    data=_data(transition),
                    context=context,
                )

        task = start_named(
            fire(), name=f"roomkit-timer-hooks-{room_id}", context=unlocked_context()
        )
        task.add_done_callback(_log_failed_announcement)
        task.add_done_callback(self._background.discard)
        self._background.add(task)
        return task

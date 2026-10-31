"""InboundLockedMixin — the locked half of inbound processing (RFC §10.1).

Everything here runs under the room lock and ends at the commit point: the
status gate, idempotency, index assignment, hooks, permission checks, the
atomic commit and the broadcast *planning* (§10.1 step 12). Execution of the
delivery set, the reentry passes and the async hooks live off the lock, in
the room's delivery lane (:mod:`roomkit.core.lanes`,
:class:`~roomkit.core.mixins.lane_execution.LaneExecutionMixin`).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Coroutine
from dataclasses import KW_ONLY, dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core._authors import People
from roomkit.core._changes import (
    NO_CHANGE,
    TARGET_NOT_FOUND,
    ChangeCheck,
    applied,
    authority_refusal,
    change_of,
    concerns_change,
    target_refusal,
    with_target,
)
from roomkit.core._observed import Observed, observed_once_over
from roomkit.core._queued_injections import QueuedInjections
from roomkit.core._reentrance import section
from roomkit.core._room_timers import due_transitions
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.hooks import kept_identity
from roomkit.core.lanes import DeliveryPlan
from roomkit.core.mixins.helpers import (
    _RECENT_EVENTS_LIMIT,
    _REFUSING_STATUSES,
    HelpersMixin,
    _RefusedOperation,
    _source_block_reason,
)
from roomkit.core.task_utils import run_to_its_end
from roomkit.core.visibility import (
    answer_scope,
    intersect_visibility,
    narrowed_to_target,
    with_committed_visibility,
)
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    EventStatus,
    EventType,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import DeleteContent, EditContent, RoomEvent
from roomkit.models.hook import InjectedEvent
from roomkit.models.identity import Identity, IdentityResult
from roomkit.models.task import Observation, Task

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.event_router import EventRouter
    from roomkit.core.hooks import HookEngine, SyncPipelineResult
    from roomkit.core.lanes import DeliveryCascade
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.room import Room
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


@dataclass(slots=True, eq=False)
class _Blocked:
    """Decision returned by ``_gate_commit``: the event is blocked and must
    be persisted BLOCKED (RFC §9.5 / §7.5 rule 2), with what its hooks
    decided (§7.5 rule 3).

    Carries everything ``_store_blocked`` needs; the inbound path writes it
    in :meth:`_process_locked`, outside the cancellable unit (§13.6).
    """

    event: RoomEvent
    context: RoomContext
    _: KW_ONLY
    reason: str | None
    blocked_by: str | None
    injected_events: list[InjectedEvent]
    tasks: list[Task]
    observations: list[Observation]

    @classmethod
    def after_hooks(
        cls,
        event: RoomEvent,
        context: RoomContext,
        sync_result: SyncPipelineResult,
        *,
        reason: str | None,
        blocked_by: str | None,
    ) -> _Blocked:
        """A block that keeps what the hooks decided (RFC §7.5 rule 3)."""
        return cls(
            event,
            context,
            reason=reason,
            blocked_by=blocked_by,
            injected_events=sync_result.injected_events,
            tasks=sync_result.tasks,
            observations=sync_result.observations,
        )


@dataclass(slots=True, eq=False)
class _Ready:
    """Decision returned by ``_gate_commit``: every gate passed and the
    event is ready to commit (RFC §10.1 step 12).

    The gate never commits. The inbound path commits in :meth:`_run_commit`,
    outside the ``process_timeout`` window, because a timeout that expires
    mid-commit would leave a committed event reported as blocked (§13.6 MUST
    NOT); a reentry pass in ``_commit_reentry``; a streamed row on its lane.
    """

    event: RoomEvent
    source_binding: Any
    sync_result: Any
    context: RoomContext
    edit_delete_target: RoomEvent | None


@dataclass(frozen=True, slots=True, eq=False)
class _AfterHooks:
    """What the gate reads of the room once an event's hooks ran (RFC §10.1
    step 9a): the context the event commits with, its source's binding as
    the hooks left it, and the room's next index, ``None`` when no hook ran
    under the lock."""

    context: RoomContext
    binding: ChannelBinding | None
    index: int | None


def _source_gate(ready: _Ready, *, source_check: bool) -> _Blocked | _Ready:
    """RFC §10.1 step 11 for *ready*, an event its hooks allowed, with its
    source's binding as the hooks left it: its right to write unless
    *source_check* is off.

    RFC §7.5 — a source that cannot write (READ_ONLY/NONE or muted) must not
    inject a DELIVERED event into the timeline. It is stored BLOCKED for
    audit, its hook side effects still collected (§7.5 rule 3), and stops
    before broadcast.
    """
    if not source_check:
        return ready
    # Delivered or stored BLOCKED in its place, the event takes the
    # visibility step 12 gives it (RFC §7.3).
    event = with_committed_visibility(ready.event, ready.source_binding, ready.context.bindings)
    reason = _source_block_reason(ready.source_binding)
    if reason is not None:
        return _Blocked.after_hooks(
            event, ready.context, ready.sync_result, reason=reason, blocked_by=reason
        )
    return replace(ready, event=event)


def at_index(event: RoomEvent, index: int | None) -> RoomEvent:
    """*event* at *index*, the room's next once its hooks ran (RFC §10.1
    step 9a); *event* itself without one, or already at it. A stored event
    takes its index again at its commit; an instruction, never stored,
    reports this one (§10.1.1)."""
    if index is None or event.index == index:
        return event
    return event.model_copy(update={"index": index})


def within_answer_scope[D: (_Ready, _Blocked)](decision: D, scope: str | None) -> D:
    """*decision*, a response's or a stream row's, its event narrowed back
    within *scope* once its hooks ran (RFC §5.12, §9.4): a hook may change
    the visibility as on any event, never past where the answers of the
    message its cascade derives from may go. A list is read against the
    categories of the room's bindings."""
    scope = answer_scope(scope)
    if scope is None:
        return decision
    event = decision.event
    categories = {binding.channel_id: binding.category for binding in decision.context.bindings}
    visibility = intersect_visibility(event.visibility, scope, categories.get)
    if visibility == event.visibility:
        return decision
    return replace(decision, event=event.model_copy(update={"visibility": visibility}))


@dataclass(frozen=True, slots=True)
class _Refused:
    """Decision returned by ``_gate_commit``: once its hooks ran under the
    lock, the room was read again (RFC §10.1 step 9a) and refused the event:
    a hook closed the room (``room_closed``), or the edit or the delete its
    hooks let through no longer had a target it may change. Nothing of it is
    stored, its hooks' results included."""

    reason: str


@dataclass(frozen=True, slots=True, eq=False)
class _TimersDue:
    """Decision returned by ``_run_precommit``: the room's timers fell due
    before its section (RFC §5.1). The caller applies them outside the
    ``process_timeout`` window, the room's state and not the event's, and
    decides again on the room they leave."""

    room: Room


async def _within[T](deadline: float, work: Coroutine[Any, Any, T]) -> T | None:
    """*work*'s result, or ``None`` when *deadline* (the loop's clock) passed
    first, the work cancelled: the ``process_timeout`` window (RFC §13.6)."""
    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
    try:
        return await asyncio.wait_for(work, timeout=remaining)
    except TimeoutError:
        return None


@dataclass(slots=True, eq=False)
class _Proceed:
    """Marker returned by ``_run_commit`` once the event has committed,
    with its broadcast planned and enqueued on the room's delivery lane
    (RFC §10.1 step 12).

    Carries the state the caller still needs after the commit — kept off the
    event so the phases stay decoupled. ``mutation_hook`` is the (trigger,
    updated target) pair for an edit/delete (RFC §10.3); it rides the plan
    and fires from the lane executor.
    """

    event: RoomEvent
    source_binding: Any
    sync_result: Any
    context: RoomContext
    mutation_hook: tuple[HookTrigger, RoomEvent] | None = None


@runtime_checkable
class InboundLockHost(Protocol):
    """Contract: capabilities a host class must provide for InboundLockedMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation store for events, bindings, participants.
        _channels: Channel registry for injected-event delivery.
        _hook_engine: Hook engine for BEFORE_BROADCAST / AFTER_BROADCAST.
        _max_chain_depth: Maximum reentry chain depth (RFC §10).

    Methods provided by the host class (RoomKit):
        _get_router: Lazily create / return the ``EventRouter`` for broadcast.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int

    def _get_router(self) -> EventRouter: ...


class InboundLockedMixin(HelpersMixin):
    """Locked inbound processing: gates, commit, broadcast planning.

    Host contract: :class:`InboundLockHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int
    _process_timeout: float
    _discussions: dict[str, Any]  # rooms a discussion holds (RFC §19.7.5)

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _commit_to_lane: Any  # see LaneExecutionMixin

    # Stub for cross-mixin call — implemented by RoomKit._get_router().
    def _get_router(self) -> EventRouter: ...

    async def _rooted(self, room_id: str, before: RoomEvent, after: RoomEvent) -> RoomEvent:
        """*after*, the event as its hooks left it, with a thread parent a
        hook set resolved to its thread root in this room (§9.4 lets a hook
        change the thread; a reply always points at a root)."""
        parent = after.parent_event_id
        if parent is None or parent == before.parent_event_id:
            return after
        root = await self._resolve_thread_root(room_id, parent)
        return after if root == parent else after.model_copy(update={"parent_event_id": root})

    async def _check_change(
        self, room_id: str, event: RoomEvent, *, by_hook: bool = False
    ) -> ChangeCheck:
        """RFC §10.1 step 8a (§10.3 validation), under the room lock: what
        *event* is as a change, the target it names and who may make it.

        A refusal is logged at warning level: the inbound path returns it
        as a blocked result, the other passes drop the event, nothing stored.
        *by_hook* is a change a hook injected (§9.4), the hook's and a
        SYSTEM change only.
        """
        if not concerns_change(event):
            return ChangeCheck()
        change = change_of(event)
        target = None if change is None else await self._store.get_event(change.target_event_id)
        reason = TARGET_NOT_FOUND if change is None else target_refusal(target, room_id)
        if reason is None and change is not None and target is not None:
            # Read now: a sender the identity pipeline has just recorded is
            # not on the context's roster yet.
            people = People(await self._store.list_participants(room_id))
            reason = authority_refusal(change, event, target, people, by_hook=by_hook)
            if reason is None:
                return ChangeCheck(target=target)
        logger.warning(
            "Edit/Delete refused in room %s: %s",
            room_id,
            reason,
            extra={"room_id": room_id, "event_id": event.id, "reason": reason},
        )
        return ChangeCheck(refusal=reason)

    async def _commit_event(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Commit an event to the timeline (RFC §10.1 step 12 / §14.3).

        Persists the event and bumps the room counters (event_count,
        latest_index, timers.last_activity_at) as ONE atomic store transaction
        (:meth:`ConversationStore.commit_event`), so an observer never sees a
        stored event the room counters do not reflect, and the authoritative
        index is assigned inside that transaction (§8.1) — safe even without a
        cross-process room lock. Returns the committed event (its index may
        differ from the provisional pre-hook value if the store serialized a
        concurrent writer).

        When the persistence policy excludes the event, nothing is stored: the
        event is delivered but not persisted, so it consumes no index and MUST
        NOT advance the room counters (latest_index must never point at an
        unstored event, §14.3). The input event is returned unchanged.
        """
        committed = await self._persist_committed(room_id, event, activity=True)
        return event if committed is None else committed

    async def _process_locked(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext | None,
        cascade: DeliveryCascade,
        *,
        queued: QueuedInjections,
        resolved_identity: Identity | None = None,
        pending_id_result: IdentityResult | None = None,
        deadline: float | None = None,
        precheck: SyncPipelineResult | None = None,
    ) -> InboundResult:
        """Process an event under the room lock (RFC §10.1).

        Split at the commit point (RFC §10.1): the pre-commit critical
        section (:meth:`_run_precommit`) is bounded by ``process_timeout``
        (§13.6) and decides — it writes nothing to the timeline. Every durable
        write happens below, after the cancellable unit has been left: a
        timeout MUST NOT be able to leave a committed event reported as
        blocked (§13.6). The commit plans the broadcast and enqueues it on the
        room's delivery lane (§10.2) — execution, the RFC §10.3 mutation
        trigger, AFTER_BROADCAST and ON_ERROR hooks all run off the lock,
        tracked by *cascade*; the caller awaits the cascade once the lock is
        released.

        *context* is the context the caller built BEFORE taking the lock — the
        inbound pipeline builds one for the channel and the identity resolver
        (RFC §10.1 steps 3-5) — and it is carried into the locked rebuild
        (:meth:`_build_context`). A caller that already holds the lock has no
        such context and passes ``None``.

        *precheck* is the outcome of the off-lock check (RFC §9.5.1), when
        the caller ran one: step 9 applies it and runs only the hooks that
        need the lock. ``None`` runs every hook here.

        The events its hooks inject, whichever way the event ended, are
        queued on *queued*: the caller runs their passes once it has
        released the room lock (RFC §9.4, §10.1 steps 10 to 14).
        """
        # What is left of the caller's single pre-commit budget (RFC §13.6).
        # A deadline rather than the raw setting: the phase starts before this
        # method — context build, ``handle_inbound``, identity, the wait for
        # the room lock — and re-reading ``process_timeout`` here would let a
        # caller configured for 30s wait 60. ``None`` means the caller did not
        # open a window (a reentry pass, a direct commit), so this region is
        # the whole of it.
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + self._process_timeout
        # The section of this pass: what its hooks send into the room is
        # nested here, on the cascade's budget, its injected events queued
        # on *queued* (RFC §9.5.1 Reentrance, §9.4).
        # Its BEFORE_BROADCAST observers start once it is over, its event's
        # set placed in the room's lane (RFC §10.1).
        with section(room_id, cascade.budget, queued):
            async with observed_once_over(
                self._hook_engine, room_id, closing=lambda: self._closed
            ) as observed:
                return await self._decide_and_commit(
                    event,
                    room_id,
                    context,
                    cascade,
                    queued=queued,
                    observed=observed,
                    resolved_identity=resolved_identity,
                    pending_id_result=pending_id_result,
                    deadline=deadline,
                    precheck=precheck,
                )

    async def _decide_and_commit(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext | None,
        cascade: DeliveryCascade,
        *,
        queued: QueuedInjections,
        observed: Observed,
        resolved_identity: Identity | None,
        pending_id_result: IdentityResult | None,
        deadline: float,
        precheck: SyncPipelineResult | None,
    ) -> InboundResult:
        """:meth:`_process_locked`'s decision within its deadline, then its
        writes past the commit point."""
        precommit = functools.partial(
            self._run_precommit,
            event,
            room_id,
            context,
            cascade,
            observed=observed,
            resolved_identity=resolved_identity,
            pending_id_result=pending_id_result,
            precheck=precheck,
        )
        decision = await _within(deadline, precommit())
        while isinstance(decision, _TimersDue):
            # RFC §5.1, §10.1 step 6: the room's due transitions apply first,
            # out of the window, and the gate reads the status they leave;
            # once, since the second pass checks no timer.
            await self._room_timers.apply_due(decision.room)
            decision = await _within(deadline, precommit(timers=False))
        if decision is None:
            return await self._refuse_on_timeout(room_id, event_id=event.id)

        # --- past this line nothing is cancellable by ``process_timeout`` ---
        if isinstance(decision, InboundResult):
            return decision
        # Nor by the caller (RFC §13.6): a cancellation waits, under the lock,
        # for the writes to end, then is raised.
        return await run_to_its_end(
            self._write_decision(decision, room_id, cascade, queued),
            held=self._decided_writes,
            name=f"roomkit-decided-write-{room_id}",
        )

    async def _write_decision(
        self,
        decision: _Blocked | _Ready,
        room_id: str,
        cascade: DeliveryCascade,
        queued: QueuedInjections,
    ) -> InboundResult:
        """Every durable write of a decided event: its BLOCKED record or its
        commit, enqueued on the room's lane, and what its hooks injected
        queued on *queued*."""
        # Read back if the kit's close cuts these writes as they land: the
        # call then reports its event stored (RFC §12.10.4).
        if isinstance(decision, _Blocked):
            self._door_runs.committing(decision.event, blocked_reason=decision.reason)
            blocked_event = await self._store_blocked(room_id, decision, queued)
            return InboundResult(event=blocked_event, blocked=True, reason=decision.reason)
        self._door_runs.committing(decision.event)
        outcome = await self._run_commit(decision, room_id, cascade)
        # Injected events from allow/modify hooks: queued, each committed
        # after the trigger once its section has released the lock (the
        # trigger is committed and must never be contradicted by a late
        # cancellation).
        queued.add(outcome.sync_result.injected_events, outcome.event)
        return InboundResult(
            event=outcome.event, unavailable_targets=list(cascade.unavailable_targets)
        )

    async def _run_precommit(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext | None,
        cascade: DeliveryCascade,
        *,
        observed: Observed,
        resolved_identity: Identity | None = None,
        pending_id_result: IdentityResult | None = None,
        precheck: SyncPipelineResult | None = None,
        timers: bool = True,
    ) -> InboundResult | _Blocked | _Ready | _TimersDue:
        """Pre-commit critical section (RFC §10.1 steps 6-11) — decides only.

        Returns an :class:`InboundResult` for a refusal that leaves no trace
        (closed room, duplicate, invalid edit/delete), a :class:`_Blocked` for
        one that owes the timeline a BLOCKED record, or a :class:`_Ready` for
        an event that passed every gate; with *timers*, a
        :class:`_TimersDue` for a room whose timer transitions fell due,
        before anything else is read.

        Writes nothing to the timeline: the caller performs every durable
        write once this coroutine has returned, so a ``process_timeout``
        expiring here aborts cleanly with nothing persisted and can never
        report a committed event as blocked (§13.6). The one write that stays
        inside the window is the deferred participant row (identity
        resolution, §11): it is idempotent and carries no index, so a timeout
        leaves no inconsistency behind.
        """
        # Re-read under the lock: the status gate must not act on an answer
        # taken before it (§10.1 step 6), and planning reads bindings under the
        # lock (step 12). The pre-lock context comes along: its history is the
        # expensive half of a context, and it is carried whenever the room's
        # counter proves nothing committed since it was read.
        # RFC §10.1.1 — an instruction is the caller's (a parser's INSTRUCTION
        # type is put back to the message's), and its address and its
        # caller's key were checked before anything was read; a key its
        # channel's parser set asks the same check of a stored event it will
        # never be.
        if event.type == EventType.INSTRUCTION:
            if event.idempotency_key:
                return InboundResult(blocked=True, reason="instruction_not_idempotent")
            # RFC §19.7.5 rule 15 — a discussion that is over gives no turn.
            discussion = self._discussions.get(room_id)
            if discussion is not None and discussion.state.over:
                return InboundResult(blocked=True, reason="discussion_over")
            # Intelligence channels only, whatever the caller asked: no
            # transport delivers it, no voice channel speaks it.
            event = event.model_copy(update={"visibility": Visibility.INTELLIGENCE})

        context = await self._build_context(room_id, carrying=context)
        if timers and due_transitions(context.room, datetime.now(UTC)):
            return _TimersDue(context.room)

        # RFC §5.1 / §10.1 step 6 — a room whose status refuses new events
        # refuses them here, at the one point every entry converges on: inbound
        # messages, send_event(), hook-injected events and the framework's own
        # re-injection all pass through this. Under the lock, because
        # close_room() takes the same one and an earlier answer can be stale by
        # the time the event would commit.
        #
        # Nothing is written, not even a BLOCKED record: appending an audit
        # event to a closed room is the thing the status forbids (§5.1).
        if context.room.status in _REFUSING_STATUSES:
            return await self._refuse_closed_room(
                room_id, status=context.room.status, operation="inbound", event=event
            )

        # Persist deferred participant creation inside the lock (Fix #1)
        if resolved_identity is not None:
            await self._ensure_identified_participant(room_id, event, resolved_identity)
        elif pending_id_result is not None:
            await self._create_pending_participant(room_id, event, pending_id_result)

        # Idempotency check (inside lock to prevent TOCTOU race). A duplicate
        # replays the first delivery's answer rather than reprocessing (RFC
        # §13.4) — the caller retried because it never saw that answer, and a
        # refusal does not tell it the message was in fact accepted. A store
        # that cannot resolve the key still reports the duplicate as blocked.
        # A key is unique per room and per source channel (RFC §13.4).
        if event.idempotency_key and await self._store.check_idempotency(
            room_id, event.idempotency_key, channel_id=event.source.channel_id
        ):
            original = await self._store.get_event_by_idempotency_key(
                room_id, event.idempotency_key, channel_id=event.source.channel_id
            )
            logger.info(
                "Duplicate event %s",
                event.idempotency_key,
                extra={
                    "room_id": room_id,
                    "idempotency_key": event.idempotency_key,
                    "original_event_id": original.id if original is not None else None,
                },
            )
            if original is not None and original.status == EventStatus.BLOCKED:
                # The first delivery was refused: so is its replay, with the
                # record and the reason it was stored with (RFC §13.4).
                return InboundResult(
                    event=original, blocked=True, reason=original.blocked_by, duplicate=True
                )
            if original is not None:
                return InboundResult(event=original, duplicate=True)
            return InboundResult(blocked=True, reason="duplicate", duplicate=True)

        # Provisional index for the hooks. The authoritative one is (re)assigned
        # inside the commit (§8.1), so this reads the counter the context
        # already carries under the lock rather than paying a COUNT(*) over the
        # room's whole timeline on every single inbound.
        event = event.model_copy(update={"index": context.room.event_count})

        # The in-app thread parent, rooted (RFC §10.1 step 8): direct
        # send_event and inbound both traverse here (§10.5), and every other
        # path that commits an event roots it at its own commit.
        event = await self._with_thread_root(room_id, event)

        # Edit/Delete validation (RFC §10.1 step 8a). The target mutation is
        # deferred to the commit, once BEFORE_BROADCAST hooks allow the event,
        # so a moderation hook that blocks an edit/delete cannot leave the
        # target already mutated.
        check = await self._check_change(room_id, event)
        if check.refusal is not None:
            return InboundResult(blocked=True, reason=check.refusal)

        decision = await self._gate_commit(
            room_id, event, context, precheck, change=check, observed=observed
        )
        if isinstance(decision, _Refused):
            return InboundResult(blocked=True, reason=decision.reason)
        return decision

    async def _gate_commit(
        self,
        room_id: str,
        event: RoomEvent,
        context: RoomContext,
        precheck: SyncPipelineResult | None = None,
        *,
        observed: Observed,
        source_check: bool = True,
        change: ChangeCheck = NO_CHANGE,
        under_lock: bool = True,
        operation: _RefusedOperation = "inbound",
    ) -> _Blocked | _Ready | _Refused:
        """RFC §10.1 steps 9 to 11 for an event about to commit: its
        BEFORE_BROADCAST hooks, then its source's right to write, unless
        *source_check* is off for an event a hook injected (§9.4), which is
        the hook's, not a channel's.

        Once a hook ran *under_lock*, the room is read again (step 9a): a
        room a hook closed refuses the event as step 6 does, announced for
        *operation* (:class:`_Refused`). A streamed row's hooks run off the
        lock, and its stream reads the room's status itself (§5.1).

        *change* is what step 8a accepted of an edit or a delete: once a
        hook ran under the lock, its target is read and validated again
        (step 9a), so an entry committed during the hooks (a delete of the
        target) refuses it rather than be undone by it (:class:`_Refused`).

        The one gate of every commit path, an inbound event and a response
        alike, buffered, streamed or regenerated (RFC §7.5 rule 2): a hook
        that blocks names the block, and a source that cannot write is
        refused only once its hooks have run, so what they decided is kept.

        Decides only: a :class:`_Blocked` carries what the hooks decided, for
        :meth:`_store_blocked`; a :class:`_Ready` carries the event as the
        hooks left it, with its source's binding as the hooks left it
        (:meth:`_after_hooks`). A chain that allowed the event owes its ASYNC
        observers that event, as its SYNC hooks left it, on *observed*: its
        pass starts them once it is over (RFC §10.1).
        """
        ran = self._hooks_run_here(room_id, event, precheck)
        # Run sync hooks (before_broadcast). After an off-lock check (RFC
        # §9.5.1) its outcome comes first and only the locked hooks run here.
        sync_result = await self._run_before_broadcast(room_id, event, context, precheck)
        if sync_result.allowed:
            final = sync_result.event if isinstance(sync_result.event, RoomEvent) else event
            observed.owe(final, context)
        await self._announce_hook_errors(room_id, event, sync_result)
        after = await self._after_hooks(
            room_id, event, context, ran=ran, under_lock=under_lock, operation=operation
        )
        if after is None:
            return _Refused("room_closed")
        event, context = at_index(event, after.index), after.context

        if not sync_result.allowed:
            if source_check:
                # A channel's record sees no more than its event would have
                # (RFC §7.3); a hook's injected event keeps its own (§9.4).
                event = with_committed_visibility(event, after.binding, context.bindings)
            return _Blocked.after_hooks(
                event,
                context,
                sync_result,
                reason=sync_result.reason,
                blocked_by=sync_result.blocked_by,
            )

        # Use potentially modified event. HookResult.event carries whatever
        # payload its trigger passed, so only a RoomEvent may replace one.
        if isinstance(sync_result.event, RoomEvent):
            # A modify keeps the index its hooks were shown (RFC §9.4), which
            # gives way to the room's next once they ran (step 9a).
            rooted = await self._rooted(room_id, event, sync_result.event)
            event = at_index(rooted, after.index)
        target = change.target
        if ran and target is not None:
            # An injected change (no source check) is the hook's (§9.4).
            change = await self._check_change(room_id, event, by_hook=not source_check)
            if change.refusal is not None:
                return _Refused(change.refusal)
            target = change.target
        ready = _Ready(event, after.binding, sync_result, context, target)
        return _source_gate(ready, source_check=source_check)

    def _hooks_run_here(
        self, room_id: str, event: RoomEvent, precheck: SyncPipelineResult | None
    ) -> bool:
        """Whether a BEFORE_BROADCAST hook runs for *event* in the gate: every
        one without an off-lock check, the ones that need the lock after it
        (RFC §9.5.1), none after a check that blocked."""
        if precheck is not None and not precheck.allowed:
            return False
        return self._hook_engine.has_sync_hooks(
            room_id,
            HookTrigger.BEFORE_BROADCAST,
            event,
            needs_lock=None if precheck is None else True,
        )

    async def _after_hooks(
        self,
        room_id: str,
        event: RoomEvent,
        context: RoomContext,
        *,
        ran: bool,
        under_lock: bool,
        operation: _RefusedOperation,
    ) -> _AfterHooks | None:
        """What the gate reads once *event*'s hooks ran (RFC §10.1 step 9a).

        *context* holds the room as it was built: under the lock for an
        inbound event and a commit pass (steps 6 and 12), where only a hook
        that ran there changes it since; at the start of its stream for a
        streamed row, whose hooks run off the lock and whose stream reads the
        room's status itself (§5.1). With no hook *ran*, it is the answer.

        Once a hook ran under the lock, the room's context is read again, on
        one connection, whatever the hook did: an entry it committed moves
        the room's counter, but a message it deleted or redacted in place
        (``delete_event``, ``update_event``) and a binding it changed with no
        record stored (a persistence policy that keeps messages only) move
        none, and the event's commit plans with the history and the bindings
        as the hook left them (steps 9a and 12, §10.3). A room a hook closed
        refuses the event as step 6 does, announced for *operation* (§5.1,
        §8.2), ``None``; the event takes the room's next index, an entry
        committed ahead of it during its hooks having taken the one step 8
        showed them, and its mutation trigger reads that entry (§10.3). A
        streamed row's hooks read the binding again alone.
        """
        source = event.source.channel_id
        if not ran:
            return _AfterHooks(context, context.get_binding(source), None)
        if not under_lock:
            return _AfterHooks(context, await self._store.get_binding(room_id, source), None)
        try:
            context = await self._build_context(room_id)
        except RoomNotFoundError:
            await self._refuse_closed_room(room_id, status=None, operation=operation, event=event)
            return None
        room = context.room
        if room.status in _REFUSING_STATUSES:
            await self._refuse_closed_room(
                room_id, status=room.status, operation=operation, event=event
            )
            return None
        return _AfterHooks(context, context.get_binding(source), room.event_count)

    async def _announce_hook_errors(
        self, room_id: str, event: RoomEvent, sync_result: SyncPipelineResult
    ) -> None:
        """Emit a ``hook_error`` framework event per hook that failed on *event*."""
        for hook_err in sync_result.hook_errors:
            await self._emit_framework_event(
                "hook_error",
                room_id=room_id,
                event_id=event.id,
                data=hook_err,
            )

    async def _run_before_broadcast(
        self,
        room_id: str,
        event: RoomEvent,
        context: RoomContext,
        precheck: SyncPipelineResult | None,
    ) -> SyncPipelineResult:
        """RFC §10.1 step 9, with the off-lock outcome folded in (§9.5.1).

        Without *precheck* every hook runs here, the off-lock ones included —
        ``needs_lock=False`` lets a hook run off the lock, never lets an
        event skip it. With one, a block stands as decided; otherwise the
        locked hooks see the (possibly modified) event, and the side effects
        of both phases are collected together. The off-lock hooks' unusable
        results were reported when met, before the lock (§9.5.1). Its ASYNC
        observers are its pass's to start (:meth:`_gate_commit`).
        """
        if precheck is None:
            return await self._hook_engine.run_sync_hooks(
                room_id, HookTrigger.BEFORE_BROADCAST, event, context, fire_observers=False
            )
        if not precheck.allowed:
            return precheck
        if isinstance(precheck.event, RoomEvent):
            # The off-lock hooks rewrote the event as it stood before the
            # lock: their changes stand on the locked event, never what the
            # locked section decided of it (RFC §9.4, §9.5.1).
            event = kept_identity(event, precheck.event, context.bindings)
        locked = await self._hook_engine.run_sync_hooks(
            room_id,
            HookTrigger.BEFORE_BROADCAST,
            event,
            context,
            needs_lock=True,
            fire_observers=False,
        )
        locked.injected_events[:0] = precheck.injected_events
        locked.tasks[:0] = precheck.tasks
        locked.observations[:0] = precheck.observations
        locked.metadata = {**precheck.metadata, **locked.metadata}
        return locked

    async def _run_commit(self, ready: _Ready, room_id: str, cascade: DeliveryCascade) -> _Proceed:
        """Commit phase (RFC §10.1 step 12) — every durable timeline write.

        Runs under the room lock but OUTSIDE the ``process_timeout`` window:
        an expiry must never leave a committed event reported as blocked
        (§13.6 MUST NOT), and the edit/delete target mutation must not be
        separable from the commit that records it (§10.3).
        """
        source_binding = ready.source_binding
        sync_result = ready.sync_result

        # The edit/delete target as the commit writes it, now that the event
        # is authorized and hook-allowed (RFC §10.3 — mutation must not
        # precede the block decision), in the event's own transaction; the
        # change reaches those who saw its target, with the target as
        # changed.
        target, mutation_hook = target_update(ready.event, ready.edit_delete_target)
        event = narrowed_to_target(ready.event, target, ready.context.bindings)
        context = with_target(ready.context, target)

        # Commit point (RFC §10.1 step 12): persist DELIVERED and bump the
        # room counters atomically, PLAN the broadcast against binding state
        # consistent with this commit, and enqueue it on the room's delivery
        # lane. Past this line the event is authoritative; its delivery set
        # executes off the lock (§10.2), tracked by *cascade*.
        router = self._get_router()

        def plan_factory(committed: RoomEvent) -> Any:
            cascade.unavailable_targets = list(committed.addressed_to or [])
            if source_binding is None:
                # No source binding: no broadcast, but the RFC §10.3 mutation
                # trigger must still fire once the commit is visible, with
                # the change in its history.
                return mutation_plan(committed, committed_in(context, committed), mutation_hook)
            # Refresh context locally by appending the committed event, so
            # the delivery set (and the AI's on_event) sees the trigger. An
            # instruction is not part of the room's history (RFC §10.1.1):
            # the agent it directs receives it as the event, not as a turn.
            broadcast_ctx = (
                context
                if committed.type == EventType.INSTRUCTION
                else context.model_copy(
                    update={
                        "recent_events": [
                            *context.recent_events[-(_RECENT_EVENTS_LIMIT - 1) :],
                            committed,
                        ]
                    }
                )
            )
            plan = router.plan(committed, source_binding, broadcast_ctx)
            reachable = {
                target.channel_id
                for target in plan.targets
                if target.category == ChannelCategory.INTELLIGENCE
                and self._channels.get(target.channel_id) is not None
            }
            cascade.unavailable_targets = [
                target for target in committed.addressed_to or [] if target not in reachable
            ]
            plan.mutation_hook = mutation_hook
            plan.response_visibility = committed.response_visibility
            plan.hook_tasks = list(sync_result.tasks)
            plan.hook_observations = list(sync_result.observations)
            plan.emit_processed = True
            return plan

        event = event.model_copy(update={"status": EventStatus.DELIVERED})
        committed = await self._commit_to_lane(
            room_id, event, cascade, plan_factory, target=target
        )
        # A persistence-policy-excluded event is delivered but not persisted;
        # it keeps its provisional identity (RFC §14.3 — no index consumed).
        event = event if committed is None else committed
        return _Proceed(event, source_binding, sync_result, context, mutation_hook)

    async def _handle_block(
        self,
        *,
        room_id: str,
        event: RoomEvent,
        reason: str | None,
        blocked_by: str | None,
    ) -> RoomEvent:
        """RFC §9.5 block handling: commit the event as BLOCKED and emit the
        framework event. Shared by every block path (main-inbound hook block,
        reentry hook block, source write-permission block) so they cannot
        drift.

        The BLOCKED event is committed atomically like any other event — index,
        status, and room counters in one store transaction (§14.3) — because a
        blocked event is still part of the timeline and consumes an index (§8.3).

        Returns the committed BLOCKED event so the caller can include it in its
        return value.
        """
        blocked_event = event.model_copy(
            update={
                "status": EventStatus.BLOCKED,
                "blocked_by": blocked_by or reason,
            }
        )
        # An instruction is never stored, blocked or not (RFC §10.1.1): the
        # timeline holds what was said in the room, and nobody said this.
        if event.type != EventType.INSTRUCTION:
            blocked_event = await self._commit_indexed(room_id, blocked_event)
        await self._announce_blocked(room_id, blocked_event, reason=reason, blocked_by=blocked_by)
        return blocked_event

    async def _announce_blocked(
        self, room_id: str, blocked: RoomEvent, *, reason: str | None, blocked_by: str | None
    ) -> None:
        """Emit ``event_blocked`` for a BLOCKED record, naming its source.

        The one announcement of a block, whichever path stored the record (a
        hook, a source that cannot write, a router block), so subscribers read
        the same fields everywhere.
        """
        await self._emit_framework_event(
            "event_blocked",
            room_id=room_id,
            event_id=blocked.id,
            channel_id=blocked.source.channel_id,
            data={"reason": reason, "blocked_by": blocked_by},
        )

    async def _store_blocked(
        self, room_id: str, blocked: _Blocked, queued: QueuedInjections
    ) -> RoomEvent:
        """Store what :meth:`_gate_commit` refused as its BLOCKED record, with
        its hook side effects (RFC §7.5 rule 3 — side effects are always
        collected), then queue the events its hooks injected on *queued*,
        committed after it once its section released the lock (§9.4).

        Shared by every commit path, the hook block and the source
        write-permission block alike, so they cannot drift.
        """
        blocked_event = await self._handle_block(
            room_id=room_id,
            event=blocked.event,
            reason=blocked.reason,
            blocked_by=blocked.blocked_by,
        )
        await self._persist_side_effects(
            room_id, blocked.tasks, blocked.observations, blocked_event, blocked.context
        )
        queued.add(blocked.injected_events, blocked_event)
        return blocked_event


def committed_in(context: RoomContext, committed: RoomEvent) -> RoomContext:
    """*context* with *committed* as its newest event, in place of a copy
    its pass's hooks read: the room as the event's commit left it, at its
    committed index and as its hooks left it, which its delivery set and a
    change's mutation trigger read (RFC §10.3, §9.2)."""
    history = [event for event in context.recent_events if event.id != committed.id]
    recent = [*history[-(_RECENT_EVENTS_LIMIT - 1) :], committed]
    return context.model_copy(update={"recent_events": recent})


def with_counters(context: RoomContext, committed: RoomEvent) -> RoomContext:
    """*context* with its room's counters as *committed*'s commit left them
    (RFC §10.1 step 12): ``event_count`` past its index, ``latest_index`` at
    it. For a change's mutation trigger alone (§10.3): a context a channel
    is handed keeps the counters it was built with, which tell a later read
    that its history may not be carried (``_carried_history``)."""
    room = context.room.model_copy(
        update={"event_count": committed.index + 1, "latest_index": committed.index}
    )
    return context.model_copy(update={"room": room})


def mutation_plan(
    committed: RoomEvent,
    context: RoomContext,
    mutation_hook: tuple[HookTrigger, RoomEvent] | None,
) -> DeliveryPlan | None:
    """The plan of a committed event that has nothing to deliver: none,
    unless it is a change, whose RFC §10.3 mutation trigger still fires once
    the commit is visible, off the lock. It fires no ``AFTER_BROADCAST`` and
    persists no hook side effects."""
    if mutation_hook is None:
        return None
    return DeliveryPlan(
        event=committed,
        source_binding=None,
        context=context,
        targets=[],
        mutation_hook=mutation_hook,
        fire_after_broadcast=False,
    )


def target_update(
    event: RoomEvent, target: RoomEvent | None
) -> tuple[RoomEvent | None, tuple[HookTrigger, RoomEvent] | None]:
    """What *event*'s commit writes on its edit or delete *target*, and the
    mutation trigger it fires once the lock is released (RFC §10.3); nothing
    without a target. The event as the hooks left it: a ``modify`` hook on
    the edit is honoured."""
    content = event.content
    if target is None or not isinstance(content, EditContent | DeleteContent):
        return None, None
    updated, trigger = applied(content, target)
    return updated, (trigger, updated)

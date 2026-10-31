"""Delivery lanes — broadcast planned under the room lock, executed off it.

RFC §10.1 steps 12-14 / §10.2: the room lock covers the commit and the
*planning* of a broadcast (resolving the delivery set against binding state
consistent with the committed timeline). *Execution* — the per-target
transcode / ``on_event`` / ``deliver`` work, which includes provider round
trips — runs after the lock is released, ordered per room by the store's
``delivered_index`` cursor (RFC §13.5 "Serialization scope").

One :class:`RoomDeliveryLane` exists per room *and per process*: a lane only
ever executes plans its own process enqueued, because a plan holds live
channel objects. Cross-process order comes from the shared cursor — a lane
may only execute the plan at ``delivered_index + 1``, and advancing the
cursor is a strict store CAS (:meth:`ConversationStore.advance_delivered_index`).
The *delivery claim* — a derived key on the existing
:class:`~roomkit.core.locks.RoomLockManager` — serializes executors of the
same room across processes, so gap decisions and "one event's set completes
before the next begins" hold deployment-wide. A crashed process releases its
claim with its connection; the events it committed but never delivered are a
cursor hole, and the waiting lane skips over it after ``gap_timeout``
(framework event ``delivery_skipped``) — the loss stays bounded to the crash
window, but it is observable and the room never wedges. The durable outbox
that turns that loss into recovery is deliberately a separate step (RFC
§13.6).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import math
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core import task_utils
from roomkit.core._reentrance import ReentryBudget, delivering
from roomkit.core.exceptions import KitClosedError
from roomkit.core.locks import _has_room_lock, _held_rooms
from roomkit.core.task_utils import spared_by_close, start_named, wait_for_unspared
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.providers.utils import _aclose_stream
from roomkit.telemetry.context import restored_span
from roomkit.turns.owner import drop_unread_streams

if TYPE_CHECKING:
    from roomkit.core.event_router import StreamingResponse
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.delivery import InboundResult
    from roomkit.models.enums import HookTrigger
    from roomkit.models.event import RoomEvent
    from roomkit.models.task import Observation, Task
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.lanes")

# Claim keys live in the same RoomLockManager as the room commit locks but in
# a disjoint key space, so the ContextVar reentrancy of ``locked()`` can never
# confuse "I hold the delivery claim" with "I hold the room lock".
DELIVERY_CLAIM_PREFIX = "__delivery__:"

# Room id whose lane executor the current task belongs to. Lets
# ``DeliveryCascade.wait()`` refuse to wait on itself: integrator code running
# inside a plan's execution (a tool handler calling ``send_event``) would
# otherwise deadlock the lane on its own queue.
_active_lane_room: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_lane_room", default=None
)


def delivery_claim_key(room_id: str) -> str:
    """The lock-manager key for a room's delivery claim."""
    return f"{DELIVERY_CLAIM_PREFIX}{room_id}"


@dataclass(slots=True)
class DeliveryPlan:
    """A broadcast resolved under the room lock, to be executed off it.

    Immutable once enqueued. ``targets`` is the delivery set (RFC §10.1 step
    12): eligible bindings resolved against state consistent with the
    committed timeline. ``injected`` marks a hook-injected event (RFC §9.4):
    its targets are transports only, so it asks no agent (a transport
    hosting a model takes it silently), and no reentry follows its delivery;
    its ``AFTER_BROADCAST`` fires like any other's.
    """

    event: RoomEvent
    source_binding: ChannelBinding | None
    context: RoomContext
    targets: list[ChannelBinding]
    injected: bool = False
    exclude_delivery: set[str] | None = None
    # (trigger, updated target) for an edit/delete (RFC §10.3), fired by the
    # executor before the event's own AFTER_BROADCAST — matching the order the
    # locked pipeline produced them.
    mutation_hook: tuple[HookTrigger, RoomEvent] | None = None
    # The context the mutation trigger reads, the room as the change's commit
    # left it (RFC §10.3): ``context`` with the room's counters past the
    # change. None reads ``context`` (a change delivered unstored moves none).
    mutation_context: RoomContext | None = None
    # The trigger's response_visibility, stamped onto reentry events so the
    # router's visibility check honours the caller's response scope.
    response_visibility: str | None = None
    # Side effects collected by the BEFORE_BROADCAST sync hooks, persisted
    # with the execution results (they always persisted post-broadcast).
    hook_tasks: list[Task] = field(default_factory=list)
    hook_observations: list[Observation] = field(default_factory=list)
    # Whether the event's AFTER_BROADCAST hooks (and side-effect persistence)
    # fire after its set executes. False preserves the path that never fired
    # them: a trigger without a source binding. A rerun keeps its side
    # effects and never repeats its AFTER_BROADCAST. An injected event fires
    # its own (RFC §9.4).
    fire_after_broadcast: bool = True
    # Trace continuity across the lane boundary: the caller's current span
    # (and its backend context), captured at plan time. The lane executor
    # runs on a fresh contextvars context, so without these the broadcast
    # span, the post-plan hook spans and the reentry pass's plan would be
    # orphans instead of children of the inbound span.
    parent_span_id: str | None = None
    parent_span_ctx: Any = None
    # True on the root pass of an inbound/send_event call: emits
    # ``event_processed`` once the delivery set has executed.
    emit_processed: bool = False
    # A committed event broadcast again (a regeneration, RFC §13.6): the
    # caller reads its replies as a root pass's, its side effects are kept,
    # and nothing of its first pass is repeated (its delivery report, its
    # AFTER_BROADCAST, ``event_processed``). The room's status is read again
    # when the lane takes it; ``refusal`` is what refused it then.
    rerun: bool = False
    refusal: InboundResult | None = None
    # A discussion takes the room's turns (RFC §19.7.5): its plans ask no agent
    # at broadcast, and a turn it gives is a rerun planned for one agent alone,
    # bounded by the discussion's own depth limit.
    defers_solicitation: bool = False
    turn_for: str | None = None
    max_chain_depth: int | None = None
    # What its delivery set has done so far, set by the set as it starts and
    # read when the kit's close cuts it (RFC §12.10.4): the targets
    # delivered, the ones still sending failed; its caller reads it as the
    # set's result.
    partial_result: Callable[[], Any] | None = None

    @property
    def answers_caller(self) -> bool:
        """Whether the pass's replies are the caller's own (RFC §10.1 step 18)."""
        return self.emit_processed or self.rerun


class DeliveryCascade:
    """Completion of a caller's delivery set, transitively.

    One unit per enqueued plan; the executor releases a unit only after the
    plan's delivery set, its post-plan effects and its reentry pass have run
    (its ASYNC hooks started, not waited for, RFC §9.3) — and the reentry
    pass retains a unit for every child plan *before* the parent's release,
    so the count never touches zero between two passes of an AI chain.
    ``wait()`` therefore resolves exactly when "the caller observes its
    event's delivery-set completion" (RFC §10.1 step 18) holds for the whole
    chain the call started.

    Cancelling it stops its turns, not its delivery (RFC §10.1, *Explicit
    delivery cancellation*): its turn work (an intelligence target's turn,
    the reading of a stream) is cancelled, and none starts after; its plans
    still deliver their event to every other target and fire their
    AFTER_BROADCAST, and no reentry follows.
    """

    __slots__ = (
        "_done",
        "_pending",
        "budget",
        "caller_logs",
        "caller_logs_streamed",
        "cancelled",
        "delivery_results",
        "unavailable_targets",
        "error",
        "response_metadata",
        "response_events",
        "room_id",
        "streams",
        "_tasks",
        "_turn_work",
        "_cleanup_task",
        "_when_over",
        "_over",
        "reader_detached",
        # The kit holds its cascades weakly (core._closing.Cascades).
        "__weakref__",
    )

    def __init__(self, room_id: str, *, reentry_budget: int | ReentryBudget) -> None:
        self.room_id = room_id
        self._pending = 0
        self._done = asyncio.Event()
        self._done.set()
        # Shared with the cascade it was drawn from when the room's own work
        # made the call (RFC §9.5.1 Reentrance), else its own.
        self.budget = (
            reentry_budget
            if isinstance(reentry_budget, ReentryBudget)
            else ReentryBudget(reentry_budget)
        )
        # Streaming responses captured during execution, consumed by the
        # caller after wait() — streaming delivery (TTS, long generations)
        # must not stall the lane. Every pass adds its own, so the list grows
        # while the caller reads it (RFC §8.3: a started response is read).
        self.streams: list[Any] = []
        # First intelligence-channel failure, surfaced on InboundResult.error.
        self.error: Exception | None = None
        # The caller waits for the streams and receives their failure (on
        # InboundResult.error, or raised) and logs it: a stream with no
        # streaming target then logs it at DEBUG only. Set by _finish_cascade.
        self.caller_logs = False
        # The caller logs a stream a streaming target rendered too (a
        # delegation): that target's line drops to DEBUG as well.
        self.caller_logs_streamed = False
        # Final response metadata produced by the root delivery set. Non-streaming
        # outputs land here when the plan finishes; detached streaming consumers
        # add their records after consuming the streams, before their handle wakes.
        self.response_metadata = ResponseMetadata()
        self.response_events: list[RoomEvent] = []
        # Reason the cascade was cancelled (close/seal), or None.
        self.cancelled: str | None = None
        # Per-channel outcome of the ROOT pass's delivery set, keyed by channel
        # id (RFC §10.1 step 18). Only the root: a reentry is a separate event
        # whose deliveries belong to its own result, and merging them here
        # would collide on any channel both passes reached.
        self.delivery_results: dict[str, Any] = {}
        self.unavailable_targets: list[str] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        # Its turn work: what a cancellation cancels and its cleanup joins.
        self._turn_work: set[asyncio.Task[Any]] = set()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._when_over: list[Callable[[], None]] = []
        self._over = False
        # A reader nobody awaits reads its streams: the delivery is over when
        # that reader ends, not when its call returns.
        self.reader_detached = False

    @property
    def drained(self) -> bool:
        """No turn work (a turn, a stream's reader) is still unwinding: once
        cancelled, a plan still delivering its event to the transports does
        not hold the turns' resources."""
        return all(task.done() for task in self._turn_work) and (
            self._cleanup_task is None or self._cleanup_task.done()
        )

    def detach_reader(self, task: asyncio.Task[Any]) -> None:
        """Own *task*, a reader of the cascade's streams nobody awaits, as
        turn work: the caller's delivery is over when it ends."""
        self.track(task, turn_work=True)
        self.reader_detached = True
        task.add_done_callback(lambda _: self.over())

    def call_returned(self) -> None:
        """The caller's call returned: its delivery is over, unless a reader
        it detached (a deferred tail, a stream read off the lane) ends it."""
        if not self.reader_detached:
            self.over()

    def when_over(self, callback: Callable[[], None]) -> None:
        """Call *callback* once the caller's delivery is over (:meth:`over`)."""
        self._when_over.append(callback)

    def over(self) -> None:
        """The caller's delivery is over: its call returned having read what
        it started, or the reader it detached ended. Calls the callbacks
        :meth:`when_over` registered, once."""
        self._over = True
        callbacks, self._when_over = self._when_over, []
        for callback in callbacks:
            callback()

    def track(self, task: asyncio.Task[Any], *, turn_work: bool = False) -> None:
        """Own a task before it can start; *turn_work* (a turn, or the
        reading of a stream) is what a cancellation cancels, once."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if not turn_work:
            return
        self._turn_work.add(task)
        task.add_done_callback(self._turn_work.discard)
        if self.cancelled is not None:
            task.cancel(self.cancelled)

    async def run[T](
        self,
        work: Coroutine[Any, Any, T],
        *,
        on_task: Callable[[asyncio.Task[T]], None] | None = None,
        turn_work: bool = False,
    ) -> T:
        """Execute owned work without giving its waiter cancellation
        ownership; *on_task* receives its task as soon as it is owned.
        *turn_work* (a turn, a stream's reader) is refused once the cascade is
        cancelled; its delivery work still runs."""
        if turn_work and self.cancelled is not None:
            work.close()
            raise asyncio.CancelledError(self.cancelled)
        task = start_named(work)
        self.track(task, turn_work=turn_work)
        if on_task is not None:
            on_task(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Framework shutdown also stops the lane's running child.
            if not task.done() and not task.cancelling():
                task.cancel()
            raise

    def turn_work(self) -> set[asyncio.Task[Any]]:
        """The cascade's turn work still running."""
        return {task for task in self._turn_work if not task.done()}

    def running(self) -> set[asyncio.Task[Any]]:
        """The work it owns still running (:meth:`run`): its turns, the
        reading of its streams, its delivery sets and their aftermath. Its
        cleanup, which waits for that work, is not."""
        return {task for task in self._tasks if not task.done()}

    async def wait_drained(self) -> None:
        """Join its turn work, including its asynchronous finalizers."""
        while pending := {task for task in self._turn_work if not task.done()}:
            await asyncio.wait(pending)
        if self._cleanup_task is not None:
            await asyncio.shield(self._cleanup_task)

    async def _cleanup(self) -> None:
        # The turns' work, not the delivery going on (RFC §10.1).
        while pending := {task for task in self._turn_work if not task.done()}:
            await asyncio.wait(pending)
        # Also close streams whose consumer was cancelled before its first
        # step. No generator is closed concurrently with its running task.
        for response in self.streams:
            await _aclose_stream(response.stream)
        # A turn whose end its executor still owes has not finished cleaning
        # up: nothing reports it done (RFC §10.6.8).
        owed = [r.source_channel_id for r in self.streams if getattr(r.stream, "end_owed", False)]
        if owed:
            raise TimeoutError(
                f"Delivery cleanup incomplete for room {self.room_id}: the turns of "
                f"{owed} did not end on their executor"
            )

    async def abandon(self, reason: str) -> None:
        """Stop the turns of a cascade its caller left (cancelled or
        failed): drained within :meth:`cancel_and_wait`'s budget, or only
        cancelled when the caller is the lane or holds the room lock."""
        if self.waiter_would_deadlock():
            self.cancel(reason)
        else:
            await self.cancel_and_wait(reason)

    async def cancel_and_wait(self, reason: str, timeout: float = 5.0) -> None:
        """Request cancellation once and join cleanup within one fixed budget.

        Repeated cancellation of the waiting task never interrupts the owned
        finalizers. An expiry reports incomplete cleanup; it does not release
        resources or claim that the turn is safe to dispose.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        if self.waiter_would_deadlock():
            raise RuntimeError("Cannot drain delivery from its lane or room lock")
        self.cancel(reason)
        deadline = asyncio.get_running_loop().time() + timeout
        interrupted = False
        assert self._cleanup_task is not None
        while not self._cleanup_task.done():
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            try:
                _, pending = await asyncio.wait({self._cleanup_task}, timeout=remaining)
            except asyncio.CancelledError:
                interrupted = True
                continue
            if pending:
                raise TimeoutError(f"Delivery cleanup incomplete for room {self.room_id}")
        self._cleanup_task.result()
        if interrupted:
            raise asyncio.CancelledError

    def retain(self) -> None:
        """Account one pending unit. Call before any await can fail. A
        cancelled cascade holds no waiter, whether it was cancelled before
        or after its units were retained (RFC §10.1)."""
        self._pending += 1
        if self.cancelled is None:
            self._done.clear()

    def release(self) -> None:
        """Release one unit; the last release resolves ``wait()``."""
        self._pending -= 1
        if self._pending <= 0:
            self._done.set()

    def consume_reentry_budget(self) -> bool:
        """Take one reentry slot; ``False`` when the chain cap is exhausted."""
        return self.budget.take()

    def add_streams(self, streams: list[StreamingResponse], *, chained: bool = False) -> None:
        """Queue streams for the caller's reader.

        ``chained`` marks answers to an answer, started by a pass other than
        the caller's own: read like the caller's, they count against the
        reentry budget and their record and failure are not the caller's.
        A cancelled cascade's are dropped unread: nobody reads them.
        """
        if self.cancelled is not None:
            drop_unread_streams(streams)
            return
        for stream in streams:
            stream.chained = chained
        self.streams.extend(streams)

    def caller_logs_failure_of(self, stream: Any, *, rendered: bool) -> bool:
        """Whether the caller logs *stream*'s failure, the framework's own
        line then dropping to DEBUG: never a chained stream's, which is not
        the caller's; one a streaming target *rendered* only for a caller that
        logs every failure of its turn (a delegation); any other when the
        caller receives it (RFC §15.2)."""
        if stream.chained:
            return False
        return self.caller_logs_streamed if rendered else self.caller_logs

    def record_error(self, exc: Exception) -> None:
        """Record an intelligence failure; the first one wins. One met once
        the kit's close cut the cascade (a turn given up as its backend
        closed) is the cut's, which the call reads as its reason, not its
        error (RFC §12.10.4)."""
        if self.error is None and self.cancelled != KitClosedError.code:
            self.error = exc

    @property
    def in_flight(self) -> bool:
        """Whether its caller's delivery is still under way: a unit not
        released, turn work running, or a detached reader not ended."""
        return self._pending > 0 or not self.drained or (self.reader_detached and not self._over)

    def cancel(
        self,
        reason: str,
        *,
        spared: set[asyncio.Task[Any]] | frozenset[Any] = frozenset(),
        wake: bool = True,
    ) -> None:
        """Stop the cascade's turns: cancel its turn work and start none
        after; wake every waiter, keep the reason. The turn work in *spared*
        (the work a close runs under, RFC §12.10.4) is left to end. Without
        *wake* (the kit's close), a waiter still waits for its delivery set,
        which the close lets end or cuts, so it reads what was delivered."""
        if self.cancelled is not None:
            return
        self.cancelled = reason
        for task in self._turn_work:
            if task in spared:
                continue
            if not task.done() and not task.cancelling():
                # The reason travels with the cancellation: a turn the
                # cascade runs tells its executor why (RFC §10.6.8).
                task.cancel(reason)
        if wake:
            self._done.set()
        self._cleanup_task = asyncio.create_task(
            self._cleanup(),
            context=contextvars.Context(),
            name=f"roomkit-delivery-cleanup-{self.room_id}",
        )
        self._cleanup_task.add_done_callback(self._cleanup_ended)

    def _cleanup_ended(self, task: asyncio.Task[None]) -> None:
        """Read how the cleanup ended, so one nobody waits for is no
        unretrieved task error. Its waiter, if any, is told; with none,
        nothing reports the cleanup done (RFC §10.6.8)."""
        if task.cancelled() or (error := task.exception()) is None:
            return
        logger.debug("Delivery cleanup for room %s ended: %s", self.room_id, error)

    async def wait(self) -> bool:
        """Wait for the cascade to complete.

        Short-circuits (returning ``False``) when the current task is the
        room's own lane executor or holds the room lock: in both cases
        waiting would deadlock (the lane cannot progress past the caller),
        so the call returns with the event committed and its delivery
        following in lane order — the detached completion RFC §10.1 step 18
        permits. Returns ``True`` when the cascade actually completed; a
        detached caller must arrange for ``streams`` to be consumed later.
        """
        if self._pending <= 0:
            return True
        if self.waiter_would_deadlock():
            return False
        await self._done.wait()
        return True

    async def wait_detached(self) -> None:
        """Wait for true completion, without the short-circuit guards.

        For the background consumer a detached caller schedules — it runs on
        its own clean task, where waiting cannot deadlock.
        """
        await self._done.wait()

    def waits_on_caller(self) -> bool:
        """Whether a turn it asked waits its place behind the current task's
        own turn (RFC §6.4): reading its streams here would wait on that
        turn, so its caller hands the reading to the background instead."""
        return any(getattr(r.stream, "waits_on_caller", False) for r in self.streams)

    def waiter_would_deadlock(self) -> bool:
        """Whether the current task must not wait on this room's delivery.

        The same condition ``wait()`` short-circuits on — the caller is the
        room's own lane executor or holds the room lock, so the lane cannot
        progress past it — exposed for waiters outside this module: a
        :class:`~roomkit.models.delivery.DeliveryHandle` awaited from a tool
        handler or a sync hook would otherwise hang on its own turn.
        """
        return _active_lane_room.get() == self.room_id or _has_room_lock(self.room_id)


def _placing_work() -> frozenset[asyncio.Task[Any]]:
    """The tasks the current code runs as and the kit-held runs it runs
    under: the work placing an entry."""
    return frozenset({*task_utils.acting_tasks(), *task_utils.held_runs()})


def anchor_index(latest_index: int | None) -> int:
    """The ``after_index`` of an index-less plan: the room's latest index, or
    -1 while the room has no event (RFC §5.1), the cursor's own start."""
    return -1 if latest_index is None else latest_index


@dataclass(slots=True)
class ExecEntry:
    """A plan waiting in a lane.

    ``index is None`` when the persistence policy excluded the event: it is
    delivered but not stored, consumes no index and must not move the cursor.
    Such entries execute in FIFO order between themselves once the cursor has
    reached ``after_index`` — the room's ``latest_index`` at plan time, i.e.
    the last event that causally precedes them.
    """

    plan: DeliveryPlan
    cascade: DeliveryCascade
    index: int | None
    after_index: int = -1
    placed_under: frozenset[asyncio.Task[Any]] = field(default_factory=_placing_work)
    """The work whose call placed it: a ``close()`` from that work drops it
    as it starts when the lane still keeps it (:meth:`RoomDeliveryLane.drop_kept`)."""


@dataclass(slots=True)
class CursorEntry:
    """A committed index with no delivery set (a BLOCKED event, a reentry
    without a binding, a stored-only injected event, a streamed segment, an
    inline-delivered commit): advances the cursor at its turn, executes
    nothing. Every committed index MUST reach its lane as exactly one entry —
    a missing one is a permanent hole the gap policy will eventually skip.
    """

    index: int


def _turn(entry: ExecEntry | CursorEntry) -> tuple[int, int]:
    """When a lane runs *entry*: once the cursor reached the index before
    its own, or its anchor when it has none, an index-less entry ahead of
    the indexed one waiting for the same cursor."""
    if isinstance(entry, CursorEntry):
        return (entry.index - 1, 1)
    if entry.index is None:
        return (entry.after_index, 0)
    return (entry.index - 1, 1)


def _runs_before(entry: ExecEntry | CursorEntry, other: ExecEntry) -> bool:
    """Whether a lane runs *entry*, placed after *other*, before it (see
    :func:`_turn`); index-less entries run in their order of arrival."""
    turn, other_turn = _turn(entry), _turn(other)
    if turn[1] == other_turn[1] == 0:
        return False
    return turn < other_turn


@dataclass(slots=True)
class _Executed:
    """A dequeued entry and, once it returns, its execution outcome.

    The record is what *owns* an entry between the queues and ``_finish``:
    an entry pulled off ``_indexed`` / ``_unindexed`` belongs to no queue any
    more, so this is the only place a cancelled round can find it again.
    """

    entry: ExecEntry
    result: Any = None


@dataclass(slots=True)
class LaneConfig:
    """Tuning for the delivery lanes (see ``RoomKit.__init__``)."""

    gap_timeout: float = 30.0
    gap_backoff_initial: float = 0.05
    gap_backoff_max: float = 0.25
    idle_ttl: float = 30.0
    close_grace: float = 5.0


@runtime_checkable
class LaneHost(Protocol):
    """What a lane executor asks of the framework.

    ``_execute_plan`` runs a plan's delivery set, its intelligence targets'
    turns as its cascade's turn work, and returns its
    ``BroadcastResult``-shaped outcome. ``_post_plan_effects`` fires the
    per-event aftermath (delivery framework events, ON_ERROR funnel, side
    effects, AFTER_BROADCAST); ``_report_cut`` reports a cut plan's result
    to its caller, and nothing else of its aftermath.
    ``_reentry_commit_pass`` turns the response events of an executed plan
    into fresh commit passes (room lock taken anew) whose plans join the
    same lane.
    """

    _store: ConversationStore

    async def _execute_plan(self, plan: DeliveryPlan, cascade: DeliveryCascade) -> Any: ...

    async def _post_plan_effects(
        self, plan: DeliveryPlan, result: Any, cascade: DeliveryCascade
    ) -> None: ...

    async def _reentry_commit_pass(
        self, room_id: str, plan: DeliveryPlan, result: Any, cascade: DeliveryCascade
    ) -> None: ...

    def _report_cut(self, plan: DeliveryPlan, result: Any, cascade: DeliveryCascade) -> None: ...

    async def _emit_framework_event(
        self,
        event_type: str,
        room_id: str | None = None,
        channel_id: str | None = None,
        event_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None: ...


class RoomDeliveryLane:
    """One room's delivery executor in this process.

    ``enqueue`` is synchronous and never blocks — it runs under the room
    lock. The lane's own task executes entries strictly in index order,
    gated by the shared cursor; entries are buffered *indexed* (a dict keyed
    by index, plus a FIFO of index-less entries anchored by ``after_index``)
    because arrival order is not index order — streamed segments commit off
    the lock and can overtake an inbound commit.
    """

    def __init__(
        self,
        room_id: str,
        host: LaneHost,
        claims: RoomLockManager,
        config: LaneConfig,
        *,
        retire: Any = None,
    ) -> None:
        self.room_id = room_id
        self._host = host
        self._claims = claims
        self._config = config
        # Called on idle timeout; returns True when the registry dropped us
        # (checked and removed without an await in between, so an enqueue
        # racing the retirement either lands before the check or creates a
        # fresh lane afterwards — never on a dead one).
        self._retire = retire or (lambda _lane: True)
        self._indexed: dict[int, ExecEntry | CursorEntry] = {}
        self._unindexed: deque[ExecEntry] = deque()
        # The entry placed while the lane was idle, until a round takes it or
        # a drop ends it: its set runs from its placement, which the kit's
        # close does not drop (RFC §12.10.4).
        self._placed_idle: ExecEntry | None = None
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._busy = False
        # Whether a round's sets are sending: once their targets answered,
        # their aftermath (the responses' passes) leaves the lane idle to a
        # set placed meanwhile (RFC §12.10.4).
        self._sending = False
        self._closed = False
        # Set by seal(): the reason the entries it drops, and the ones a cut
        # round took, end with.
        self._reason = "lane_cancelled"
        self._cut = False

    @property
    def idle(self) -> bool:
        """Whether the lane holds no pending work."""
        return not self._indexed and not self._unindexed and not self._wakeup.is_set()

    def start(self) -> None:
        """Start the executor on a FRESH contextvars context.

        The lane is created while the enqueuing task holds the room lock;
        a plainly-created task would inherit ``_held_rooms`` and the reentry
        pass would take the lock manager's reentrant fast path *without
        holding the lock*. An empty context makes the executor a stranger to
        every lock its creator held. The executor is named in its own
        context (:func:`start_named`): a ``close()`` from the work it runs (a
        delivery's hook) spares it rather than wait for itself.
        """
        if self._task is None and not self._closed:
            self._task = start_named(
                self._run(),
                name=f"roomkit-delivery-lane-{self.room_id}",
                context=contextvars.Context(),
            )
            self._task.add_done_callback(self._ended)

    def _ended(self, _task: asyncio.Task[None]) -> None:
        """The executor ended: once sealed, the entry placed while it was
        idle and never taken (its round cut first) ends as the queued ones
        did (RFC §12.10.4)."""
        if self._closed:
            self._abort_pending(self._reason)

    def enqueue(self, entry: ExecEntry | CursorEntry) -> None:
        """Accept an entry. Synchronous — runs under the room lock."""
        if self._closed and not (isinstance(entry, CursorEntry) and self._runs_on):
            self._drop_entry(entry, "lane_closed")
            return
        self._note_placement(entry)
        if isinstance(entry, CursorEntry):
            self._indexed[entry.index] = entry
        elif entry.index is None:
            self._unindexed.append(entry)
        else:
            self._indexed[entry.index] = entry
        self._wakeup.set()

    def _note_placement(self, entry: ExecEntry | CursorEntry) -> None:
        """Note *entry* as running from its placement when the lane is idle,
        no set sending and nothing queued (RFC §12.10.4): a set ahead whose
        targets all answered leaves the lane idle while its responses' passes
        run, so an answer placed then runs from its placement. A set placed
        behind another is queued. A set placed later that the room's order
        runs first takes the lead: the entry noted is queued behind it. A
        cursor entry runs nothing and the close keeps it, so the index it
        fills never costs the noted entry its place."""
        if not self._sending and not self._indexed and not self._unindexed:
            self._placed_idle = entry if isinstance(entry, ExecEntry) else None
        elif (
            isinstance(entry, ExecEntry)
            and self._placed_idle is not None
            and _runs_before(entry, self._placed_idle)
        ):
            self._placed_idle = None

    @property
    def _runs_on(self) -> bool:
        """Whether the lane takes its entries: unsealed, or sealed with the
        entry placed while it was idle still to run (RFC §12.10.4)."""
        return not self._closed or self._placed_idle is not None

    async def drain(self) -> None:
        """Wait until every queued entry has been processed."""
        while self._indexed or self._unindexed or self._busy:
            await asyncio.sleep(0.01)

    @property
    def task(self) -> asyncio.Task[None] | None:
        """The lane's executor, while it runs."""
        return self._task

    def seal(self, reason: str) -> None:
        """Take no more work (RFC §12.10.4): the queued entries are dropped
        with *reason*, their cascades cancelled, but the one placed while the
        lane was idle, which runs from its placement; the round under way
        ends after the entry it runs, or after that one; what it ran is then
        finished as usual. The kit's close cuts the cascades under way
        first, sparing what it is called from: a drop cancels nothing of
        them again."""
        self._closed = True
        self._reason = reason
        self._wakeup.set()
        self._abort_pending(reason, keep=self._placed_idle)

    def cut(self) -> None:
        """Cut the round under way, once sealed: its task is cancelled, and
        every entry it took reports what it delivered (a target cut mid-send
        reported failed) and ends with the seal's reason; nothing of their
        aftermath runs."""
        task = self._task
        if task is None or task.done() or self._cut:
            return
        self._cut = True
        task.cancel()

    def drop_kept(self, placed_by: set[asyncio.Task[Any]] | None = None) -> None:
        """Once sealed, drop the entry kept as running from its placement and
        not taken yet: this lane's round runs the close, which can give it no
        turn before it releases the store (RFC §12.10.4). The lane then ends
        with its round. With *placed_by*, only an entry that work placed:
        the close's own, dropped as the close starts, whether or not the
        lane runs the close."""
        kept = self._placed_idle
        if not self._closed or kept is None:
            return
        if placed_by is not None and not kept.placed_under & placed_by:
            return
        self._placed_idle = None
        if kept.index is None:
            if kept in self._unindexed:
                self._unindexed.remove(kept)
        elif self._indexed.get(kept.index) is kept:
            del self._indexed[kept.index]
        self._drop_entry(kept, self._reason)
        self._wakeup.set()

    def _drop_entry(self, entry: ExecEntry | CursorEntry, reason: str) -> None:
        if entry is self._placed_idle:
            self._placed_idle = None
        if isinstance(entry, ExecEntry):
            # The kit's close drops what it did not start (RFC §12.10.4): an
            # expected end, not a warning.
            logger.log(
                logging.INFO if reason == KitClosedError.code else logging.WARNING,
                "Delivery dropped (%s) for room %s index %s",
                reason,
                self.room_id,
                entry.index,
            )
            entry.cascade.cancel(reason)
            entry.cascade.release()

    def _abort_pending(self, reason: str, *, keep: ExecEntry | None = None) -> None:
        """Drop every queued entry with *reason*, but *keep* and, while it is
        kept, the cursor entries: they run nothing, and *keep* may wait on
        an index one of them fills."""
        for index, entry in list(self._indexed.items()):
            if entry is not keep and not (keep is not None and isinstance(entry, CursorEntry)):
                del self._indexed[index]
                self._drop_entry(entry, reason)
        unindexed, self._unindexed = self._unindexed, deque()
        for entry in unindexed:
            if entry is keep:
                self._unindexed.append(entry)
            else:
                self._drop_entry(entry, reason)

    async def _run(self) -> None:
        # Belt and braces with start(): this task must hold no inherited lock
        # membership, and everything it calls must know it runs inside this
        # room's lane.
        _held_rooms.set(frozenset())
        _active_lane_room.set(self.room_id)
        loop = asyncio.get_running_loop()
        gap_since: float | None = None
        backoff = self._config.gap_backoff_initial
        while self._runs_on:
            self._wakeup.clear()
            if not self._indexed and not self._unindexed:
                gap_since = None
                backoff = self._config.gap_backoff_initial
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=self._config.idle_ttl)
                except TimeoutError:
                    if self.idle and self._retire(self):
                        return
                continue
            force_skip = (
                gap_since is not None and (loop.time() - gap_since) >= self._config.gap_timeout
            )
            try:
                progressed = await self._process_ready(force_skip=force_skip)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A store/claim failure must not kill the executor; back off
                # and retry — the entries are still queued.
                logger.exception("Delivery lane for room %s failed a round", self.room_id)
                progressed = False
            if progressed:
                gap_since = None
                backoff = self._config.gap_backoff_initial
                continue
            # Blocked on a cursor hole owned elsewhere (another process's
            # in-flight or crashed delivery). The gap clock runs across claim
            # acquisitions: while the hole's owner is actively executing it
            # HOLDS the claim, so our next round blocks on acquisition instead
            # of measuring — only an absent owner lets the clock accumulate.
            if gap_since is None:
                gap_since = loop.time()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=backoff)
            backoff = min(backoff * 2, self._config.gap_backoff_max)

    async def _process_ready(self, *, force_skip: bool) -> bool:
        """One round: a cursor pre-check, then a claim tenure if warranted.

        The pre-check runs OFF the claim: when the cursor says none of this
        lane's entries are up next, entering the claim queue would only add
        churn — with N workers holding plans for one room, every wrong-turn
        acquisition delays the right worker's grant behind grant-check-release
        cycles that do nothing. A lane therefore polls the cheap cursor read
        and only queues for the claim when it has work the cursor allows —
        or when its gap clock expired, because the skip decision must be
        made under the claim: acquiring it blocks behind a live owner, so a
        slow delivery is re-checked, never skipped.

        Post-plan effects and reentry passes run AFTER the claim is
        released — the reentry pass takes the room lock, and holding claim
        and room lock together is the two-connection deadlock this design
        forbids (a lane never holds both).
        """
        delivered = await self._read_cursor()
        progressed = self._drop_stale(delivered)
        if not force_skip and not self._has_turn(delivered):
            return progressed

        executed: list[_Executed] = []
        self._busy = self._sending = True
        try:
            progressed |= await self._claimed_round(executed, force_skip=force_skip)
            self._sending = False
            while executed:
                run = executed.pop(0)
                await self._finish(run.entry, run.result)
        finally:
            self._busy = self._sending = False
            # A cancellation mid-round must not leak cascade units: every
            # entry pulled off the queues but not finished reports what it
            # delivered and is dropped here.
            for run in executed:
                self._report_cut(run.entry, run.result)
                self._drop_entry(run.entry, self._reason)
        return progressed

    def _report_cut(self, entry: ExecEntry, result: Any) -> None:
        """What a cut entry delivered, to its caller: the result of its
        delivery set, or what the set had delivered when the kit's close cut
        it, the sends still under way failed."""
        partial = entry.plan.partial_result
        if (
            result is None
            and partial is not None
            and entry.cascade.cancelled == KitClosedError.code
        ):
            result = partial()
        if result is not None:
            self._host._report_cut(entry.plan, result, entry.cascade)

    def _has_turn(self, delivered: int) -> bool:
        """Whether the cursor allows any of this lane's entries to run."""
        if delivered + 1 in self._indexed:
            return True
        return bool(self._unindexed) and self._unindexed[0].after_index <= delivered

    def _drop_stale(self, delivered: int) -> bool:
        """Drop entries another worker's gap-skip declared lost: their
        delivery was skipped past, executing them now would break the
        per-room order. Needs no claim — nothing is written."""
        progressed = False
        for idx in [i for i in self._indexed if i <= delivered]:
            stale = self._indexed.pop(idx)
            if isinstance(stale, ExecEntry):
                self._drop_entry(stale, "delivery_stale")
            progressed = True
        return progressed

    async def _claimed_round(self, executed: list[_Executed], *, force_skip: bool) -> bool:
        progressed = False
        async with self._claims.locked(delivery_claim_key(self.room_id)):
            delivered = await self._read_cursor()
            while self._runs_on:
                progressed |= self._drop_stale(delivered)

                while self._unindexed and self._unindexed[0].after_index <= delivered:
                    await self._execute_owned(self._unindexed.popleft(), executed)
                    progressed = True

                nxt = self._indexed.pop(delivered + 1, None)
                if nxt is not None:
                    if isinstance(nxt, ExecEntry):
                        await self._execute_owned(nxt, executed)
                    await self._host._store.advance_delivered_index(self.room_id, delivered + 1)
                    delivered += 1
                    progressed = True
                    continue
                if not self._indexed and not self._unindexed:
                    break
                # Hole. Re-read — another process may have advanced the
                # cursor while we were executing.
                refreshed = await self._read_cursor()
                if refreshed > delivered:
                    delivered = refreshed
                    progressed = True
                    continue
                if force_skip and not progressed:
                    target = self._skip_target(delivered)
                    if target > delivered and await self._host._store.advance_delivered_index(
                        self.room_id, target, force=True
                    ):
                        logger.warning(
                            "Delivery gap skipped for room %s: indexes %d..%d declared lost",
                            self.room_id,
                            delivered + 1,
                            target,
                        )
                        await self._host._emit_framework_event(
                            "delivery_skipped",
                            room_id=self.room_id,
                            data={"from_index": delivered + 1, "to_index": target},
                        )
                        delivered = target
                        progressed = True
                        continue
                break
        return progressed

    async def _read_cursor(self) -> int:
        return await self._host._store.get_delivered_index(self.room_id)

    def _skip_target(self, delivered: int) -> int:
        """The smallest cursor value that unblocks the earliest local entry.

        Never past an index this lane holds a plan for — a lane must not
        skip its own deliveries, only holes owned by absent workers.
        """
        needs: list[int] = []
        if self._indexed:
            needs.append(min(self._indexed) - 1)
        if self._unindexed:
            needs.append(min(e.after_index for e in self._unindexed))
        candidates = [n for n in needs if n > delivered]
        return min(candidates) if candidates else delivered

    async def _execute_owned(self, entry: ExecEntry, executed: list[_Executed]) -> None:
        """Execute a dequeued entry, recorded in ``executed`` BEFORE the await.

        Recording after the call would evaluate the await first: a
        cancellation there (``aclose`` cancelling the executor) leaves the
        entry in no queue and in no record, its cascade unit forever
        unreleased and every waiter on that cascade hung.
        """
        if entry is self._placed_idle:
            self._placed_idle = None
        run = _Executed(entry)
        executed.append(run)
        try:
            run.result = await entry.cascade.run(self._execute(entry))
        except asyncio.CancelledError:
            if self._cut:
                # The entry reports what its set has done (partial_result);
                # the set unwinds as the cascade's work, which the kit's
                # close waits for within its unwind.
                raise
            if entry.cascade.cancelled is None:
                raise
            # Still consume the committed index; cancellation must not leave
            # a cursor hole or stop another cascade in the same lane.

    async def _execute(self, entry: ExecEntry) -> Any:
        """Run one plan's delivery set; a failure is recorded, never raised.

        ``_execute_plan`` collects per-target failures into its result — an
        exception escaping it is a framework defect, logged and turned into
        a completed-empty execution so the cursor and the cascade still
        advance.

        Its deliveries and turns are the room's work for the cascade: what
        they send into the room draws on its budget (RFC §9.5.1).
        """
        try:
            with delivering(self.room_id, entry.cascade.budget):
                return await self._host._execute_plan(entry.plan, entry.cascade)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Delivery execution failed for room %s index %s", self.room_id, entry.index
            )
            entry.cascade.record_error(exc)
            return None

    async def _finish(self, entry: ExecEntry, result: Any) -> None:
        """Post-claim aftermath of one executed plan, then the unit release.

        Runs under the planner's span (``DeliveryPlan.parent_span_id``), the
        way ``execute_plan`` does for the delivery set: this task's context
        is fresh, so without the restore the AFTER_BROADCAST hook spans and
        the reentry pass — whose own plan captures the current span as its
        parent — would be trace roots instead of children of the inbound or
        send_event span that started the turn.
        """
        try:
            if result is not None and entry.cascade.cancelled is not None:
                # Cancelled once the plan ran: nobody reads what it started,
                # and its event's aftermath still follows (RFC §10.1).
                drop_unread_streams(getattr(result, "streaming_responses", ()))
            if result is not None:
                await entry.cascade.run(self._finish_effects(entry, result))
        except asyncio.CancelledError:
            if self._cut:
                # Cut in its aftermath: its caller still reads what it
                # delivered (RFC §12.10.4).
                self._report_cut(entry, result)
                entry.cascade.cancel(self._reason)
                raise
            if entry.cascade.cancelled is None:
                raise
        except Exception as exc:
            # Post-plan effects include a real write (side-effect persistence):
            # the failure must reach the caller, not just the log.
            logger.exception("Post-delivery effects failed for room %s", self.room_id)
            entry.cascade.record_error(exc)
        finally:
            entry.cascade.release()

    async def _finish_effects(self, entry: ExecEntry, result: Any) -> None:
        plan = entry.plan
        with restored_span(plan.parent_span_id, telemetry_ctx=plan.parent_span_ctx):
            await self._host._post_plan_effects(plan, result, entry.cascade)
            if entry.cascade.cancelled is None:
                await self._host._reentry_commit_pass(self.room_id, plan, result, entry.cascade)


class RoomLaneRegistry:
    """The framework's lanes, created lazily and retired when idle."""

    def __init__(self, host: LaneHost, claims: RoomLockManager, config: LaneConfig) -> None:
        self._host = host
        self._claims = claims
        self._config = config
        self._lanes: dict[str, RoomDeliveryLane] = {}
        self._sealed = False

    def lane(self, room_id: str) -> RoomDeliveryLane:
        """Get or create (and start) the room's lane."""
        lane = self._lanes.get(room_id)
        if lane is None:
            lane = RoomDeliveryLane(
                room_id, self._host, self._claims, self._config, retire=self._try_retire
            )
            self._lanes[room_id] = lane
            lane.start()
        return lane

    def enqueue(self, room_id: str, entry: ExecEntry | CursorEntry) -> None:
        """Route an entry to its room's lane.

        After the registry is sealed (framework close) an ExecEntry is
        dropped with its cascade cancelled — the process is going away, and
        this is the crash-window loss class, made explicit.
        """
        if self._sealed:
            if isinstance(entry, ExecEntry):
                logger.info(
                    "Delivery dropped (kit closing) for room %s index %s",
                    room_id,
                    entry.index,
                )
                entry.cascade.cancel(KitClosedError.code)
                entry.cascade.release()
            elif (lane := self._lanes.get(room_id)) is not None:
                # A committed index still fills the cursor of a lane running
                # the set it keeps (RFC §12.10.4); it opens no lane.
                lane.enqueue(entry)
            return
        self.lane(room_id).enqueue(entry)

    def note_committed(self, room_id: str, index: int) -> None:
        """Record a committed index with no delivery set (cursor no-op)."""
        self.enqueue(room_id, CursorEntry(index))

    def _try_retire(self, lane: RoomDeliveryLane) -> bool:
        """Idle-timeout retirement; no await between the check and the drop."""
        if not lane.idle:
            return False
        if self._lanes.get(lane.room_id) is lane:
            del self._lanes[lane.room_id]
        return True

    def seal(self, reason: str = "kit_closed") -> None:
        """Take no more work (RFC §12.10.4): every entry queued or enqueued
        from now on is dropped with *reason*, its cascade cancelled, but the
        one each lane holds placed while it was idle, which runs from its
        placement; the rounds under way end after the entry each runs.
        Idempotent."""
        self._sealed = True
        for lane in self._lanes.values():
            lane.seal(reason)

    def running(self) -> set[asyncio.Task[None]]:
        """The executors still running, the rounds under way included."""
        return {
            lane.task
            for lane in self._lanes.values()
            if lane.task is not None and not lane.task.done()
        }

    def drop_placed_by(self, spared: set[asyncio.Task[Any]]) -> None:
        """Once sealed, drop the entry each lane keeps that the close's own
        work, *spared*, placed (an ``AFTER_BROADCAST`` hook that sends, then
        closes): what the closing work sent is never delivered, the lane
        idle or not (RFC §12.10.4)."""
        for lane in self._lanes.values():
            lane.drop_kept(placed_by=spared)

    def cut(self, spared: set[asyncio.Task[Any]]) -> None:
        """Cut every round still under way but those *spared* (a lane whose
        work runs the close, RFC §12.10.4), which drop the entry they kept
        instead."""
        for lane in self._lanes.values():
            if lane.task not in spared:
                lane.cut()
            else:
                lane.drop_kept()

    async def aclose(self) -> None:
        """Seal the registry, give the rounds under way the lanes'
        ``close_grace`` to end, then cut what is left and let it unwind, at
        most :data:`~roomkit.core.task_utils.CUT_UNWIND_S`. A lane whose work
        runs this close is neither
        waited for nor cut. Idempotent. The kit's close runs these steps
        itself, on the one grace it gives all its work."""
        self.seal()

        def unspared() -> set[asyncio.Task[None]]:
            return self.running() - spared_by_close()

        await wait_for_unspared(unspared, timeout=self._config.close_grace)
        self.cut(spared_by_close())
        await wait_for_unspared(unspared, timeout=task_utils.CUT_UNWIND_S)

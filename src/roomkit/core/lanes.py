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
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.locks import _held_rooms

if TYPE_CHECKING:
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
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
    committed timeline. ``injected`` marks a hook-injected event, whose
    delivery is deliberately bare — direct ``on_event`` / ``deliver`` to the
    named channels, no transcoding and no response collection: an injected
    event must not be able to trigger an AI reply.
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
    # The trigger's response_visibility, stamped onto reentry events so the
    # router's visibility check honours the caller's response scope.
    response_visibility: str | None = None
    # Side effects collected by the BEFORE_BROADCAST sync hooks, persisted
    # with the execution results (they always persisted post-broadcast).
    hook_tasks: list[Task] = field(default_factory=list)
    hook_observations: list[Observation] = field(default_factory=list)
    # Whether the event's AFTER_BROADCAST hooks (and side-effect persistence)
    # fire after its set executes. False preserves the paths that never fired
    # them: injected events and a trigger without a source binding.
    fire_after_broadcast: bool = True
    # Trace continuity across the lane boundary: the caller's current span
    # (and its backend context), captured at plan time. The lane executor
    # runs on a fresh contextvars context, so without these the broadcast
    # span would be an orphan instead of a child of the inbound span.
    parent_span_id: str | None = None
    parent_span_ctx: Any = None
    # True on the root pass of an inbound/send_event call: emits
    # ``event_processed`` once the delivery set has executed.
    emit_processed: bool = False


class DeliveryCascade:
    """Completion of a caller's delivery set, transitively.

    One unit per enqueued plan; the executor releases a unit only after the
    plan's delivery set, its post-plan effects (AFTER_BROADCAST included) and
    its reentry pass have run — and the reentry pass retains a unit for every
    child plan *before* the parent's release, so the count never touches zero
    between two passes of an AI chain. ``wait()`` therefore resolves exactly
    when "the caller observes its event's delivery-set completion" (RFC §10.1
    step 18) holds for the whole chain the call started.
    """

    __slots__ = (
        "_done",
        "_pending",
        "_reentry_budget",
        "cancelled",
        "error",
        "room_id",
        "streams",
    )

    def __init__(self, room_id: str, *, reentry_budget: int) -> None:
        self.room_id = room_id
        self._pending = 0
        self._done = asyncio.Event()
        self._done.set()
        self._reentry_budget = reentry_budget
        # Streaming responses captured during execution, consumed by the
        # caller after wait() — streaming delivery (TTS, long generations)
        # must not stall the lane.
        self.streams: list[Any] = []
        # First intelligence-channel failure, surfaced on InboundResult.error.
        self.error: Exception | None = None
        # Reason the cascade was cancelled (close/seal), or None.
        self.cancelled: str | None = None

    def retain(self) -> None:
        """Account one pending unit. Call before any await can fail."""
        self._pending += 1
        self._done.clear()

    def release(self) -> None:
        """Release one unit; the last release resolves ``wait()``."""
        self._pending -= 1
        if self._pending <= 0:
            self._done.set()

    def consume_reentry_budget(self) -> bool:
        """Take one reentry slot; ``False`` when the chain cap is exhausted."""
        if self._reentry_budget <= 0:
            return False
        self._reentry_budget -= 1
        return True

    def add_streams(self, streams: list[Any]) -> None:
        self.streams.extend(streams)

    def record_error(self, exc: Exception) -> None:
        """Record an intelligence failure; the first one wins."""
        if self.error is None:
            self.error = exc

    def cancel(self, reason: str) -> None:
        """Abort the cascade: wake every waiter, keep the reason."""
        self.cancelled = reason
        self._done.set()

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
        if _active_lane_room.get() == self.room_id or self.room_id in _held_rooms.get():
            return False
        await self._done.wait()
        return True

    async def wait_detached(self) -> None:
        """Wait for true completion, without the short-circuit guards.

        For the background consumer a detached caller schedules — it runs on
        its own clean task, where waiting cannot deadlock.
        """
        await self._done.wait()


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


@dataclass(slots=True)
class CursorEntry:
    """A committed index with no delivery set (a BLOCKED event, a reentry
    without a binding, a stored-only injected event, a streamed segment, an
    inline-delivered commit): advances the cursor at its turn, executes
    nothing. Every committed index MUST reach its lane as exactly one entry —
    a missing one is a permanent hole the gap policy will eventually skip.
    """

    index: int


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

    ``_execute_plan`` runs a plan's delivery set and returns its
    ``BroadcastResult``-shaped outcome. ``_post_plan_effects`` fires the
    per-event aftermath (delivery framework events, ON_ERROR funnel, side
    effects, AFTER_BROADCAST). ``_reentry_commit_pass`` turns the response
    events of an executed plan into fresh commit passes (room lock taken
    anew) whose plans join the same lane.
    """

    _store: ConversationStore

    async def _execute_plan(self, plan: DeliveryPlan) -> Any: ...

    async def _post_plan_effects(
        self, plan: DeliveryPlan, result: Any, cascade: DeliveryCascade
    ) -> None: ...

    async def _reentry_commit_pass(
        self, room_id: str, plan: DeliveryPlan, result: Any, cascade: DeliveryCascade
    ) -> None: ...

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
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._busy = False
        self._closed = False

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
        every lock its creator held.
        """
        if self._task is None and not self._closed:
            self._task = asyncio.get_running_loop().create_task(
                self._run(),
                name=f"roomkit-delivery-lane-{self.room_id}",
                context=contextvars.Context(),
            )

    def enqueue(self, entry: ExecEntry | CursorEntry) -> None:
        """Accept an entry. Synchronous — runs under the room lock."""
        if self._closed:
            self._drop_entry(entry, "lane_closed")
            return
        if isinstance(entry, CursorEntry):
            self._indexed[entry.index] = entry
        elif entry.index is None:
            self._unindexed.append(entry)
        else:
            self._indexed[entry.index] = entry
        self._wakeup.set()

    async def drain(self) -> None:
        """Wait until every queued entry has been processed."""
        while self._indexed or self._unindexed or self._busy:
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        """Stop the lane: bounded drain, then cancel, then abort leftovers.

        The drain runs first, while the executor is still willing — a close
        must not throw away work it had time to do. It is bounded (a cursor
        hole owned by an absent worker would otherwise hold the whole
        framework close), as is the cancellation grace — a provider that
        swallows cancellation cannot extend it. Whatever remains is aborted:
        every pending unit is released and its cascade cancelled so no
        waiter hangs.
        """
        if not self._closed:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.drain(), timeout=self._config.close_grace)
        self._closed = True
        self._wakeup.set()
        task, self._task = self._task, None
        if task is not None and task is not asyncio.current_task():
            _, pending = await asyncio.wait({task}, timeout=1.0)
            if pending:
                task.cancel()
                await asyncio.wait({task}, timeout=1.0)
        self._abort_pending("kit_closed")

    def _drop_entry(self, entry: ExecEntry | CursorEntry, reason: str) -> None:
        if isinstance(entry, ExecEntry):
            logger.warning(
                "Delivery dropped (%s) for room %s index %s",
                reason,
                self.room_id,
                entry.index,
            )
            entry.cascade.cancel(reason)
            entry.cascade.release()

    def _abort_pending(self, reason: str) -> None:
        for entry in list(self._indexed.values()):
            self._drop_entry(entry, reason)
        self._indexed.clear()
        while self._unindexed:
            self._drop_entry(self._unindexed.popleft(), reason)

    async def _run(self) -> None:
        # Belt and braces with start(): this task must hold no inherited lock
        # membership, and everything it calls must know it runs inside this
        # room's lane.
        _held_rooms.set(frozenset())
        _active_lane_room.set(self.room_id)
        loop = asyncio.get_running_loop()
        gap_since: float | None = None
        backoff = self._config.gap_backoff_initial
        while not self._closed:
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

        executed: list[tuple[ExecEntry, Any]] = []
        self._busy = True
        try:
            progressed |= await self._claimed_round(executed, force_skip=force_skip)
            while executed:
                entry, result = executed.pop(0)
                await self._finish(entry, result)
        finally:
            self._busy = False
            # A cancellation mid-round must not leak cascade units: every
            # entry pulled off the queues but not finished is aborted here.
            for entry, _ in executed:
                self._drop_entry(entry, "lane_cancelled")
        return progressed

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

    async def _claimed_round(
        self, executed: list[tuple[ExecEntry, Any]], *, force_skip: bool
    ) -> bool:
        progressed = False
        async with self._claims.locked(delivery_claim_key(self.room_id)):
            delivered = await self._read_cursor()
            while not self._closed:
                progressed |= self._drop_stale(delivered)

                while self._unindexed and self._unindexed[0].after_index <= delivered:
                    entry = self._unindexed.popleft()
                    executed.append((entry, await self._execute(entry)))
                    progressed = True

                nxt = self._indexed.pop(delivered + 1, None)
                if nxt is not None:
                    if isinstance(nxt, ExecEntry):
                        executed.append((nxt, await self._execute(nxt)))
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

    async def _execute(self, entry: ExecEntry) -> Any:
        """Run one plan's delivery set; a failure is recorded, never raised.

        ``_execute_plan`` collects per-target failures into its result — an
        exception escaping it is a framework defect, logged and turned into
        a completed-empty execution so the cursor and the cascade still
        advance.
        """
        try:
            return await self._host._execute_plan(entry.plan)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Delivery execution failed for room %s index %s", self.room_id, entry.index
            )
            entry.cascade.record_error(exc)
            return None

    async def _finish(self, entry: ExecEntry, result: Any) -> None:
        """Post-claim aftermath of one executed plan, then the unit release."""
        try:
            if result is not None:
                await self._host._post_plan_effects(entry.plan, result, entry.cascade)
                await self._host._reentry_commit_pass(
                    self.room_id, entry.plan, result, entry.cascade
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Post-plan effects include a real write (side-effect persistence):
            # the failure must reach the caller, not just the log.
            logger.exception("Post-delivery effects failed for room %s", self.room_id)
            entry.cascade.record_error(exc)
        finally:
            entry.cascade.release()


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
                logger.warning(
                    "Delivery dropped (registry sealed) for room %s index %s",
                    room_id,
                    entry.index,
                )
                entry.cascade.cancel("kit_closed")
                entry.cascade.release()
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

    async def aclose(self) -> None:
        """Seal the registry and close every lane. Idempotent."""
        self._sealed = True
        lanes, self._lanes = list(self._lanes.values()), {}
        for lane in lanes:
            await lane.aclose()

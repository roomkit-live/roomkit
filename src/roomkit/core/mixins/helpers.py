"""HelpersMixin — internal helpers shared across framework mixins.

``_RECENT_EVENTS_LIMIT`` is the hard ceiling on how many events the
in-memory ``RoomContext.recent_events`` carries. Memory providers that
care about token budget (``BudgetAwareMemory``) trim further per turn,
so this number is the safety upper bound — large enough that a long
chat never trips it, small enough that the worst-case memory footprint
stays sane. The mixins below import it for the per-event append slices
in ``inbound_locked`` / ``inbound_streaming`` so all three sites stay
aligned with the initial store fetch.

The previous value (50) predates ``BudgetAwareMemory`` and was
calibrated for ``SlidingWindowMemory`` (event-count trimming). With
the token-aware memory provider doing the real work, the event cap
was both redundant and harmful — it dropped older turns even when
the token budget had plenty of headroom, producing the visible
"context shrinks when a long past message rolls off" behavior on
long conversations.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core._authors import own_record_id, reached, stored_register, with_author
from roomkit.core._participant_channels import channels_reached, warn_cross_channel
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.hooks import SyncPipelineResult
from roomkit.core.locks import RoomLockManager, lent_while_it_runs
from roomkit.core.task_utils import run_to_its_end
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
    IdentificationStatus,
    RoomStatus,
    Visibility,
)
from roomkit.models.event import EventSource, RoomEvent, SystemContent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.identity import Identity, IdentityHookResult, IdentityResult
from roomkit.models.participant import Participant
from roomkit.models.task import Observation, Task
from roomkit.models.tool_call import (
    ToolCallEvent,
    ToolCallVerdict,
)
from roomkit.tools.external import BeforeToolDecision

_RECENT_EVENTS_LIMIT = 2_000
"""Hard ceiling on events kept in ``RoomContext.recent_events`` in memory."""

_RECENT_EVENTS_FLOOR = 50
"""Events loaded for a room whose channels declare no recent-history need
(transport-only rooms, e.g. realtime voice) while something else will read
them: a hook, regular or identity, or a caller that scans the tail itself.
Enough for a hook that glances at recent context without paying the
full-ceiling deserialisation per turn; with nobody to read them, none are
loaded."""

# RFC §5.1 — statuses that refuse new events. CLOSED and ARCHIVED refuse
# identically; they differ in intent, not in what they accept. Lives here
# because every path that can grow a timeline must consult it (§5.1: "at
# EVERY point where the timeline can grow"), and those paths span mixins.
_REFUSING_STATUSES = frozenset({RoomStatus.CLOSED, RoomStatus.ARCHIVED})

_RefusedOperation = Literal["inbound", "reentry", "regenerate"]
"""The path a ``room_refused_event`` names in ``data["operation"]`` (RFC §8.2)."""


def _warn_top_level(room_id: str, parent_event_id: str) -> None:
    """Say that an event naming *parent_event_id* is posted at the top level."""
    logger.warning(
        "Thread parent %s not found in room %s; posting at top level",
        parent_event_id,
        room_id,
        extra={"room_id": room_id, "parent_event_id": parent_event_id},
    )


def _refuses_writes(room: Room | None) -> bool:
    """Whether *room* refuses new timeline events (RFC §5.1). A missing room
    refuses too: nothing can be appended to a room that is not there."""
    return room is None or room.status in _REFUSING_STATUSES


def _source_block_reason(binding: ChannelBinding | None) -> str | None:
    """Why an event from a source bound by *binding* is stored BLOCKED
    (RFC §7.5 rule 2), ``None`` for a source that can write.

    ``source_unbound`` for a source with no binding in the room, such as a
    channel the integrator detached (rule 7); ``source_read_only`` for an
    access without writing, ``NONE`` included, which wins over a mute since
    it is the lasting cause; ``source_muted`` for a muted source that could
    write."""
    if binding is None:
        return "source_unbound"
    if binding.can_write:
        return None
    if binding.access not in (Access.READ_WRITE, Access.WRITE_ONLY):
        return "source_read_only"
    return "source_muted"


if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import (
        ThinkingHook,
        ToolUsageLoader,
    )
    from roomkit.channels._task_planner import PlanUpdatedCallback
    from roomkit.channels.base import Channel
    from roomkit.core._closing import Cascades, DoorRuns
    from roomkit.core._timer_transitions import TimerTransitions
    from roomkit.core.hooks import HookEngine, IdentityHookRegistration
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.room import Room
    from roomkit.models.tool_call import (
        ToolCallCallback,
        ToolCallObserver,
    )
    from roomkit.store.base import ConversationStore
    from roomkit.tools.external import BeforeToolCallback

logger = logging.getLogger("roomkit.framework")

FrameworkEventHandler = Callable[[FrameworkEvent], Coroutine[Any, Any, None]]
IdentityHookFn = Callable[
    [RoomEvent, RoomContext, IdentityResult],
    Coroutine[Any, Any, IdentityHookResult | None],
]


@runtime_checkable
class FrameworkHelpers(Protocol):
    """Contract: capabilities a host class must provide for HelpersMixin.

    Every attribute listed here is initialized by ``RoomKit.__init__``.
    This Protocol also serves as the base contract for the 12 framework
    mixins that inherit from HelpersMixin — their own Protocols extend
    these requirements with mixin-specific attributes and methods.

    Attributes:
        _store: Persistent storage backend for rooms, events, participants.
        _channels: Registry of all registered channels, keyed by channel ID.
        _hook_engine: Engine for sync/async hook pipeline execution.
        _event_handlers: List of ``(event_type, handler)`` pairs for
            framework event dispatch.
        _identity_hooks: Per-trigger identity hook registrations.
        _pending_traces: Buffered protocol traces for rooms that don't
            exist yet — flushed when the room is created.
        _pending_hook_tasks: Fire-and-forget async tasks awaiting cleanup.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _turn_hooks: HookEngine  # the kit's own turn executor's (RFC §10.6.6)
    _event_handlers: list[tuple[str, FrameworkEventHandler]]
    _identity_hooks: dict[HookTrigger, list[IdentityHookRegistration]]
    _pending_traces: dict[str, list[object]]
    _pending_hook_tasks: set[asyncio.Task[Any]]


class HelpersMixin:
    """Internal helpers used by other framework mixins.

    Host contract: :class:`FrameworkHelpers`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _turn_hooks: HookEngine  # the kit's own turn executor's (RFC §10.6.6)
    _event_handlers: list[tuple[str, FrameworkEventHandler]]
    _identity_hooks: dict[HookTrigger, list[IdentityHookRegistration]]
    _pending_traces: dict[str, list[object]]  # room_id -> [ProtocolTrace, ...]
    _pending_hook_tasks: set[asyncio.Task[Any]]
    _decided_writes: set[asyncio.Task[Any]]  # writes past a decision — set by RoomKit.__init__
    _persistence_policy: Any  # PersistencePolicy | None — set by RoomKit.__init__
    _resource_lease: Any  # RoomKit._resource_lease — the close()-ordering hold on the store
    _ensure_status_bus_subscribed: Any  # RoomKit's — the StatusBus listened to from now on
    _lanes: Any  # RoomLaneRegistry — set by RoomKit.__init__
    _cascades: Cascades  # the deliveries the kit's close cuts — set by RoomKit.__init__
    _door_runs: DoorRuns  # the calls into a room the kit's close cuts — set by RoomKit.__init__
    _room_close_epoch: int  # counted by _store_refusing_room — set by RoomKit.__init__
    _room_timers: TimerTransitions  # the rooms' due timer transitions — set by RoomKit.__init__
    _process_timeout: float  # the pre-commit bound (RFC §13.6) — set by RoomKit.__init__
    _lock_manager: RoomLockManager  # the per-room lock — set by RoomKit.__init__
    _closed: bool  # set once close() starts — set by RoomKit.__init__
    turn_executor: Any  # RoomKit's own TurnExecutor, whose wiring the turn triggers run on
    get_room: Any  # RoomLifecycleMixin's, the room read with the caller's scope (RFC §17.2)

    # -- Persistence helpers (policy-aware) --
    #
    # Every committed index must be accounted on the room's delivery cursor
    # exactly once (RFC §10.2 — a missing entry is a permanent hole the gap
    # policy eventually skips). These two methods and ``_commit_to_lane``
    # (the planned variant, LaneExecutionMixin) are therefore the ONLY paths
    # to ``store.commit_event`` in the framework; a static guard test
    # enforces it.
    #
    # The two here are for events with NO delivery set. An event that gets
    # delivered belongs to ``_commit_to_lane`` / ``_commit_and_deliver``,
    # because accounting it here publishes its index on the cursor at commit
    # time — which is precisely what releases the lane to execute the *next*
    # index while this one is still undelivered.

    async def _room_refuses_writes(self, room_id: str) -> bool:
        """Whether the room's status refuses new timeline events (RFC §5.1).

        The status gate applies at *every* point where the timeline can grow —
        an inbound message, a hook's injected event, the framework's own
        re-injection and its lifecycle system events alike. Callers outside
        the locked pipeline (which reads the status off the context it already
        holds) ask here; a missing room refuses too.
        """
        return _refuses_writes(await self._store.get_room(room_id))

    async def _store_refusing_room(self, room: Room) -> Room:
        """Store *room* in a status that refuses writes (CLOSED, ARCHIVED).

        It counts the change, as a timer's close does
        (:meth:`_count_room_closed`), and a stream, which reads the status
        once and again only when that count moved, refuses its next row (RFC
        §5.1). A status changed elsewhere (another process, another kit
        sharing the store, the store directly) is seen by the next response.
        """
        stored = await self._store.update_room(room)
        self._count_room_closed()
        return stored

    def _count_room_closed(self) -> None:
        """Count a room stored in a status that refuses writes, which a
        stream reads to know it must read the status again (RFC §5.1)."""
        self._room_close_epoch += 1

    async def _refuse_on_timeout(
        self,
        room_id: str | None,
        *,
        channel_id: str | None = None,
        event_id: str | None = None,
        operation: _RefusedOperation = "inbound",
    ) -> InboundResult:
        """Refuse what ``process_timeout`` stopped before its commit point
        (RFC §13.6), with the ``process_timeout`` framework event: an inbound
        event before the room lock (*channel_id*) or under it (*event_id*),
        or a regeneration still waiting for the lock (``operation``). Its
        room is null when the deadline passed while the message was routed
        (RFC §8.2)."""
        logger.error(
            "Pre-commit of %s timed out after %.1fs",
            operation,
            self._process_timeout,
            extra={"room_id": room_id, "channel_id": channel_id, "event_id": event_id},
        )
        data: dict[str, Any] = {"timeout": self._process_timeout}
        if operation != "inbound":
            data["operation"] = operation
        await self._emit_framework_event(
            "process_timeout",
            room_id=room_id,
            channel_id=channel_id,
            event_id=event_id,
            data=data,
        )
        return InboundResult(blocked=True, reason="process_timeout")

    async def _refuse_closed_room(
        self,
        room_id: str,
        *,
        status: RoomStatus | None,
        operation: _RefusedOperation,
        event: RoomEvent | None,
    ) -> InboundResult:
        """Refuse a write to a room whose status refuses new events (RFC §5.1).

        The one place the refusing paths converge — the inbound gate (§10.1
        step 6), the reentry pass and a regenerate — so a refusal reads the
        same from each: a log line, the ``room_refused_event`` framework event
        with the one ``data`` contract §8.2 specifies (``status``,
        ``operation``, ``event_type``), and the blocked result. Nothing is
        written, not even a BLOCKED record: appending an audit event to a
        closed room is the thing the status forbids.

        ``event`` is the refused event — for a regenerate, the message it
        would have replayed, ``None`` when nothing qualified. ``status`` is
        ``None`` only when the room no longer exists (a reentry whose room was
        deleted while its trigger's delivery set ran).
        """
        event_id = event.id if event is not None else None
        event_type = str(event.type) if event is not None else None
        logger.info(
            "Refused %s: room %s is %s",
            operation,
            room_id,
            status if status is not None else "gone",
            extra={"room_id": room_id, "event_id": event_id},
        )
        await self._emit_framework_event(
            "room_refused_event",
            room_id=room_id,
            event_id=event_id,
            data={
                "status": str(status) if status is not None else None,
                "operation": operation,
                "event_type": event_type,
            },
        )
        return InboundResult(blocked=True, reason="room_closed")

    async def _persist_committed(
        self,
        room_id: str,
        event: RoomEvent,
        *,
        target: RoomEvent | None = None,
        activity: bool,
    ) -> RoomEvent | None:
        """Atomically commit an event (index + insert + room counters, RFC
        §10.1 step 12 / §14.3) if the policy allows it, and account its
        index on the room's delivery cursor. *target* is what an edit or a
        delete writes on its target, in the same transaction, and
        *activity* whether the commit is the room's activity (RFC §5.1;
        :meth:`_commit`).

        Returns the committed event, or ``None`` if the policy excluded it.
        """
        if self._persistence_policy is not None and not self._persistence_policy.should_persist(
            event.type
        ):
            await self._write_unstored_change(room_id, target)
            return None
        return await self._commit_noted(room_id, event, target=target, activity=activity)

    async def _with_thread_root(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """*event* with its in-app thread parent rooted, as every path that
        commits an event stores it (RFC §10.1 step 8): a parent naming a reply
        is replaced by that reply's thread root, and one the room does not
        hold is dropped, the event posted at the top level. The same event
        when its parent needs no change."""
        parent = event.parent_event_id
        if parent is None:
            return event
        root = await self._resolve_thread_root(room_id, parent)
        return event if root == parent else event.model_copy(update={"parent_event_id": root})

    async def _resolve_thread_root(self, room_id: str, parent_event_id: str) -> str | None:
        """Resolve an in-app thread parent to its thread root.

        Flat two-level threading: a reply always points at the thread ROOT.
        If the referenced event is itself a reply, its root is returned so
        replying-to-a-reply stays in a single thread. A parent that, once
        rooted, names no event of the room (none, another room's, a root a
        hard delete removed while its reply stayed) drops to ``None`` (top
        level) with a warning — a stale reference must not lose the sender's
        message (RFC §10.1 step 8).
        """
        parent = await self._store.get_event(parent_event_id)
        if parent is None or parent.room_id != room_id:
            _warn_top_level(room_id, parent_event_id)
            return None
        root_id = parent.parent_event_id
        if root_id is None:
            return parent_event_id
        root = await self._store.get_event(root_id)
        if root is None or root.room_id != room_id:
            _warn_top_level(room_id, root_id)
            return None
        return root_id

    async def _commit_indexed(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Commit an event unconditionally (policy-exempt) and account its
        index on the delivery cursor.

        For commits that must always be stored regardless of the persistence
        policy and that nobody receives: BLOCKED events (part of the
        timeline, they consume an index — RFC §8.3), records committed
        outside the pipeline (``commit_event``, §10.5), child-room traces.
        None is the room's activity (§5.1): ``last_activity_at`` stays.
        """
        return await self._commit_noted(room_id, event, activity=False)

    async def _commit_noted(
        self, room_id: str, event: RoomEvent, *, target: RoomEvent | None = None, activity: bool
    ) -> RoomEvent:
        """Commit *event* and account its index on the delivery cursor, as one
        unit its caller's cancellation cannot cut (RFC §13.5, §13.6): a
        caller cancelled between the two would leave an index the lane never
        reaches, and the room's next event waiting ``delivery_gap_timeout``
        for it. A cancellation waits for both, under the lock the caller
        holds, then is raised; the kit's close gives them its grace, then
        cuts them (RFC §12.10.4)."""
        return await run_to_its_end(
            self._commit_and_note(room_id, event, target=target, activity=activity),
            held=self._decided_writes,
            name=f"roomkit-decided-write-{room_id}",
        )

    async def _commit_and_note(
        self, room_id: str, event: RoomEvent, *, target: RoomEvent | None, activity: bool
    ) -> RoomEvent:
        """:meth:`_commit_noted`'s commit, then its index's note."""
        committed = await self._commit(room_id, event, target=target, activity=activity)
        await self._note_committed_index(room_id, committed.index)
        return committed

    async def _commit(
        self,
        room_id: str,
        event: RoomEvent,
        *,
        target: RoomEvent | None = None,
        activity: bool,
    ) -> RoomEvent:
        """Commit *event* (RFC §10.1 step 12) carrying the record of its author.

        *target* is the target of the edit or the delete *event* is, as its
        commit writes it: the store writes both in one transaction (§10.3,
        §14.3). It is passed only with a change, so a custom store that does
        not take it still commits every other event. *activity* is whether
        the commit is the room's activity (RFC §5.1), an event the pipeline
        stores DELIVERED; it is passed only when it is not, the store's
        default.
        """
        authored = await self._with_author(room_id, event)
        change: dict[str, Any] = {} if target is None else {"target": target}
        if not activity:
            change["activity"] = False
        return await self._store.commit_event(room_id, authored, **change)

    async def _write_unstored_change(self, room_id: str, target: RoomEvent | None) -> None:
        """Write *target* as the edit or the delete the persistence policy
        keeps out of the timeline changed it: with no row of its own to
        commit, the change is its target's update alone, a delete marking the
        target's edits deleted with it (RFC §10.3)."""
        if target is not None:
            await self._store.write_change_target(room_id, target)

    async def _with_author(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """*event* carrying the record of its author, under the room lock,
        before the commit (:func:`~roomkit.core._authors.with_author`)."""
        return await with_author(self._store, room_id, event)

    async def _note_committed_index(self, room_id: str, index: int) -> None:
        """Account a committed index that carries NO delivery set.

        Only ever for an event nobody receives — a BLOCKED event, a system
        or trace event, a greeting the voice path speaks itself. An index
        accounted here is declared delivered, so routing a real delivery
        through it would let the lane run the next index first.

        Opportunistic strict CAS first — the common case for inline paths
        (greeting, child rooms, streamed segments) where the cursor already
        sits at ``index - 1``. When it does not (a lane is in flight for the
        room), the index becomes a cursor entry the lane advances at its
        turn.
        """
        if await self._store.advance_delivered_index(room_id, index):
            return
        self._lanes.note_committed(room_id, index)

    # -- Error surfacing --

    async def _fire_error_hook(
        self,
        room_id: str,
        context: RoomContext,
        source: EventSource,
        *,
        error: str,
        error_type: str,
        error_category: str,
        chain_depth: int = 0,
        visibility: str = "all",
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
        responds_to: str | None = None,
    ) -> None:
        """Hand a turn-level failure to the ON_ERROR hooks, started and not
        waited for (RFC §9.3).

        RoomKit does not persist the error itself — it fires ON_ERROR with a
        synthetic error :class:`RoomEvent` so hosts can classify and surface it
        (e.g. render an error card). Every provider/inference failure path
        funnels through here so no failure vanishes with only a log line.
        """
        error_event = RoomEvent(
            room_id=room_id,
            source=source,
            content=TextContent(body=error),
            metadata={
                "error": error,
                "error_type": error_type,
                "error_category": error_category,
            },
            chain_depth=chain_depth,
            visibility=visibility,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            responds_to=responds_to,
        )
        self._hook_engine._start_async_hooks(room_id, HookTrigger.ON_ERROR, error_event, context)

    @staticmethod
    def _first_intelligence_error(broadcast_result: Any, context: RoomContext) -> Exception | None:
        """The first intelligence-channel generation failure from a broadcast,
        as the live exception (cause chain intact) so a headless caller can
        classify it. Transport-delivery failures are excluded — they are not
        turn-level agent errors.
        """
        errors_exc = getattr(broadcast_result, "errors_exc", {})
        for binding in context.bindings:
            if binding.category != ChannelCategory.INTELLIGENCE:
                continue
            exc = errors_exc.get(binding.channel_id)
            if exc is not None:
                return exc
        return None

    # -- Internal helpers --

    def _identity_hook_matches_event(
        self, hook: IdentityHookRegistration, event: RoomEvent
    ) -> bool:
        """Check if an identity hook's filters match the given event."""
        source = event.source

        # All filters must pass (None means "match all")
        type_ok = hook.channel_types is None or source.channel_type in hook.channel_types
        id_ok = hook.channel_ids is None or source.channel_id in hook.channel_ids
        dir_ok = hook.directions is None or source.direction in hook.directions

        return type_ok and id_ok and dir_ok

    async def _run_identity_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent,
        context: RoomContext,
        id_result: IdentityResult,
    ) -> IdentityHookResult | None:
        """Run identity hooks for *trigger*, return the first non-None result."""
        hooks = self._identity_hooks.get(trigger, [])
        for hook_reg in hooks:
            # Apply filters
            if not self._identity_hook_matches_event(hook_reg, event):
                continue
            try:
                # Lent the caller's room locks while it runs, as any hook's
                # body (RFC §9.5.1).
                with lent_while_it_runs():
                    result: IdentityHookResult | None = await hook_reg.fn(
                        event, context, id_result
                    )
                if result is not None:
                    return result
            except Exception:
                logger.exception(
                    "Identity hook failed for trigger %s",
                    trigger,
                    extra={"room_id": room_id, "trigger": str(trigger)},
                )
        return None

    async def _create_pending_participant(
        self,
        room_id: str,
        event: RoomEvent,
        id_result: IdentityResult,
    ) -> Participant:
        """Create a participant with pending identification status.

        Idempotent: if the sender's id already names a participant on the
        channel it arrived on, that record is returned without creating a
        duplicate. An id naming a record reached through other channels only
        names nobody here (RFC §5.5): a sender nobody identified does not adopt
        it, and gets a record of its own (:func:`own_record_id`), the one the
        identity step sends the event under and their next message finds.
        """
        participant_id = event.source.participant_id or f"pending-{uuid4().hex[:8]}"
        channel_id = event.source.channel_id
        existing = await self._store.get_participant(room_id, participant_id)
        if existing is not None and not reached(existing, channel_id):
            # Reached elsewhere since the identity step read the roster.
            participant_id = own_record_id(channel_id, participant_id)
            existing = await self._store.get_participant(room_id, participant_id)
        if existing is not None:
            warn_cross_channel(existing, channel_id, rehomed=False)
            return existing
        candidate_ids = [c.id for c in id_result.candidates] if id_result.candidates else None
        participant = Participant(
            id=participant_id,
            room_id=room_id,
            channel_id=channel_id,
            connected_via=[channel_id],
            identification=IdentificationStatus.PENDING,
            candidates=candidate_ids,
        )
        participant = await self._store.add_participant(participant)
        await self._emit_system_event(
            room_id,
            EventType.PARTICIPANT_JOINED,
            code="participant_joined_pending",
            message=f"Participant {participant.id} joined with pending identification",
            data={"participant_id": participant.id, "status": "pending"},
        )
        return participant

    async def _ensure_identified_participant(
        self,
        room_id: str,
        event: RoomEvent,
        identity: Identity,
    ) -> Participant:
        """Ensure a participant record exists for an identified identity.

        Idempotent: if a participant with the identity's ID already exists in the
        room, the existing record is returned without creating a duplicate. An
        identity reachable on several channels is the ordinary case here, so the
        channel this event came in on is recorded on that one record (RFC §5.5).
        """
        channel_id = event.source.channel_id
        existing = await self._store.get_participant(room_id, identity.id)
        if existing is not None:
            update: dict[str, Any] = {}
            # Update identification status if it was pending
            if existing.identification != IdentificationStatus.IDENTIFIED:
                update = {
                    "identification": IdentificationStatus.IDENTIFIED,
                    "identity_id": identity.id,
                    "display_name": identity.display_name or existing.display_name,
                }
            warn_cross_channel(existing, channel_id, rehomed=False)
            channels = channels_reached(existing, channel_id)
            if channels is not None:
                update["connected_via"] = channels
            if update:
                revised = existing.model_copy(update=update)
                existing = await self._store.update_participant(revised)
            return existing

        participant = Participant(
            id=identity.id,
            room_id=room_id,
            channel_id=channel_id,
            connected_via=[channel_id],
            display_name=identity.display_name,
            identification=IdentificationStatus.IDENTIFIED,
            identity_id=identity.id,
        )
        participant = await self._store.add_participant(participant)
        await self._emit_system_event(
            room_id,
            EventType.PARTICIPANT_JOINED,
            code="participant_joined_identified",
            message=f"Participant {participant.id} joined as identified",
            data={"participant_id": participant.id, "status": "identified"},
        )
        return participant

    async def _fire_lifecycle_hook(
        self,
        room_id: str,
        trigger: HookTrigger,
        event_type: EventType,
        code: str,
        message: str,
        data: dict[str, Any] | None = None,
        *,
        context: RoomContext | None = None,
    ) -> None:
        """Fire an async lifecycle hook with a synthetic system event, in
        *context*, the room as the change left it, read now when not given.

        The hooks are started and not waited for (RFC §9.3), but for
        ``ON_ROOM_CREATED``'s, which the room's creation awaits: a hook that
        attaches the room's agent has done so before anything is sent into
        the room."""
        event = RoomEvent(
            room_id=room_id,
            type=event_type,
            source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
            content=SystemContent(body=message, code=code, data=data or {}),
            status=EventStatus.DELIVERED,
            visibility=Visibility.INTERNAL,
        )
        if context is None:
            try:
                context = await self._build_context(room_id)
            except Exception:
                # Room may not exist yet (e.g. ON_ROOM_CREATED before bindings exist)
                with self._resource_lease():
                    room = await self._store.get_room(room_id)
                if room is None:
                    return
                context = RoomContext(room=room, bindings=[])
        if trigger is HookTrigger.ON_ROOM_CREATED:
            await self._hook_engine.run_async_hooks(room_id, trigger, event, context)
            return
        self._hook_engine._start_async_hooks(room_id, trigger, event, context)

    async def _persist_side_effects(
        self,
        room_id: str,
        tasks: list[Task],
        observations: list[Observation],
        event: RoomEvent,
        context: RoomContext,
        metadata_updates: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist tasks and observations, write the room metadata updates
        the channels returned (RFC §10.1 step 15), start the ON_TASK_CREATED
        hooks of the new tasks (§9.3)."""
        persisted_tasks: list[Task] = []
        for task in tasks:
            try:
                await self._store.add_task(task)
                persisted_tasks.append(task)
            except Exception:
                logger.exception(
                    "Failed to persist task %s",
                    task.id,
                    extra={"room_id": room_id, "task_id": task.id},
                )
        for observation in observations:
            try:
                await self._store.add_observation(observation)
            except Exception:
                logger.exception(
                    "Failed to persist observation %s",
                    observation.id,
                    extra={"room_id": room_id, "observation_id": observation.id},
                )
        await self._write_metadata_updates(room_id, metadata_updates)
        # Fire ON_TASK_CREATED hooks only for successfully persisted tasks
        for task in persisted_tasks:
            task_event = RoomEvent(
                room_id=room_id,
                type=EventType.TASK_CREATED,
                source=event.source,
                content=event.content,
                status=EventStatus.DELIVERED,
                visibility=Visibility.INTERNAL,
                metadata={"task_id": task.id, "task_title": task.title},
            )
            self._hook_engine._start_async_hooks(
                room_id, HookTrigger.ON_TASK_CREATED, task_event, context
            )

    async def _write_metadata_updates(
        self, room_id: str, updates: Mapping[str, Any] | None
    ) -> None:
        """Write *updates*, the room metadata updates a delivery set's
        channels returned, merged in the order they answered, on the room
        once, as a partial update under the room lock (RFC §10.1 step 15):
        a concurrent write of the room's metadata loses none of them. A
        failure is logged; the delivery set stands."""
        if not updates:
            return
        try:
            async with self._lock_manager.locked(room_id):
                await self._store.patch_room_metadata(room_id, dict(updates))
        except Exception:
            logger.exception(
                "Failed to write the room metadata updates %s",
                sorted(updates),
                extra={"room_id": room_id},
            )

    async def _emit_system_event(
        self,
        room_id: str,
        event_type: EventType,
        code: str,
        message: str,
        data: dict[str, Any] | None = None,
        *,
        records_transition: bool = False,
    ) -> None:
        """Emit a system event to the room timeline (internal/audit).

        Passes the RFC §5.1 status gate like any other write: a CLOSED or
        ARCHIVED room refuses lifecycle records too — the timeline of a closed
        room does not keep growing because a member was renamed.

        ``records_transition`` exempts the one legitimate exception: the event
        that *records* the closing transition itself, written after the status
        has already flipped. Nothing else may claim it.
        """
        if not records_transition and await self._room_refuses_writes(room_id):
            logger.debug(
                "System event %s refused: room %s no longer accepts writes",
                code,
                room_id,
                extra={"room_id": room_id},
            )
            return
        event = RoomEvent(
            room_id=room_id,
            type=event_type,
            source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
            content=SystemContent(body=message, code=code, data=data or {}),
            status=EventStatus.DELIVERED,
            visibility=Visibility.INTERNAL,
        )
        # Commit atomically (index + room counters, §14.3): a system event is a
        # DELIVERED timeline event and must be reflected in the counters too.
        # A lifecycle record is no activity (RFC §5.1): a mute does not
        # restart the room's inactivity timer.
        await self._persist_committed(room_id, event, activity=False)

    def _follow_discussion(self, room: Room) -> None:
        """Join, or leave, the discussion *room* holds as stored (RFC §19.7.5
        rule 16): the DiscussionMixin's, run for every context built."""

    async def _build_context(
        self,
        room_id: str,
        *,
        recent_limit: int | None = None,
        carrying: RoomContext | None = None,
        reads_history: bool = False,
        read: bool = False,
    ) -> RoomContext:
        """Build a RoomContext for the given room.

        ``recent_limit`` caps how many recent events are loaded into
        ``RoomContext.recent_events``. When omitted it is derived from the room's
        bound channels — the largest ``recent_events_window`` any of them
        declares, floored for hooks and capped at ``_RECENT_EVENTS_LIMIT``. A
        transport-only room (e.g. realtime voice) whose channels read no history
        loads just the floor instead of deserialising the whole ceiling per
        turn, and nothing at all when nothing will read it: the read is
        skipped, not merely emptied. The room's register of authors
        (``RoomContext.author_register``) is read whatever the history: an
        agent whose memory reads none still names the people taking part with
        their ranks, a realtime session labels what is broadcast to it, and
        read as empty the register would give a seated participant the label
        of the sender who took the name first. ``reads_history`` says the caller itself
        scans ``recent_events`` — ``regenerate_response`` looking for its
        trigger — so the floor applies whether or not a hook is registered.

        ``carrying`` hands over a context an earlier pass of the same message
        already built, so its history is not deserialised twice — see
        :meth:`_carried_history` for when it can be honoured. It must be a
        context of the same room built with the derived window (no explicit
        ``recent_limit``), which is what the inbound pipeline builds. Room,
        bindings and participants are always re-read: the room lock exists to
        make the status gate and the delivery plan read fresh state (RFC §10.1
        steps 6 and 12), and the history is the one part of a context the lock
        does not protect.

        Runs whole under the framework's resource lease: it is store reads and
        nothing else, and ``close()`` promises not to release the store while
        an operation it was given is still in flight — a context built for a
        hook announcement is one of the reads that promise covers. *read*
        says one of the kit's reads builds it: after the close, it reads the
        store while the store is open (RFC §12.10.4).

        "Store reads and nothing else" is also what lets the reads share one
        connection (``store.connection()``): the recent-events limit is derived
        from the bindings just read, so the two reads cannot be merged into one
        round trip, but they need not pay two checkouts either.

        Before the reads, the room's first activity in this process subscribes
        the framework to its StatusBus, so a status naming a room fires the
        room's ON_STATUS_POSTED hooks (RFC §19.8).
        """
        await self._ensure_status_bus_subscribed()
        with self._resource_lease(read=read):
            async with self._store.connection():
                room, bindings, participants = await self._store.load_room_context(room_id)
                if room is None:
                    raise RoomNotFoundError(f"Room {room_id} not found")
                self._follow_discussion(room)
                if recent_limit is None:
                    recent_limit = self._resolve_recent_events_limit(
                        bindings, reads_history=reads_history
                    )
                if recent_limit <= 0:
                    # Nothing bound reads history and no hook is registered:
                    # no query, no deserialisation.
                    recent: list[RoomEvent] = []
                else:
                    carried = self._carried_history(carrying, room, recent_limit)
                    if carried is None:
                        carried = await self._store.get_conversation(room_id, limit=recent_limit)
                    recent = carried
                register = await stored_register(self._store, room)
        return RoomContext(
            room=room,
            bindings=bindings,
            participants=participants,
            recent_events=recent,
            author_register=register,
        )

    def _carried_history(
        self, carrying: RoomContext | None, room: Room, recent_limit: int
    ) -> list[RoomEvent] | None:
        """The history *carrying* may hand over as-is, or ``None`` to read it.

        The window is carried only when it is provably identical to what a fresh
        read returns, so a hook and an AI channel see exactly the history they
        would otherwise be given: *room*'s counter has not moved since
        *carrying* was built, so nothing committed in between — a check that
        holds however long the gap is, which matters because a realtime tool
        call carries a context across its handler's whole execution, not across
        a pipeline step — and the timeline
        is append-only (RFC §8.1), so nothing else could have changed it. An
        edit or a delete is itself a committed event (RFC §10.3), which moves
        the counter and sends this back to a fresh read; the one thing the
        counter cannot see is a host calling ``update_event`` / hard
        ``delete_event`` in that window, and RFC §14.4 already calls a read
        event a snapshot. The floor is the other input the counter does not
        cover: it follows the hook registries, so a hook registered between a
        message's two passes reads the window the first pass built — empty if
        nothing read history then — and the next message is the first it sees
        whole.

        The window itself is the second question: a channel bound since
        *carrying* was built may declare a wider ``recent_events_window`` than
        the carried events can cover, and a wider window has to be read. This
        compares against the window *carrying*'s own bindings derive, which is
        why it must have been built without an explicit ``recent_limit``.

        The room check is not paranoia: handing one room's history to another
        room's hooks would leak a conversation, and a mismatch is a caller bug
        no read would catch.
        """
        if carrying is None:
            return None
        if carrying.room.id != room.id or carrying.room.event_count != room.event_count:
            return None
        if recent_limit > self._resolve_recent_events_limit(carrying.bindings):
            return None
        return carrying.recent_events[-recent_limit:]

    def _resolve_recent_events_limit(
        self, bindings: list[ChannelBinding], *, reads_history: bool = False
    ) -> int:
        """Events to load = the largest window any bound channel needs.

        Floored at ``_RECENT_EVENTS_FLOOR`` while something besides a
        declaring channel will read the tail — a hook, from either registry
        (the engine's index holds the regular ones, the framework its identity
        hooks), or the caller itself (``reads_history``) — and capped at
        ``_RECENT_EVENTS_LIMIT`` (the in-memory ceiling). A missing or
        unregistered channel contributes 0. With no history-reading channel
        and nobody to read the floor, the room loads none: on a transport-only
        room that read was a Postgres round trip and fifty pydantic models per
        message, for nobody (RMK-103). One hook anywhere in the process, on any
        trigger, re-arms the floor for every room it serves.
        """
        windows = [
            getattr(self._channels.get(b.channel_id), "recent_events_window", 0) for b in bindings
        ]
        largest = max(windows, default=0)
        read = (
            reads_history
            or self._hook_engine.has_hooks()
            or self._turn_hooks.has_hooks()
            or any(self._identity_hooks.values())
        )
        floor = _RECENT_EVENTS_FLOOR if read else 0
        return min(_RECENT_EVENTS_LIMIT, max(floor, largest))

    # -- Protocol trace --

    def _on_channel_trace(self, trace: object) -> None:
        """Forward a ProtocolTrace to ON_PROTOCOL_TRACE hooks for the room."""
        from roomkit.models.trace import ProtocolTrace

        if not isinstance(trace, ProtocolTrace):
            return

        room_id = trace.room_id
        if room_id is None and trace.session_id is not None:
            room_id = self._resolve_trace_room(trace)
        if room_id is None:
            return

        with contextlib.suppress(RuntimeError):
            task = asyncio.get_running_loop().create_task(self._fire_trace_hook(trace, room_id))
            task.add_done_callback(self._pending_hook_tasks.discard)
            self._pending_hook_tasks.add(task)

    def _resolve_trace_room(self, trace: object) -> str | None:
        """Try to resolve a room_id for a trace via the originating channel."""
        from roomkit.models.trace import ProtocolTrace

        if not isinstance(trace, ProtocolTrace):
            return None
        channel = self._channels.get(trace.channel_id)
        if channel is not None:
            result: str | None = channel.resolve_trace_room(trace.session_id)
            return result
        return None

    async def _fire_trace_hook(self, trace: object, room_id: str) -> None:
        """Fire ON_PROTOCOL_TRACE hooks for the given room.

        If the room does not exist yet (e.g. SIP INVITE trace fires
        before ``process_inbound`` creates the room), the trace is
        buffered and replayed when :meth:`_flush_pending_traces` is
        called from ``attach_channel``.
        """
        try:
            context = await self._build_context(room_id)
        except Exception:
            self._pending_traces.setdefault(room_id, []).append(trace)
            return
        await self._hook_engine.run_async_hooks(
            room_id,
            HookTrigger.ON_PROTOCOL_TRACE,
            trace,
            context,
            skip_event_filter=True,
        )

    async def _flush_pending_traces(self, room_id: str) -> None:
        """Replay buffered traces for a room that now exists."""
        traces = self._pending_traces.pop(room_id, None)
        if not traces:
            return
        try:
            context = await self._build_context(room_id)
        except Exception:
            return
        for trace in traces:
            await self._hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_PROTOCOL_TRACE,
                trace,
                context,
                skip_event_filter=True,
            )

    async def _hook_context(
        self, room_id: str, trigger: HookTrigger, *, carrying: RoomContext | None = None
    ) -> RoomContext | None:
        """The room's context for *trigger*'s hooks, or ``None`` when it will not build.

        A context is store reads (the room, its bindings, its participants, its
        history), and some triggers fire on every tool call of every round: a
        caller builds one only when ``has_hooks`` says a hook will read it.
        """
        try:
            return await self._build_context(room_id, carrying=carrying)
        except Exception:
            logger.warning(
                "Failed to build context for %s hook in room %s",
                trigger.name,
                room_id,
                exc_info=True,
            )
            return None

    # -- The turn triggers' wiring, on the kit's own executor (RFC §10.6.6) -----
    #
    # The wiring is the executor's (``roomkit.turns._wiring``); these reach the
    # kit's own executor's, for the doors that fire a turn trigger in the kit's
    # process outside a turn (a speech-to-speech session's tool calls, a
    # conference's, RFC §10.6.6) and for code that held these names.

    def _build_tool_usage_loader(self, channel_id: str) -> ToolUsageLoader:
        return self.turn_executor.wiring.build_tool_usage_loader(channel_id)

    def _build_tool_call_hook(self, channel_id: str) -> ToolCallCallback:
        return self.turn_executor.wiring.tools.build_tool_call_hook(channel_id)

    async def _judge_tool_call(
        self,
        event: ToolCallEvent,
        channel_id: str,
        *,
        carrying: RoomContext | None = None,
        claim: Callable[[], bool] | None = None,
    ) -> ToolCallVerdict | None:
        tools = self.turn_executor.wiring.tools
        return await tools.judge_tool_call(event, channel_id, carrying=carrying, claim=claim)

    async def _tell_tool_call(
        self, observed: ToolCallEvent, context: RoomContext | None, channel_id: str
    ) -> None:
        await self.turn_executor.wiring.tools.tell_tool_call(observed, context, channel_id)

    async def _run_tool_call_chain(
        self, event: ToolCallEvent, room_id: str, *, carrying: RoomContext | None = None
    ) -> tuple[SyncPipelineResult, RoomContext | None] | None:
        tools = self.turn_executor.wiring.tools
        return await tools.run_tool_call_chain(event, room_id, carrying=carrying)

    def _build_tool_report_hook(self, channel_id: str) -> ToolCallObserver:
        return self.turn_executor.wiring.tools.build_tool_report_hook(channel_id)

    async def _report_tool_call(
        self, event: ToolCallEvent, channel_id: str, *, claim: Callable[[], bool] | None = None
    ) -> None:
        await self.turn_executor.wiring.tools.report_tool_call(event, channel_id, claim=claim)

    def _build_tool_observer_hook(self, channel_id: str) -> ToolCallObserver:
        return self.turn_executor.wiring.tools.build_tool_observer_hook(channel_id)

    async def _observe_failed_tool_call(
        self, event: ToolCallEvent, channel_id: str, *, claim: Callable[[], bool] | None = None
    ) -> None:
        tools = self.turn_executor.wiring.tools
        await tools.observe_failed_tool_call(event, channel_id, claim=claim)

    def _build_before_tool_call_hook(self, channel_id: str) -> BeforeToolCallback:
        return self.turn_executor.wiring.tools.build_before_tool_call_hook(channel_id)

    async def _decide_before_tool_use(
        self, event: ToolCallEvent, channel_id: str, *, carrying: RoomContext | None = None
    ) -> tuple[BeforeToolDecision, RoomContext | None]:
        tools = self.turn_executor.wiring.tools
        return await tools.decide_before_tool_use(event, channel_id, carrying=carrying)

    def _build_thinking_hook(self, channel_id: str) -> ThinkingHook:
        return self.turn_executor.wiring.build_thinking_hook(channel_id)

    def _build_plan_updated_hook(self, channel_id: str) -> PlanUpdatedCallback:
        return self.turn_executor.wiring.build_plan_updated_hook(channel_id)

    def _build_on_user_input_required_hook(self, channel_id: str) -> Any:
        return self.turn_executor.wiring.build_on_user_input_required_hook(channel_id)

    def _emit_framework_event_soon(
        self,
        event_type: str,
        room_id: str | None = None,
        channel_id: str | None = None,
        event_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a framework event from synchronous code, best effort.

        Some mandated §8.2 events are raised by synchronous API
        (``register_channel``, ``unregister_channel``). Emission is scheduled
        on the running loop; called with no loop running — before the
        application starts, where no handler can have observed anything yet —
        it is a no-op rather than an error.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(
            self._emit_framework_event(
                event_type,
                room_id=room_id,
                channel_id=channel_id,
                event_id=event_id,
                data=data,
            )
        )
        task.add_done_callback(self._pending_hook_tasks.discard)
        self._pending_hook_tasks.add(task)

    async def _emit_framework_event(
        self,
        event_type: str,
        room_id: str | None = None,
        channel_id: str | None = None,
        event_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a framework event to handlers registered for *event_type*."""
        fw_event = FrameworkEvent(
            type=event_type,
            room_id=room_id,
            channel_id=channel_id,
            event_id=event_id,
            data=data or {},
        )
        for filter_type, handler in self._event_handlers:
            if filter_type == fw_event.type:
                try:
                    await handler(fw_event)
                except Exception:
                    logger.exception(
                        "Framework event handler failed",
                        extra={"event_type": fw_event.type, "room_id": fw_event.room_id},
                    )

    async def submit_feedback(
        self,
        room_id: str,
        rating: float,
        *,
        event_id: str | None = None,
        channel_id: str | None = None,
        comment: str = "",
        dimension: str = "overall",
        metadata: dict[str, Any] | None = None,
        organization_id: str | None = None,
    ) -> None:
        """Submit user feedback for a conversation or specific response.

        Stores feedback as an :class:`~roomkit.models.task.Observation`
        in the conversation store and fires the ``ON_FEEDBACK`` hook.

        Args:
            room_id: Room the feedback applies to.
            rating: Quality rating between 0.0 and 1.0.
            event_id: Optional specific event being rated.
            channel_id: Optional channel being rated.
            comment: Optional free-text comment.
            dimension: What is being rated (default "overall").
            metadata: Arbitrary metadata to attach.
            organization_id: The organization the caller acts for (RFC
                §17.2): another organization's room raises
                :class:`RoomNotFoundError`, nothing stored or fired.
        """
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        from roomkit.models.enums import HookTrigger
        from roomkit.models.task import Observation

        rating = max(0.0, min(1.0, rating))

        obs = Observation(
            id=uuid4().hex,
            room_id=room_id,
            channel_id=channel_id or "",
            content=f"[{dimension}] {rating:.2f}: {comment}"
            if comment
            else f"[{dimension}] {rating:.2f}",
            category=f"feedback:{dimension}",
            confidence=rating,
            metadata={
                "type": "feedback",
                "dimension": dimension,
                "rating": rating,
                "comment": comment,
                "event_id": event_id,
                **(metadata or {}),
            },
        )
        await self._store.add_observation(obs)

        # Fire ON_FEEDBACK hook
        try:
            context = await self._build_context(room_id)
        except Exception:
            return
        await self._hook_engine.run_async_hooks(
            room_id,
            HookTrigger.ON_FEEDBACK,
            obs,
            context,
            skip_event_filter=True,
        )

        await self._emit_framework_event(
            "feedback",
            room_id=room_id,
            channel_id=channel_id,
            event_id=event_id,
            data={"dimension": dimension, "rating": rating},
        )

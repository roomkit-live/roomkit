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
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core._participant_channels import channels_reached, warn_cross_channel
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.hooks import SyncPipelineResult
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import (
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
from roomkit.models.plan_event import PlanUpdatedEvent
from roomkit.models.store_filter import EventFilter
from roomkit.models.task import Observation, Task
from roomkit.models.thinking_event import ThinkingEvent
from roomkit.models.tool_call import (
    ToolCallEvent,
    ToolCallVerdict,
    observed_call_event,
    tool_call_chain_fold,
    withheld_call_event,
)
from roomkit.tools._outcome import kept_in_tool_memory
from roomkit.tools.context import turn_report_claim
from roomkit.tools.external import BeforeToolDecision, note_handler_report
from roomkit.tools.result import before_tool_use_detail, hook_errors_detail, tool_call_verdict

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


def _refuses_writes(room: Room | None) -> bool:
    """Whether *room* refuses new timeline events (RFC §5.1). A missing room
    refuses too: nothing can be appended to a room that is not there."""
    return room is None or room.status in _REFUSING_STATUSES


def _source_block_reason(binding: ChannelBinding | None) -> str | None:
    """Why an event from a source bound by *binding* is stored BLOCKED
    (RFC §7.5 rule 2): ``source_muted`` or ``source_read_only`` for a source
    that cannot write, ``None`` for one that can or has no binding."""
    if binding is None or binding.can_write:
        return None
    return "source_muted" if binding.muted else "source_read_only"


if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import (
        AfterToolRoundHook,
        BeforeGenerationHook,
        ThinkingHook,
        ToolUsageLoader,
    )
    from roomkit.channels._task_planner import PlanUpdatedCallback
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine, IdentityHookRegistration
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.room import Room
    from roomkit.models.tool_call import (
        AfterResponseCallback,
        ToolCallCallback,
        ToolCallObserver,
        ToolRoundEvent,
    )
    from roomkit.store.base import ConversationStore
    from roomkit.tools.external import BeforeToolCallback

logger = logging.getLogger("roomkit.framework")

FrameworkEventHandler = Callable[[FrameworkEvent], Coroutine[Any, Any, None]]
IdentityHookFn = Callable[
    [RoomEvent, RoomContext, IdentityResult],
    Coroutine[Any, Any, IdentityHookResult | None],
]


def _remembered_calls(events: list[RoomEvent]) -> list[dict[str, Any]]:
    """The calls a tool memory keeps, read off a channel's stored tool rows,
    in the order they were written."""
    # The arguments the model sent, from the call's start: the end carries the
    # ones that ran, which a BEFORE_TOOL_USE hook may have de-tokenised, and
    # the digest goes back into the model's prompt. An end pairs with the
    # start of the same call in the same turn (its response's correlation
    # id): a provider may reuse an id from one turn to the next, and two turns
    # of a room may run at once.
    requested: dict[tuple[str | None, str], dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    for ev in events:
        content = ev.content
        call = (ev.correlation_id, getattr(content, "tool_id", ""))
        if ev.type == EventType.TOOL_CALL_START:
            requested[call] = getattr(content, "arguments", {}) or {}
            continue
        name = getattr(content, "tool_name", "")
        if ev.type != EventType.TOOL_CALL_END or not name:
            continue
        arguments = requested.pop(call, {})
        # The live memory's rule, read off the outcome the row states: a
        # refusal or a call nothing served is not kept.
        if not kept_in_tool_memory(getattr(content, "outcome", None)):
            continue
        calls.append(
            {
                "name": name,
                "arguments": arguments,
                "result": getattr(content, "result", "") or "",
                "outcome": getattr(content, "outcome", None),
            }
        )
    return calls


def _before_tool_decision(name: str, hook_result: Any) -> BeforeToolDecision:
    """What the BEFORE_TOOL_USE chain decided about a call of *name*.

    Rewritten arguments that are not an object refuse the call. A hook that
    failed closed refused it too (RFC §9.3): its error rides the decision for
    the observers, never the model.
    """
    rewritten = hook_result.metadata.get("arguments")
    if "arguments" in hook_result.metadata and not isinstance(rewritten, dict):
        logger.error(
            "BEFORE_TOOL_USE hook returned non-object arguments for %s — denying tool call",
            name,
        )
        return BeforeToolDecision(allowed=False)
    blocked = not hook_result.allowed and not hook_result.failed_closed
    return BeforeToolDecision(
        allowed=hook_result.allowed,
        arguments=rewritten if isinstance(rewritten, dict) else None,
        detail=before_tool_use_detail(hook_result),
        reason=hook_result.reason if blocked else None,
    )


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
    _event_handlers: list[tuple[str, FrameworkEventHandler]]
    _identity_hooks: dict[HookTrigger, list[IdentityHookRegistration]]
    _pending_traces: dict[str, list[object]]  # room_id -> [ProtocolTrace, ...]
    _pending_hook_tasks: set[asyncio.Task[Any]]
    _persistence_policy: Any  # PersistencePolicy | None — set by RoomKit.__init__
    _resource_lease: Any  # RoomKit._resource_lease — the close()-ordering hold on the store
    _lanes: Any  # RoomLaneRegistry — set by RoomKit.__init__
    _room_close_epoch: int  # counted by _store_refusing_room — set by RoomKit.__init__

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

        The one writer of such a status: it counts the change, and a stream,
        which reads the status once and again only when that count moved,
        refuses its next row (RFC §5.1). A status changed elsewhere (another
        process, another kit sharing the store, the store directly) is seen
        by the next response.
        """
        stored = await self._store.update_room(room)
        self._room_close_epoch += 1
        return stored

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

    async def _persist_committed(self, room_id: str, event: RoomEvent) -> RoomEvent | None:
        """Atomically commit an event (index + insert + room counters, RFC
        §10.1 step 12 / §14.3) if the policy allows it, and account its
        index on the room's delivery cursor.

        Returns the committed event, or ``None`` if the policy excluded it.
        """
        if self._persistence_policy is not None and not self._persistence_policy.should_persist(
            event.type
        ):
            return None
        committed = await self._store.commit_event(room_id, event)
        await self._note_committed_index(room_id, committed.index)
        return committed

    async def _commit_indexed(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Commit an event unconditionally (policy-exempt) and account its
        index on the delivery cursor.

        For commits that must always be stored regardless of the persistence
        policy: BLOCKED events (part of the timeline, they consume an index —
        RFC §8.3), injected events, child-room traces.
        """
        committed = await self._store.commit_event(room_id, event)
        await self._note_committed_index(room_id, committed.index)
        return committed

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
    ) -> None:
        """Hand a turn-level failure to the ON_ERROR hooks.

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
        )
        await self._hook_engine.run_async_hooks(
            room_id, HookTrigger.ON_ERROR, error_event, context
        )

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
                result: IdentityHookResult | None = await hook_reg.fn(event, context, id_result)
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

        Idempotent: if a participant with the same ID already exists in the room,
        the existing record is returned without creating a duplicate — the
        channel it arrived on is recorded on that record (RFC §5.5), never used
        to fork a second one.
        """
        participant_id = event.source.participant_id or f"pending-{uuid4().hex[:8]}"
        channel_id = event.source.channel_id
        existing = await self._store.get_participant(room_id, participant_id)
        if existing is not None:
            warn_cross_channel(existing, channel_id, rehomed=False)
            channels = channels_reached(existing, channel_id)
            if channels is None:
                return existing
            return await self._store.update_participant(
                existing.model_copy(update={"connected_via": channels})
            )
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
    ) -> None:
        """Fire an async lifecycle hook with a synthetic system event."""
        event = RoomEvent(
            room_id=room_id,
            type=event_type,
            source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
            content=SystemContent(body=message, code=code, data=data or {}),
            status=EventStatus.DELIVERED,
            visibility=Visibility.INTERNAL,
        )
        try:
            context = await self._build_context(room_id)
        except Exception:
            # Room may not exist yet (e.g. ON_ROOM_CREATED before bindings exist)
            with self._resource_lease():
                room = await self._store.get_room(room_id)
            if room is None:
                return
            context = RoomContext(room=room, bindings=[])
        await self._hook_engine.run_async_hooks(room_id, trigger, event, context)

    async def _persist_side_effects(
        self,
        room_id: str,
        tasks: list[Task],
        observations: list[Observation],
        event: RoomEvent,
        context: RoomContext,
    ) -> None:
        """Persist tasks and observations, fire ON_TASK_CREATED hooks for new tasks."""
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
            await self._hook_engine.run_async_hooks(
                room_id, HookTrigger.ON_TASK_CREATED, task_event, context
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
        await self._persist_committed(room_id, event)

    async def _build_context(
        self,
        room_id: str,
        *,
        recent_limit: int | None = None,
        carrying: RoomContext | None = None,
        reads_history: bool = False,
    ) -> RoomContext:
        """Build a RoomContext for the given room.

        ``recent_limit`` caps how many recent events are loaded into
        ``RoomContext.recent_events``. When omitted it is derived from the room's
        bound channels — the largest ``recent_events_window`` any of them
        declares, floored for hooks and capped at ``_RECENT_EVENTS_LIMIT``. A
        transport-only room (e.g. realtime voice) whose channels read no history
        loads just the floor instead of deserialising the whole ceiling per
        turn, and nothing at all when nothing will read it: the read is
        skipped, not merely emptied. ``reads_history`` says the caller itself
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
        hook announcement is one of the reads that promise covers.

        "Store reads and nothing else" is also what lets the reads share one
        connection (``store.connection()``): the recent-events limit is derived
        from the bindings just read, so the two reads cannot be merged into one
        round trip, but they need not pay two checkouts either.
        """
        with self._resource_lease():
            async with self._store.connection():
                room, bindings, participants = await self._store.load_room_context(room_id)
                if room is None:
                    raise RoomNotFoundError(f"Room {room_id} not found")
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
        return RoomContext(
            room=room,
            bindings=bindings,
            participants=participants,
            recent_events=recent,
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
        read = reads_history or self._hook_engine.has_hooks() or any(self._identity_hooks.values())
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

    def _build_tool_usage_loader(self, channel_id: str) -> ToolUsageLoader:
        """Build the tool-usage hydration loader for AIChannel *channel_id*.

        Fetches the channel's most recent persisted tool rows in a room so its
        in-memory ToolUsageMemory (digest + re-reveal set) and its skill
        activations survive channel-object lifetimes — the store dies with the
        object (process restart, cache expiry) while conversations outlive it.
        Only the channel's own rows: another agent's calls in the room are not
        its memory, and their results may be withheld from it (RFC §7.5 rule
        8). Called at most once per room per process (the channel marks the
        room hydrated).
        """
        kit_ref = self
        # Enough to refill both windows (digest 8 + reveal 12) after the
        # infra-tool rows are filtered out by ToolUsageMemory.record().
        limit = 30

        async def _load(room_id: str) -> list[dict[str, Any]]:
            with kit_ref._resource_lease():
                events = await kit_ref._store.get_timeline(
                    room_id,
                    event_filter=EventFilter(
                        event_types=[EventType.TOOL_CALL_START, EventType.TOOL_CALL_END],
                        source_channel_id=channel_id,
                    ),
                    limit=limit * 2,  # a start and an end per call
                    newest_first=True,  # most recent N, returned ascending
                )
            return _remembered_calls(events)

        return _load

    def _build_tool_call_hook(self, channel_id: str) -> ToolCallCallback:
        """Build the ON_TOOL_CALL callback for a call a channel served.

        The returned callback runs ON_TOOL_CALL's SYNC hooks as a chain on the
        call's outcome (RFC §9.3, ``tool_call_chain_fold``) and returns their
        :class:`ToolCallVerdict` (see :func:`tool_call_verdict`). The ASYNC
        observers see the final outcome as the model reads it, or, after a
        BLOCK, the failure. Emits a ``tool_call`` framework event.

        A call nothing served arrives with no result: the chain is the hooks'
        chance to serve it, not a report on it. The observers then see the
        result a hook supplied, or the block; when none serves it, the channel
        reports the failure itself, once (RFC §9.3).
        """
        kit_ref = self

        async def _callback(
            event: ToolCallEvent, *, claim: Callable[[], bool] | None = None
        ) -> ToolCallVerdict | None:
            return await kit_ref._judge_tool_call(event, channel_id, claim=claim)

        return _callback

    async def _judge_tool_call(
        self,
        event: ToolCallEvent,
        channel_id: str,
        *,
        carrying: RoomContext | None = None,
        claim: Callable[[], bool] | None = None,
    ) -> ToolCallVerdict | None:
        """ON_TOOL_CALL's verdict on a call a channel served, its observers told.

        What :meth:`_build_tool_call_hook`'s callback runs, for every channel;
        *carrying* is a context the caller already built for this call, which
        spares the room history a second read. *claim* claims the call's one
        report between the chain and the observers: when it answers ``False``
        the outcome was already reported (a cancellation that landed while
        the chain ran) and nothing more is (RFC §12.4).
        """
        if not event.room_id:
            return None
        chain = await self._run_tool_call_chain(event, event.room_id, carrying=carrying)
        if chain is None:
            unreachable = self._unreachable_tool_call_verdict(event.room_id)
            # The framework event says what the model reads: the failure of
            # a hook that fails closed (RFC §9.3).
            reported = (
                withheld_call_event(event, str(unreachable.result)) if unreachable else event
            )
            await self._report_unreachable_tool_call(reported, channel_id, claim)
            return unreachable
        hook_result, context = chain
        verdict = tool_call_verdict(hook_result, event)
        read = verdict.result if verdict.result is not None else event.result
        if read is None:
            # Served by nothing: the channel reports the failure, once,
            # with its own framework event, and what any hook that
            # failed said for the observers.
            return replace(verdict, error_detail=hook_errors_detail(hook_result))
        if claim is not None and not claim():
            return verdict
        # Observers and the framework event see what the model reads: the
        # result, or the failure of a call a hook withheld (RFC §9.3).
        observed = observed_call_event(hook_result, event, read)
        if context is not None:
            await self._observe_tool_call(observed, context)
        await self._emit_tool_call_event(observed, channel_id)
        return verdict

    async def _run_tool_call_chain(
        self, event: ToolCallEvent, room_id: str, *, carrying: RoomContext | None = None
    ) -> tuple[SyncPipelineResult, RoomContext | None] | None:
        """ON_TOOL_CALL's SYNC chain on a served call, and the context it ran with.

        The one runner of the chain, for every channel (RFC §9.3). With no
        ON_TOOL_CALL hook registered the call stands as served, and no
        context is built. ``None`` when the context would not build.
        *carrying* is a context the caller already built for this call, which
        spares the room history a second read.
        """
        if not self._hook_engine.has_hooks(HookTrigger.ON_TOOL_CALL):
            return SyncPipelineResult(event=event), None
        context = await self._hook_context(room_id, HookTrigger.ON_TOOL_CALL, carrying=carrying)
        if context is None:
            return None
        hook_result = await self._hook_engine.run_sync_hooks(
            room_id,
            HookTrigger.ON_TOOL_CALL,
            event,
            context,
            skip_event_filter=True,
            fold=tool_call_chain_fold(event),
            fire_observers=False,
        )
        return hook_result, context

    def _build_tool_report_hook(self, channel_id: str) -> ToolCallObserver:
        """Build the ON_TOOL_CALL callback for a call an external handler ran.

        A report, by construction (RFC §9.3): the agent already read the
        provider's result, so nothing a hook returns reaches it, no rewrite is
        folded in, and a BLOCK withholds nothing. The observers see the
        provider's outcome and its own ``is_error``, a block included. The
        channel's turn that announced the call claims its one report once the
        observers hear it.
        """
        kit_ref = self

        async def _callback(event: ToolCallEvent) -> None:
            # The turn that announced the call claims its one report where
            # the observers hear it (RFC §9.3).
            turn_claim = turn_report_claim(event.tool_call_id, channel_id)

            def claim() -> bool:
                # Where the observers hear it: a handler that raises past
                # this point has made the call's report, and its door does
                # not make it again.
                note_handler_report(event.tool_call_id)
                return turn_claim() if turn_claim is not None else True

            await kit_ref._report_tool_call(event, channel_id, claim=claim)

        return _callback

    async def _report_tool_call(
        self,
        event: ToolCallEvent,
        channel_id: str,
        *,
        claim: Callable[[], bool] | None = None,
    ) -> None:
        """Fire ON_TOOL_CALL as a report on a call whose outcome the model already read.

        Every hook runs and nothing it returns is applied (RFC §9.3): the
        observers see *event* as it stands, a BLOCK included. For a call an
        external handler or a provider ran. A call the turn cut or a
        gate or handler refused never ran: its ASYNC observers alone hear of
        it, as of a local call cut or refused.
        *claim* claims the call's one report between the chain and the
        observers, as :meth:`_judge_tool_call` does: a report cut while the
        chain ran is still owed, and one the observers heard is made.
        """
        if not event.room_id:
            return
        if event.cancelled or event.refused:
            # Neither ran: the observers alone hear of it, on every door.
            await self._observe_failed_tool_call(event, channel_id, claim=claim)
            return
        context: RoomContext | None = None
        if self._hook_engine.has_hooks(HookTrigger.ON_TOOL_CALL):
            # The one runner of the chain, so each hook sees the call as the
            # previous one left it, whichever form its rewrite took; nothing it
            # returns reaches the agent.
            chain = await self._run_tool_call_chain(event, event.room_id)
            if chain is None:
                await self._report_unreachable_tool_call(event, channel_id, claim)
                return
            _, context = chain
        if claim is not None and not claim():
            return
        if context is not None:
            await self._observe_tool_call(event, context)
        await self._emit_tool_call_event(event, channel_id)

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

    async def _observe_tool_call(self, event: ToolCallEvent, context: RoomContext) -> None:
        """Fire ON_TOOL_CALL's ASYNC observers alone on *event*."""
        await self._hook_engine.run_observers(
            str(event.room_id),
            HookTrigger.ON_TOOL_CALL,
            event,
            context,
            skip_event_filter=True,
        )

    async def _emit_tool_call_event(self, event: ToolCallEvent, channel_id: str) -> None:
        """The ``tool_call`` framework event of one reported call, its failure
        and cancellation markers included, whichever path reported it."""
        data: dict[str, Any] = {
            "tool_name": event.name,
            "tool_call_id": event.tool_call_id,
            "channel_type": str(event.channel_type),
        }
        if event.is_error:
            data["is_error"] = True
        if event.cancelled:
            data["cancelled"] = True
        if event.refused:
            data["refused"] = True
        if event.refused_but_ran:
            data["refused_but_ran"] = True
        await self._emit_framework_event(
            "tool_call", room_id=event.room_id, channel_id=channel_id, data=data
        )

    async def _report_unreachable_tool_call(
        self, event: ToolCallEvent, channel_id: str, claim: Callable[[], bool] | None = None
    ) -> None:
        """Report a call whose ON_TOOL_CALL hooks could not run, as *event* stands.

        The observers need the context that failed: the ``tool_call``
        framework event is then the call's one report, on every channel,
        saying what the model reads (a fail-closed hook's failure included,
        which the judge hands it) (RFC §9.3). A call with no result yet is the
        channel's to report, as the failure it is.
        """
        if event.result is not None and (claim is None or claim()):
            await self._emit_tool_call_event(event, channel_id)

    def _unreachable_tool_call_verdict(self, room_id: str) -> ToolCallVerdict | None:
        """The verdict on a call whose ON_TOOL_CALL hooks could not run.

        None, keeping the result, unless a hook there fails closed: then the
        result is withheld as that hook's own failure would withhold it, since
        the redaction it exists for must not be skipped.
        """
        closed = self._hook_engine.fail_closed_hook(room_id, HookTrigger.ON_TOOL_CALL)
        if closed is None:
            return None
        reason = json.dumps({"error": f"hook_error:{closed}"})
        return ToolCallVerdict(result=reason, blocked=True)

    def _build_tool_observer_hook(self, channel_id: str) -> ToolCallObserver:
        """Build a ToolCallObserver closure for an AIChannel.

        The counterpart of :meth:`_build_tool_call_hook` for a call that failed
        or was refused: it runs the ASYNC observers of ON_TOOL_CALL and returns
        nothing. A refused call has no result for a hook to provide or correct,
        and must not reach a hook that would serve it.
        """
        kit_ref = self

        async def _callback(event: ToolCallEvent) -> None:
            claim = turn_report_claim(event.tool_call_id, channel_id)
            await kit_ref._observe_failed_tool_call(event, channel_id, claim=claim)

        return _callback

    async def _observe_failed_tool_call(
        self,
        event: ToolCallEvent,
        channel_id: str,
        *,
        claim: Callable[[], bool] | None = None,
    ) -> None:
        """Tell ON_TOOL_CALL's ASYNC observers a call failed, was refused or was
        cancelled, or one whose report a cut left unmade, and emit its
        ``tool_call`` framework event, both as *event* stands (RFC §9.3).

        The one report of such a call, for every channel: nothing that could
        serve the call reads it. *claim* claims that report once the context
        the observers read is built, as every report runner does: a report cut
        before then is still owed.
        """
        if not event.room_id:
            return
        context: RoomContext | None = None
        if self._hook_engine.has_hooks(HookTrigger.ON_TOOL_CALL):
            context = await self._hook_context(event.room_id, HookTrigger.ON_TOOL_CALL)
            if context is None:
                # The observers need the context that failed; the framework
                # event still reports the call once, as for a served call.
                await self._report_unreachable_tool_call(event, channel_id, claim)
                return
        if claim is not None and not claim():
            return
        if context is not None:
            await self._hook_engine.run_observers(
                event.room_id,
                HookTrigger.ON_TOOL_CALL,
                event,
                context,
                skip_event_filter=True,
            )
        await self._emit_tool_call_event(event, channel_id)

    def _build_thinking_hook(self, channel_id: str) -> ThinkingHook:
        """Build an ON_AI_THINKING callback closure for an AIChannel.

        RFC §9.2. The same reasoning also goes out as an ephemeral event for
        live UIs; the hook is what makes it observable to a host that runs no
        realtime backend.
        """
        kit_ref = self

        async def _callback(room_id: str, thinking: str, round_idx: int) -> None:
            if not room_id or not kit_ref._hook_engine.has_hooks(HookTrigger.ON_AI_THINKING):
                return
            context = await kit_ref._hook_context(room_id, HookTrigger.ON_AI_THINKING)
            if context is None:
                return

            await kit_ref._hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_AI_THINKING,
                ThinkingEvent(
                    room_id=room_id,
                    channel_id=channel_id,
                    thinking=thinking,
                    round_index=round_idx,
                ),
                context,
                skip_event_filter=True,
            )

        return _callback

    def _build_plan_updated_hook(self, channel_id: str) -> PlanUpdatedCallback:
        """Build an ON_PLAN_UPDATED callback closure for an AIChannel."""
        kit_ref = self

        async def _callback(room_id: str, tasks: list[dict[str, Any]]) -> None:
            if not room_id or not kit_ref._hook_engine.has_hooks(HookTrigger.ON_PLAN_UPDATED):
                return
            context = await kit_ref._hook_context(room_id, HookTrigger.ON_PLAN_UPDATED)
            if context is None:
                return

            await kit_ref._hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_PLAN_UPDATED,
                PlanUpdatedEvent(room_id=room_id, channel_id=channel_id, tasks=list(tasks)),
                context,
                skip_event_filter=True,
            )

        return _callback

    def _build_before_tool_call_hook(self, channel_id: str) -> BeforeToolCallback:
        """Build a BEFORE_TOOL_USE callback closure for an AIChannel.

        The returned callback runs BEFORE_TOOL_USE sync hooks against the
        framework's hook engine. If any hook blocks, the tool call is denied.
        A hook may also rewrite the call's arguments by returning them under
        ``metadata["arguments"]`` — the mirror of what ON_TOOL_CALL already
        does with ``metadata["result"]`` on the way out.
        """
        kit_ref = self

        async def _callback(event: ToolCallEvent) -> BeforeToolDecision:
            decision, _ = await kit_ref._decide_before_tool_use(event, channel_id)
            return decision

        return _callback

    async def _decide_before_tool_use(
        self, event: ToolCallEvent, channel_id: str, *, carrying: RoomContext | None = None
    ) -> tuple[BeforeToolDecision, RoomContext | None]:
        """What BEFORE_TOOL_USE decides about *event*, for every channel, and
        the context its hooks ran with, ``None`` when none was built.

        *carrying* is a context the caller already built for this call. A
        context that will not build denies the call: an authorization failure
        MUST NOT silently permit it (RFC §9.3).
        """
        if not event.room_id:
            return BeforeToolDecision(allowed=True), None  # Allow if no room context
        ran = await self._run_before_tool_use(event, event.room_id, carrying=carrying)
        if ran is None:
            logger.warning("BEFORE_TOOL_USE could not run: tool %s denied", event.name)
            return BeforeToolDecision(allowed=False), None
        hook_result, context = ran
        await self._emit_framework_event(
            "before_tool_use",
            room_id=event.room_id,
            channel_id=channel_id,
            data={
                "tool_name": event.name,
                "tool_call_id": event.tool_call_id,
                "allowed": hook_result.allowed,
                "reason": hook_result.reason,
            },
        )
        return _before_tool_decision(event.name, hook_result), context

    async def _run_before_tool_use(
        self, event: ToolCallEvent, room_id: str, *, carrying: RoomContext | None = None
    ) -> tuple[SyncPipelineResult, RoomContext | None] | None:
        """What the BEFORE_TOOL_USE hooks decide about *event*, and the context
        they ran with; ``None`` when the room's context could not be built.

        With no hook registered there is nothing to run and no context is built.
        """
        if not self._hook_engine.has_hooks(HookTrigger.BEFORE_TOOL_USE):
            return SyncPipelineResult(event=event), None
        context = await self._hook_context(room_id, HookTrigger.BEFORE_TOOL_USE, carrying=carrying)
        if context is None:
            return None
        hook_result = await self._hook_engine.run_sync_hooks(
            room_id,
            HookTrigger.BEFORE_TOOL_USE,
            event,
            context,
            skip_event_filter=True,
        )
        return hook_result, context

    def _build_on_user_input_required_hook(self, channel_id: str) -> Any:
        """Build an ON_USER_INPUT_REQUIRED callback closure.

        The returned callback runs ON_USER_INPUT_REQUIRED **sync** hooks
        against the framework's hook engine and emits a
        ``user_input_required`` framework event.

        Sync execution is what gives these hooks their order and their
        veto — a BLOCK rejects the request. It does not gate the request
        being answerable: ``HumanInputHandler`` arms the request first and
        runs this callback off the waiting path, so a human who answers
        while a slow notification is still in flight is answering a
        request that is already listening.
        """
        from roomkit.models.enums import HookTrigger
        from roomkit.models.pending_input import PendingInputEvent

        kit_ref = self

        async def _callback(event: PendingInputEvent) -> bool:
            if not event.room_id:
                return True  # Allow if no room context
            hook_result = SyncPipelineResult(event=event)
            if kit_ref._hook_engine.has_hooks(HookTrigger.ON_USER_INPUT_REQUIRED):
                context = await kit_ref._hook_context(
                    event.room_id, HookTrigger.ON_USER_INPUT_REQUIRED
                )
                if context is None:
                    return True  # Allow on error (fail-open)
                hook_result = await kit_ref._hook_engine.run_sync_hooks(
                    event.room_id,
                    HookTrigger.ON_USER_INPUT_REQUIRED,
                    event,
                    context,
                    skip_event_filter=True,
                )

            await kit_ref._emit_framework_event(
                "user_input_required",
                room_id=event.room_id,
                channel_id=event.channel_id or channel_id,
                data={
                    "pending_id": event.pending_id,
                    "tool_name": event.tool_name,
                    "tool_call_id": event.tool_call_id,
                    "allowed": hook_result.allowed,
                    "reason": hook_result.reason,
                },
            )

            return hook_result.allowed

        return _callback

    def _build_after_response_hook(self, channel_id: str) -> AfterResponseCallback:
        """Build an AfterResponseCallback closure for an AIChannel.

        The returned callback runs ON_AI_RESPONSE async hooks against
        the framework's hook engine and emits an ``ai_response`` framework
        event.  Observational only — does not block the response path.
        """
        from roomkit.models.enums import HookTrigger
        from roomkit.models.tool_call import AIResponseEvent

        kit_ref = self

        async def _callback(event: AIResponseEvent) -> None:
            if not event.room_id:
                return
            if kit_ref._hook_engine.has_hooks(HookTrigger.ON_AI_RESPONSE):
                context = await kit_ref._hook_context(event.room_id, HookTrigger.ON_AI_RESPONSE)
                if context is None:
                    return
                await kit_ref._hook_engine.run_async_hooks(
                    event.room_id,
                    HookTrigger.ON_AI_RESPONSE,
                    event,
                    context,
                    skip_event_filter=True,
                )

            await kit_ref._emit_framework_event(
                "ai_response",
                room_id=event.room_id,
                channel_id=channel_id,
                data={
                    "tool_calls_count": event.tool_calls_count,
                    "latency_ms": event.latency_ms,
                    "streaming": event.streaming,
                },
            )

        return _callback

    def _build_after_tool_round_hook(self) -> AfterToolRoundHook:
        """AFTER_TOOL_ROUND for an AIChannel: the room's SYNC hooks on a round
        its loop ran, between that round and the next (RFC §6.4).

        The hooks act on the event they receive, in place (withdrawals,
        messages); a MODIFY's returned event is not read. A BLOCK stops the
        hooks after it, as on any SYNC trigger, and changes nothing of the
        round, which has run.
        """
        kit_ref = self

        async def _callback(event: ToolRoundEvent) -> None:
            trigger = HookTrigger.AFTER_TOOL_ROUND
            if not event.room_id or not kit_ref._hook_engine.has_hooks(trigger):
                return
            context = await kit_ref._hook_context(event.room_id, trigger)
            if context is None:
                return
            await kit_ref._hook_engine.run_sync_hooks(
                event.room_id, trigger, event, context, skip_event_filter=True
            )

        return _callback

    def _build_before_generation_hook(self, channel_id: str) -> BeforeGenerationHook:
        """Build a BeforeGenerationCallback closure for an AIChannel.

        The returned callback runs BEFORE_AI_GENERATION sync hooks against
        the framework's hook engine.  Returns a :class:`SyncPipelineResult`
        that indicates whether generation should proceed or be blocked.
        """
        from roomkit.models.enums import HookTrigger
        from roomkit.models.tool_call import AIGenerationEvent

        kit_ref = self

        async def _callback(event: AIGenerationEvent) -> SyncPipelineResult:
            if not event.room_id:
                return SyncPipelineResult(allowed=True)
            sync_result = SyncPipelineResult(event=event)
            if kit_ref._hook_engine.has_hooks(HookTrigger.BEFORE_AI_GENERATION):
                context = await kit_ref._hook_context(
                    event.room_id, HookTrigger.BEFORE_AI_GENERATION
                )
                if context is None:
                    return SyncPipelineResult(allowed=True)
                sync_result = await kit_ref._hook_engine.run_sync_hooks(
                    event.room_id,
                    HookTrigger.BEFORE_AI_GENERATION,
                    event,
                    context,
                    skip_event_filter=True,
                )

            await kit_ref._emit_framework_event(
                "before_ai_generation",
                room_id=event.room_id,
                channel_id=channel_id,
                data={
                    "allowed": sync_result.allowed,
                    "blocked_by": sync_result.blocked_by,
                },
            )

            return sync_result

        return _callback

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
        """
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

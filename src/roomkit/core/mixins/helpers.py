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
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
    IdentificationStatus,
    Visibility,
)
from roomkit.models.event import EventSource, RoomEvent, SystemContent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.identity import Identity, IdentityHookResult, IdentityResult
from roomkit.models.participant import Participant
from roomkit.models.task import Observation, Task

_RECENT_EVENTS_LIMIT = 2_000
"""Hard ceiling on events kept in ``RoomContext.recent_events`` in memory."""

_RECENT_EVENTS_FLOOR = 50
"""Events loaded for a room whose channels declare no recent-history need
(transport-only rooms, e.g. realtime voice). Enough for hooks that glance at
recent context without paying the full-ceiling deserialisation cost per turn."""

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine, IdentityHookRegistration
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.room import Room
    from roomkit.store.base import ConversationStore

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

    @staticmethod
    def _record_channel_use(
        existing: Participant, channel_id: str, *, rehomes: bool = False
    ) -> list[str] | None:
        """The channels to store now that *channel_id* has asked for *existing*.

        A participant is one record per (room, id) — the same person reached by
        SMS and then by email is one participant, not two (RFC §4.4, §5.5). So
        a caller naming a channel the record was not created on gets that record
        back, which is legitimate and must not fail; what it must not be is
        silent. The record handed over still names another channel in
        ``channel_id``, and a caller that goes on to keep a lifecycle or a status
        on it is driving another channel's record without having been told —
        which is how leaving a conference came to erase a team-channel
        membership. Hence the warning naming both channels.

        Returns the new ``connected_via`` (primary channel included, order of
        first sight preserved), or ``None`` when the record already says
        everything there is to say. Bookkeeping, not presentation: no
        ``PARTICIPANT_UPDATED``, no ``ON_PARTICIPANT_UPDATED`` (RFC §5.5).

        *rehomes* says the caller is a deliberate join, which MAY move the
        primary channel to the one being joined through; a lazy get-or-create
        leaves it where it is. Either way the channel it replaces stays on the
        list.
        """
        channels = list(dict.fromkeys([existing.channel_id, *existing.connected_via, channel_id]))
        if channel_id != existing.channel_id:
            if rehomes:
                logger.warning(
                    "Participant %s of room %s is joining through channel %r; the primary "
                    "channel of their record was %r and becomes %r — one record still, and "
                    "%r stays in connected_via (RFC 5.5).",
                    existing.id,
                    existing.room_id,
                    channel_id,
                    existing.channel_id,
                    channel_id,
                    existing.channel_id,
                    extra={"room_id": existing.room_id},
                )
            else:
                logger.warning(
                    "Participant %s of room %s is recorded on channel %r; %r asked for them "
                    "and gets that record as it stands, primary channel included (RFC 5.5). "
                    "Reaching one participant on several channels is the point — but a "
                    "lifecycle or a status kept on this record from %r moves it for %r too.",
                    existing.id,
                    existing.room_id,
                    existing.channel_id,
                    channel_id,
                    channel_id,
                    existing.channel_id,
                    extra={"room_id": existing.room_id},
                )
        return channels if channels != existing.connected_via else None

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
            channels = self._record_channel_use(existing, channel_id)
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
            channels = self._record_channel_use(existing, channel_id)
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
    ) -> None:
        """Emit a system event to the room timeline (internal/audit)."""
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
    ) -> RoomContext:
        """Build a RoomContext for the given room.

        ``recent_limit`` caps how many recent events are loaded into
        ``RoomContext.recent_events``. When omitted it is derived from the room's
        bound channels — the largest ``recent_events_window`` any of them
        declares, floored for hooks and capped at ``_RECENT_EVENTS_LIMIT``. A
        transport-only room (e.g. realtime voice) whose channels read no history
        loads just the floor instead of deserialising the whole ceiling per turn.

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
                    recent_limit = self._resolve_recent_events_limit(bindings)
                recent = self._carried_history(carrying, room, recent_limit)
                if recent is None:
                    recent = await self._store.get_conversation(room_id, limit=recent_limit)
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
        *carrying* was built, so nothing committed in between — and the timeline
        is append-only (RFC §8.1), so nothing else could have changed it. An
        edit or a delete is itself a committed event (RFC §10.3), which moves
        the counter and sends this back to a fresh read; the one thing the
        counter cannot see is a host calling ``update_event`` / hard
        ``delete_event`` in that window, and RFC §14.4 already calls a read
        event a snapshot.

        The window itself is the second question: a channel bound since
        *carrying* was built may declare a wider ``recent_events_window`` than
        the carried events can cover, and a wider window has to be read. This
        compares against the window *carrying*'s own bindings derive, which is
        why it must have been built without an explicit ``recent_limit``.

        The room check is not paranoia: handing one room's history to another
        room's hooks would leak a conversation, and a mismatch is a caller bug
        no read would catch.
        """
        if carrying is None or recent_limit <= 0:
            return None
        if carrying.room.id != room.id or carrying.room.event_count != room.event_count:
            return None
        if recent_limit > self._resolve_recent_events_limit(carrying.bindings):
            return None
        return carrying.recent_events[-recent_limit:]

    def _resolve_recent_events_limit(self, bindings: list[ChannelBinding]) -> int:
        """Events to load = the largest window any bound channel needs.

        Floored at ``_RECENT_EVENTS_FLOOR`` (hooks) and capped at
        ``_RECENT_EVENTS_LIMIT`` (the in-memory ceiling). A missing/unregistered
        channel contributes 0, so a room with no history-reading channel —
        or no bindings at all — loads only the floor.
        """
        windows = [
            getattr(self._channels.get(b.channel_id), "recent_events_window", 0) for b in bindings
        ]
        largest = max(windows, default=0)
        return min(_RECENT_EVENTS_LIMIT, max(_RECENT_EVENTS_FLOOR, largest))

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

    def _build_tool_usage_loader(self) -> Any:
        """Build the tool-usage hydration loader for an AIChannel.

        Fetches a room's most recent persisted ``TOOL_CALL_END`` events so the
        channel's in-memory ToolUsageMemory (digest + re-reveal set) survives
        channel-object lifetimes — the store dies with the object (process
        restart, cache expiry) while conversations outlive it. Called at most
        once per room per process (the channel marks the room hydrated).
        """
        from roomkit.models.enums import EventType
        from roomkit.models.store_filter import EventFilter

        kit_ref = self
        # Enough to refill both windows (digest 8 + reveal 12) after the
        # infra-tool rows are filtered out by ToolUsageMemory.record().
        limit = 30

        async def _load(room_id: str) -> list[dict[str, Any]]:
            with kit_ref._resource_lease():
                events = await kit_ref._store.get_timeline(
                    room_id,
                    event_filter=EventFilter(event_types=[EventType.TOOL_CALL_END]),
                    limit=limit,
                    newest_first=True,  # most recent N, returned ascending
                )
            calls: list[dict[str, Any]] = []
            for ev in events:
                content = ev.content
                name = getattr(content, "tool_name", "")
                if not name:
                    continue
                calls.append(
                    {
                        "name": name,
                        "arguments": getattr(content, "arguments", {}) or {},
                        "result": getattr(content, "result", "") or "",
                    }
                )
            return calls

        return _load

    def _build_tool_call_hook(self, channel_id: str) -> Any:
        """Build a ToolCallCallback closure for an AIChannel.

        The returned callback runs ON_TOOL_CALL sync hooks against the
        framework's hook engine and emits a ``tool_call`` framework event.
        Returns the hook-provided result (str) or None to keep the original.
        """
        from roomkit.models.enums import HookTrigger
        from roomkit.models.tool_call import ToolCallEvent

        kit_ref = self

        async def _callback(event: ToolCallEvent) -> str | None:
            if not event.room_id:
                return None
            try:
                context = await kit_ref._build_context(event.room_id)
            except Exception:
                logger.warning(
                    "Failed to build context for ON_TOOL_CALL hook in room %s",
                    event.room_id,
                    exc_info=True,
                )
                return None

            hook_result = await kit_ref._hook_engine.run_sync_hooks(
                event.room_id,
                HookTrigger.ON_TOOL_CALL,
                event,
                context,
                skip_event_filter=True,
            )

            await kit_ref._emit_framework_event(
                "tool_call",
                room_id=event.room_id,
                channel_id=channel_id,
                data={
                    "tool_name": event.name,
                    "tool_call_id": event.tool_call_id,
                    "channel_type": str(event.channel_type),
                },
            )

            if not hook_result.allowed:
                import json

                return json.dumps({"error": hook_result.reason or "blocked"})
            return hook_result.metadata.get("result")

        return _callback

    def _build_before_tool_call_hook(self, channel_id: str) -> Any:
        """Build a BEFORE_TOOL_USE callback closure for an AIChannel.

        The returned callback runs BEFORE_TOOL_USE sync hooks against the
        framework's hook engine. If any hook blocks, the tool call is denied.
        Returns True if the tool call is allowed, False if denied.
        """
        from roomkit.models.enums import HookTrigger
        from roomkit.models.tool_call import ToolCallEvent

        kit_ref = self

        async def _callback(event: ToolCallEvent) -> bool:
            if not event.room_id:
                return True  # Allow if no room context
            try:
                context = await kit_ref._build_context(event.room_id)
            except Exception:
                logger.warning(
                    "Failed to build context for BEFORE_TOOL_USE hook in room %s "
                    "— denying tool call (fail-closed)",
                    event.room_id,
                    exc_info=True,
                )
                # Fail-closed: an authorization failure MUST NOT silently permit
                # the tool call. Denying is the safe default.
                return False

            hook_result = await kit_ref._hook_engine.run_sync_hooks(
                event.room_id,
                HookTrigger.BEFORE_TOOL_USE,
                event,
                context,
                skip_event_filter=True,
            )

            await kit_ref._emit_framework_event(
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

            return hook_result.allowed

        return _callback

    def _build_on_user_input_required_hook(self, channel_id: str) -> Any:
        """Build an ON_USER_INPUT_REQUIRED callback closure.

        The returned callback runs ON_USER_INPUT_REQUIRED **sync** hooks
        against the framework's hook engine and emits a
        ``user_input_required`` framework event.

        Sync execution ensures the notification (e.g. WebSocket broadcast)
        completes before :meth:`HumanInputHandler.wait` starts blocking —
        avoiding a race where the user never sees the question.
        """
        from roomkit.models.enums import HookTrigger
        from roomkit.models.pending_input import PendingInputEvent

        kit_ref = self

        async def _callback(event: PendingInputEvent) -> bool:
            if not event.room_id:
                return True  # Allow if no room context
            try:
                context = await kit_ref._build_context(event.room_id)
            except Exception:
                logger.warning(
                    "Failed to build context for ON_USER_INPUT_REQUIRED hook in room %s",
                    event.room_id,
                    exc_info=True,
                )
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
                channel_id=channel_id,
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

    def _build_after_response_hook(self, channel_id: str) -> Any:
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
            try:
                context = await kit_ref._build_context(event.room_id)
            except Exception:
                logger.warning(
                    "Failed to build context for ON_AI_RESPONSE hook in room %s",
                    event.room_id,
                    exc_info=True,
                )
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

    def _build_before_generation_hook(self, channel_id: str) -> Any:
        """Build a BeforeGenerationCallback closure for an AIChannel.

        The returned callback runs BEFORE_AI_GENERATION sync hooks against
        the framework's hook engine.  Returns a :class:`SyncPipelineResult`
        that indicates whether generation should proceed or be blocked.
        """
        from roomkit.core.hooks import SyncPipelineResult
        from roomkit.models.enums import HookTrigger
        from roomkit.models.tool_call import AIGenerationEvent

        kit_ref = self

        async def _callback(event: AIGenerationEvent) -> SyncPipelineResult:
            if not event.room_id:
                return SyncPipelineResult(allowed=True)
            try:
                context = await kit_ref._build_context(event.room_id)
            except Exception:
                logger.warning(
                    "Failed to build context for BEFORE_AI_GENERATION hook in room %s",
                    event.room_id,
                    exc_info=True,
                )
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

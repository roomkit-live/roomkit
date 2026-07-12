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

    # -- Persistence helpers (policy-aware) --

    async def _persist_event(self, event: RoomEvent) -> RoomEvent | None:
        """Persist an event if the persistence policy allows it.

        Returns the stored event, or ``None`` if the policy excluded it.
        """
        if self._persistence_policy is not None and not self._persistence_policy.should_persist(
            event.type
        ):
            return None
        return await self._store.add_event(event)

    async def _persist_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent | None:
        """Atomically index and persist an event if the policy allows it.

        Returns the stored event, or ``None`` if the policy excluded it.
        """
        if self._persistence_policy is not None and not self._persistence_policy.should_persist(
            event.type
        ):
            return None
        return await self._store.add_event_auto_index(room_id, event)

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
        the existing record is returned without creating a duplicate.
        """
        participant_id = event.source.participant_id or f"pending-{uuid4().hex[:8]}"
        existing = await self._store.get_participant(room_id, participant_id)
        if existing is not None:
            return existing
        candidate_ids = [c.id for c in id_result.candidates] if id_result.candidates else None
        participant = Participant(
            id=participant_id,
            room_id=room_id,
            channel_id=event.source.channel_id,
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

        Idempotent: if a participant with the identity's ID already exists in the room,
        the existing record is returned without creating a duplicate.
        """
        existing = await self._store.get_participant(room_id, identity.id)
        if existing is not None:
            # Update identification status if it was pending
            if existing.identification != IdentificationStatus.IDENTIFIED:
                existing = existing.model_copy(
                    update={
                        "identification": IdentificationStatus.IDENTIFIED,
                        "identity_id": identity.id,
                        "display_name": identity.display_name or existing.display_name,
                    }
                )
                existing = await self._store.update_participant(existing)
            return existing

        participant = Participant(
            id=identity.id,
            room_id=room_id,
            channel_id=event.source.channel_id,
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
        await self._persist_event_auto_index(room_id, event)

    async def _build_context(
        self, room_id: str, *, recent_limit: int | None = None
    ) -> RoomContext:
        """Build a RoomContext for the given room.

        ``recent_limit`` caps how many recent events are loaded into
        ``RoomContext.recent_events``. When omitted it is derived from the room's
        bound channels — the largest ``recent_events_window`` any of them
        declares, floored for hooks and capped at ``_RECENT_EVENTS_LIMIT``. A
        transport-only room (e.g. realtime voice) whose channels read no history
        loads just the floor instead of deserialising the whole ceiling per turn.
        """
        room = await self._store.get_room(room_id)
        if room is None:
            raise RoomNotFoundError(f"Room {room_id} not found")
        bindings = await self._store.list_bindings(room_id)
        participants = await self._store.list_participants(room_id)
        if recent_limit is None:
            recent_limit = self._resolve_recent_events_limit(bindings)
        recent = await self._store.get_conversation(room_id, limit=recent_limit)
        return RoomContext(
            room=room,
            bindings=bindings,
            participants=participants,
            recent_events=recent,
        )

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

"""HelpersMixin — internal helpers shared across framework mixins."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
    IdentificationStatus,
)
from roomkit.models.event import EventSource, RoomEvent, SystemContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.identity import Identity, IdentityHookResult, IdentityResult
from roomkit.models.participant import Participant
from roomkit.models.task import Observation, Task

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine, IdentityHookRegistration
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")

FrameworkEventHandler = Callable[[FrameworkEvent], Coroutine[Any, Any, None]]
IdentityHookFn = Callable[
    [RoomEvent, RoomContext, IdentityResult],
    Coroutine[Any, Any, IdentityHookResult | None],
]


class HelpersMixin:
    """Internal helpers used by other framework mixins."""

    _store: ConversationStore
    _hook_engine: HookEngine
    _event_handlers: list[tuple[str, FrameworkEventHandler]]
    _identity_hooks: dict[HookTrigger, list[IdentityHookRegistration]]
    _pending_traces: dict[str, list[object]]  # room_id -> [ProtocolTrace, ...]

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
            visibility="internal",
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
                visibility="internal",
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
        count = await self._store.get_event_count(room_id)
        event = RoomEvent(
            room_id=room_id,
            type=event_type,
            source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
            content=SystemContent(body=message, code=code, data=data or {}),
            status=EventStatus.DELIVERED,
            visibility="internal",
            index=count,
        )
        await self._store.add_event(event)

    async def _build_context(self, room_id: str) -> RoomContext:
        """Build a RoomContext for the given room."""
        room = await self._store.get_room(room_id)
        if room is None:
            from roomkit.core.framework import RoomNotFoundError

            raise RoomNotFoundError(f"Room {room_id} not found")
        bindings = await self._store.list_bindings(room_id)
        participants = await self._store.list_participants(room_id)
        recent = await self._store.list_events(room_id, offset=0, limit=50)
        return RoomContext(
            room=room,
            bindings=bindings,
            participants=participants,
            recent_events=recent,
        )

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
            asyncio.get_running_loop().create_task(self._fire_trace_hook(trace, room_id))

    def _resolve_trace_room(self, trace: object) -> str | None:
        """Try to resolve a room_id for a trace via the originating channel."""
        from roomkit.models.trace import ProtocolTrace

        if not isinstance(trace, ProtocolTrace):
            return None
        channel = self._channels.get(trace.channel_id)  # type: ignore[attr-defined]
        if channel is not None:
            return channel.resolve_trace_room(trace.session_id)
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

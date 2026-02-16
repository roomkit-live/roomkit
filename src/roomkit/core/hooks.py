"""Hook engine for sync and async hook pipelines."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, cast

from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelDirection, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.task import Observation, Task

logger = logging.getLogger("roomkit.hooks")

SyncHookFn = Callable[[RoomEvent, RoomContext], Coroutine[Any, Any, HookResult]]
AsyncHookFn = Callable[[RoomEvent, RoomContext], Coroutine[Any, Any, None]]


@dataclass
class HookRegistration:
    """A registered hook function.

    Attributes:
        trigger: When the hook fires (BEFORE_BROADCAST, AFTER_BROADCAST, etc.)
        execution: SYNC (can block/modify) or ASYNC (fire-and-forget)
        fn: The hook function
        priority: Lower numbers run first (default: 0)
        name: Optional name for logging and removal
        timeout: Max execution time in seconds (default: 30.0)
        channel_types: Only run for events from these channel types (None = all)
        channel_ids: Only run for events from these channel IDs (None = all)
        directions: Only run for events with these directions (None = all)
    """

    trigger: HookTrigger
    execution: HookExecution
    fn: SyncHookFn | AsyncHookFn
    priority: int = 0
    name: str = ""
    timeout: float = 30.0
    # Filters (None = match all)
    channel_types: set[ChannelType] | None = None
    channel_ids: set[str] | None = None
    directions: set[ChannelDirection] | None = None


@dataclass
class IdentityHookRegistration:
    """A registered identity hook function.

    Attributes:
        trigger: When the hook fires (ON_IDENTITY_AMBIGUOUS, ON_IDENTITY_UNKNOWN)
        fn: The hook function
        channel_types: Only run for events from these channel types (None = all)
        channel_ids: Only run for events from these channel IDs (None = all)
        directions: Only run for events with these directions (None = all)
    """

    trigger: HookTrigger
    fn: Any  # IdentityHookFn - using Any to avoid circular import
    channel_types: set[ChannelType] | None = None
    channel_ids: set[str] | None = None
    directions: set[ChannelDirection] | None = None


@dataclass
class SyncPipelineResult:
    """Result of running the sync hook pipeline."""

    allowed: bool = True
    event: RoomEvent | None = None
    reason: str | None = None
    blocked_by: str | None = None
    injected_events: list[InjectedEvent] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    hook_errors: list[dict[str, str]] = field(default_factory=list)


class HookEngine:
    """Manages global and per-room hook registration and execution."""

    def __init__(self) -> None:
        self._global_hooks: list[HookRegistration] = []
        self._room_hooks: dict[str, list[HookRegistration]] = {}
        self._telemetry: Any = None  # Set by RoomKit after init
        self._suppressed_triggers: set[str] = {
            "on_input_audio_level",
            "on_output_audio_level",
            "on_vad_audio_level",
        }

    def register(self, hook: HookRegistration) -> None:
        """Register a global hook."""
        self._global_hooks.append(hook)

    def add_room_hook(self, room_id: str, hook: HookRegistration) -> None:
        """Register a hook for a specific room."""
        self._room_hooks.setdefault(room_id, []).append(hook)

    def remove_room_hook(self, room_id: str, name: str) -> bool:
        """Remove a room hook by name."""
        hooks = self._room_hooks.get(room_id, [])
        for i, h in enumerate(hooks):
            if h.name == name:
                hooks.pop(i)
                return True
        return False

    def _hook_matches_event(self, hook: HookRegistration, event: RoomEvent) -> bool:
        """Check if a hook's filters match the given event."""
        source = event.source

        # All filters must pass (None means "match all")
        type_ok = hook.channel_types is None or source.channel_type in hook.channel_types
        id_ok = hook.channel_ids is None or source.channel_id in hook.channel_ids
        dir_ok = hook.directions is None or source.direction in hook.directions

        return type_ok and id_ok and dir_ok

    def _get_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        execution: HookExecution | None,
        event: RoomEvent | None = None,
    ) -> list[HookRegistration]:
        """Get merged global + room hooks filtered and sorted by priority.

        Args:
            room_id: The room ID to get hooks for
            trigger: The hook trigger to filter by
            execution: The execution mode to filter by, or ``None`` to
                match all execution modes.
            event: Optional event to filter hooks by channel_type/id/direction
        """
        all_hooks = [
            h
            for h in self._global_hooks
            if h.trigger == trigger and (execution is None or h.execution == execution)
        ]
        room_hooks = [
            h
            for h in self._room_hooks.get(room_id, [])
            if h.trigger == trigger and (execution is None or h.execution == execution)
        ]
        all_hooks.extend(room_hooks)

        # Apply event-based filters if event is provided
        if event is not None:
            all_hooks = [h for h in all_hooks if self._hook_matches_event(h, event)]

        all_hooks.sort(key=lambda h: h.priority)
        return all_hooks

    async def run_sync_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        skip_event_filter: bool = False,
    ) -> SyncPipelineResult:
        """Run sync hooks sequentially. Stops on block, passes modified events.

        Args:
            room_id: The room ID to run hooks for.
            trigger: The hook trigger type.
            event: The event to pass to hooks. For voice hooks, this may be
                a VoiceSession or str instead of RoomEvent.
            context: The room context.
            skip_event_filter: If True, skip channel-based event filtering.
                Use this for voice hooks where event is not a RoomEvent.
        """
        filter_event = None if skip_event_filter else event
        hooks = self._get_hooks(room_id, trigger, HookExecution.SYNC, event=filter_event)
        result = SyncPipelineResult(event=event)

        for hook in hooks:
            span_id = None
            should_trace = (
                self._telemetry is not None and str(trigger) not in self._suppressed_triggers
            )
            if should_trace:
                from roomkit.telemetry.base import Attr, SpanKind
                from roomkit.telemetry.context import get_current_span

                span_id = self._telemetry.start_span(
                    SpanKind.HOOK_SYNC,
                    f"hook.sync.{hook.name or 'unnamed'}",
                    parent_id=get_current_span(),
                    room_id=room_id,
                    attributes={
                        Attr.HOOK_NAME: hook.name or "unnamed",
                        Attr.HOOK_TRIGGER: str(trigger),
                    },
                )
            try:
                current_event = result.event or event
                fn = cast(SyncHookFn, hook.fn)
                hook_result: HookResult = await asyncio.wait_for(
                    fn(current_event, context), timeout=hook.timeout
                )
            except TimeoutError:
                logger.warning(
                    "Sync hook %s timed out after %.1fs",
                    hook.name,
                    hook.timeout,
                    extra={"room_id": room_id},
                )
                result.hook_errors.append(
                    {"hook": hook.name, "error": f"timeout ({hook.timeout}s)"}
                )
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message="timeout")
                continue
            except Exception as exc:
                logger.exception("Sync hook %s failed", hook.name, extra={"room_id": room_id})
                result.hook_errors.append({"hook": hook.name, "error": str(exc)})
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message=str(exc))
                continue

            if not isinstance(hook_result, HookResult):
                logger.error(
                    "Sync hook %s returned %s instead of HookResult — skipping",
                    hook.name,
                    type(hook_result).__name__,
                    extra={"room_id": room_id},
                )
                result.hook_errors.append(
                    {
                        "hook": hook.name,
                        "error": f"expected HookResult, got {type(hook_result).__name__}",
                    }
                )
                if span_id is not None:
                    self._telemetry.end_span(
                        span_id, status="error", error_message="invalid return type"
                    )
                continue

            if span_id is not None:
                self._telemetry.end_span(
                    span_id,
                    attributes={Attr.HOOK_RESULT: hook_result.action},
                )

            result.injected_events.extend(hook_result.injected_events)
            result.tasks.extend(hook_result.tasks)
            result.observations.extend(hook_result.observations)

            if hook_result.action == "block":
                result.allowed = False
                result.reason = hook_result.reason
                result.blocked_by = hook.name
                return result

            if hook_result.action == "modify" and hook_result.event is not None:
                result.event = hook_result.event

        return result

    async def run_async_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        skip_event_filter: bool = False,
    ) -> None:
        """Run async hooks concurrently. Errors are logged, never raised.

        Finds hooks regardless of their declared execution mode so that
        hooks registered with the default ``SYNC`` execution still fire
        for triggers that are only invoked asynchronously (e.g.
        ``AFTER_BROADCAST``, lifecycle hooks, voice hooks).

        Args:
            room_id: The room ID to run hooks for.
            trigger: The hook trigger type.
            event: The event to pass to hooks. For voice hooks, this may be
                a VoiceSession or str instead of RoomEvent.
            context: The room context.
            skip_event_filter: If True, skip channel-based event filtering.
                Use this for voice hooks where event is not a RoomEvent.
        """
        filter_event = None if skip_event_filter else event
        hooks = self._get_hooks(room_id, trigger, None, event=filter_event)
        if not hooks:
            return

        async def _run_one(hook: HookRegistration) -> None:
            span_id = None
            should_trace = (
                self._telemetry is not None and str(trigger) not in self._suppressed_triggers
            )
            if should_trace:
                from roomkit.telemetry.base import Attr, SpanKind
                from roomkit.telemetry.context import get_current_span

                span_id = self._telemetry.start_span(
                    SpanKind.HOOK_ASYNC,
                    f"hook.async.{hook.name or 'unnamed'}",
                    parent_id=get_current_span(),
                    room_id=room_id,
                    attributes={
                        Attr.HOOK_NAME: hook.name or "unnamed",
                        Attr.HOOK_TRIGGER: str(trigger),
                    },
                )
            try:
                await asyncio.wait_for(
                    hook.fn(event, context),
                    timeout=hook.timeout,
                )
                if span_id is not None:
                    self._telemetry.end_span(span_id)
            except TimeoutError:
                logger.warning(
                    "Async hook %s timed out after %.1fs",
                    hook.name,
                    hook.timeout,
                    extra={"room_id": room_id},
                )
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message="timeout")
            except Exception:
                logger.exception(
                    "Async hook %s failed",
                    hook.name,
                    extra={"room_id": room_id},
                )
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message="failed")

        await asyncio.gather(*[_run_one(hook) for hook in hooks], return_exceptions=True)

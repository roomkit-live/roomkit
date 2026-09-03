"""Hook engine for sync and async hook pipelines."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

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
    event: Any = None
    reason: str | None = None
    blocked_by: str | None = None
    injected_events: list[InjectedEvent] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    hook_errors: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class HookEngine:
    """Manages global and per-room hook registration and execution."""

    def __init__(self) -> None:
        self._global_hooks: list[HookRegistration] = []
        self._room_hooks: dict[str, list[HookRegistration]] = {}
        self._trigger_index: set[HookTrigger] = set()
        self._telemetry: Any = None  # Set by RoomKit after init
        # Set by RoomKit after init — RFC §8.2 mandates a ``hook_timeout``
        # framework event, which the engine alone knows how to raise.
        self._framework_emitter: Any = None
        self._suppressed_triggers: set[str] = {
            "on_input_audio_level",
            "on_output_audio_level",
            "on_vad_audio_level",
        }

    async def _emit_hook_timeout(
        self, room_id: str, hook: HookRegistration, trigger: HookTrigger
    ) -> None:
        """Raise the RFC §8.2 ``hook_timeout`` framework event.

        A timeout is its own observable condition, distinct from
        ``hook_error``: an operator reading the event stream can tell a hook
        that raised from one that never came back.
        """
        if self._framework_emitter is None:
            return
        await self._framework_emitter(
            "hook_timeout",
            room_id=room_id,
            data={
                "hook_name": hook.name,
                "trigger": str(trigger),
                "timeout": hook.timeout,
            },
        )

    def register(self, hook: HookRegistration) -> None:
        """Register a global hook."""
        self._global_hooks.append(hook)
        self._trigger_index.add(hook.trigger)

    def add_room_hook(self, room_id: str, hook: HookRegistration) -> None:
        """Register a hook for a specific room."""
        self._room_hooks.setdefault(room_id, []).append(hook)
        self._trigger_index.add(hook.trigger)

    def remove_global_hook(self, name: str) -> bool:
        """Remove a global hook by name."""
        for i, h in enumerate(self._global_hooks):
            if h.name == name:
                self._global_hooks.pop(i)
                self._rebuild_trigger_index()
                return True
        return False

    def remove_room_hook(self, room_id: str, name: str) -> bool:
        """Remove a room hook by name."""
        hooks = self._room_hooks.get(room_id, [])
        for i, h in enumerate(hooks):
            if h.name == name:
                hooks.pop(i)
                self._rebuild_trigger_index()
                return True
        return False

    def has_hooks(self, trigger: HookTrigger | None = None) -> bool:
        """Whether a hook is registered — for ``trigger``, or for any trigger at all.

        O(1) either way: a set lookup, or the set's emptiness. The framework
        skips the work only a hook would consume when nothing is listening —
        the context built for ``BEFORE_DELIVER``, the room history loaded for
        the inbound pipeline — so it runs a handful of times per message,
        never per hook. Global and room hooks alike are in the index; identity
        hooks live in the framework's own registry, and a caller that needs
        both asks both.
        """
        if trigger is None:
            return bool(self._trigger_index)
        return trigger in self._trigger_index

    def _rebuild_trigger_index(self) -> None:
        """Rebuild the trigger index from all registered hooks."""
        self._trigger_index = {h.trigger for h in self._global_hooks}
        for hooks in self._room_hooks.values():
            self._trigger_index.update(h.trigger for h in hooks)

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

    #: Triggers whose payload is content that a hook may be there to withhold —
    #: redacting a transcript, holding back speech. On those, a hook that raises
    #: blocks rather than letting the original payload through: logging the
    #: error and carrying on would publish exactly what the hook existed to
    #: suppress. Everywhere else a failing hook stays non-fatal, so a broken
    #: hook cannot take a room down.
    FAIL_CLOSED_TRIGGERS: ClassVar[frozenset[HookTrigger]] = frozenset(
        {HookTrigger.BEFORE_TTS, HookTrigger.ON_TRANSCRIPTION}
    )

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
                # ``is not None`` rather than truthiness: redacting to an empty
                # string is a modification, and treating it as absent would hand
                # the next hook the original secret.
                current_event = result.event if result.event is not None else event
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
                await self._emit_hook_timeout(room_id, hook, trigger)
                result.hook_errors.append(
                    {"hook": hook.name, "error": f"timeout ({hook.timeout}s)"}
                )
                if trigger in self.FAIL_CLOSED_TRIGGERS:
                    result.allowed = False
                    result.reason = f"hook {hook.name} timed out after {hook.timeout}s"
                    return result
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message="timeout")
                continue
            except Exception as exc:
                logger.exception("Sync hook %s failed", hook.name, extra={"room_id": room_id})
                result.hook_errors.append({"hook": hook.name, "error": str(exc)})
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message=str(exc))
                if trigger in self.FAIL_CLOSED_TRIGGERS:
                    result.allowed = False
                    result.reason = f"hook {hook.name} failed: {exc}"
                    return result
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
                if trigger in self.FAIL_CLOSED_TRIGGERS:
                    result.allowed = False
                    result.reason = (
                        f"hook {hook.name} returned {type(hook_result).__name__} "
                        "instead of HookResult"
                    )
                    return result
                continue

            if span_id is not None:
                self._telemetry.end_span(
                    span_id,
                    attributes={Attr.HOOK_RESULT: hook_result.action},
                )

            result.injected_events.extend(hook_result.injected_events)
            result.tasks.extend(hook_result.tasks)
            result.observations.extend(hook_result.observations)
            if hook_result.metadata:
                result.metadata.update(hook_result.metadata)

            if hook_result.action == "block":
                result.allowed = False
                result.reason = hook_result.reason
                result.blocked_by = hook.name
                return result

            if hook_result.action == "modify" and hook_result.event is not None:
                if trigger in self.FAIL_CLOSED_TRIGGERS and not isinstance(
                    hook_result.event, type(current_event)
                ):
                    # The consumer would silently ignore a payload it cannot use
                    # and carry on with the original — which for a redaction hook
                    # publishes the very content it meant to replace.
                    logger.error(
                        "Sync hook %s returned a %s where a %s was expected",
                        hook.name,
                        type(hook_result.event).__name__,
                        type(current_event).__name__,
                        extra={"room_id": room_id},
                    )
                    result.hook_errors.append(
                        {
                            "hook": hook.name,
                            "error": (
                                f"modify returned {type(hook_result.event).__name__}, "
                                f"expected {type(current_event).__name__}"
                            ),
                        }
                    )
                    result.allowed = False
                    result.reason = f"hook {hook.name} returned an unusable payload"
                    return result
                result.event = hook_result.event

        # Fire ASYNC observers for the same trigger (fire-and-forget).
        # This allows ASYNC hooks to observe events from triggers that
        # are only invoked via run_sync_hooks (e.g. ON_TRANSCRIPTION,
        # ON_VISION_RESULT, ON_TOOL_CALL).  Only ASYNC hooks are fired
        # — SYNC hooks already ran above.
        final_event = result.event if result.event is not None else event
        filter_ev = None if skip_event_filter else final_event
        async_hooks = self._get_hooks(
            room_id,
            trigger,
            HookExecution.ASYNC,
            event=filter_ev,
        )
        if async_hooks:
            await self._run_async_hooks_list(
                async_hooks,
                room_id,
                trigger,
                final_event,
                context,
            )

        return result

    async def run_async_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        skip_event_filter: bool = False,
        name_prefix: str | None = None,
        exclude_name_prefix: str | None = None,
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
            name_prefix: Only run hooks whose name starts with this prefix.
            exclude_name_prefix: Skip hooks whose name starts with this prefix.
        """
        filter_event = None if skip_event_filter else event
        hooks = self._get_hooks(room_id, trigger, None, event=filter_event)
        if name_prefix is not None:
            hooks = [h for h in hooks if h.name.startswith(name_prefix)]
        if exclude_name_prefix is not None:
            hooks = [h for h in hooks if not h.name.startswith(exclude_name_prefix)]
        if not hooks:
            return

        await self._run_async_hooks_list(hooks, room_id, trigger, event, context)

    async def _run_async_hooks_list(
        self,
        hooks: list[HookRegistration],
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
    ) -> None:
        """Run a list of hooks concurrently. Errors are logged, never raised."""
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
                await self._emit_hook_timeout(room_id, hook, trigger)
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

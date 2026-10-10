"""Hook engine for sync and async hook pipelines."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelDirection,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.task import Observation, Task

logger = logging.getLogger("roomkit.hooks")

SyncHookFn = Callable[[RoomEvent, RoomContext], Coroutine[Any, Any, HookResult]]
AsyncHookFn = Callable[[RoomEvent, RoomContext], Coroutine[Any, Any, None]]

_ROUTER_MARK = "_roomkit_router"


def mark_router_hook(fn: SyncHookFn) -> SyncHookFn:
    """Mark *fn* as a conversation router's hook (RFC §19.4), which a room a
    discussion holds cannot have (RFC §19.7.5 rule 1)."""
    setattr(fn, _ROUTER_MARK, True)
    return fn


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
        event_types: Only run for events of these types (None = all)
        fail_closed: SYNC only — a timeout, an exception or an unusable result
            blocks instead of allowing (RFC §9.3). For a content check that
            must never let an unchecked payload through.
        needs_lock: BEFORE_BROADCAST SYNC only — ``False`` runs the hook off
            the room lock, before it is taken, ordered by the room's
            admission ticket (RFC §9.5.1). Only for a hook that reads the
            event, not the room's state: its context is the pre-lock one.
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
    event_types: set[EventType] | None = None
    fail_closed: bool = False
    needs_lock: bool = True


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
    failed_closed: bool = False
    """Whether a hook's failure (a raise, a timeout, an unusable result)
    blocked, as opposed to a hook's deliberate BLOCK (RFC §9.3)."""


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
        # Set by RoomKit after init: whether a room (any room, for None) holds
        # a discussion (RFC §19.7.5 rule 1), which a router's hook cannot join.
        self._holds_discussion: Callable[[str | None], bool] | None = None
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

    def _check_lock_placement(self, hook: HookRegistration, room_id: str | None) -> None:
        """Enforce RFC §9.1's registration rules before *hook* is added.

        ``fail_closed`` only exists on a SYNC hook: an ASYNC one cannot block,
        and accepting the flag there would promise a guarantee nobody keeps.
        ``needs_lock=False`` only exists on a SYNC ``BEFORE_BROADCAST`` hook.
        And off-lock hooks run first, so a locked hook ordered before an
        off-lock one could not run in its declared place — a consent or
        budget gate meant to refuse a message before it is scanned would run
        after the scan. That is refused here, loudly, rather than reordered.
        """
        if hook.fail_closed and hook.execution != HookExecution.SYNC:
            raise ValueError(
                f"Hook {hook.name!r}: fail_closed=True needs a SYNC hook; an ASYNC hook "
                "cannot block"
            )
        is_check = (
            hook.trigger == HookTrigger.BEFORE_BROADCAST and hook.execution == HookExecution.SYNC
        )
        if not hook.needs_lock and not is_check:
            raise ValueError(
                f"Hook {hook.name!r}: needs_lock=False is only supported on a SYNC "
                f"BEFORE_BROADCAST hook, not {hook.execution} {hook.trigger}"
            )
        if not is_check:
            return
        # A global hook runs in every room, so it meets every registered hook;
        # a room hook meets the global ones and its own room's.
        peers = list(self._global_hooks)
        if room_id is None:
            for hooks in self._room_hooks.values():
                peers.extend(hooks)
        else:
            peers.extend(self._room_hooks.get(room_id, []))
        for peer in peers:
            if peer.trigger != hook.trigger or peer.execution != HookExecution.SYNC:
                continue
            locked, off = (peer, hook) if not hook.needs_lock else (hook, peer)
            if locked.needs_lock and not off.needs_lock and locked.priority < off.priority:
                raise ValueError(
                    f"Hook {locked.name!r} (priority {locked.priority}) needs the room "
                    f"lock but is ordered before off-lock hook {off.name!r} (priority "
                    f"{off.priority}). Off-lock hooks run first (RFC §9.5.1): give "
                    f"{off.name!r} a priority at or below {locked.priority}, or keep "
                    f"it needs_lock=True. Channel and event filters are not considered."
                )

    def has_off_lock_hooks(self, room_id: str, event: RoomEvent) -> bool:
        """Whether a ``needs_lock=False`` check applies to *event* (RFC §9.5.1)."""
        return any(
            not h.needs_lock
            for h in self._get_hooks(
                room_id, HookTrigger.BEFORE_BROADCAST, HookExecution.SYNC, event=event
            )
        )

    def room_hook_names(self, room_id: str) -> list[str]:
        """The names of the hooks registered for *room_id* alone."""
        return [h.name for h in self._room_hooks.get(room_id, [])]

    def has_router_hook(self, room_id: str) -> bool:
        """Whether a conversation router's hook applies to *room_id*."""
        hooks = [*self._global_hooks, *self._room_hooks.get(room_id, [])]
        return any(getattr(h.fn, _ROUTER_MARK, False) for h in hooks)

    def has_sync_hooks(self, room_id: str, trigger: HookTrigger, event: RoomEvent) -> bool:
        """Whether a SYNC hook of *trigger* applies to *event* in *room_id*."""
        return bool(self._get_hooks(room_id, trigger, HookExecution.SYNC, event=event))

    def _check_router_placement(self, hook: HookRegistration, room_id: str | None) -> None:
        """Refuse a router's hook where a discussion takes the turns (RFC
        §19.7.5 rule 1): one rule decides who speaks, never two."""
        holds = self._holds_discussion
        if holds is None or not getattr(hook.fn, _ROUTER_MARK, False) or not holds(room_id):
            return
        where = f"Room {room_id} holds" if room_id is not None else "A room holds"
        raise ValueError(f"{where} a discussion: a router cannot be installed there")

    def register(self, hook: HookRegistration) -> None:
        """Register a global hook."""
        self._check_router_placement(hook, None)
        self._check_lock_placement(hook, None)
        self._global_hooks.append(hook)
        self._trigger_index.add(hook.trigger)

    def add_room_hook(self, room_id: str, hook: HookRegistration) -> None:
        """Register a hook for a specific room."""
        self._check_router_placement(hook, room_id)
        self._check_lock_placement(hook, room_id)
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
        event_ok = hook.event_types is None or event.type in hook.event_types

        return type_ok and id_ok and dir_ok and event_ok

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
    #: redacting a transcript, holding back speech — or an action it may be
    #: there to prevent: a tool call behind an approval hook. On those, a hook
    #: that raises blocks rather than letting the payload through: logging the
    #: error and carrying on would publish, or run, exactly what the hook
    #: existed to stop (RFC §9.3). Everywhere else a failing hook stays
    #: non-fatal, so a broken hook cannot take a room down.
    FAIL_CLOSED_TRIGGERS: ClassVar[frozenset[HookTrigger]] = frozenset(
        {HookTrigger.BEFORE_TTS, HookTrigger.ON_TRANSCRIPTION, HookTrigger.BEFORE_TOOL_USE}
    )

    def _apply_rewrite(
        self,
        result: SyncPipelineResult,
        hook: HookRegistration,
        trigger: HookTrigger,
        room_id: str,
        original: Any,
        hook_result: HookResult,
        fold: Callable[[Any, dict[str, Any]], Any] | None,
    ) -> bool:
        """Carry *hook_result*'s rewrite into the event the next hook sees.

        A MODIFY replaces the payload; with a *fold*, the hook's metadata
        rewrite is written into it too, and the fold runs after every hook
        that rewrote the payload, either way. A MODIFY whose payload is not of the
        type the trigger passed in replaces nothing (RFC §9.3): it blocks when
        the hook fails closed, and under a *fold* the chain carries on from
        the previous outcome, the hook's metadata rewrite still applied.
        Without a fold on a fail-open hook the payload is taken as it is.
        Returns whether the pipeline was closed.
        """
        current = result.event if result.event is not None else original
        payload = hook_result.event
        if hook_result.action == "modify" and payload is not None:
            usable = isinstance(payload, type(current))
            if usable or (fold is None and not self._fails_closed(hook, trigger)):
                result.event = payload
            else:
                # The consumer would silently ignore a payload it cannot use
                # and carry on with the original — which for a redaction hook
                # publishes the very content it meant to replace.
                expected, got = type(current).__name__, type(payload).__name__
                logger.error(
                    "Sync hook %s returned a %s where a %s was expected",
                    hook.name,
                    got,
                    expected,
                    extra={"room_id": room_id},
                )
                result.hook_errors.append(
                    {"hook": hook.name, "error": f"modify returned {got}, expected {expected}"}
                )
                if self._close(
                    result,
                    hook,
                    trigger,
                    "hook_invalid_result",
                    f"hook {hook.name} returned an unusable payload",
                ):
                    return True
        rewrote = hook_result.action == "modify" and payload is not None
        if fold is not None and (hook_result.metadata or rewrote):
            latest = result.event if result.event is not None else original
            result.event = fold(latest, hook_result.metadata or {})
        return False

    def _fails_closed(self, hook: HookRegistration, trigger: HookTrigger) -> bool:
        """Whether an unusable outcome of *hook* blocks (RFC §9.3)."""
        return hook.fail_closed or trigger in self.FAIL_CLOSED_TRIGGERS

    def fail_closed_hook(self, room_id: str, trigger: HookTrigger) -> str | None:
        """The name of the first SYNC hook on *trigger* that fails closed, or None.

        For a caller that cannot run the pipeline at all (the room's context
        would not build): with such a hook registered, the payload must be
        withheld as the hook's own failure would withhold it.
        """
        for hook in self._get_hooks(room_id, trigger, HookExecution.SYNC):
            if self._fails_closed(hook, trigger):
                return hook.name
        return None

    def _close(
        self,
        result: SyncPipelineResult,
        hook: HookRegistration,
        trigger: HookTrigger,
        outcome: str,
        trigger_reason: str,
    ) -> bool:
        """Block *result* when *hook* fails closed; return whether it did.

        A hook that declared ``fail_closed`` names itself and the outcome
        (``hook_timeout:<name>``) so the sender can be told why the message
        did not go out. A fail-closed *trigger* reports a human-readable
        reason instead.
        """
        if not self._fails_closed(hook, trigger):
            return False
        result.allowed = False
        result.failed_closed = True
        if hook.fail_closed:
            result.reason = f"{outcome}:{hook.name}"
            result.blocked_by = hook.name
        else:
            result.reason = trigger_reason
        return True

    async def run_sync_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        skip_event_filter: bool = False,
        needs_lock: bool | None = None,
        fold: Callable[[Any, dict[str, Any]], Any] | None = None,
        fire_observers: bool = True,
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
            needs_lock: ``None`` runs every SYNC hook. ``False`` runs only the
                off-lock ones (RFC §9.5.1) and fires no ASYNC observer — the
                locked pass that follows fires them once, on the final event.
                ``True`` runs only the hooks that need the lock.
            fold: For a trigger whose hooks may also rewrite the payload
                through ``metadata`` (ON_TOOL_CALL's result override),
                ``fold(event, metadata)`` returns the event as a hook's
                rewrite left it (a MODIFY's payload or the event, with the
                metadata written in; ``metadata`` is empty after a MODIFY
                that set none), so the next hook, the ASYNC observers and the caller
                all see the chain's latest state. ``None`` leaves metadata
                beside the event.
            fire_observers: ``False`` leaves the ASYNC observers to the
                caller, for a firing whose observers must see something other
                than the event the chain left: a served tool call's result as
                the model reads it, or a report no SYNC hook can change.
        """
        filter_event = None if skip_event_filter else event
        hooks = self._get_hooks(room_id, trigger, HookExecution.SYNC, event=filter_event)
        if needs_lock is not None:
            hooks = [h for h in hooks if h.needs_lock == needs_lock]
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
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message="timeout")
                if self._close(
                    result,
                    hook,
                    trigger,
                    "hook_timeout",
                    f"hook {hook.name} timed out after {hook.timeout}s",
                ):
                    return result
                continue
            except Exception as exc:
                logger.exception("Sync hook %s failed", hook.name, extra={"room_id": room_id})
                result.hook_errors.append({"hook": hook.name, "error": str(exc)})
                if span_id is not None:
                    self._telemetry.end_span(span_id, status="error", error_message=str(exc))
                if self._close(
                    result, hook, trigger, "hook_error", f"hook {hook.name} failed: {exc}"
                ):
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
                if self._close(
                    result,
                    hook,
                    trigger,
                    "hook_invalid_result",
                    f"hook {hook.name} returned {type(hook_result).__name__} "
                    "instead of HookResult",
                ):
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

            if self._apply_rewrite(result, hook, trigger, room_id, event, hook_result, fold):
                return result

        # Fire ASYNC observers for the same trigger (fire-and-forget).
        # This allows ASYNC hooks to observe events from triggers that
        # are only invoked via run_sync_hooks (e.g. ON_TRANSCRIPTION,
        # ON_VISION_RESULT, ON_TOOL_CALL).  Only ASYNC hooks are fired
        # — SYNC hooks already ran above.
        if needs_lock is False or not fire_observers:
            return result
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

    async def run_observers(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent | Any,
        context: RoomContext,
        *,
        skip_event_filter: bool = False,
    ) -> None:
        """Run only the ASYNC-registered hooks for *trigger*, fire-and-forget.

        The counterpart of :meth:`run_async_hooks`, which deliberately ignores
        the declared execution mode. Here the mode is the whole point: it is
        what separates a hook that *observes* a call from one that *serves* it.

        A refused tool call must still be observable — an audit trail that
        cannot see a denial cannot tell a denied agent from an idle one — but
        it must not reach a hook that would serve it, or the denial would
        merely hide the side effect instead of preventing it. Only a SYNC hook
        can serve a call (it is the one that returns a result), so dispatching
        the ASYNC hooks alone makes the distinction structural rather than a
        rule each hook author has to remember.
        """
        hooks = self._get_hooks(
            room_id,
            trigger,
            HookExecution.ASYNC,
            event=None if skip_event_filter else event,
        )
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

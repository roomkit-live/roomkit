"""The wiring an executor gives the channels it runs (RFC §10.6.2, §10.6.6).

A channel needs what the kit used to set on it at registration: the
callbacks its loop fires the turn triggers through, a realtime backend, its
readers of the store and of the owner's process, the host's media readers.
The executor gives them, the kit's own as a host's, so a turn fires the same
hooks wherever it runs: the executor's, against a room context built when
they fire, their framework events and the channel's publications crossing to
the owner during a turn (``ephemeral`` and ``framework_event`` frames).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from roomkit.channels.acp import ACPChannel
from roomkit.channels.ai import AIChannel
from roomkit.core._changes import withheld
from roomkit.core.hooks import SyncPipelineResult
from roomkit.models.enums import ChannelCategory, EventType, HookTrigger
from roomkit.models.pending_input import PendingInputEvent
from roomkit.models.plan_event import PlanUpdatedEvent
from roomkit.models.thinking_event import ThinkingEvent
from roomkit.models.tool_call import AIGenerationEvent, AIResponseEvent
from roomkit.speaking.base import SpeakDecision, SpeakDecisionEvent
from roomkit.tools._outcome import kept_in_tool_memory
from roomkit.turns._outlet import TurnRealtime, emit_framework_event
from roomkit.turns._tool_hooks import ToolHooks
from roomkit.turns.host import current_turn_host

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import (
        AfterToolRoundHook,
        BeforeGenerationHook,
        RoomTasksLoader,
        RoomVisionLoader,
        SpeakDecisionHook,
        ThinkingHook,
        ThoughtHook,
        ToolHistoryLoader,
        ToolUsageLoader,
    )
    from roomkit.channels._task_planner import PlanUpdatedCallback
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.models.tool_call import AfterResponseCallback, ToolRoundEvent
    from roomkit.speaking.thought import ThoughtEvent
    from roomkit.tools.external import ExternalToolHandler
    from roomkit.turns._wiring_env import WiringEnv

logger = logging.getLogger("roomkit.turns")

TOOL_MEMORY_CALLS = 30
"""Calls a channel's stored tool rows are read for (the 8 the history replays,
the 12 tools re-revealed), after the infra rows are filtered out."""


def _standing_decision(event: SpeakDecisionEvent, result: SyncPipelineResult) -> SpeakDecision:
    """The decision *event*'s SYNC hooks leave: silent when one blocked, the
    decision a MODIFY's event carries, else the policy's. Only the decision
    is read from a MODIFY: the event decided on and the room stay the
    channel's."""
    if not result.allowed:
        hook = result.blocked_by or "a hook"
        reason = f"hook_error:{hook}" if result.failed_closed else f"blocked by {hook}"
        return SpeakDecision("silent", reason=reason)
    changed = result.event
    if changed is None or changed is event:
        return event.decision
    if isinstance(changed, SpeakDecisionEvent):
        return changed.decision
    logger.error(
        "An ON_SPEAK_DECISION hook returned a %s where a SpeakDecisionEvent was "
        "expected; the decision stands",
        type(changed).__name__,
    )
    return event.decision


def remembered_calls(events: list[RoomEvent]) -> list[dict[str, Any]]:
    """The calls a tool memory keeps, read off a channel's stored tool rows,
    in the order they were written."""
    # The arguments the model sent, from the call's start: the end carries the
    # ones that ran, which a BEFORE_TOOL_USE hook may have de-tokenised, and
    # what the memory keeps may go back into the model's prompt. An end pairs with the
    # start of the same call in the same turn (its response's correlation
    # id): a provider may reuse an id from one turn to the next, and two turns
    # of a room may run at once.
    requested: dict[tuple[str | None, str], dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    # A deleted row is withheld, as from any history (RFC §7.5 rule 8, §10.3).
    for ev in withheld(events):
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


class TurnWiring:
    """What one executor gives each channel it runs.

    Args:
        hooks: The executor's hook engine.
        env: What the wiring reads beside the hooks.
    """

    def __init__(self, hooks: HookEngine, env: WiringEnv) -> None:
        self._hooks = hooks
        self.env = env
        self.realtime = TurnRealtime(env.realtime)
        self.tools = ToolHooks(hooks, self.hook_context, self.emit)

    # -- What it rests on ---------------------------------------------------------

    async def hook_context(
        self, room_id: str, trigger: HookTrigger, *, carrying: RoomContext | None = None
    ) -> RoomContext | None:
        """The room's context for *trigger*'s hooks, or ``None`` when it will not build.

        A context is store reads (the room, its bindings, its participants, its
        history), and some triggers fire on every tool call of every round: a
        caller builds one only when ``has_hooks`` says a hook will read it.
        """
        try:
            return await self.env.context(room_id, carrying=carrying)
        except Exception:
            logger.warning(
                "Failed to build context for %s hook in room %s",
                trigger.name,
                room_id,
                exc_info=True,
            )
            return None

    async def emit(
        self,
        event_type: str,
        *,
        room_id: str | None = None,
        channel_id: str | None = None,
        event_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a framework event: to the owner during a turn (RFC §10.6.3)."""
        await emit_framework_event(
            self.env.emitter(),
            event_type,
            room_id=room_id,
            channel_id=channel_id,
            event_id=event_id,
            data=data,
        )

    # -- The channels -------------------------------------------------------------

    def wire(self, channel: Channel) -> None:
        """Give *channel* what its turns fire the executor's hooks through."""
        # ON_AI_RESPONSE reports a fact (an agent finished a turn) which every
        # INTELLIGENCE channel produces, whichever class it descends from.
        if channel.category is ChannelCategory.INTELLIGENCE:
            channel._after_response_hook = (  # ty: ignore[unresolved-attribute]
                self.build_after_response_hook(channel.channel_id)
            )
        if isinstance(channel, AIChannel):
            self._wire_ai_channel(channel)
        if isinstance(channel, ACPChannel):
            # ACP agents own their tool loop, but use the same permission hooks
            # and realtime activity surface as in-process AI providers.
            channel._realtime = self.realtime
            channel._tool_report_hook = self.tools.build_tool_report_hook(channel.channel_id)
            channel._tool_observer_hook = self.tools.build_tool_observer_hook(channel.channel_id)
            if channel._external_tool_handler is not None:
                self.wire_external_tool_handler(channel.channel_id, channel._external_tool_handler)

    def _wire_ai_channel(self, channel: AIChannel) -> None:
        """Hand an AI channel the callbacks its tool loop runs the hooks through."""
        channel_id = channel.channel_id
        channel._realtime = self.realtime
        channel._tool_call_hook = self.tools.build_tool_call_hook(channel_id)
        channel._tool_observer_hook = self.tools.build_tool_observer_hook(channel_id)
        channel._tool_report_hook = self.tools.build_tool_report_hook(channel_id)
        channel._before_tool_call_hook = self.tools.build_before_tool_call_hook(channel_id)
        channel._tool_usage_loader = self.build_tool_usage_loader(channel_id)
        channel._tool_history_loader = self.build_tool_history_loader(channel_id)
        channel._room_tasks_loader = self.build_room_tasks_loader()
        channel._room_vision_loader = self.build_room_vision_loader()
        channel._media_resolver = self.env.media_resolver
        channel._attached_media = self.env.attached_media
        channel._before_generation_hook = self.build_before_generation_hook(channel_id)
        channel._after_tool_round_hook = self.build_after_tool_round_hook()
        channel._thinking_hook = self.build_thinking_hook(channel_id)
        channel._speak_decision_hook = self.build_speak_decision_hook()
        channel._thought_hook = self.build_thought_hook()
        channel._plan_updated_hook = self.build_plan_updated_hook(channel_id)
        if channel._external_tool_handler is not None:
            self.wire_external_tool_handler(channel_id, channel._external_tool_handler)
        # The person's tools announce their requests through the room's
        # ON_USER_INPUT_REQUIRED hooks; registering makes this object the
        # owner of its id's requests (a session channel registers its own
        # from set_framework).
        if channel._human_input.given:
            channel._human_input.register(
                channel_id, self.build_on_user_input_required_hook(channel_id)
            )

    def wire_external_tool_handler(self, channel_id: str, handler: ExternalToolHandler) -> None:
        """Inject hook callbacks so external tool calls fire the tool hooks."""
        if handler._channel_id and handler._channel_id != channel_id:
            # The injected callbacks are per-channel, so the second wiring
            # silently re-attributes the first channel's tool hooks — and
            # anything reading handler.channel_id then names the wrong agent.
            logger.warning(
                "External tool handler %s was already wired to channel %r; "
                "re-wiring to %r. Give each channel its own handler instance.",
                type(handler).__name__,
                handler._channel_id,
                channel_id,
            )
        handler._channel_id = channel_id
        handler._before_tool_hook = self.tools.build_before_tool_call_hook(channel_id)
        handler._on_tool_hook = self.tools.build_tool_report_hook(channel_id)

    # -- Readers ----------------------------------------------------------------------

    def build_tool_usage_loader(self, channel_id: str) -> ToolUsageLoader:
        """The tool-usage hydration loader for AI channel *channel_id*.

        Fetches the channel's most recent stored tool rows in a room so its
        in-memory ToolUsageMemory (the re-reveal set) and its skill
        activations survive channel-object lifetimes — the store dies with the
        object (process restart, cache expiry) while conversations outlive it.
        Only the channel's own rows: another agent's calls in the room are not
        its memory, and their results may be withheld from it (RFC §7.5 rule
        8). Called at most once per room per process (the channel marks the
        room hydrated).
        """
        env = self.env

        async def _load(room_id: str) -> list[dict[str, Any]]:
            # A start and an end per call.
            rows = await env.tool_rows(room_id, channel_id, TOOL_MEMORY_CALLS * 2)
            return remembered_calls(rows)

        return _load

    def build_tool_history_loader(self, channel_id: str) -> ToolHistoryLoader:
        """The reader of AI channel *channel_id*'s latest tool rows in a room,
        for the turn's history (RFC §6.4).

        Only the channel's own rows, for the reason the usage loader gives;
        a deleted row is withheld, as from any history (RFC §7.5 rule 8).
        Read every turn: the store, not the channel object, is what outlives
        a restart.
        """
        env = self.env

        async def _load(room_id: str) -> list[RoomEvent]:
            # A start and an end per call, with room for calls never replayed.
            return withheld(await env.tool_rows(room_id, channel_id, TOOL_MEMORY_CALLS * 2))

        return _load

    def build_room_tasks_loader(self) -> RoomTasksLoader:
        """The reader of a room's background tasks for a turn's notes (RFC
        §23.4): state of the owner's process, read through the turn's host."""
        env = self.env

        async def _load(room_id: str) -> list[dict[str, Any]]:
            host = current_turn_host()
            # A reader needs the turn: after it, the executor's own reads it.
            if host is not None and host.open:
                return await host.room_tasks()
            return await env.room_tasks(room_id)

        return _load

    def build_room_vision_loader(self) -> RoomVisionLoader:
        """The reader of what a room's video last showed, for a turn's notes
        (RFC §12.8.7): state of the owner's process, read through the turn's
        host."""
        env = self.env

        async def _load(context: RoomContext) -> str | None:
            host = current_turn_host()
            # A reader needs the turn: after it, the executor's own reads it.
            if host is not None and host.open:
                return await host.room_vision()
            return env.room_vision(context)

        return _load

    # -- The turn triggers ----------------------------------------------------------

    def build_thinking_hook(self, channel_id: str) -> ThinkingHook:
        """An ON_AI_THINKING callback (RFC §9.2). The same reasoning also goes
        out as an ephemeral event for live UIs; the hook is what makes it
        observable to a host that runs no realtime backend."""

        async def _callback(room_id: str, thinking: str, round_idx: int) -> None:
            trigger = HookTrigger.ON_AI_THINKING
            event = ThinkingEvent(
                room_id=room_id, channel_id=channel_id, thinking=thinking, round_index=round_idx
            )
            await self._observe(room_id, trigger, event)

        return _callback

    def build_speak_decision_hook(self) -> SpeakDecisionHook:
        """An ON_SPEAK_DECISION callback (RFC §6.4): the room's SYNC hooks, in
        order, may change the decision before the channel acts on it (a
        MODIFY's event carries the decision that replaces it, a BLOCK leaves
        the agent silent); the ASYNC hooks then observe the decision that
        stands. Returns that decision."""

        async def _callback(event: SpeakDecisionEvent) -> SpeakDecision:
            trigger = HookTrigger.ON_SPEAK_DECISION
            if not event.room_id or not self._hooks.has_hooks(trigger):
                return event.decision
            context = await self.hook_context(event.room_id, trigger)
            if context is None:
                return self._unreachable_speak_decision(event)
            result = await self._hooks.run_sync_hooks(
                event.room_id, trigger, event, context, fire_observers=False
            )
            standing = dataclasses.replace(event, decision=_standing_decision(event, result))
            await self._hooks.run_observers(event.room_id, trigger, standing, context)
            return standing.decision

        return _callback

    def _unreachable_speak_decision(self, event: SpeakDecisionEvent) -> SpeakDecision:
        """The decision on an event whose ON_SPEAK_DECISION hooks could not run
        (the room's context would not build): the policy's, unless a hook there
        fails closed: then silent, as that hook's own failure would leave it."""
        trigger = HookTrigger.ON_SPEAK_DECISION
        closed = self._hooks.fail_closed_hook(event.room_id, trigger, event)
        if closed is None:
            return event.decision
        return SpeakDecision("silent", reason=f"hook_error:{closed}")

    def build_thought_hook(self) -> ThoughtHook:
        """An ON_THOUGHT callback (RFC §6.4)."""

        async def _callback(event: ThoughtEvent) -> None:
            await self._observe(event.room_id, HookTrigger.ON_THOUGHT, event)

        return _callback

    def build_plan_updated_hook(self, channel_id: str) -> PlanUpdatedCallback:
        """An ON_PLAN_UPDATED callback."""

        async def _callback(room_id: str, tasks: list[dict[str, Any]]) -> None:
            event = PlanUpdatedEvent(room_id=room_id, channel_id=channel_id, tasks=list(tasks))
            await self._observe(room_id, HookTrigger.ON_PLAN_UPDATED, event)

        return _callback

    def build_after_tool_round_hook(self) -> AfterToolRoundHook:
        """AFTER_TOOL_ROUND: the room's SYNC hooks on a round its loop ran,
        between that round and the next (RFC §6.4).

        The hooks act on the event they receive, in place (withdrawals,
        messages); a MODIFY's returned event is not read. A BLOCK stops the
        hooks after it, as on any SYNC trigger, and changes nothing of the
        round, which has run.
        """

        async def _callback(event: ToolRoundEvent) -> None:
            trigger = HookTrigger.AFTER_TOOL_ROUND
            if not event.room_id or not self._hooks.has_hooks(trigger):
                return
            context = await self.hook_context(event.room_id, trigger)
            if context is None:
                return
            await self._hooks.run_sync_hooks(event.room_id, trigger, event, context)

        return _callback

    def build_on_user_input_required_hook(self, channel_id: str) -> Any:
        """An ON_USER_INPUT_REQUIRED callback, which also emits a
        ``user_input_required`` framework event.

        Sync execution is what gives these hooks their order and their
        veto — a BLOCK rejects the request. It does not gate the request
        being answerable: ``HumanInputHandler`` arms the request first and
        runs this callback off the waiting path, so a human who answers
        while a slow notification is still in flight is answering a
        request that is already listening.
        """

        async def _callback(event: PendingInputEvent) -> bool:
            if not event.room_id:
                return True  # Allow if no room context
            hook_result = SyncPipelineResult(event=event)
            trigger = HookTrigger.ON_USER_INPUT_REQUIRED
            if self._hooks.has_hooks(trigger):
                context = await self.hook_context(event.room_id, trigger)
                if context is None:
                    return True  # Allow on error (fail-open)
                hook_result = await self._hooks.run_sync_hooks(
                    event.room_id, trigger, event, context
                )
            await self.emit(
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

    def build_after_response_hook(self, channel_id: str) -> AfterResponseCallback:
        """An ON_AI_RESPONSE callback, which also emits an ``ai_response``
        framework event once they have ended. Observational only: it starts
        the hooks and does not wait for them, so they never hold the response
        up (RFC §9.3)."""

        async def _callback(event: AIResponseEvent) -> None:
            if not event.room_id:
                return
            room_id = event.room_id

            async def reported() -> None:
                await self.emit(
                    "ai_response",
                    room_id=room_id,
                    channel_id=channel_id,
                    data={
                        "tool_calls_count": event.tool_calls_count,
                        "latency_ms": event.latency_ms,
                        "streaming": event.streaming,
                    },
                )

            trigger = HookTrigger.ON_AI_RESPONSE
            if not self._hooks.has_hooks(trigger):
                await reported()
                return
            context = await self.hook_context(room_id, trigger)
            if context is None:
                return
            await self._hooks._fire_async_hooks(room_id, trigger, event, context, then=reported)

        return _callback

    def build_before_generation_hook(self, channel_id: str) -> BeforeGenerationHook:
        """A BEFORE_AI_GENERATION callback, which also emits a
        ``before_ai_generation`` framework event: whether generation proceeds
        or is blocked."""

        async def _callback(event: AIGenerationEvent) -> SyncPipelineResult:
            if not event.room_id:
                return SyncPipelineResult(allowed=True)
            sync_result = SyncPipelineResult(event=event)
            trigger = HookTrigger.BEFORE_AI_GENERATION
            if self._hooks.has_hooks(trigger):
                context = await self.hook_context(event.room_id, trigger)
                if context is None:
                    return SyncPipelineResult(allowed=True)
                sync_result = await self._hooks.run_sync_hooks(
                    event.room_id, trigger, event, context
                )
            await self.emit(
                "before_ai_generation",
                room_id=event.room_id,
                channel_id=channel_id,
                data={"allowed": sync_result.allowed, "blocked_by": sync_result.blocked_by},
            )
            return sync_result

        return _callback

    async def _observe(self, room_id: str | None, trigger: HookTrigger, event: Any) -> None:
        """Start *trigger*'s ASYNC hooks on *event* when one is registered,
        and return without waiting for them (RFC §9.3)."""
        if not room_id or not self._hooks.has_hooks(trigger):
            return
        context = await self.hook_context(room_id, trigger)
        if context is None:
            return
        self._hooks._start_async_hooks(room_id, trigger, event, context)

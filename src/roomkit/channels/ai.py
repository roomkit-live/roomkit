"""AI channel implementation.

This module defines :class:`AIChannel`, the intelligence channel that generates
responses using an AI provider.  Behaviour is composed from focused mixins:

- :class:`~._ai_events.AIEventsMixin` — ephemeral tool/thinking events
- :class:`~._ai_steering.AISteeringMixin` — mid-run steering directives
- :class:`~._ai_policy.AIToolPolicyMixin` — tool policy & skill gating
- :class:`~._ai_resilience.AIResilienceMixin` — retry / fallback / compaction
- :class:`~._ai_context.AIContextMixin` — AI context building
- :class:`~._ai_tools.AIToolsMixin` — tool execution & dispatch
- :class:`~._ai_generation.AIGenerationMixin` — generation hook, telemetry, provider errors
- :class:`~._ai_streaming.AIStreamingMixin` — the tool loop every turn runs
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_context import AIContextMixin
from roomkit.channels._ai_events import AIEventsMixin
from roomkit.channels._ai_generation import AIGenerationMixin
from roomkit.channels._ai_loop_rules import (
    _EMPTY_RETRY_NUDGE as _EMPTY_RETRY_NUDGE,
)
from roomkit.channels._ai_loop_rules import (
    _FORCE_STOP_NUDGE as _FORCE_STOP_NUDGE,
)
from roomkit.channels._ai_policy import AIToolPolicyMixin
from roomkit.channels._ai_resilience import AIResilienceMixin
from roomkit.channels._ai_speaking import AISpeakingMixin, speak_notes
from roomkit.channels._ai_steering import AISteeringMixin
from roomkit.channels._ai_streaming import AIStreamingMixin
from roomkit.channels._ai_thinking import AIThinkingMixin
from roomkit.channels._ai_tools import AIToolsMixin
from roomkit.channels._discussion_turn import discussion_turn
from roomkit.channels._served_tools import (
    CollisionLog,
    refuse_host_tools,
)
from roomkit.channels._skill_activation import SkillActivationMemory
from roomkit.channels._task_planner import TaskPlanner
from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._tool_registry import ChannelRegistry, ToolSource
from roomkit.channels._tool_search import checked_threshold_tokens
from roomkit.channels._tool_search_constants import (
    DEFAULT_TOOL_SEARCH_THRESHOLD,
    DEFAULT_TOOL_SEARCH_THRESHOLD_PCT,
    DEFAULT_TOOL_SEARCH_THRESHOLD_TOKENS,
)
from roomkit.channels._tool_usage import ToolUsageMemory
from roomkit.channels._turn_budget import turn_budget
from roomkit.channels._turn_config import ConfigProvider
from roomkit.channels.base import Channel
from roomkit.memory.base import MemoryProvider
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.channel import (
    ChannelBinding,
    ChannelCapabilities,
    ChannelOutput,
    RetryPolicy,
)
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    EventType,
)
from roomkit.models.event import RoomEvent, is_tool_call_record
from roomkit.providers.ai.base import (
    AIImagePart,
    AIProvider,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.ai.json_schema import check_portable_schema
from roomkit.realtime.base import RealtimeBackend
from roomkit.tools._human_input_channel import ChannelHumanInput, warn_plain_handler
from roomkit.tools.compose import compose_tool_handlers, extract_tools
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext
from roomkit.tools.policy import ToolPolicy
from roomkit.tools.timeout import ToolTimeouts

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import AfterToolRoundHook
    from roomkit.models.tool_call import ContinuationPolicy, ToolCallCallback, ToolCallObserver
    from roomkit.providers.ai.base import AIContext
    from roomkit.sandbox.executor import SandboxExecutor
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry
    from roomkit.speaking.base import SpeakDecision, SpeakPolicy
    from roomkit.speaking.thinker import Thinker
    from roomkit.tools.base import Tool
    from roomkit.tools.external import ExternalToolHandler
    from roomkit.tools.human_input import HumanInputToolHandler

# A tool may answer with plain text or a content-part list (text + images,
# e.g. a screenshot) — providers without image support flatten via as_text().
ToolResult = str | list[AITextPart | AIImagePart]
ToolHandler = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]
# What a handler returns is the tool's answer. To decline a call instead, raise
# ``roomkit.ToolRefusedError``; to say it ran and failed, ``roomkit.ToolFailedError``:
# the message reaches the model verbatim and the call is marked failed (refused, for
# the first), where a returned body would read as work that was done.


# What an AI channel's transcript says of an event whose content extracts to
# nothing (an upload without a caption, say): a text, or None to omit it.
EmptyEventDescriber = Callable[[RoomEvent], str | None]

# Content part union — matches AIMessage.content list type
_ContentPart = AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart

logger = logging.getLogger("roomkit.channels.ai")

# Largest tool result the usage memory keeps for later turns, in characters.
_TOOL_MEMORY_RESULT_CHARS = 6000


def _portable_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """*schema*, checked against the portable subset now: one outside it fails
    at construction rather than on the first message."""
    if schema is not None:
        check_portable_schema(schema)
    return schema


class AIChannel(
    AIThinkingMixin,
    AISpeakingMixin,
    AIStreamingMixin,
    AIGenerationMixin,
    AIToolsMixin,
    AIContextMixin,
    AIResilienceMixin,
    AIToolPolicyMixin,
    AISteeringMixin,
    AIEventsMixin,
    Channel,
):
    """AI intelligence channel that generates responses using an AI provider."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE
    direction = ChannelDirection.BIDIRECTIONAL

    def __init__(
        self,
        channel_id: str,
        provider: AIProvider,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_context_events: int = 50,
        tool_handler: ToolHandler | None = None,
        tools: list[AITool | Tool] | None = None,
        max_tool_rounds: int = 50,
        tool_loop_timeout_seconds: float | None = 300.0,
        tool_loop_warn_after: int = 25,
        max_empty_retries: int = 1,
        continuation: ContinuationPolicy | None = None,
        thinking_coalesce_ms: float = 80.0,
        thinking_coalesce_chars: int = 256,
        retry_policy: RetryPolicy | None = None,
        fallback_provider: AIProvider | None = None,
        skills: SkillRegistry | None = None,
        skills_in_prompt: bool = True,
        script_executor: ScriptExecutor | None = None,
        sandbox: SandboxExecutor | None = None,
        external_tool_handler: ExternalToolHandler | None = None,
        human_input_handler: HumanInputToolHandler | None = None,
        memory: MemoryProvider | None = None,
        tool_policy: ToolPolicy | None = None,
        thinking_budget: int | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        response_schema: dict[str, Any] | None = None,
        evict_threshold_tokens: int = 5000,
        enable_planning: bool = False,
        config_provider: ConfigProvider | None = None,
        tool_search: bool | None = None,
        tool_search_pinned: list[str] | None = None,
        tool_search_threshold: int = DEFAULT_TOOL_SEARCH_THRESHOLD,
        tool_search_threshold_pct: float = DEFAULT_TOOL_SEARCH_THRESHOLD_PCT,
        tool_search_miss_hint: str | None = None,
        turn_budget_tokens: int | None = None,
        turn_budget_usd: float | None = None,
        tool_search_threshold_tokens: int | None = DEFAULT_TOOL_SEARCH_THRESHOLD_TOKENS,
        tool_timeout_seconds: float | None = 30.0,
        tool_timeouts: Mapping[str, float | None] | None = None,
        describe_empty_event: EmptyEventDescriber | None = None,
        speak_policy: SpeakPolicy | None = None,
        speak_timeout: float = 2.0,
        thinker: Thinker | None = None,
        think_wait: float = 1.5,
    ) -> None:
        super().__init__(channel_id)
        # Whether the agent speaks on an event (RFC §6.4): none, it answers every one.
        self._store_speak_policy(speak_policy, speak_timeout)
        # What it thinks while it listens (RFC §6.4): none, no thought.
        self._store_thinker(thinker, think_wait, speak_policy)
        self._store_turn_budget(turn_budget_tokens, turn_budget_usd, provider, fallback_provider)
        self._provider = provider
        self._system_prompt = system_prompt
        # Per-turn config resolution — see channels/_turn_config.py. When
        # set, system prompt / tools / sampling are resolved fresh at the
        # start of every turn instead of living as attach-time snapshots.
        self._config_provider = config_provider
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._store_transcript_rules(max_context_events, memory, describe_empty_event)
        self._thinking_budget = thinking_budget
        self._enable_thinking = enable_thinking
        self._reasoning_effort = reasoning_effort
        # Every turn's answer is constrained to this schema unless the binding
        # or the config provider says otherwise (RFC §6.7).
        self._response_schema = _portable_schema(response_schema)
        self._store_loop_rules(
            max_tool_rounds, tool_loop_warn_after, max_empty_retries, continuation
        )
        self._store_tool_bounds(tool_loop_timeout_seconds, tool_timeout_seconds, tool_timeouts)
        # Reasoning-stream coalescing window — see _ThinkingCoalescer. Per-token
        # thinking deltas are batched into one realtime publish per window so a
        # long reasoning trace costs 10-100x fewer ephemeral events + WS sends
        # while staying visibly real-time. 0 ms disables (publish every delta).
        self._thinking_coalesce_ms = thinking_coalesce_ms
        self._thinking_coalesce_chars = thinking_coalesce_chars
        self._retry_policy = retry_policy
        self._fallback_provider = fallback_provider
        self._skills = skills
        # Hosts that render their own skills manifest inside ``system_prompt``
        # (e.g. positioned above a prompt-cache boundary) set this to False to
        # skip the automatic preamble+XML injection while keeping the skill
        # activation tools.
        self._skills_in_prompt = skills_in_prompt
        self._script_executor = script_executor
        self._sandbox = sandbox
        self._tool_policy = tool_policy
        self._eviction = ToolEviction(threshold_tokens=evict_threshold_tokens)
        # Per-conversation record of tools the agent has called — feeds the
        # "tools you've already used" digest and re-reveals used tools each turn
        # so a tool used once stays callable under Tool Search. See _tool_usage.
        # A result kept for later turns never exceeds what the eviction
        # threshold lets the model see in the turn itself (~4 chars a token).
        self._tool_usage = ToolUsageMemory(
            result_keep_chars=min(_TOOL_MEMORY_RESULT_CHARS, 4 * evict_threshold_tokens),
            recorded=self._in_usage_digest,
        )
        # Per-conversation record of the skills the model activated. Their bodies
        # ride the system prompt from the next turn on, so ``activate_skill``
        # answers with a short ACK instead of re-sending a multi-KB body every
        # turn (the rebuilt context drops tool results). See _skill_activation.
        self._skill_activation = SkillActivationMemory()
        self._planner = TaskPlanner() if enable_planning else None
        # Tool Search: ``None`` auto-enables it when the deferrable schemas
        # pass ``tool_search_threshold_tokens`` (cost, whatever the window) or
        # ``tool_search_threshold_pct`` % of the window (fit); with the window
        # unknown, past the ``tool_search_threshold`` tool count. True/False
        # force it. The text loop re-sends its tool list every round, so a
        # reveal is a per-round re-filter, unlike the realtime channel's
        # provider.reconfigure. See ``should_activate_tool_search``.
        self._tool_search = tool_search
        self._tool_search_pinned: set[str] = set(tool_search_pinned or [])
        self._tool_search_threshold = tool_search_threshold
        self._tool_search_threshold_pct = tool_search_threshold_pct
        self._tool_search_threshold_tokens = checked_threshold_tokens(tool_search_threshold_tokens)
        self._tool_search_miss_hint = tool_search_miss_hint

        self._init_tool_surface(tool_handler, tools, human_input_handler)

        # Active tool loops for steering (loop_id -> context)
        self._active_loops: dict[str, _ToolLoopContext] = {}

        self._init_framework_callbacks()
        # External tool handler for provider-executed tools (e.g. Claude Code)
        self._external_tool_handler = external_tool_handler

    def _store_turn_budget(
        self,
        tokens: int | None,
        usd: float | None,
        provider: AIProvider,
        fallback: AIProvider | None,
    ) -> None:
        """Keep the channel's default turn budget, which the binding and the
        config provider may override per turn; a budget that is not a positive
        number, or a cost budget for an unpriced model, fails here (RFC §6.4)."""
        turn_budget(tokens, usd, provider, fallback)
        self._turn_budget_tokens = tokens
        self._turn_budget_usd = usd

    def _store_transcript_rules(
        self,
        max_events: int,
        memory: MemoryProvider | None,
        describe_empty_event: EmptyEventDescriber | None,
    ) -> None:
        """Keep how a turn's transcript is built: the history it reads (the
        channel's memory, a sliding window of *max_events* by default) and
        what it says of an event whose content extracts to nothing."""
        self._max_context_events = max_events
        self._memory = memory or SlidingWindowMemory(max_events=max_events)
        self._describe_empty_event = describe_empty_event

    def _store_loop_rules(
        self,
        max_rounds: int,
        warn_after: int,
        max_empty_retries: int,
        continuation: ContinuationPolicy | None,
    ) -> None:
        """Keep the rules the tool loop applies between rounds: its round cap,
        when it warns, and the tries an empty round and the continuation policy
        share, the policy going on an answer that did not act (RFC §6.4)."""
        self._max_tool_rounds = max_rounds
        self._tool_loop_warn_after = warn_after
        self._max_empty_retries = max_empty_retries
        self._continuation = continuation

    def _store_tool_bounds(
        self,
        loop_seconds: float | None,
        call_seconds: float | None,
        per_tool: Mapping[str, float | None] | None,
    ) -> None:
        """Keep the turn's tool-loop deadline and each call's own bound (RFC §21.6).

        The deadline is read between rounds, so it cannot stop a handler that
        never answers: the call bound does, costing the call rather than the
        turn. A bound that is not positive fails here.
        """
        self._tool_loop_timeout_seconds = loop_seconds
        self._tool_timeouts = ToolTimeouts(call_seconds, dict(per_tool or {}))

    def _init_tool_surface(
        self,
        tool_handler: ToolHandler | None,
        tools: list[AITool | Tool] | None,
        human_input_handler: HumanInputToolHandler | None,
    ) -> None:
        """The tools this channel declares and the handlers that serve them."""
        # The person's tools, which the channel serves itself, before the
        # host's handler, under their own timeout (RFC §9.3, §21.6).
        self._human_input = ChannelHumanInput(human_input_handler, self.channel_type)
        warn_plain_handler(tool_handler, self.channel_id)
        extracted_defs, effective_handler = self._compose_host_tools(tool_handler, tools)

        # The host's handler, kept apart: all dispatch goes through
        # _channel_tool_handler, which routes to the registry's entries (the
        # channel's own tools, orchestration's), the sandbox, the person's
        # tools, then to this.
        self._user_tool_handler = effective_handler

        # The host's tools (from the constructor), served by its handler.
        self._user_tools: list[AITool] = extracted_defs
        # What the channel serves itself and what orchestration sets up on it,
        # each tool with its traits, for every room or one (RFC §19.7, §21.1).
        self._registry = ChannelRegistry(self.channel_id, self._host_tool_names)
        self._register_channel_tools()

        # Host tools that collide with the channel's own (RFC §21.1), each
        # reported once.
        self._collisions = CollisionLog(self.channel_id)
        names = (tool.name for tool in self._user_tools)
        refuse_host_tools(names, self._channel_tool_names(), self.channel_id)
        self._human_input.refuse_collisions(self._channel_own_names(), self.channel_id)

    def _compose_host_tools(
        self, tool_handler: ToolHandler | None, tools: list[AITool | Tool] | None
    ) -> tuple[list[AITool], ToolHandler | None]:
        """The host's tool definitions, and the one handler that serves them."""
        # Extract Tool objects: split into AITool definitions + composed handler
        extracted_defs: list[AITool] = []
        extracted_handler: ToolHandler | None = None
        if tools:
            extracted_defs, extracted_handler = extract_tools(list(tools))

        # Merge explicit tool_handler with handlers extracted from Tool objects
        effective_handler = tool_handler
        if extracted_handler and tool_handler:
            effective_handler = compose_tool_handlers(tool_handler, extracted_handler)
        elif extracted_handler:
            effective_handler = extracted_handler
        return extracted_defs, effective_handler

    def _host_tool_names(self) -> list[str]:
        """The names the host's own tools carry: its definitions and its
        human-input tools, served by the handlers it gave."""
        names = [tool.name for tool in self._user_tools]
        names.extend(self._human_input.declared_names)
        return names

    def _in_usage_digest(self, name: str) -> bool:
        """Whether a call to *name* is work the agent did, for the usage digest:
        not a discovery or housekeeping tool of the channel's own."""
        traits = self._registry.traits(name)
        return traits is None or traits.in_digest

    def _init_framework_callbacks(self) -> None:
        """The callbacks the framework injects on ``register_channel``, unset until then."""
        # Realtime backend for ephemeral tool call events
        self._realtime: RealtimeBackend | None = None
        # Unified tool call hook callback
        self._tool_call_hook: ToolCallCallback | None = None
        # Fired for a call that failed or was refused — observers only.
        self._tool_observer_hook: ToolCallObserver | None = None
        # Fired for a call a provider already ran — a report, nothing applied.
        self._tool_report_hook: ToolCallObserver | None = None
        self._before_tool_call_hook = None
        self._after_response_hook = None
        self._before_generation_hook = None
        # AFTER_TOOL_ROUND, between two rounds of the tool loop (RFC §6.4).
        self._after_tool_round_hook: AfterToolRoundHook | None = None
        self._thinking_hook = None
        self._plan_updated_hook = None
        # Tool-usage hydration loader: fetches this channel's persisted tool
        # rows in a room so ToolUsageMemory survives channel-object lifetimes
        # (restarts, cache expiry) — the in-memory store dies with the object
        # while conversations outlive it.
        self._tool_usage_loader = None
        # The room's background tasks for the turn's notes (RFC §23.4), read
        # from the framework's StatusBus: wired at registration.
        self._room_tasks_loader = None
        # What the room's cameras last saw, for the turn's notes (RFC §12.8.7):
        # wired at registration.
        self._room_vision_loader = None

    @property
    def tool_handler(self) -> ToolHandler | None:
        """The host's tool handler: it serves every tool neither the channel
        nor orchestration serves (RFC §21.1).

        Replacing it replaces the host's handler only: the channel's own tools
        and the ones orchestration set up keep being served.
        """
        return self._user_tool_handler

    @tool_handler.setter
    def tool_handler(self, value: ToolHandler | None) -> None:
        self._user_tool_handler = value

    @property
    def provider(self) -> AIProvider:
        """The underlying AI provider."""
        return self._provider

    @property
    def system_prompt(self) -> str | None:
        """The system prompt used to build request context each turn."""
        return self._system_prompt

    def set_system_prompt(self, prompt: str | None) -> None:
        """Replace the system prompt for subsequent turns.

        ``AIChannel`` rebuilds its request context from ``self._system_prompt``
        at the start of every turn, so the new prompt takes effect on the next
        turn with no reconnect and no loss of memory or tool state — the
        supported way to swap personas/attitudes mid-conversation.

        Note: when a ``config_provider`` is set the system prompt is resolved
        fresh per turn, so this value is overridden on the next turn.
        """
        self._system_prompt = prompt

    @property
    def extra_tools(self) -> list[AITool]:
        """The host's tools, then the ones orchestration set up for every room."""
        return self._user_tools + self._orchestration_tools(None)

    def active_skill_names(self, room_id: str | None) -> set[str]:
        """Skills whose instructions are binding in *room_id* right now.

        Runtime state, not the catalogue: these are the activations recorded for
        this conversation, which is exactly what the system prompt already
        carries under "Active skill instructions". A host rendering its own
        manifest (``skills_in_prompt=False``) cannot otherwise tell an available
        skill from an active one, and pushing the model to load what is already
        binding costs a tool round and contradicts the rules in front of it.
        """
        return self._skill_activation.active_names(room_id)

    def _propagate_telemetry(self) -> None:
        """Propagate telemetry to AI provider."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            self._provider._telemetry = telemetry

    @property
    def info(self) -> dict[str, Any]:
        return {"provider": type(self._provider).__name__, "active_turns": self.active_turns}

    @property
    def active_turns(self) -> int:
        """Turns being produced right now.

        Every turn's tool loop registers itself in ``_active_loops`` for
        steering, from the start of its generation to its ``finally``. The
        span starts when the turn is *consumed*: a streaming output handed
        back by ``on_event`` and not yet iterated reads 0, a window the
        caller's own wait has to cover. ``close()`` tears the provider down
        under whichever turn is running, so a caller retiring this object
        waits for zero first.
        """
        return len(self._active_loops)

    def capabilities(self) -> ChannelCapabilities:
        media_types = [ChannelMediaType.TEXT, ChannelMediaType.RICH]
        if self._provider.supports_vision:
            media_types.append(ChannelMediaType.MEDIA)
        return ChannelCapabilities(
            media_types=media_types,
            supports_rich_text=True,
            supports_media=self._provider.supports_vision,
        )

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError("AI channel does not accept inbound messages")

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """React to an event: the room's turn runner takes it, or the channel answers.

        A strategy installed in a room (a loop, a supervisor's delegation
        passes) may take that room's turns in place of the agent's own answer
        (RFC §19.7); every other room gets the channel's (:meth:`_respond`).
        """
        room_id = context.room.id if context.room else event.room_id
        runner = self._registry.turn_runner(room_id)
        if runner is not None:
            return await runner(event, binding, context)
        decision = await self._speak_decision(event, context, self._thought_of(room_id))
        if decision is None or decision.mode != "silent":
            notes = await self._speaking_notes(room_id, decision)
            return await self._respond(event, binding, context, notes=notes)
        # No turn, but the conversation is the agent's memory all the same.
        await self._ingest_event(event, context)
        decision = await self._think_while_listening(event, binding, context, decision)
        if decision.mode == "silent":
            return ChannelOutput.empty()
        notes = await self._speaking_notes(room_id, decision)
        return await self._respond(event, binding, context, notes=notes, ingested=True)

    async def _speaking_notes(
        self, room_id: str, decision: SpeakDecision | None
    ) -> tuple[str, ...]:
        """What a decided turn's notes carry: the agent's thought, then what the
        decision asks (RFC §6.4)."""
        if decision is None:
            return ()
        return (*await self._thought_notes(room_id), *speak_notes(decision))

    async def on_room_attached(self, room_id: str, binding: ChannelBinding) -> None:
        """A room the channel joins starts from an empty thought, and open to the
        agent, whatever an earlier room of the same id left (RFC §6.4)."""
        await self._forget_room(room_id)

    async def on_room_detached(self, room_id: str) -> None:
        """The room's thought, and what the speak policy kept of it, go with the
        binding (RFC §6.4)."""
        await self._forget_room(room_id)

    async def _forget_room(self, room_id: str) -> None:
        await self._forget_thought(room_id)
        if self._speak_policy is not None:
            self._speak_policy.forget_room(room_id)

    async def _thinking_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext | None:
        """What the thinker reads on *event*: its context, built as for an answer
        and passed through BEFORE_AI_GENERATION as a thought (RFC §6.4), which a
        hook may change; ``None`` when a hook blocks it."""
        loop_ctx = self._turn_loop_ctx(event, context)
        loop_ctx.names_every_speaker = True  # the thinker knows who speaks (§6.4)
        token = _current_loop_ctx.set(loop_ctx)
        try:
            ai_context = await self._build_context(event, binding, context)
            ai_context, blocked = await self._fire_before_generation_hook(
                ai_context, event, purpose="thought"
            )
        finally:
            _current_loop_ctx.reset(token)
        return None if blocked else ai_context

    async def _respond(
        self,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
        *,
        notes: tuple[str, ...] = (),
        ingested: bool = False,
    ) -> ChannelOutput:
        """Answer an event with this channel's own turn.

        Skips events from this channel to prevent self-loops. Every other
        event runs the one tool loop (RFC §6.4): it yields text deltas round
        by round and executes tool calls between rounds, whatever the provider
        streams (one that does not is read through its ``generate()``) and
        whether the turn carries tools (a turn without any is one round).
        *notes* join the turn's notes: what a speak policy's decision asks.
        *ingested*: the memory provider already has the event.
        """
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()

        if is_tool_call_record(event):
            return ChannelOutput.empty()

        # A discussion handed the event to memory when it committed (RFC
        # §19.7.5 rule 3): its turn does not hand it again.
        if not ingested and discussion_turn(event) is None:
            await self._ingest_event(event, context)

        token = _current_loop_ctx.set(self._turn_loop_ctx(event, context))
        try:
            return await self._start_streaming_tool_response(event, binding, context, notes=notes)
        finally:
            _current_loop_ctx.reset(token)

    async def _ingest_event(self, event: RoomEvent, context: RoomContext) -> None:
        """Hand *event* to the memory provider (a stateful one, a vector store,
        indexes content as it arrives), whether the agent answers it or not.

        An instruction is the application's, not the conversation's (RFC
        §10.1.1): a memory provider never learns it as something said.
        """
        room_id = context.room.id if context.room else event.room_id
        if not room_id or event.type == EventType.INSTRUCTION:
            return
        try:
            await self._memory.ingest(room_id, event, channel_id=self.channel_id)
        except Exception:
            logger.warning("Memory ingestion failed", exc_info=True)

    def _turn_loop_ctx(self, event: RoomEvent, context: RoomContext) -> _ToolLoopContext:
        """The per-turn context the turn's tool handlers read (RFC §21.4).

        Set on a per-invocation _ToolLoopContext visible via contextvar so that
        _build_context and the tool loop methods can read the participant's
        role (role-based tool policy). ``room_id`` rides the same contextvar:
        the channel object is registered once per channel_id and shared by
        every room it serves, so per-call room resolution
        (``current_tool_room_id``) is the only safe way for tool handlers to
        learn the originating room. The chain depth is the one this turn's
        response carries (RFC §8.3).
        """
        ctx = _ToolLoopContext()
        ctx.current_participant_role = self._resolve_participant_role(event, context)
        ctx.actor_id = event.source.participant_id
        ctx.chain_depth = event.chain_depth + 1
        ctx.room_id = context.room.id if context.room else event.room_id
        ctx.room = context.room
        return ctx

    def _orchestration_tools(self, room_id: str | None) -> list[AITool]:
        """The tools orchestration set up for every room and for *room_id*
        (RFC §19.7), declared in *room_id*'s turns."""
        entries = self._registry.entries(room_id, source=ToolSource.ORCHESTRATION)
        return [entry.definition for entry in entries]

    def _orchestration_tool_names(self, room_id: str | None) -> set[str]:
        """The tools declared at every round of *room_id*'s turns, outside the
        catalogue Tool Search measures (RFC §21.1)."""
        return self._registry.names(room_id, lambda traits: traits.always_declared)

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Intelligence channels are not called via deliver by the router."""
        return ChannelOutput.empty()

    @property
    def recent_events_window(self) -> int:
        """Recent-events need = this channel's memory provider's window."""
        return self._memory.recent_events_window

    async def close(self) -> None:
        """Close the channel: its running turns first (their calls cancelled
        and reported, RFC §9.3), then its provider, memory, and executors."""
        await self._end_running_turns()
        await self._close_minds()
        await self._human_input.close(self.channel_id)
        await super().close()
        await self._memory.close()
        if self._script_executor is not None:
            await self._script_executor.close()
        if self._sandbox is not None:
            await self._sandbox.close()

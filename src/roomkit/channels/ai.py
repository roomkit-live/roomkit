"""AI channel implementation.

This module defines :class:`AIChannel`, the intelligence channel that generates
responses using an AI provider.  Behaviour is composed from focused mixins:

- :class:`~._ai_events.AIEventsMixin` — ephemeral tool/thinking events
- :class:`~._ai_steering.AISteeringMixin` — mid-run steering directives
- :class:`~._ai_policy.AIToolPolicyMixin` — tool policy & skill gating
- :class:`~._ai_resilience.AIResilienceMixin` — retry / fallback / compaction
- :class:`~._ai_context.AIContextMixin` — AI context building
- :class:`~._ai_tools.AIToolsMixin` — tool execution & dispatch
- :class:`~._ai_generation.AIGenerationMixin` — non-streaming generation
- :class:`~._ai_streaming.AIStreamingMixin` — streaming generation
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_context import AIContextMixin
from roomkit.channels._ai_events import AIEventsMixin
from roomkit.channels._ai_generation import AIGenerationMixin
from roomkit.channels._ai_policy import AIToolPolicyMixin
from roomkit.channels._ai_resilience import AIResilienceMixin
from roomkit.channels._ai_steering import AISteeringMixin
from roomkit.channels._ai_streaming import AIStreamingMixin
from roomkit.channels._ai_tools import AIToolsMixin
from roomkit.channels._task_planner import TaskPlanner
from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._tool_search_constants import (
    DEFAULT_TOOL_SEARCH_THRESHOLD,
    DEFAULT_TOOL_SEARCH_THRESHOLD_PCT,
)
from roomkit.channels._tool_usage import ToolUsageMemory
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
from roomkit.models.event import RoomEvent
from roomkit.models.steering import SteeringDirective
from roomkit.providers.ai.base import (
    AIImagePart,
    AIProvider,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.realtime.base import RealtimeBackend
from roomkit.tools.policy import ToolPolicy

if TYPE_CHECKING:
    from roomkit.models.tool_call import ToolCallCallback
    from roomkit.sandbox.executor import SandboxExecutor
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry
    from roomkit.tools.base import Tool
    from roomkit.tools.external import ExternalToolHandler
    from roomkit.tools.human_input import HumanInputToolHandler

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

# Content part union — matches AIMessage.content list type
_ContentPart = AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart

logger = logging.getLogger("roomkit.channels.ai")


@dataclass
class _ToolLoopContext:
    """Per-invocation state for a tool loop, scoped via contextvar."""

    activated_skills: set[str] = field(default_factory=set)
    # Tool Search: names revealed by ``find_tools`` this loop. Accrues during
    # the loop (NOT inherited across for_loop) exactly like ``activated_skills``
    # — round 0 starts empty, a find_tools call reveals matches, and the next
    # round's tool re-filter shows them.
    revealed_tools: set[str] = field(default_factory=set)
    # Tools the agent already CALLED this conversation, seeded once per turn in
    # ``_build_context`` from ToolUsageMemory. Unlike ``revealed_tools`` (per-loop
    # find_tools discovery, starts empty), these are INHERITED across for_loop and
    # unioned into the Tool Search keep-set by ``_apply_tool_filters`` — so a tool
    # used once stays callable without the model having to re-run find_tools.
    sticky_tools: set[str] = field(default_factory=set)
    all_context_tools: list[Any] = field(default_factory=list)
    # Whether Tool Search is active for this turn (catalogue over threshold).
    # Decided once in ``_build_context`` and read by ``_apply_tool_filters`` on
    # every round, so it is inherited across for_loop like ``all_context_tools``.
    tool_search_active: bool = False
    current_participant_role: str | None = None
    room_id: str | None = None
    steering_queue: asyncio.Queue[SteeringDirective] = field(default_factory=asyncio.Queue)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    loop_id: str = ""

    @classmethod
    def for_loop(cls, parent: _ToolLoopContext | None, room_id: str | None) -> _ToolLoopContext:
        """Create a tool-loop context inheriting per-turn state from *parent*.

        _build_context ran under the parent (handle_event) ctx and stamped the
        turn's full toolset there — without this inheritance the per-round
        tools re-application never fires (skill-gated tools would stay hidden
        after activation) and per-call allowlist accessors see nothing.
        """
        ctx = cls()
        ctx.loop_id = str(id(ctx))
        if parent is not None:
            ctx.current_participant_role = parent.current_participant_role
            ctx.all_context_tools = parent.all_context_tools
            ctx.tool_search_active = parent.tool_search_active
            # Carry the used-tools re-exposition seeded in _build_context into the
            # loop: the per-round re-filter runs under THIS child ctx, so without
            # this the seed is dropped at round 0 and the model must re-find_tools.
            ctx.sticky_tools = set(parent.sticky_tools)
        ctx.room_id = room_id or (parent.room_id if parent else None)
        return ctx


_current_loop_ctx: contextvars.ContextVar[_ToolLoopContext | None] = contextvars.ContextVar(
    "_current_loop_ctx", default=None
)


# Corrective nudge re-injected when a generation round ends after tool calls
# without any final text (common with small local models): the tool results
# are in context, the model just failed to verbalize the answer. Re-prompting
# for the final answer recovers it. Bounded by ``max_empty_retries``.
_EMPTY_RETRY_NUDGE = (
    "You called tools and already have their results above. Now write your "
    "final answer to the user in plain text. Do not call any more tools."
)


class AIChannel(
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
        max_tokens: int = 1024,
        max_context_events: int = 50,
        tool_handler: ToolHandler | None = None,
        tools: list[AITool | Tool] | None = None,
        max_tool_rounds: int = 200,
        tool_loop_timeout_seconds: float | None = 300.0,
        tool_loop_warn_after: int = 50,
        max_empty_retries: int = 1,
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
        evict_threshold_tokens: int = 5000,
        enable_planning: bool = False,
        config_provider: ConfigProvider | None = None,
        tool_search: bool | None = None,
        tool_search_pinned: list[str] | None = None,
        tool_search_threshold: int = DEFAULT_TOOL_SEARCH_THRESHOLD,
        tool_search_threshold_pct: float = DEFAULT_TOOL_SEARCH_THRESHOLD_PCT,
    ) -> None:
        super().__init__(channel_id)
        self._provider = provider
        self._system_prompt = system_prompt
        # Per-turn config resolution — see channels/_turn_config.py. When
        # set, system prompt / tools / sampling are resolved fresh at the
        # start of every turn instead of living as attach-time snapshots.
        self._config_provider = config_provider
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_context_events = max_context_events
        self._thinking_budget = thinking_budget
        self._max_tool_rounds = max_tool_rounds
        self._tool_loop_timeout_seconds = tool_loop_timeout_seconds
        self._tool_loop_warn_after = tool_loop_warn_after
        self._max_empty_retries = max_empty_retries
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
        self._memory = memory or SlidingWindowMemory(max_events=max_context_events)
        self._tool_policy = tool_policy
        self._eviction = ToolEviction(threshold_tokens=evict_threshold_tokens)
        # Per-conversation record of tools the agent has called — feeds the
        # "tools you've already used" digest and re-reveals used tools each turn
        # so a tool used once stays callable under Tool Search. See _tool_usage.
        self._tool_usage = ToolUsageMemory()
        self._planner = TaskPlanner() if enable_planning else None
        # Tool Search — progressive tool disclosure for large catalogues.
        # ``None`` auto-enables when the deferrable tools would exceed
        # ``tool_search_threshold_pct`` % of the model's context window (so it
        # self-tunes: a big model is a no-op, a small one defers early); when
        # the window is unknown it falls back to the ``tool_search_threshold``
        # tool count. True/False force on/off. Unlike the realtime channel
        # (which pushes matches via provider.reconfigure), the text loop
        # re-sends its tool list every round, so revealing a tool is just a
        # per-round re-filter — no provider capability is required. See
        # ``_should_activate_tool_search``.
        self._tool_search = tool_search
        self._tool_search_pinned: set[str] = set(tool_search_pinned or [])
        self._tool_search_threshold = tool_search_threshold
        self._tool_search_threshold_pct = tool_search_threshold_pct

        # Extract Tool objects: split into AITool definitions + composed handler
        from roomkit.tools.compose import extract_tools

        extracted_defs: list[AITool] = []
        extracted_handler: ToolHandler | None = None
        if tools:
            extracted_defs, extracted_handler = extract_tools(list(tools))

        # Merge explicit tool_handler with handlers extracted from Tool objects
        effective_handler = tool_handler
        if extracted_handler and tool_handler:
            from roomkit.tools.compose import compose_tool_handlers

            effective_handler = compose_tool_handlers(tool_handler, extracted_handler)
        elif extracted_handler:
            effective_handler = extracted_handler

        # Human-input handler: compose first (highest priority) so it
        # intercepts matching tools before the user handler chain.
        self._human_input_handler = human_input_handler
        if human_input_handler:
            from roomkit.tools.compose import compose_tool_handlers

            if effective_handler:
                effective_handler = compose_tool_handlers(human_input_handler, effective_handler)
            else:
                effective_handler = human_input_handler

        # Store the user/orchestration tool handler separately; all dispatch
        # goes through _channel_tool_handler which routes to channel-managed
        # tools (eviction, planning), skill tools, then user tools.
        self._user_tool_handler = effective_handler
        # Set _tool_handler to the unified dispatcher only when tools actually
        # exist.  Keeping it None when no tools are configured preserves the
        # "no tools" fast-path guard in the tool loop (lines that check
        # ``self._tool_handler is None``).
        has_any_tools = bool(
            extracted_defs
            or effective_handler
            or (skills and skills.skill_count > 0)
            or enable_planning
            or sandbox is not None
        )
        self._tool_handler: ToolHandler | None = (
            self._channel_tool_handler if has_any_tools else None
        )

        # User-provided tools (from constructor) and orchestration-injected
        # tools (e.g. HANDOFF_TOOL, DELEGATE_TOOL) are kept separate so that
        # orchestration code can inspect/modify injected tools independently.
        self._user_tools: list[AITool] = extracted_defs
        self._injected_tools: list[AITool] = []

        # Active tool loops for steering (loop_id -> context)
        self._active_loops: dict[str, _ToolLoopContext] = {}

        # Realtime backend for ephemeral tool call events (set by register_channel)
        self._realtime: RealtimeBackend | None = None
        # Unified tool call hook callback (injected by framework on register_channel)
        self._tool_call_hook: ToolCallCallback | None = None
        # Pre-tool-use hook callback (injected by framework on register_channel)
        self._before_tool_call_hook: Any = None
        # AI response hook callback (injected by framework on register_channel)
        self._after_response_hook: Any = None
        # BEFORE_AI_GENERATION hook callback (injected by framework on register_channel)
        self._before_generation_hook: Any = None
        # Current room ID (set per on_event invocation for tool hook context)
        self._current_room_id: str | None = None
        # External tool handler for provider-executed tools (e.g. Claude Code)
        self._external_tool_handler = external_tool_handler

    @property
    def tool_handler(self) -> ToolHandler | None:
        """The current tool handler (may be wrapped by orchestration)."""
        return self._tool_handler

    @tool_handler.setter
    def tool_handler(self, value: ToolHandler | None) -> None:
        self._tool_handler = value

    @property
    def extra_tools(self) -> list[AITool]:
        """All extra tools (user-provided + orchestration-injected)."""
        return self._user_tools + self._injected_tools

    def _propagate_telemetry(self) -> None:
        """Propagate telemetry to AI provider."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            self._provider._telemetry = telemetry

    @property
    def info(self) -> dict[str, Any]:
        return {"provider": type(self._provider).__name__}

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
        """React to an event by generating an AI response.

        Skips events from this channel to prevent self-loops.
        When the provider supports streaming or structured streaming:
        - With tools: uses the streaming tool loop that executes tool calls
          between generation rounds while yielding text deltas progressively.
        - Without tools: returns a plain streaming response.
        Otherwise falls back to the non-streaming generate path.
        """
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()

        # Skip tool call events — these are activity records, not messages
        # to respond to. Prevents multi-agent rooms from generating
        # spurious responses to another agent's tool calls.
        if event.type in (EventType.TOOL_CALL_START, EventType.TOOL_CALL_END):
            return ChannelOutput.empty()

        # Track current room for tool call hooks
        self._current_room_id = context.room.id if context.room else None

        # Ingest event into memory provider (enables stateful providers
        # like vector stores to index content as it arrives).
        _ingest_room_id = context.room.id if context.room else event.room_id
        if _ingest_room_id:
            try:
                await self._memory.ingest(_ingest_room_id, event, channel_id=self.channel_id)
            except Exception:
                logger.warning("Memory ingestion failed", exc_info=True)

        # Resolve participant role for role-based tool policy.
        # Set on a per-invocation _ToolLoopContext visible via contextvar so that
        # _build_context and the tool loop methods can read it. ``room_id``
        # rides the same contextvar: the channel object is registered once
        # per channel_id and shared by every room it serves, so per-call
        # room resolution (``current_tool_room_id``) is the only safe way
        # for tool handlers to learn the originating room.
        event_ctx = _ToolLoopContext()
        event_ctx.current_participant_role = self._resolve_participant_role(event, context)
        event_ctx.room_id = context.room.id if context.room else event.room_id
        token = _current_loop_ctx.set(event_ctx)
        try:
            raw_tools = binding.metadata.get("tools", [])
            # A config_provider may deliver tools at _build_context time even
            # when the binding carries no snapshot — route to the tool loop
            # so those tools are executable. An empty turn toolset just runs
            # the loop for a single round.
            has_tools = (
                bool(raw_tools)
                or self._config_provider is not None
                or bool(self._user_tools)
                or bool(self._injected_tools)
                or (self._skills is not None and self._skills.skill_count > 0)
                or self._planner is not None
                or self._sandbox is not None
                or self._external_tool_handler is not None
                or (
                    self._human_input_handler is not None and bool(self._human_input_handler.tools)
                )
            )

            if self._provider.supports_streaming or self._provider.supports_structured_streaming:
                if has_tools:
                    return await self._start_streaming_tool_response(event, binding, context)
                return await self._start_streaming_response(event, binding, context)

            return await self._generate_response(event, binding, context)
        finally:
            _current_loop_ctx.reset(token)

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
        """Close the channel, its provider, memory, and executors."""
        await super().close()
        await self._memory.close()
        if self._script_executor is not None:
            await self._script_executor.close()
        if self._sandbox is not None:
            await self._sandbox.close()

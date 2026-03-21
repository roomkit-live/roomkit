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
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry
    from roomkit.tools.base import Tool

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

# Content part union — matches AIMessage.content list type
_ContentPart = AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart

logger = logging.getLogger("roomkit.channels.ai")

# Skill tool names
_TOOL_ACTIVATE_SKILL = "activate_skill"
_TOOL_READ_REFERENCE = "read_skill_reference"
_TOOL_RUN_SCRIPT = "run_skill_script"

_SKILLS_PREAMBLE = (
    "You have access to Agent Skills — specialized knowledge packages. "
    "Use the activate_skill tool to load a skill's full instructions before "
    "using it. Available skills are listed below."
)

_SKILLS_NO_SCRIPTS_NOTE = " Note: Script execution is not available in this environment."


@dataclass
class _ToolLoopContext:
    """Per-invocation state for a tool loop, scoped via contextvar."""

    activated_skills: set[str] = field(default_factory=set)
    all_context_tools: list[Any] = field(default_factory=list)
    current_participant_role: str | None = None
    steering_queue: asyncio.Queue[SteeringDirective] = field(default_factory=asyncio.Queue)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    loop_id: str = ""


_current_loop_ctx: contextvars.ContextVar[_ToolLoopContext | None] = contextvars.ContextVar(
    "_current_loop_ctx", default=None
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
        retry_policy: RetryPolicy | None = None,
        fallback_provider: AIProvider | None = None,
        skills: SkillRegistry | None = None,
        script_executor: ScriptExecutor | None = None,
        memory: MemoryProvider | None = None,
        tool_policy: ToolPolicy | None = None,
        thinking_budget: int | None = None,
        evict_threshold_tokens: int = 5000,
        enable_planning: bool = False,
    ) -> None:
        super().__init__(channel_id)
        self._provider = provider
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_context_events = max_context_events
        self._thinking_budget = thinking_budget
        self._max_tool_rounds = max_tool_rounds
        self._tool_loop_timeout_seconds = tool_loop_timeout_seconds
        self._tool_loop_warn_after = tool_loop_warn_after
        self._retry_policy = retry_policy
        self._fallback_provider = fallback_provider
        self._skills = skills
        self._script_executor = script_executor
        self._memory = memory or SlidingWindowMemory(max_events=max_context_events)
        self._tool_policy = tool_policy
        self._eviction = ToolEviction(threshold_tokens=evict_threshold_tokens)
        self._planner = TaskPlanner() if enable_planning else None

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
        # AI response hook callback (injected by framework on register_channel)
        self._after_response_hook: Any = None
        # Current room ID (set per on_event invocation for tool hook context)
        self._current_room_id: str | None = None

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
        # _build_context and the tool loop methods can read it.
        event_ctx = _ToolLoopContext()
        event_ctx.current_participant_role = self._resolve_participant_role(event, context)
        token = _current_loop_ctx.set(event_ctx)
        try:
            raw_tools = binding.metadata.get("tools", [])
            has_tools = (
                bool(raw_tools)
                or bool(self._user_tools)
                or bool(self._injected_tools)
                or (self._skills is not None and self._skills.skill_count > 0)
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

    async def close(self) -> None:
        """Close the channel, its provider, and the memory provider."""
        await super().close()
        await self._memory.close()

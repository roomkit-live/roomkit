"""AIChannel mixin for building the AI provider context from room events."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._dangling_recovery import patch_dangling_tool_calls
from roomkit.channels._skill_constants import (
    SKILLS_NO_SCRIPTS_NOTE as _SKILLS_NO_SCRIPTS_NOTE,
)
from roomkit.channels._skill_constants import (
    SKILLS_PREAMBLE as _SKILLS_PREAMBLE,
)
from roomkit.channels._task_planner import TaskPlanner
from roomkit.channels._tool_eviction import ToolEviction
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import CompositeContent, MediaContent, TextContent
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AITool,
)

if TYPE_CHECKING:
    from roomkit.channels.ai import _ContentPart, _ToolLoopContext
    from roomkit.memory.base import MemoryProvider
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.providers.ai.base import AIProvider
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry

logger = logging.getLogger("roomkit.channels.ai")


@runtime_checkable
class AIContextHost(Protocol):
    """Contract: capabilities a host class must provide for AIContextMixin.

    Attributes provided by the host's ``__init__``:
        _provider: AI provider for generation and capability queries.
        _system_prompt: Default system prompt (overridable per room).
        _temperature: Default temperature (overridable per room).
        _max_tokens: Default max tokens (overridable per room).
        _thinking_budget: Optional thinking budget for extended thinking.
        _skills: Skill registry for tool injection.
        _script_executor: Script executor for skill scripts.
        _memory: Memory provider for conversation retrieval.
        _eviction: Tool result eviction / truncation strategy.
        _planner: Optional task planner for planning tools.
        _user_tools: User-provided tool definitions.
        _injected_tools: Orchestration-injected tool definitions.
        channel_id: Unique identifier for this channel.

    Properties / methods provided by other mixins:
        extra_tools: ``AIChannel`` property returning user + injected tools.
        _skill_tools: ``AIToolsMixin`` — builds skill tool definitions.
        _apply_tool_filters: ``AIToolPolicyMixin`` — applies policy + gating.
        _get_loop_ctx: ``AISteeringMixin`` — returns the current tool-loop context.
    """

    _provider: AIProvider
    _system_prompt: str | None
    _temperature: float
    _max_tokens: int
    _thinking_budget: int | None
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _memory: MemoryProvider
    _eviction: ToolEviction
    _planner: TaskPlanner | None
    _user_tools: list[AITool]
    _injected_tools: list[AITool]
    channel_id: str

    @property
    def extra_tools(self) -> list[AITool]: ...
    def _skill_tools(self) -> list[AITool]: ...
    def _apply_tool_filters(self, tools: list[AITool]) -> list[AITool]: ...
    def _get_loop_ctx(self) -> _ToolLoopContext: ...


class AIContextMixin:
    """Builds the AIContext passed to the provider from room state and events.

    Host contract: :class:`AIContextHost`.
    """

    _provider: AIProvider
    _system_prompt: str | None
    _temperature: float
    _max_tokens: int
    _thinking_budget: int | None
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _memory: MemoryProvider
    _eviction: ToolEviction
    _planner: TaskPlanner | None
    _user_tools: list[AITool]
    _injected_tools: list[AITool]
    channel_id: str

    # Cross-mixin methods — Any annotations avoid MRO shadowing
    extra_tools: Any  # see AIContextHost
    _skill_tools: Any  # see AIContextHost
    _apply_tool_filters: Any  # see AIContextHost
    _get_loop_ctx: Any  # see AIContextHost

    async def _build_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext:
        """Build AI context from room events.

        Per-room overrides can be set via binding.metadata:
        - system_prompt: Override the channel's default system prompt
        - temperature: Override the channel's default temperature
        - max_tokens: Override the channel's default max_tokens
        - tools: List of tool definitions for function calling
        """
        # Per-room overrides from binding metadata
        system_prompt = binding.metadata.get("system_prompt", self._system_prompt)
        temperature = binding.metadata.get("temperature", self._temperature)
        max_tokens = binding.metadata.get("max_tokens", self._max_tokens)
        thinking_budget = binding.metadata.get("thinking_budget", self._thinking_budget)
        raw_tools = binding.metadata.get("tools", [])

        # Convert raw tool dicts to AITool instances
        tools = [
            AITool(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
            )
            for t in raw_tools
        ]

        # Inject extra tools (user-provided + orchestration handoff, etc.)
        tools.extend(self.extra_tools)

        # Inject skill tools and prompt (infra tools added here, gated tools later)
        if self._skills and self._skills.skill_count > 0:
            tools.extend(self._skill_tools())
            preamble = _SKILLS_PREAMBLE
            if not self._script_executor:
                preamble += _SKILLS_NO_SCRIPTS_NOTE
            skills_xml = self._skills.to_prompt_xml()
            skill_block = f"\n\n{preamble}\n\n{skills_xml}"
            system_prompt = (system_prompt or "") + skill_block

        # Inject eviction re-read tool when large results have been stored
        if self._eviction.has_evicted:
            tools.append(ToolEviction.tool_definition())

        # Inject planning tool and plan context when enabled
        if self._planner is not None:
            tools.append(TaskPlanner.tool_definition())
            if self._planner.current_plan:
                system_prompt = (system_prompt or "") + TaskPlanner.format_plan_prompt(
                    self._planner.current_plan
                )

        # Store unfiltered tool list for re-application after skill activation
        loop_ctx = self._get_loop_ctx()
        loop_ctx.all_context_tools = list(tools)

        # Apply tool policy + skill gating visibility filters
        tools = self._apply_tool_filters(tools)

        # Retrieve memory
        memory_result = await self._memory.retrieve(
            event.room_id,
            event,
            context,
            channel_id=self.channel_id,
        )

        messages: list[AIMessage] = []

        # Pre-built messages from memory (e.g. summaries)
        messages.extend(memory_result.messages)

        # Convert memory events using AIChannel content extraction
        for past_event in memory_result.events:
            role = self._determine_role(past_event)
            content = self._extract_content(past_event)
            if content:
                messages.append(AIMessage(role=role, content=content))

        # Patch orphaned tool calls from interrupted tool loops (barge-in)
        messages = patch_dangling_tool_calls(messages)

        # Add current event
        content = self._extract_content(event)
        if content:
            messages.append(AIMessage(role="user", content=content))

        # Determine target channel capabilities for capability-aware generation
        # Use intersection of all transport bindings' media types (weakest common)
        transport_bindings = [
            b
            for b in context.bindings
            if b.category == ChannelCategory.TRANSPORT and b.channel_id != self.channel_id
        ]
        if transport_bindings:
            common_types = set(transport_bindings[0].capabilities.media_types)
            for b in transport_bindings[1:]:
                common_types &= set(b.capabilities.media_types)
            target_media = list(common_types)
            # Intersect capabilities: AND for booleans, MIN for numeric limits
            caps0 = transport_bindings[0].capabilities
            merged = caps0.model_dump()
            merged["media_types"] = target_media
            for b in transport_bindings[1:]:
                other = b.capabilities
                for field_name in (
                    "supports_threading",
                    "supports_reactions",
                    "supports_edit",
                    "supports_delete",
                    "supports_read_receipts",
                    "supports_typing",
                    "supports_templates",
                    "supports_rich_text",
                    "supports_buttons",
                    "supports_cards",
                    "supports_quick_replies",
                    "supports_media",
                    "supports_audio",
                    "supports_video",
                ):
                    merged[field_name] = merged[field_name] and getattr(other, field_name)
                for field_name in (
                    "max_length",
                    "max_buttons",
                    "max_media_size_bytes",
                    "max_audio_duration_seconds",
                    "max_video_duration_seconds",
                ):
                    a, b_val = merged[field_name], getattr(other, field_name)
                    if a is not None and b_val is not None:
                        merged[field_name] = min(a, b_val)
                    elif b_val is not None:
                        merged[field_name] = b_val
            target_caps = ChannelCapabilities(**merged)
        else:
            target_media = []
            target_caps = None

        return AIContext(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            tools=tools,
            room=context,
            target_capabilities=target_caps,
            target_media_types=target_media,
        )

    def _determine_role(self, event: RoomEvent) -> str:
        if event.source.channel_id == self.channel_id:
            return "assistant"
        return "user"

    def _extract_content(
        self,
        event: RoomEvent,
    ) -> str | list[_ContentPart]:
        """Extract content, including images if provider supports vision."""
        content = event.content

        if not self._provider.supports_vision:
            # Text-only fallback (existing behavior)
            return self._extract_text(event)

        # Build multimodal content
        if isinstance(content, TextContent):
            return content.body  # Simple case: just text

        if isinstance(content, MediaContent):
            parts: list[_ContentPart] = []
            if content.caption:
                parts.append(AITextPart(text=content.caption))
            parts.append(AIImagePart(url=content.url, mime_type=content.mime_type))
            return parts

        if isinstance(content, CompositeContent):
            cparts: list[_ContentPart] = []
            for part in content.parts:
                if isinstance(part, TextContent):
                    cparts.append(AITextPart(text=part.body))
                elif isinstance(part, MediaContent):
                    if part.caption:
                        cparts.append(AITextPart(text=part.caption))
                    cparts.append(AIImagePart(url=part.url, mime_type=part.mime_type))
            return cparts if cparts else ""

        # Fallback for other types
        return self._extract_text(event)

    def _extract_text(self, event: RoomEvent) -> str:
        if isinstance(event.content, TextContent):
            return event.content.body
        return ""

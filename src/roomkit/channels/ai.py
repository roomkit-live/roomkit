"""AI channel implementation."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from roomkit.channels.base import Channel
from roomkit.memory.base import MemoryProvider
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.channel import (
    ChannelBinding,
    ChannelCapabilities,
    ChannelOutput,
)
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
)
from roomkit.models.event import (
    CompositeContent,
    EventSource,
    MediaContent,
    RoomEvent,
    TextContent,
)
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.noop import NoopTelemetryProvider

if TYPE_CHECKING:
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

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


class AIChannel(Channel):
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
        max_tool_rounds: int = 10,
        skills: SkillRegistry | None = None,
        script_executor: ScriptExecutor | None = None,
        memory: MemoryProvider | None = None,
    ) -> None:
        super().__init__(channel_id)
        self._provider = provider
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_context_events = max_context_events
        self._max_tool_rounds = max_tool_rounds
        self._skills = skills
        self._script_executor = script_executor
        self._memory = memory or SlidingWindowMemory(max_events=max_context_events)

        # Wrap the user's tool handler with skill-aware dispatch
        self._user_tool_handler = tool_handler
        self._tool_handler: ToolHandler | None
        if skills and skills.skill_count > 0:
            self._tool_handler = self._skill_aware_tool_handler
        else:
            self._tool_handler = tool_handler

    def _propagate_telemetry(self) -> None:
        """Propagate telemetry to AI provider."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            self._provider._telemetry = telemetry

    async def close(self) -> None:
        """Close the channel, its provider, and the memory provider."""
        await super().close()
        await self._memory.close()

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

        raw_tools = binding.metadata.get("tools", [])
        has_tools = bool(raw_tools) or (self._skills is not None and self._skills.skill_count > 0)

        if self._provider.supports_streaming or self._provider.supports_structured_streaming:
            if has_tools:
                return await self._start_streaming_tool_response(event, binding, context)
            return await self._start_streaming_response(event, binding, context)

        return await self._generate_response(event, binding, context)

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Intelligence channels are not called via deliver by the router."""
        return ChannelOutput.empty()

    async def _start_streaming_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response handle (generator starts on consumption)."""
        ai_context = await self._build_context(event, binding, context)
        return ChannelOutput(
            responded=True,
            response_stream=self._provider.generate_stream(ai_context),
        )

    async def _start_streaming_tool_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response that handles tool calls between rounds."""
        ai_context = await self._build_context(event, binding, context)
        return ChannelOutput(
            responded=True,
            response_stream=self._run_streaming_tool_loop(ai_context),
        )

    async def _run_streaming_tool_loop(self, context: AIContext) -> AsyncIterator[str]:
        """Stream text deltas, executing tool calls between generation rounds."""
        telemetry = self._telemetry_provider

        for _round_idx in range(self._max_tool_rounds + 1):
            text_parts: list[str] = []
            tool_calls: list[StreamToolCall] = []

            async for event in self._provider.generate_structured_stream(context):
                if isinstance(event, StreamTextDelta):
                    text_parts.append(event.text)
                    yield event.text
                elif isinstance(event, StreamToolCall):
                    tool_calls.append(event)

            if not tool_calls or self._tool_handler is None:
                return

            # Don't execute tools on the final iteration — no generation follows
            if _round_idx >= self._max_tool_rounds:
                logger.warning(
                    "Streaming tool loop reached max_tool_rounds=%d",
                    self._max_tool_rounds,
                )
                return

            logger.info(
                "Streaming tool round %d: %d call(s)",
                _round_idx + 1,
                len(tool_calls),
            )

            # Append assistant message with text + tool calls
            parts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
            accumulated_text = "".join(text_parts)
            if accumulated_text:
                parts.append(AITextPart(text=accumulated_text))
            for tc in tool_calls:
                parts.append(AIToolCallPart(id=tc.id, name=tc.name, arguments=tc.arguments))
            context.messages.append(AIMessage(role="assistant", content=parts))

            # Execute each tool
            result_parts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
            for tc in tool_calls:
                logger.info("Executing tool: %s(%s)", tc.name, tc.id)
                tool_span_id = telemetry.start_span(
                    SpanKind.LLM_TOOL_CALL,
                    f"tool.{tc.name}",
                    attributes={"tool.name": tc.name, "tool.id": tc.id},
                )
                try:
                    result = await self._tool_handler(tc.name, tc.arguments)
                    telemetry.end_span(tool_span_id)
                except Exception as exc:
                    telemetry.end_span(tool_span_id, status="error", error_message=str(exc))
                    raise
                result_parts.append(
                    AIToolResultPart(
                        tool_call_id=tc.id,
                        name=tc.name,
                        result=result,
                    )
                )

            context.messages.append(AIMessage(role="tool", content=result_parts))

    @property
    def _telemetry_provider(self) -> NoopTelemetryProvider:
        """Access telemetry provider (set by register_channel)."""
        return getattr(self, "_telemetry", None) or NoopTelemetryProvider()

    async def _generate_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Generate an AI response, executing tool calls if needed."""
        from roomkit.telemetry.context import get_current_span

        ai_context = await self._build_context(event, binding, context)
        telemetry = self._telemetry_provider
        span_id = telemetry.start_span(
            SpanKind.LLM_GENERATE,
            "llm.generate",
            parent_id=get_current_span(),
            room_id=event.room_id,
            channel_id=self.channel_id,
            attributes={
                Attr.PROVIDER: type(self._provider).__name__,
                Attr.LLM_STREAMING: False,
            },
        )
        try:
            response = await self._run_tool_loop(ai_context, parent_span_id=span_id)
        except ProviderError as exc:
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            if exc.status_code == 404:
                logger.error(
                    "AI model not found (channel=%s, provider=%s): %s",
                    self.channel_id,
                    exc.provider,
                    exc,
                )
            elif exc.status_code and exc.status_code >= 500:
                logger.error(
                    "AI provider server error (channel=%s, provider=%s, status=%s): %s",
                    self.channel_id,
                    exc.provider,
                    exc.status_code,
                    exc,
                )
            else:
                logger.exception(
                    "AI provider error for channel %s",
                    self.channel_id,
                    extra={
                        "provider": exc.provider,
                        "retryable": exc.retryable,
                        "status_code": exc.status_code,
                    },
                )
            return ChannelOutput.empty()
        except Exception:
            telemetry.end_span(span_id, status="error", error_message="AI provider failed")
            logger.exception("AI provider failed for channel %s", self.channel_id)
            return ChannelOutput.empty()

        # End LLM span with usage attributes
        usage = response.usage or {}
        telemetry.end_span(
            span_id,
            attributes={
                Attr.LLM_INPUT_TOKENS: usage.get("input_tokens", 0),
                Attr.LLM_OUTPUT_TOKENS: usage.get("output_tokens", 0),
                Attr.LLM_TOOL_COUNT: len(response.tool_calls) if response.tool_calls else 0,
            },
        )

        response_event = RoomEvent(
            room_id=event.room_id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                provider=self.provider_name,
            ),
            content=TextContent(body=response.content),
            chain_depth=event.chain_depth + 1,
            metadata={"ai_usage": response.usage},
        )

        return ChannelOutput(
            responded=True,
            response_events=[response_event],
        )

    async def _run_tool_loop(
        self, context: AIContext, *, parent_span_id: str | None = None
    ) -> AIResponse:
        """Generate → execute tools → re-generate until a text response."""
        response: AIResponse = await self._provider.generate(context)
        telemetry = self._telemetry_provider

        for round_idx in range(self._max_tool_rounds):
            if not response.tool_calls or self._tool_handler is None:
                break

            logger.info(
                "Tool round %d: %d call(s)",
                round_idx + 1,
                len(response.tool_calls),
            )

            # Append assistant message with tool calls
            parts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
            if response.content:
                parts.append(AITextPart(text=response.content))
            for tc in response.tool_calls:
                parts.append(AIToolCallPart(id=tc.id, name=tc.name, arguments=tc.arguments))
            context.messages.append(AIMessage(role="assistant", content=parts))

            # Execute each tool and collect results
            result_parts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
            for tc in response.tool_calls:
                logger.info("Executing tool: %s(%s)", tc.name, tc.id)
                tool_span_id = telemetry.start_span(
                    SpanKind.LLM_TOOL_CALL,
                    f"tool.{tc.name}",
                    parent_id=parent_span_id,
                    attributes={"tool.name": tc.name, "tool.id": tc.id},
                )
                try:
                    result = await self._tool_handler(tc.name, tc.arguments)
                    telemetry.end_span(tool_span_id)
                except Exception as exc:
                    telemetry.end_span(tool_span_id, status="error", error_message=str(exc))
                    raise
                result_parts.append(
                    AIToolResultPart(
                        tool_call_id=tc.id,
                        name=tc.name,
                        result=result,
                    )
                )

            context.messages.append(AIMessage(role="tool", content=result_parts))

            # Re-generate with tool results
            response = await self._provider.generate(context)

        return response

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

        # Inject skill tools and prompt
        if self._skills and self._skills.skill_count > 0:
            tools.extend(self._skill_tools())
            preamble = _SKILLS_PREAMBLE
            if not self._script_executor:
                preamble += _SKILLS_NO_SCRIPTS_NOTE
            skills_xml = self._skills.to_prompt_xml()
            skill_block = f"\n\n{preamble}\n\n{skills_xml}"
            system_prompt = (system_prompt or "") + skill_block

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
            target_caps = transport_bindings[0].capabilities
        else:
            target_media = []
            target_caps = None

        return AIContext(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
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
    ) -> str | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart]:
        """Extract content, including images if provider supports vision."""
        content = event.content

        if not self._provider.supports_vision:
            # Text-only fallback (existing behavior)
            return self._extract_text(event)

        # Build multimodal content
        if isinstance(content, TextContent):
            return content.body  # Simple case: just text

        if isinstance(content, MediaContent):
            parts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
            if content.caption:
                parts.append(AITextPart(text=content.caption))
            parts.append(AIImagePart(url=content.url, mime_type=content.mime_type))
            return parts

        if isinstance(content, CompositeContent):
            cparts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
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

    # -- Skill integration --------------------------------------------------

    def _skill_tools(self) -> list[AITool]:
        """Build the list of AITool definitions for skill operations."""
        tools = [
            AITool(
                name=_TOOL_ACTIVATE_SKILL,
                description=(
                    "Activate a skill to get its full instructions, "
                    "available scripts, and reference files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the skill to activate.",
                        },
                    },
                    "required": ["name"],
                },
            ),
            AITool(
                name=_TOOL_READ_REFERENCE,
                description="Read a reference file from a skill.",
                parameters={
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill.",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Reference filename to read.",
                        },
                    },
                    "required": ["skill_name", "filename"],
                },
            ),
        ]
        if self._script_executor:
            tools.append(
                AITool(
                    name=_TOOL_RUN_SCRIPT,
                    description="Run a script from a skill's scripts/ directory.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "Name of the skill.",
                            },
                            "script_name": {
                                "type": "string",
                                "description": "Script filename to run.",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Optional key-value arguments.",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["skill_name", "script_name"],
                    },
                )
            )
        return tools

    async def _skill_aware_tool_handler(self, name: str, arguments: dict[str, Any]) -> str:
        """Intercept skill tool calls; delegate the rest to user handler."""
        if name == _TOOL_ACTIVATE_SKILL:
            return await self._handle_activate_skill(arguments)
        if name == _TOOL_READ_REFERENCE:
            return await self._handle_read_reference(arguments)
        if name == _TOOL_RUN_SCRIPT:
            return await self._handle_run_script(arguments)
        if self._user_tool_handler:
            return await self._user_tool_handler(name, arguments)
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def _handle_activate_skill(self, arguments: dict[str, Any]) -> str:
        """Load and return full skill instructions."""
        skill_name = arguments.get("name", "")
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})

        skill = self._skills.get_skill(skill_name)
        if skill is None:
            available = self._skills.skill_names
            return json.dumps(
                {
                    "error": f"Skill {skill_name!r} not found",
                    "available_skills": available,
                }
            )

        result: dict[str, Any] = {
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.instructions,
        }
        scripts = skill.list_scripts()
        if scripts:
            result["scripts"] = scripts
        refs = skill.list_references()
        if refs:
            result["references"] = refs
        return json.dumps(result)

    async def _handle_read_reference(self, arguments: dict[str, Any]) -> str:
        """Read a reference file from a skill."""
        skill_name = arguments.get("skill_name", "")
        filename = arguments.get("filename", "")
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})

        skill = self._skills.get_skill(skill_name)
        if skill is None:
            return json.dumps({"error": f"Skill {skill_name!r} not found"})

        try:
            content = skill.read_reference(filename)
            return json.dumps({"filename": filename, "content": content})
        except (ValueError, FileNotFoundError) as exc:
            return json.dumps({"error": str(exc)})

    async def _handle_run_script(self, arguments: dict[str, Any]) -> str:
        """Execute a script via the configured ScriptExecutor."""
        skill_name = arguments.get("skill_name", "")
        script_name = arguments.get("script_name", "")
        script_args = arguments.get("arguments")
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        if not self._script_executor:
            return json.dumps({"error": "Script execution is not available"})

        skill = self._skills.get_skill(skill_name)
        if skill is None:
            return json.dumps({"error": f"Skill {skill_name!r} not found"})

        try:
            result = await self._script_executor.execute(skill, script_name, arguments=script_args)
            return result.model_dump_json()
        except Exception as exc:
            logger.exception("Script execution failed: %s/%s", skill_name, script_name)
            return json.dumps({"error": f"Script execution failed: {exc}"})

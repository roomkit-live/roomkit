"""AI channel implementation."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.channels.base import Channel
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
)

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

logger = logging.getLogger("roomkit.channels.ai")


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
    ) -> None:
        super().__init__(channel_id)
        self._provider = provider
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_context_events = max_context_events
        self._tool_handler = tool_handler
        self._max_tool_rounds = max_tool_rounds

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
        When the provider supports streaming and no tools are configured,
        returns a streaming response for the framework to pipe to channels.
        """
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()

        raw_tools = binding.metadata.get("tools", [])
        if self._provider.supports_streaming and not raw_tools:
            return self._start_streaming_response(event, binding, context)

        return await self._generate_response(event, binding, context)

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Intelligence channels are not called via deliver by the router."""
        return ChannelOutput.empty()

    def _start_streaming_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response handle (generator starts on consumption)."""
        ai_context = self._build_context(event, binding, context)
        return ChannelOutput(
            responded=True,
            response_stream=self._provider.generate_stream(ai_context),
        )

    async def _generate_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Generate an AI response, executing tool calls if needed."""
        ai_context = self._build_context(event, binding, context)
        try:
            response = await self._run_tool_loop(ai_context)
        except ProviderError as exc:
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
            logger.exception("AI provider failed for channel %s", self.channel_id)
            return ChannelOutput.empty()

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

    async def _run_tool_loop(self, context: AIContext) -> AIResponse:
        """Generate → execute tools → re-generate until a text response."""
        response = await self._provider.generate(context)

        for round_idx in range(self._max_tool_rounds):
            if not response.tool_calls or not self._tool_handler:
                break

            logger.info(
                "Tool round %d: %d call(s)",
                round_idx + 1,
                len(response.tool_calls),
            )

            # Append assistant message with tool calls
            parts: list[AITextPart | AIToolCallPart] = []
            if response.content:
                parts.append(AITextPart(text=response.content))
            for tc in response.tool_calls:
                parts.append(AIToolCallPart(id=tc.id, name=tc.name, arguments=tc.arguments))
            context.messages.append(AIMessage(role="assistant", content=parts))

            # Execute each tool and collect results
            result_parts: list[AIToolResultPart] = []
            for tc in response.tool_calls:
                logger.info("Executing tool: %s(%s)", tc.name, tc.id)
                result = await self._tool_handler(tc.name, tc.arguments)
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

    def _build_context(
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

        messages: list[AIMessage] = []

        for past_event in context.recent_events[-self._max_context_events :]:
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

    def _extract_content(self, event: RoomEvent) -> str | list[AITextPart | AIImagePart]:
        """Extract content, including images if provider supports vision."""
        content = event.content

        if not self._provider.supports_vision:
            # Text-only fallback (existing behavior)
            return self._extract_text(event)

        # Build multimodal content
        if isinstance(content, TextContent):
            return content.body  # Simple case: just text

        if isinstance(content, MediaContent):
            parts: list[AITextPart | AIImagePart] = []
            if content.caption:
                parts.append(AITextPart(text=content.caption))
            parts.append(AIImagePart(url=content.url, mime_type=content.mime_type))
            return parts

        if isinstance(content, CompositeContent):
            parts = []
            for part in content.parts:
                if isinstance(part, TextContent):
                    parts.append(AITextPart(text=part.body))
                elif isinstance(part, MediaContent):
                    if part.caption:
                        parts.append(AITextPart(text=part.caption))
                    parts.append(AIImagePart(url=part.url, mime_type=part.mime_type))
            return parts if parts else ""

        # Fallback for other types
        return self._extract_text(event)

    def _extract_text(self, event: RoomEvent) -> str:
        if isinstance(event.content, TextContent):
            return event.content.body
        return ""

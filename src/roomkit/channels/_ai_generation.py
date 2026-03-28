"""AIChannel mixin for non-streaming response generation with tool loop."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.models.channel import ChannelOutput
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITextPart,
    AIThinkingPart,
    AIToolCallPart,
    ProviderError,
)
from roomkit.realtime.base import EphemeralEventType
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.noop import NoopTelemetryProvider

if TYPE_CHECKING:
    from roomkit.channels.ai import _ContentPart, _ToolLoopContext
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.enums import ChannelType
    from roomkit.providers.ai.base import AIProvider

logger = logging.getLogger("roomkit.channels.ai")


@runtime_checkable
class AIGenerationHost(Protocol):
    """Contract: capabilities a host class must provide for AIGenerationMixin.

    Attributes provided by the host's ``__init__``:
        _provider: AI provider for generation.
        _max_tool_rounds: Maximum tool-loop iterations.
        _tool_loop_timeout_seconds: Optional wall-clock timeout for the loop.
        _tool_loop_warn_after: Log a warning after this many rounds.
        _tool_handler: Tool call handler (or ``None`` if tools disabled).
        _active_loops: Registry of currently running tool loops.
        _after_response_hook: Optional callback fired after response generation.
        channel_id: Unique identifier for this channel.
        provider_name: Human-readable provider name.
        channel_type: Channel type enum value.

    Methods provided by other mixins:
        _build_context: ``AIContextMixin`` — builds AI context from room state.
        _drain_steering_queue: ``AISteeringMixin`` — drains pending directives.
        _generate_with_retry: ``AIResilienceMixin`` — generate with retry/fallback.
        _execute_tools_parallel: ``AIToolsMixin`` — execute tool calls concurrently.
        _apply_tool_filters: ``AIToolPolicyMixin`` — apply policy + gating filters.
        _publish_thinking_event: ``AIEventsMixin`` — publish thinking events.
        _publish_tool_event: ``AIEventsMixin`` — publish tool call events.
        _is_context_overflow: ``AIResilienceMixin`` static — detect overflow errors.
        _compact_context: ``AIResilienceMixin`` — emergency context compaction.
        _extract_accumulated_text: ``AIResilienceMixin`` static — extract text.
    """

    _provider: AIProvider
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _tool_handler: Any
    _active_loops: dict[str, _ToolLoopContext]
    _after_response_hook: Any
    _before_generation_hook: Any
    channel_id: str
    provider_name: str
    channel_type: ChannelType

    async def _build_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext: ...
    def _drain_steering_queue(
        self, context: AIContext, loop_ctx: _ToolLoopContext
    ) -> tuple[AIContext, bool]: ...
    async def _generate_with_retry(self, context: AIContext) -> AIResponse: ...
    async def _execute_tools_parallel(
        self,
        tool_calls: list[Any],
        telemetry: Any,
        *,
        parent_span_id: str | None = ...,
    ) -> list[_ContentPart]: ...
    def _apply_tool_filters(self, tools: list[Any]) -> list[Any]: ...
    async def _publish_thinking_event(
        self,
        event_type: EphemeralEventType,
        room_id: str,
        thinking: str,
        round_idx: int,
    ) -> None: ...
    async def _publish_tool_event(
        self,
        event_type: EphemeralEventType,
        room_id: str,
        tool_calls: list[Any],
        round_idx: int,
        *,
        duration_ms: int | None = ...,
    ) -> None: ...
    @staticmethod
    def _is_context_overflow(exc: ProviderError) -> bool: ...
    async def _compact_context(self, context: AIContext) -> AIContext: ...
    @staticmethod
    def _extract_accumulated_text(messages: list[AIMessage]) -> str: ...


class AIGenerationMixin:
    """Non-streaming AI response generation with tool loop.

    Host contract: :class:`AIGenerationHost`.
    """

    _provider: Any
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _tool_handler: Any
    _active_loops: dict[str, _ToolLoopContext]
    _after_response_hook: Any
    _before_generation_hook: Any
    channel_id: str
    provider_name: str
    channel_type: Any

    # Cross-mixin methods — Any annotations avoid MRO shadowing.
    # _build_context is NOT annotated here: it's a real typed method on
    # AIContextMixin whose return type must be preserved for subclasses
    # (Agent.super()._build_context()). Call sites use type: ignore instead.
    _drain_steering_queue: Any  # see AIGenerationHost
    _generate_with_retry: Any  # see AIGenerationHost
    _execute_tools_parallel: Any  # see AIGenerationHost
    _apply_tool_filters: Any  # see AIGenerationHost
    _publish_thinking_event: Any  # see AIGenerationHost
    _publish_tool_event: Any  # see AIGenerationHost
    _is_context_overflow: Any  # see AIGenerationHost
    _compact_context: Any  # see AIGenerationHost
    _extract_accumulated_text: Any  # see AIGenerationHost

    @property
    def _telemetry_provider(self) -> NoopTelemetryProvider:
        """Access telemetry provider (set by register_channel)."""
        return getattr(self, "_telemetry", None) or NoopTelemetryProvider()

    async def _fire_before_generation_hook(
        self, ai_context: AIContext, event: RoomEvent
    ) -> tuple[AIContext, bool]:
        """Fire BEFORE_AI_GENERATION hook. Returns ``(context, blocked)``."""
        if not self._before_generation_hook:
            return ai_context, False
        gen_event = AIGenerationEvent(
            ai_context=ai_context,
            channel_id=self.channel_id,
            room_id=event.room_id,
            provider_name=self.provider_name,
        )
        sync_result = await self._before_generation_hook(gen_event)
        if not sync_result.allowed:
            logger.info(
                "AI generation blocked by hook (reason=%s, blocked_by=%s)",
                sync_result.reason,
                sync_result.blocked_by,
            )
            return ai_context, True
        return gen_event.ai_context, False

    async def _generate_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Generate an AI response, executing tool calls if needed."""
        from roomkit.telemetry.context import get_current_span

        ai_context = await self._build_context(event, binding, context)  # ty: ignore[unresolved-attribute]
        ai_context, blocked = await self._fire_before_generation_hook(ai_context, event)
        if blocked:
            return ChannelOutput.empty()
        telemetry = self._telemetry_provider
        _t0 = time.monotonic()
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

        if self._after_response_hook:
            try:
                from roomkit.models.tool_call import AIResponseEvent

                await self._after_response_hook(
                    AIResponseEvent(
                        channel_id=self.channel_id,
                        response_content=response.content or "",
                        room_id=event.room_id,
                        tool_calls_count=(len(response.tool_calls) if response.tool_calls else 0),
                        usage=response.usage or {},
                        thinking=response.thinking or "",
                        latency_ms=int((time.monotonic() - _t0) * 1000),
                    )
                )
            except Exception:
                logger.debug("After-response hook failed", exc_info=True)

        return ChannelOutput(
            responded=True,
            response_events=[response_event],
        )

    async def _run_tool_loop(
        self, context: AIContext, *, parent_span_id: str | None = None
    ) -> AIResponse:
        """Generate -> execute tools -> re-generate until a text response."""
        from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext

        loop_ctx = _ToolLoopContext()
        loop_ctx.loop_id = str(id(loop_ctx))
        parent_lctx = _current_loop_ctx.get()
        if parent_lctx is not None:
            loop_ctx.current_participant_role = parent_lctx.current_participant_role
        _current_loop_ctx.set(loop_ctx)
        self._active_loops[loop_ctx.loop_id] = loop_ctx
        try:
            context, should_cancel = self._drain_steering_queue(context, loop_ctx)
            if should_cancel:
                return AIResponse(content="", tool_calls=[])
            response: AIResponse = await self._generate_with_retry(context)
            telemetry = self._telemetry_provider
            room_id = context.room.room.id if context.room else None
            deadline = (
                asyncio.get_running_loop().time() + self._tool_loop_timeout_seconds
                if self._tool_loop_timeout_seconds
                else None
            )

            if response.thinking and room_id:
                await self._publish_thinking_event(
                    EphemeralEventType.THINKING_START, room_id, "", 0
                )
                await self._publish_thinking_event(
                    EphemeralEventType.THINKING_END, room_id, response.thinking, 0
                )

            for round_idx in range(self._max_tool_rounds):
                if not response.tool_calls or self._tool_handler is None:
                    break

                if loop_ctx.cancel_event.is_set():
                    logger.info("Tool loop cancelled before round %d", round_idx)
                    break

                if deadline and asyncio.get_running_loop().time() >= deadline:
                    logger.warning(
                        "Tool loop timeout after %d rounds (%.0fs)",
                        round_idx,
                        self._tool_loop_timeout_seconds,
                    )
                    break

                if round_idx == self._tool_loop_warn_after:
                    logger.warning("Tool loop reached %d rounds, still running", round_idx)

                logger.info(
                    "Tool round %d: %d call(s)",
                    round_idx + 1,
                    len(response.tool_calls),
                )

                parts: list[_ContentPart] = []
                if response.thinking:
                    parts.append(
                        AIThinkingPart(
                            thinking=response.thinking,
                            signature=response.thinking_signature,
                        )
                    )
                if response.content:
                    parts.append(AITextPart(text=response.content))
                for tc in response.tool_calls:
                    parts.append(
                        AIToolCallPart(
                            id=tc.id,
                            name=tc.name,
                            arguments=tc.arguments,
                            metadata=tc.metadata,
                        )
                    )
                context.messages.append(AIMessage(role="assistant", content=parts))

                if room_id:
                    await self._publish_tool_event(
                        EphemeralEventType.TOOL_CALL_START,
                        room_id,
                        response.tool_calls,
                        round_idx,
                    )

                t0 = time.monotonic()
                result_parts = await self._execute_tools_parallel(
                    response.tool_calls, telemetry, parent_span_id=parent_span_id
                )
                duration_ms = int((time.monotonic() - t0) * 1000)
                context.messages.append(AIMessage(role="tool", content=result_parts))

                if room_id:
                    await self._publish_tool_event(
                        EphemeralEventType.TOOL_CALL_END,
                        room_id,
                        result_parts,
                        round_idx,
                        duration_ms=duration_ms,
                    )

                context, should_cancel = self._drain_steering_queue(context, loop_ctx)
                if should_cancel:
                    logger.info("Tool loop cancelled after round %d", round_idx)
                    break

                if loop_ctx.all_context_tools:
                    context = context.model_copy(
                        update={"tools": self._apply_tool_filters(loop_ctx.all_context_tools)}
                    )

                try:
                    response = await self._generate_with_retry(context)
                except ProviderError as exc:
                    if self._is_context_overflow(exc):
                        logger.warning("Context overflow at round %d. Compacting.", round_idx)
                        context = await self._compact_context(context)
                        response = await self._generate_with_retry(context)
                    else:
                        accumulated = self._extract_accumulated_text(context.messages)
                        if accumulated:
                            return AIResponse(
                                content=accumulated + f"\n\n[Agent interrupted: {exc}]",
                                tool_calls=[],
                            )
                        raise

                if response.thinking and room_id:
                    await self._publish_thinking_event(
                        EphemeralEventType.THINKING_START, room_id, "", round_idx + 1
                    )
                    await self._publish_thinking_event(
                        EphemeralEventType.THINKING_END,
                        room_id,
                        response.thinking,
                        round_idx + 1,
                    )

            return response
        finally:
            self._active_loops.pop(loop_ctx.loop_id, None)
            _current_loop_ctx.set(None)

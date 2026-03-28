"""AIChannel mixin for streaming response generation with tool loops."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.models.channel import ChannelOutput
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AIToolCallPart,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.realtime.base import EphemeralEventType
from roomkit.telemetry.base import Attr, SpanKind

if TYPE_CHECKING:
    from roomkit.channels.ai import _ContentPart, _ToolLoopContext
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.providers.ai.base import StreamEvent
    from roomkit.telemetry.noop import NoopTelemetryProvider

logger = logging.getLogger("roomkit.channels.ai")


@runtime_checkable
class AIStreamingHost(Protocol):
    """Contract: capabilities a host class must provide for AIStreamingMixin.

    Attributes provided by the host's ``__init__``:
        _provider: AI provider for generation.
        _max_tool_rounds: Maximum tool-loop iterations.
        _tool_loop_timeout_seconds: Optional wall-clock timeout for the loop.
        _tool_loop_warn_after: Log a warning after this many rounds.
        _tool_handler: Tool call handler (or ``None`` if tools disabled).
        _active_loops: Registry of currently running tool loops.
        _after_response_hook: Optional callback fired after response generation.
        channel_id: Unique identifier for this channel.

    Properties / methods provided by other mixins:
        _build_context: ``AIContextMixin`` — builds AI context from room state.
        _drain_steering_queue: ``AISteeringMixin`` — drains pending directives.
        _generate_stream_with_retry: ``AIResilienceMixin`` — stream with retry.
        _execute_tools_parallel: ``AIToolsMixin`` — execute tool calls concurrently.
        _apply_tool_filters: ``AIToolPolicyMixin`` — apply policy + gating filters.
        _publish_thinking_event: ``AIEventsMixin`` — publish thinking events.
        _publish_tool_event: ``AIEventsMixin`` — publish tool call events.
        _telemetry_provider: ``AIGenerationMixin`` property — telemetry provider.
    """

    _provider: Any
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _tool_handler: Any
    _active_loops: dict[str, _ToolLoopContext]
    _after_response_hook: Any
    channel_id: str

    async def _build_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext: ...
    def _drain_steering_queue(
        self, context: AIContext, loop_ctx: _ToolLoopContext
    ) -> tuple[AIContext, bool]: ...
    async def _generate_stream_with_retry(
        self, context: AIContext
    ) -> AsyncIterator[StreamEvent]: ...
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
    @property
    def _telemetry_provider(self) -> NoopTelemetryProvider: ...


class AIStreamingMixin:
    """Streaming AI response generation with tool loop and deduplication.

    Host contract: :class:`AIStreamingHost`.
    """

    _provider: Any
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _tool_handler: Any
    _active_loops: dict[str, Any]
    _after_response_hook: Any
    channel_id: str

    # Cross-mixin methods — Any annotations avoid MRO shadowing.
    # _build_context is NOT annotated here: it's a real typed method on
    # AIContextMixin whose return type must be preserved for subclasses
    # (Agent.super()._build_context()). Call sites use type: ignore instead.
    _drain_steering_queue: Any  # see AIStreamingHost
    _generate_stream_with_retry: Any  # see AIStreamingHost
    _execute_tools_parallel: Any  # see AIStreamingHost
    _apply_tool_filters: Any  # see AIStreamingHost
    _publish_thinking_event: Any  # see AIStreamingHost
    _publish_tool_event: Any  # see AIStreamingHost
    _telemetry_provider: Any  # see AIStreamingHost

    async def _start_streaming_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response handle (generator starts on consumption)."""
        ai_context = await self._build_context(event, binding, context)  # ty: ignore[unresolved-attribute]
        return ChannelOutput(
            responded=True,
            response_stream=self._provider.generate_stream(ai_context),
        )

    async def _start_streaming_tool_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response that handles tool calls between rounds."""
        ai_context = await self._build_context(event, binding, context)  # ty: ignore[unresolved-attribute]
        return ChannelOutput(
            responded=True,
            response_stream=self._run_streaming_tool_loop(ai_context),
        )

    async def _run_streaming_tool_loop(self, context: AIContext) -> AsyncIterator[str]:
        """Stream text deltas, executing tool calls between generation rounds."""
        from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext
        from roomkit.telemetry.context import get_current_span

        loop_ctx = _ToolLoopContext()
        loop_ctx.loop_id = str(id(loop_ctx))
        parent_ctx = _current_loop_ctx.get()
        if parent_ctx is not None:
            loop_ctx.current_participant_role = parent_ctx.current_participant_role
        _current_loop_ctx.set(loop_ctx)
        self._active_loops[loop_ctx.loop_id] = loop_ctx
        telemetry = self._telemetry_provider
        span_id = telemetry.start_span(
            SpanKind.LLM_GENERATE,
            "llm.generate",
            parent_id=get_current_span(),
            room_id=context.room.room.id if context.room else None,
            channel_id=self.channel_id,
            attributes={
                Attr.PROVIDER: type(self._provider).__name__,
                Attr.LLM_STREAMING: True,
            },
        )
        _round_usage: dict[str, Any] | None = None
        _total_input_tokens = 0
        _total_output_tokens = 0
        _span_errored = False
        _t0_stream = time.monotonic()
        _accumulated_text: list[str] = []
        try:
            context, should_cancel = self._drain_steering_queue(context, loop_ctx)
            if should_cancel:
                return
            room_id = context.room.room.id if context.room else None
            deadline = (
                asyncio.get_running_loop().time() + self._tool_loop_timeout_seconds
                if self._tool_loop_timeout_seconds
                else None
            )

            _dedup_prefix = ""

            for _round_idx in range(self._max_tool_rounds + 1):
                if loop_ctx.cancel_event.is_set():
                    logger.info("Streaming tool loop cancelled before round %d", _round_idx)
                    return

                if loop_ctx.all_context_tools:
                    context = context.model_copy(
                        update={"tools": self._apply_tool_filters(loop_ctx.all_context_tools)}
                    )

                thinking_parts: list[str] = []
                text_parts: list[str] = []
                tool_calls: list[StreamToolCall] = []
                thinking_started = False
                _dedup_active = bool(_dedup_prefix)
                _dedup_offset = 0
                _dedup_buffer: list[str] = []

                async for event in self._generate_stream_with_retry(context):
                    if isinstance(event, StreamThinkingDelta):
                        if not thinking_started and room_id:
                            thinking_started = True
                            await self._publish_thinking_event(
                                EphemeralEventType.THINKING_START,
                                room_id,
                                "",
                                _round_idx,
                            )
                        thinking_parts.append(event.thinking)
                    elif isinstance(event, StreamTextDelta):
                        if thinking_started and thinking_parts and room_id:
                            thinking_started = False
                            await self._publish_thinking_event(
                                EphemeralEventType.THINKING_END,
                                room_id,
                                "".join(thinking_parts),
                                _round_idx,
                            )
                        text_parts.append(event.text)
                        _accumulated_text.append(event.text)

                        # --- Dedup: skip text that repeats previous rounds ---
                        if _dedup_active:
                            end = _dedup_offset + len(event.text)
                            if end <= len(_dedup_prefix):
                                if _dedup_prefix[_dedup_offset:end] == event.text:
                                    _dedup_offset = end
                                    _dedup_buffer.append(event.text)
                                    continue
                                _dedup_active = False
                                for buf in _dedup_buffer:
                                    yield buf
                                _dedup_buffer.clear()
                                yield event.text
                            else:
                                prefix_tail = _dedup_prefix[_dedup_offset:]
                                if event.text[: len(prefix_tail)] == prefix_tail:
                                    _dedup_active = False
                                    _dedup_buffer.clear()
                                    new_text = event.text[len(prefix_tail) :]
                                    if new_text:
                                        yield new_text
                                else:
                                    _dedup_active = False
                                    for buf in _dedup_buffer:
                                        yield buf
                                    _dedup_buffer.clear()
                                    yield event.text
                            continue

                        yield event.text
                    elif isinstance(event, StreamToolCall):
                        tool_calls.append(event)
                    elif isinstance(event, StreamDone):
                        if event.usage:
                            _round_usage = event.usage
                            round_in = _round_usage.get("input_tokens", 0)
                            round_out = _round_usage.get("output_tokens", 0)
                            _total_input_tokens += round_in
                            _total_output_tokens += round_out
                            telemetry.record_metric(
                                "roomkit.llm.input_tokens",
                                float(round_in),
                                unit="tokens",
                                attributes={"channel_id": self.channel_id},
                            )
                            telemetry.record_metric(
                                "roomkit.llm.output_tokens",
                                float(round_out),
                                unit="tokens",
                                attributes={"channel_id": self.channel_id},
                            )

                if _dedup_buffer:
                    for buf in _dedup_buffer:
                        yield buf
                    _dedup_buffer.clear()

                if thinking_started and thinking_parts and room_id:
                    await self._publish_thinking_event(
                        EphemeralEventType.THINKING_END,
                        room_id,
                        "".join(thinking_parts),
                        _round_idx,
                    )

                if not tool_calls or self._tool_handler is None:
                    return

                if _round_idx >= self._max_tool_rounds:
                    logger.warning(
                        "Streaming tool loop reached max_tool_rounds=%d",
                        self._max_tool_rounds,
                    )
                    return

                if deadline and asyncio.get_running_loop().time() >= deadline:
                    logger.warning(
                        "Streaming tool loop timeout after %d rounds (%.0fs)",
                        _round_idx,
                        self._tool_loop_timeout_seconds,
                    )
                    return

                if _round_idx == self._tool_loop_warn_after:
                    logger.warning(
                        "Streaming tool loop reached %d rounds, still running", _round_idx
                    )

                logger.info(
                    "Streaming tool round %d: %d call(s)",
                    _round_idx + 1,
                    len(tool_calls),
                )

                parts: list[_ContentPart] = []
                if thinking_parts:
                    parts.append(AIThinkingPart(thinking="".join(thinking_parts)))
                accumulated_text = "".join(text_parts)
                if accumulated_text:
                    parts.append(AITextPart(text=accumulated_text))
                    _dedup_prefix = accumulated_text
                for tc in tool_calls:
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
                        tool_calls,
                        _round_idx,
                    )

                t0 = time.monotonic()
                result_parts = await self._execute_tools_parallel(tool_calls, telemetry)
                duration_ms = int((time.monotonic() - t0) * 1000)
                context.messages.append(AIMessage(role="tool", content=result_parts))

                if room_id:
                    await self._publish_tool_event(
                        EphemeralEventType.TOOL_CALL_END,
                        room_id,
                        result_parts,
                        _round_idx,
                        duration_ms=duration_ms,
                    )

                context, should_cancel = self._drain_steering_queue(context, loop_ctx)
                if should_cancel:
                    logger.info("Streaming tool loop cancelled after round %d", _round_idx)
                    return
        except Exception as exc:
            _span_errored = True
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            raise
        finally:
            if not _span_errored:
                usage_attrs: dict[str, Any] = {}
                if _total_input_tokens or _total_output_tokens:
                    usage_attrs[Attr.LLM_INPUT_TOKENS] = _total_input_tokens
                    usage_attrs[Attr.LLM_OUTPUT_TOKENS] = _total_output_tokens
                telemetry.end_span(span_id, attributes=usage_attrs)

            if self._after_response_hook and not _span_errored:
                try:
                    from roomkit.models.tool_call import AIResponseEvent

                    await self._after_response_hook(
                        AIResponseEvent(
                            channel_id=self.channel_id,
                            response_content="".join(_accumulated_text),
                            room_id=room_id,
                            usage={
                                "input_tokens": _total_input_tokens,
                                "output_tokens": _total_output_tokens,
                            },
                            latency_ms=int((time.monotonic() - _t0_stream) * 1000),
                            streaming=True,
                        )
                    )
                except Exception:
                    logger.debug("After-response hook failed (streaming)", exc_info=True)

            self._active_loops.pop(loop_ctx.loop_id, None)
            _current_loop_ctx.set(None)

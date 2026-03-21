"""AIChannel mixin for AI response generation (streaming and non-streaming)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from roomkit.models.channel import ChannelOutput
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITextPart,
    AIThinkingPart,
    AIToolCallPart,
    ProviderError,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.realtime.base import EphemeralEventType
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.noop import NoopTelemetryProvider

if TYPE_CHECKING:
    from roomkit.channels.ai import _ContentPart, _ToolLoopContext
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext

logger = logging.getLogger("roomkit.channels.ai")


class AIGenerationMixin:
    """Streaming and non-streaming AI response generation with tool loops."""

    _provider: Any
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _tool_handler: Any
    _active_loops: dict[str, _ToolLoopContext]
    _after_response_hook: Any
    channel_id: str
    provider_name: str
    channel_type: Any

    @property
    def _telemetry_provider(self) -> NoopTelemetryProvider:
        """Access telemetry provider (set by register_channel)."""
        return getattr(self, "_telemetry", None) or NoopTelemetryProvider()

    async def _start_streaming_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response handle (generator starts on consumption)."""
        ai_context = await self._build_context(event, binding, context)  # type: ignore[attr-defined]
        return ChannelOutput(
            responded=True,
            response_stream=self._provider.generate_stream(ai_context),
        )

    async def _start_streaming_tool_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response that handles tool calls between rounds."""
        ai_context = await self._build_context(event, binding, context)  # type: ignore[attr-defined]
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
        # Inherit participant role from the on_event-level context
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
            # Apply pre-queued directives (e.g. cancel enqueued before loop started)
            context, should_cancel = self._drain_steering_queue(context, loop_ctx)  # type: ignore[attr-defined]
            if should_cancel:
                return
            room_id = context.room.room.id if context.room else None
            deadline = (
                asyncio.get_event_loop().time() + self._tool_loop_timeout_seconds
                if self._tool_loop_timeout_seconds
                else None
            )

            # Track text yielded across tool rounds for dedup.
            # Some providers (e.g. Gemini) repeat pre-tool text in
            # continuation rounds.  We speculatively buffer incoming
            # deltas and verify they match the previously-yielded prefix
            # before skipping them.
            _dedup_prefix = ""  # text yielded in prior rounds

            for _round_idx in range(self._max_tool_rounds + 1):
                # Steering checkpoint 1: fast-path cancel before generate
                if loop_ctx.cancel_event.is_set():
                    logger.info("Streaming tool loop cancelled before round %d", _round_idx)
                    return

                # Re-apply gating so newly-activated skills expose their tools
                if loop_ctx.all_context_tools:
                    context = context.model_copy(
                        update={"tools": self._apply_tool_filters(loop_ctx.all_context_tools)}  # type: ignore[attr-defined]
                    )

                thinking_parts: list[str] = []
                text_parts: list[str] = []
                tool_calls: list[StreamToolCall] = []

                # Publish THINKING_START when first thinking delta arrives
                thinking_started = False

                # Dedup state for this round
                _dedup_active = bool(_dedup_prefix)
                _dedup_offset = 0  # chars matched so far
                _dedup_buffer: list[str] = []  # held-back deltas

                async for event in self._generate_stream_with_retry(context):  # type: ignore[attr-defined]
                    if isinstance(event, StreamThinkingDelta):
                        if not thinking_started and room_id:
                            thinking_started = True
                            await self._publish_thinking_event(  # type: ignore[attr-defined]
                                EphemeralEventType.THINKING_START,
                                room_id,
                                "",
                                _round_idx,
                            )
                        thinking_parts.append(event.thinking)
                    elif isinstance(event, StreamTextDelta):
                        # Publish THINKING_END when we transition from thinking to text
                        if thinking_started and thinking_parts and room_id:
                            thinking_started = False
                            await self._publish_thinking_event(  # type: ignore[attr-defined]
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
                                # Entire delta within the prefix region
                                if _dedup_prefix[_dedup_offset:end] == event.text:
                                    _dedup_offset = end
                                    _dedup_buffer.append(event.text)
                                    continue  # matches — keep buffering
                                # Mismatch — flush buffer and yield normally
                                _dedup_active = False
                                for buf in _dedup_buffer:
                                    yield buf
                                _dedup_buffer.clear()
                                yield event.text
                            else:
                                # Delta crosses the prefix boundary
                                prefix_tail = _dedup_prefix[_dedup_offset:]
                                if event.text[: len(prefix_tail)] == prefix_tail:
                                    # Full prefix matched — commit dedup
                                    _dedup_active = False
                                    _dedup_buffer.clear()
                                    new_text = event.text[len(prefix_tail) :]
                                    if new_text:
                                        yield new_text
                                else:
                                    # Mismatch at boundary — flush buffer
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
                        # Capture usage from the stream for telemetry
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

                # Flush any remaining dedup buffer (dedup matched to end
                # of prefix but no new text followed — shouldn't happen
                # in practice but be safe).
                if _dedup_buffer:
                    for buf in _dedup_buffer:
                        yield buf
                    _dedup_buffer.clear()

                # Publish THINKING_END if thinking was the last block (no text followed)
                if thinking_started and thinking_parts and room_id:
                    await self._publish_thinking_event(  # type: ignore[attr-defined]
                        EphemeralEventType.THINKING_END,
                        room_id,
                        "".join(thinking_parts),
                        _round_idx,
                    )

                if not tool_calls or self._tool_handler is None:
                    return

                # Don't execute tools on the final iteration — no generation follows
                if _round_idx >= self._max_tool_rounds:
                    logger.warning(
                        "Streaming tool loop reached max_tool_rounds=%d",
                        self._max_tool_rounds,
                    )
                    return

                # Timeout check
                if deadline and asyncio.get_event_loop().time() >= deadline:
                    logger.warning(
                        "Streaming tool loop timeout after %d rounds (%.0fs)",
                        _round_idx,
                        self._tool_loop_timeout_seconds,
                    )
                    return

                # Soft warning
                if _round_idx == self._tool_loop_warn_after:
                    logger.warning(
                        "Streaming tool loop reached %d rounds, still running", _round_idx
                    )

                logger.info(
                    "Streaming tool round %d: %d call(s)",
                    _round_idx + 1,
                    len(tool_calls),
                )

                # Append assistant message with thinking + text + tool calls.
                # Thinking blocks must be preserved in history for providers
                # that require round-trip fidelity (e.g. Anthropic).
                parts: list[_ContentPart] = []
                if thinking_parts:
                    parts.append(AIThinkingPart(thinking="".join(thinking_parts)))
                accumulated_text = "".join(text_parts)
                if accumulated_text:
                    parts.append(AITextPart(text=accumulated_text))
                    # Set dedup prefix for the next round — the provider
                    # may regenerate this text in the continuation.
                    _dedup_prefix = accumulated_text
                for tc in tool_calls:
                    parts.append(AIToolCallPart(id=tc.id, name=tc.name, arguments=tc.arguments))
                context.messages.append(AIMessage(role="assistant", content=parts))

                # Publish TOOL_CALL_START ephemeral event
                if room_id:
                    await self._publish_tool_event(  # type: ignore[attr-defined]
                        EphemeralEventType.TOOL_CALL_START,
                        room_id,
                        tool_calls,
                        _round_idx,
                    )

                # Execute tools concurrently
                t0 = time.monotonic()
                result_parts = await self._execute_tools_parallel(tool_calls, telemetry)  # type: ignore[attr-defined]
                duration_ms = int((time.monotonic() - t0) * 1000)
                context.messages.append(AIMessage(role="tool", content=result_parts))

                # Publish TOOL_CALL_END ephemeral event
                if room_id:
                    await self._publish_tool_event(  # type: ignore[attr-defined]
                        EphemeralEventType.TOOL_CALL_END,
                        room_id,
                        result_parts,
                        _round_idx,
                        duration_ms=duration_ms,
                    )

                # Steering checkpoint 2: drain queue after tool execution
                context, should_cancel = self._drain_steering_queue(context, loop_ctx)  # type: ignore[attr-defined]
                if should_cancel:
                    logger.info("Streaming tool loop cancelled after round %d", _round_idx)
                    return
        except Exception as exc:
            _span_errored = True
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            raise
        finally:
            # end_span in finally because async generator exits via return
            # (which skips the else clause of try/except/else).
            if not _span_errored:
                usage_attrs: dict[str, Any] = {}
                if _total_input_tokens or _total_output_tokens:
                    usage_attrs[Attr.LLM_INPUT_TOKENS] = _total_input_tokens
                    usage_attrs[Attr.LLM_OUTPUT_TOKENS] = _total_output_tokens
                telemetry.end_span(span_id, attributes=usage_attrs)

            # Fire ON_AI_RESPONSE hook with accumulated streaming data
            if self._after_response_hook and not _span_errored:
                try:
                    from roomkit.models.tool_call import AIResponseEvent

                    await self._after_response_hook(
                        AIResponseEvent(
                            channel_id=self.channel_id,
                            room_id=room_id,
                            response_content="".join(_accumulated_text),
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

    async def _generate_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Generate an AI response, executing tool calls if needed."""
        from roomkit.telemetry.context import get_current_span

        ai_context = await self._build_context(event, binding, context)  # type: ignore[attr-defined]
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

        # Fire ON_AI_RESPONSE hook for evaluation/scoring (best-effort)
        if self._after_response_hook:
            try:
                from roomkit.models.tool_call import AIResponseEvent

                await self._after_response_hook(
                    AIResponseEvent(
                        channel_id=self.channel_id,
                        room_id=event.room_id,
                        response_content=response.content or "",
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
        # Inherit participant role from the on_event-level context
        parent_lctx = _current_loop_ctx.get()
        if parent_lctx is not None:
            loop_ctx.current_participant_role = parent_lctx.current_participant_role
        _current_loop_ctx.set(loop_ctx)
        self._active_loops[loop_ctx.loop_id] = loop_ctx
        try:
            # Apply pre-queued directives (e.g. cancel enqueued before loop started)
            context, should_cancel = self._drain_steering_queue(context, loop_ctx)  # type: ignore[attr-defined]
            if should_cancel:
                return AIResponse(content="", tool_calls=[])
            response: AIResponse = await self._generate_with_retry(context)  # type: ignore[attr-defined]
            telemetry = self._telemetry_provider
            room_id = context.room.room.id if context.room else None
            deadline = (
                asyncio.get_event_loop().time() + self._tool_loop_timeout_seconds
                if self._tool_loop_timeout_seconds
                else None
            )

            # Publish thinking ephemeral events for the initial response
            if response.thinking and room_id:
                await self._publish_thinking_event(  # type: ignore[attr-defined]
                    EphemeralEventType.THINKING_START, room_id, "", 0
                )
                await self._publish_thinking_event(  # type: ignore[attr-defined]
                    EphemeralEventType.THINKING_END, room_id, response.thinking, 0
                )

            for round_idx in range(self._max_tool_rounds):
                if not response.tool_calls or self._tool_handler is None:
                    break

                # Steering checkpoint 1: fast-path cancel before generate
                if loop_ctx.cancel_event.is_set():
                    logger.info("Tool loop cancelled before round %d", round_idx)
                    break

                # Timeout check (after current round finishes)
                if deadline and asyncio.get_event_loop().time() >= deadline:
                    logger.warning(
                        "Tool loop timeout after %d rounds (%.0fs)",
                        round_idx,
                        self._tool_loop_timeout_seconds,
                    )
                    break

                # Soft warning
                if round_idx == self._tool_loop_warn_after:
                    logger.warning("Tool loop reached %d rounds, still running", round_idx)

                logger.info(
                    "Tool round %d: %d call(s)",
                    round_idx + 1,
                    len(response.tool_calls),
                )

                # Append assistant message with thinking + tool calls.
                # Thinking blocks must be preserved in history for providers
                # that require round-trip fidelity (e.g. Anthropic).
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
                    parts.append(AIToolCallPart(id=tc.id, name=tc.name, arguments=tc.arguments))
                context.messages.append(AIMessage(role="assistant", content=parts))

                # Publish TOOL_CALL_START ephemeral event
                if room_id:
                    await self._publish_tool_event(  # type: ignore[attr-defined]
                        EphemeralEventType.TOOL_CALL_START,
                        room_id,
                        response.tool_calls,
                        round_idx,
                    )

                # Execute tools concurrently
                t0 = time.monotonic()
                result_parts = await self._execute_tools_parallel(  # type: ignore[attr-defined]
                    response.tool_calls, telemetry, parent_span_id=parent_span_id
                )
                duration_ms = int((time.monotonic() - t0) * 1000)
                context.messages.append(AIMessage(role="tool", content=result_parts))

                # Publish TOOL_CALL_END ephemeral event
                if room_id:
                    await self._publish_tool_event(  # type: ignore[attr-defined]
                        EphemeralEventType.TOOL_CALL_END,
                        room_id,
                        result_parts,
                        round_idx,
                        duration_ms=duration_ms,
                    )

                # Steering checkpoint 2: drain queue after tool execution
                context, should_cancel = self._drain_steering_queue(context, loop_ctx)  # type: ignore[attr-defined]
                if should_cancel:
                    logger.info("Tool loop cancelled after round %d", round_idx)
                    break

                # Re-apply gating so newly-activated skills expose their tools
                if loop_ctx.all_context_tools:
                    context = context.model_copy(
                        update={"tools": self._apply_tool_filters(loop_ctx.all_context_tools)}  # type: ignore[attr-defined]
                    )

                # Re-generate with tool results (with retry)
                try:
                    response = await self._generate_with_retry(context)  # type: ignore[attr-defined]
                except ProviderError as exc:
                    if self._is_context_overflow(exc):  # type: ignore[attr-defined]
                        logger.warning("Context overflow at round %d. Compacting.", round_idx)
                        context = await self._compact_context(context)  # type: ignore[attr-defined]
                        response = await self._generate_with_retry(context)  # type: ignore[attr-defined]
                    else:
                        # Return partial result if we have accumulated text
                        accumulated = self._extract_accumulated_text(context.messages)  # type: ignore[attr-defined]
                        if accumulated:
                            return AIResponse(
                                content=accumulated + f"\n\n[Agent interrupted: {exc}]",
                                tool_calls=[],
                            )
                        raise

                # Publish thinking ephemeral events for this round
                if response.thinking and room_id:
                    await self._publish_thinking_event(  # type: ignore[attr-defined]
                        EphemeralEventType.THINKING_START, room_id, "", round_idx + 1
                    )
                    await self._publish_thinking_event(  # type: ignore[attr-defined]
                        EphemeralEventType.THINKING_END,
                        room_id,
                        response.thinking,
                        round_idx + 1,
                    )

            return response
        finally:
            self._active_loops.pop(loop_ctx.loop_id, None)
            _current_loop_ctx.set(None)

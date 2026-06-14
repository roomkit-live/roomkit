"""AIChannel mixin for streaming response generation with tool loops."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._ai_events import THINKING_PREVIEW_LIMIT
from roomkit.models.channel import ChannelOutput
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.streaming import (
    StreamDelta,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
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


class _ThinkingCoalescer:
    """Batches per-token thinking deltas into one ``THINKING_DELTA`` publish per window.

    Reasoning models emit one ``StreamThinkingDelta`` per token, and publishing
    each on the realtime bus is one ephemeral event + fan-out + WS serialise per
    token — thousands for a long trace, all on the shared event loop. Buffering
    and publishing once per time/size window (~80 ms / ~256 chars) cuts that
    10-100x while keeping the reasoning visibly real-time: the UI appends deltas,
    so a coalesced delta renders identically to many small ones. The complete
    trace is still published verbatim at ``THINKING_END``.

    A window of ``0`` ms disables batching — every delta publishes immediately.
    Flushes larger than ``_publish_thinking_event``'s preview cap are split
    into multiple publishes so a coalesced delta is never truncated, whatever
    the configured size threshold.
    """

    def __init__(
        self,
        publish: Any,
        room_id: str | None,
        round_idx: int,
        *,
        flush_ms: float,
        flush_chars: int,
    ) -> None:
        self._publish = publish
        self._room_id = room_id
        self._round_idx = round_idx
        self._flush_ms = flush_ms
        self._flush_chars = flush_chars
        self._pending: list[str] = []
        self._pending_len = 0
        self._last_publish = time.monotonic()

    async def add(self, delta: str) -> None:
        """Buffer a delta; publish the batch once the window is exceeded."""
        self._pending.append(delta)
        self._pending_len += len(delta)
        if self._flush_ms <= 0:
            await self.flush()
            return
        elapsed_ms = (time.monotonic() - self._last_publish) * 1000.0
        if self._pending_len >= self._flush_chars or elapsed_ms >= self._flush_ms:
            await self.flush()

    async def flush(self) -> None:
        """Publish whatever is buffered, then reset the window."""
        if not self._pending:
            return
        text = "".join(self._pending)
        self._pending.clear()
        self._pending_len = 0
        self._last_publish = time.monotonic()
        for i in range(0, len(text), THINKING_PREVIEW_LIMIT):
            await self._publish(
                EphemeralEventType.THINKING_DELTA,
                self._room_id,
                text[i : i + THINKING_PREVIEW_LIMIT],
                self._round_idx,
            )


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
    _max_empty_retries: int
    _thinking_coalesce_ms: float
    _thinking_coalesce_chars: int
    _tool_handler: Any
    _active_loops: dict[str, _ToolLoopContext]
    _after_response_hook: Any
    _before_generation_hook: Any
    _before_tool_call_hook: Any
    _tool_call_hook: Any
    _external_tool_handler: Any
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
    _max_empty_retries: int
    _thinking_coalesce_ms: float
    _thinking_coalesce_chars: int
    _tool_handler: Any
    _active_loops: dict[str, Any]
    _after_response_hook: Any
    _before_generation_hook: Any
    _before_tool_call_hook: Any
    _tool_call_hook: Any
    _external_tool_handler: Any
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

    def _new_thinking_coalescer(self, room_id: str | None, round_idx: int) -> _ThinkingCoalescer:
        """Coalescer bound to this channel's publish hook and window config."""
        return _ThinkingCoalescer(
            self._publish_thinking_event,
            room_id,
            round_idx,
            flush_ms=self._thinking_coalesce_ms,
            flush_chars=self._thinking_coalesce_chars,
        )

    async def _start_streaming_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response handle (generator starts on consumption)."""
        ai_context = await self._build_context(event, binding, context)  # ty: ignore[unresolved-attribute]
        ai_context, blocked = await self._fire_before_generation_hook(ai_context, event)  # ty: ignore[unresolved-attribute]
        if blocked:
            return ChannelOutput.empty()
        return ChannelOutput(
            responded=True,
            response_stream=self._stream_text_with_thinking(ai_context),
            response_metadata=ai_context.response_metadata,
        )

    async def _stream_text_with_thinking(
        self, ai_context: AIContext
    ) -> AsyncIterator[StreamDelta]:
        """Yield text deltas + thinking markers, publish realtime events.

        Two parallel mechanisms by design:

        * **Inline (channel stream)** — every ``StreamThinkingDelta`` becomes
          a :class:`ThinkingDeltaMarker` yielded in arrival order alongside
          text deltas. Channels that want to render reasoning in line with
          the answer (CLI, web) consume them; text-only channels filter
          them out via ``isinstance(chunk, str)``.

        * **Out-of-band (realtime bus)** — a single ``THINKING_END`` event
          carrying the full accumulated reasoning is published for
          observers (dashboards, audit logs). This matches the tool-loop
          and non-streaming paths so subscribers see consistent payloads.

        Falls back to ``generate_stream`` for providers that don't expose
        a structured stream.
        """
        if not self._provider.supports_structured_streaming:
            async for chunk in self._provider.generate_stream(ai_context):
                yield chunk
            return

        room_id = ai_context.room.room.id if ai_context.room else None
        thinking_parts: list[str] = []
        thinking_started = False
        coalescer = self._new_thinking_coalescer(room_id, round_idx=0)

        async for ev in self._provider.generate_structured_stream(ai_context):
            if isinstance(ev, StreamThinkingDelta):
                if not thinking_started and room_id:
                    thinking_started = True
                    await self._publish_thinking_event(
                        EphemeralEventType.THINKING_START, room_id, "", 0
                    )
                thinking_parts.append(ev.thinking)
                # Buffer each delta and publish in windows on the realtime bus so
                # remote subscribers (browser WS clients, etc.) stream the
                # reasoning as it arrives, not only the buffered text at
                # THINKING_END. The ``thinking`` field carries the delta, not the
                # accumulator — clients append to their own buffer.
                await coalescer.add(ev.thinking)
                yield ThinkingDeltaMarker(thinking=ev.thinking)
            elif isinstance(ev, StreamTextDelta):
                if thinking_started and thinking_parts and room_id:
                    thinking_started = False
                    await coalescer.flush()
                    await self._publish_thinking_event(
                        EphemeralEventType.THINKING_END,
                        room_id,
                        "".join(thinking_parts),
                        0,
                    )
                    thinking_parts = []
                yield ev.text

        # Thinking with no following text — close the boundary anyway so
        # subscribers see the reasoning even if the model emitted nothing else.
        if thinking_started and thinking_parts and room_id:
            await coalescer.flush()
            await self._publish_thinking_event(
                EphemeralEventType.THINKING_END,
                room_id,
                "".join(thinking_parts),
                0,
            )

    async def _start_streaming_tool_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response that handles tool calls between rounds."""
        from roomkit.channels.ai import _current_loop_ctx

        ai_context = await self._build_context(event, binding, context)  # ty: ignore[unresolved-attribute]
        ai_context, blocked = await self._fire_before_generation_hook(ai_context, event)  # ty: ignore[unresolved-attribute]
        if blocked:
            return ChannelOutput.empty()
        # The generator below executes when the CONSUMER iterates the
        # stream — by then handle_event has reset the loop contextvar, so
        # the parent ctx (participant role, room, the toolset stamped by
        # _build_context) must be captured NOW and passed explicitly.
        return ChannelOutput(
            responded=True,
            response_stream=self._run_streaming_tool_loop(
                ai_context, parent_loop_ctx=_current_loop_ctx.get()
            ),
            response_metadata=ai_context.response_metadata,
        )

    async def _run_streaming_tool_loop(
        self, context: AIContext, *, parent_loop_ctx: Any | None = None
    ) -> AsyncIterator[StreamDelta]:
        """Stream text deltas, executing tool calls between generation rounds."""
        from roomkit.channels.ai import (
            _EMPTY_RETRY_NUDGE,
            _current_loop_ctx,
            _ToolLoopContext,
        )
        from roomkit.telemetry.context import get_current_span

        # The handle_event ctx is gone from the contextvar by the time this
        # generator runs (reset in handle_event's finally); the caller
        # captured it at stream creation.
        parent_ctx = parent_loop_ctx if parent_loop_ctx is not None else _current_loop_ctx.get()
        loop_ctx = _ToolLoopContext.for_loop(
            parent_ctx, context.room.room.id if context.room else None
        )
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
        room_id = context.room.room.id if context.room else None
        try:
            context, should_cancel = self._drain_steering_queue(context, loop_ctx)
            if should_cancel:
                return
            deadline = (
                asyncio.get_running_loop().time() + self._tool_loop_timeout_seconds
                if self._tool_loop_timeout_seconds
                else None
            )

            _dedup_prefix = ""
            _saw_tool_call_any = False
            _empty_retries = 0

            for _round_idx in range(self._max_tool_rounds + 1):
                if loop_ctx.cancel_event.is_set():
                    logger.info("Streaming tool loop cancelled before round %d", _round_idx)
                    return

                if loop_ctx.all_context_tools:
                    context = context.model_copy(
                        update={"tools": self._apply_tool_filters(loop_ctx.all_context_tools)}
                    )

                thinking_parts: list[str] = []
                thinking_signature: str | None = None
                text_parts: list[str] = []
                tool_calls: list[StreamToolCall] = []
                thinking_started = False
                coalescer = self._new_thinking_coalescer(room_id, round_idx=_round_idx)
                _dedup_active = bool(_dedup_prefix)
                _dedup_offset = 0
                _dedup_buffer: list[str] = []

                async for event in self._generate_stream_with_retry(context):
                    # Check cancel between every stream event — allows immediate
                    # cancellation instead of waiting for the full stream to finish.
                    if loop_ctx.cancel_event.is_set():
                        logger.info("Streaming cancelled mid-generation at round %d", _round_idx)
                        return

                    if isinstance(event, StreamThinkingDelta):
                        if event.signature:
                            # Signature arrives as its own delta (empty text);
                            # capture it so the thinking block round-trips.
                            thinking_signature = event.signature
                        if not event.thinking:
                            continue
                        if not thinking_started and room_id:
                            thinking_started = True
                            await self._publish_thinking_event(
                                EphemeralEventType.THINKING_START,
                                room_id,
                                "",
                                _round_idx,
                            )
                        thinking_parts.append(event.thinking)
                        # Buffer the per-chunk delta and publish in windows on the
                        # realtime bus so remote WS subscribers stream the reasoning
                        # live; the buffered THINKING_END below still fires so
                        # observers joining mid-stream recover the complete trace.
                        await coalescer.add(event.thinking)
                        # Inline marker so channels can render reasoning in
                        # arrival order with text deltas.
                        yield ThinkingDeltaMarker(thinking=event.thinking)
                    elif isinstance(event, StreamTextDelta):
                        if thinking_started and thinking_parts and room_id:
                            thinking_started = False
                            await coalescer.flush()
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
                        # External tools: fire hooks and yield persistence markers
                        if self._tool_handler is None and self._external_tool_handler is not None:
                            handler = self._external_tool_handler
                            # Extract result from arguments if embedded by proxy
                            args = dict(event.arguments)
                            tool_result = args.pop("_result", None)
                            tool_is_error = args.pop("_is_error", False)

                            # Yield start marker for store persistence
                            yield ToolCallStartMarker(
                                tool_name=event.name,
                                tool_id=event.id,
                                arguments=args,
                            )

                            t0_ext = time.monotonic()
                            decision = await handler.process_tool_call(
                                event.name,
                                args,
                                tool_call_id=event.id,
                                room_id=room_id,
                            )
                            # Use decision.result if handler provided one
                            # (e.g. human input answer), otherwise use
                            # the result embedded by the external provider.
                            effective_result = tool_result or ""
                            if decision and decision.result is not None:
                                effective_result = decision.result
                                tool_is_error = False
                            # Fire on_tool_result with actual result
                            await handler.on_tool_result(
                                event.name,
                                args,
                                effective_result,
                                is_error=bool(tool_is_error),
                                tool_call_id=event.id,
                                room_id=room_id,
                            )

                            # Yield end marker for store persistence
                            ext_duration_ms = int((time.monotonic() - t0_ext) * 1000)
                            yield ToolCallEndMarker(
                                tool_name=event.name,
                                tool_id=event.id,
                                arguments=args,
                                result=effective_result,
                                status="failed" if tool_is_error else "completed",
                                duration_ms=ext_duration_ms,
                                error=effective_result if tool_is_error else None,
                            )
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
                    await coalescer.flush()
                    await self._publish_thinking_event(
                        EphemeralEventType.THINKING_END,
                        room_id,
                        "".join(thinking_parts),
                        _round_idx,
                    )

                if not tool_calls:
                    # Final answer round. If it produced no text *after* a tool
                    # round, the model skipped verbalizing the result — re-prompt
                    # once (bounded) for the final answer instead of ending empty.
                    if (
                        _saw_tool_call_any
                        and not "".join(text_parts).strip()
                        and _empty_retries < self._max_empty_retries
                        and not (deadline and asyncio.get_running_loop().time() >= deadline)
                    ):
                        _empty_retries += 1
                        logger.warning(
                            "Streaming empty response after tools; re-prompting for "
                            "final answer (retry %d/%d)",
                            _empty_retries,
                            self._max_empty_retries,
                        )
                        context.messages.append(AIMessage(role="user", content=_EMPTY_RETRY_NUDGE))
                        continue
                    return

                _saw_tool_call_any = True

                # External tools: provider executed them internally.
                # If ExternalToolHandler is set, hooks were already fired
                # inline during streaming (see StreamToolCall handling above).
                # Only fire hooks here if no handler was set.
                if self._tool_handler is None:
                    if self._external_tool_handler is None:
                        # No handler — fire hooks directly for observability
                        for tc in tool_calls:
                            if self._before_tool_call_hook is not None:
                                from roomkit.models.tool_call import ToolCallEvent

                                pre_event = ToolCallEvent(
                                    channel_id=self.channel_id,
                                    channel_type=ChannelType.AI,
                                    tool_call_id=tc.id,
                                    name=tc.name,
                                    arguments=tc.arguments,
                                    result=None,
                                    room_id=room_id,
                                )
                                await self._before_tool_call_hook(pre_event)

                    # External tools were handled inline during streaming.
                    # Persistence markers were yielded alongside hook callbacks.
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
                if thinking_parts or thinking_signature:
                    parts.append(
                        AIThinkingPart(
                            thinking="".join(thinking_parts),
                            signature=thinking_signature,
                        )
                    )
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

                # Yield start markers for each tool call (persistence boundary)
                for tc in tool_calls:
                    yield ToolCallStartMarker(
                        tool_name=tc.name,
                        tool_id=tc.id,
                        arguments=tc.arguments,
                    )

                t0 = time.monotonic()
                result_parts = await self._execute_tools_parallel(tool_calls, telemetry)
                duration_ms = int((time.monotonic() - t0) * 1000)
                context.messages.append(AIMessage(role="tool", content=result_parts))

                # Yield end markers with results (persistence boundary)
                for tc, rp in zip(tool_calls, result_parts, strict=False):
                    result_val = getattr(rp, "result", None)
                    is_error = isinstance(result_val, str) and result_val.startswith(
                        "Error executing tool"
                    )
                    yield ToolCallEndMarker(
                        tool_name=tc.name,
                        tool_id=tc.id,
                        arguments=tc.arguments,
                        result=result_val,
                        status="failed" if is_error else "completed",
                        duration_ms=duration_ms,
                        error=result_val if is_error else None,
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

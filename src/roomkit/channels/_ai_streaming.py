"""AIChannel mixin for streaming response generation with tool loops."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._ai_coalescers import _ThinkingCoalescer, _ToolCallDeltaCoalescer
from roomkit.channels._ai_loop_rules import (
    AIToolLoopRulesMixin,
    _accumulate_usage,
    final_round_reason,
)
from roomkit.models.channel import ChannelOutput
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.streaming import (
    LoopEndMarker,
    LoopEndReason,
    StreamDelta,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.models.tool_call import AIResponseEvent, ToolCallEvent
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIToolResultPart,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.realtime.base import EphemeralEventType
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.context import get_current_span

if TYPE_CHECKING:
    from roomkit.channels.ai import _ToolLoopContext
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
        _publish_thinking_event: ``AIEventsMixin`` — publish thinking events.
        _publish_tool_event: ``AIEventsMixin`` — publish tool call events.
        _telemetry_provider: ``AIGenerationMixin`` property — telemetry provider.

    The shared per-round loop rules (force-stop, empty-retry, budget, parts
    assembly, tool execution) come from :class:`AIToolLoopRulesMixin`, the
    mixin's own base — see :class:`AIToolLoopRulesHost` for that contract.
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


class AIStreamingMixin(AIToolLoopRulesMixin):
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

    async def _close_thinking_window(
        self,
        coalescer: _ThinkingCoalescer,
        room_id: str,
        thinking_parts: list[str],
        round_idx: int,
        published: int,
    ) -> int:
        """Flush the buffered reasoning, publish ``THINKING_END``, return the offset.

        The window closes whenever the model stops reasoning and starts
        producing — a text delta, a tool call's first fragment, or the end of
        the round's stream. One place, so a fourth producer cannot close it
        differently from the other three.

        A round can open several windows (reason, answer, reason again), and
        each ``THINKING_END`` must carry its own block. ``published`` is how
        many of ``thinking_parts`` earlier windows already sent; the caller
        keeps the returned value and hands it back at the next close. The list
        itself is never truncated — the tool loop replays it whole into the
        assistant message it sends back to the model.
        """
        await coalescer.flush()
        await self._publish_thinking_event(
            EphemeralEventType.THINKING_END,
            room_id,
            "".join(thinking_parts[published:]),
            round_idx,
        )
        return len(thinking_parts)

    def _new_tool_call_coalescer(
        self, room_id: str | None, round_idx: int
    ) -> _ToolCallDeltaCoalescer:
        """Coalescer bound to this channel's publish hook and window config.

        It shares the thinking windows on purpose: both bound the rate at which
        one round's in-progress work reaches the bus, and a second pair of knobs
        would be public surface with no demonstrated need behind it.
        """
        return _ToolCallDeltaCoalescer(
            self._publish_tool_event,
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
        thinking_published = 0
        thinking_started = False
        coalescer = self._new_thinking_coalescer(room_id, round_idx=0)

        # Through the resilience wrapper, like every structured generation:
        # retry, fallback and overflow compaction are the wrapper's to give,
        # never a per-path courtesy.
        async for ev in self._generate_stream_with_retry(ai_context):
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
                    thinking_published = await self._close_thinking_window(
                        coalescer, room_id, thinking_parts, 0, thinking_published
                    )
                yield ev.text

        # Thinking with no following text — close the boundary anyway so
        # subscribers see the reasoning even if the model emitted nothing else.
        if thinking_started and thinking_parts and room_id:
            await self._close_thinking_window(
                coalescer, room_id, thinking_parts, 0, thinking_published
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
            _current_loop_ctx,
            _ToolLoopContext,
        )

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
        _total_usage: dict[str, int] = {}
        _span_errored = False
        _t0_stream = time.monotonic()
        _accumulated_text: list[str] = []
        room_id = context.room.room.id if context.room else None
        try:
            context, should_cancel = self._drain_steering_queue(context, loop_ctx)
            if should_cancel:
                yield LoopEndMarker(reason="cancelled", rounds=0)
                return
            state = self._new_loop_state("Streaming tool loop")

            _dedup_prefix = ""
            _saw_tool_call_any = False

            for _round_idx in range(self._max_tool_rounds + 1):
                if loop_ctx.cancel_event.is_set():
                    logger.info("Streaming tool loop cancelled before round %d", _round_idx)
                    yield LoopEndMarker(reason="cancelled", rounds=_round_idx)
                    return

                # Anti-loop ripcord (force_stop): strip tools + nudge once so
                # this round must produce a plain-text answer instead of
                # hammering the same call to the round limit.
                context = self._prepare_round_context(context, loop_ctx, state, _round_idx)

                thinking_parts: list[str] = []
                thinking_published = 0
                thinking_signature: str | None = None
                text_parts: list[str] = []
                tool_calls: list[StreamToolCall] = []
                thinking_started = False
                round_finish_reason: str | None = None
                coalescer = self._new_thinking_coalescer(room_id, round_idx=_round_idx)
                tool_coalescer = self._new_tool_call_coalescer(room_id, round_idx=_round_idx)
                _dedup_active = bool(_dedup_prefix)
                _dedup_offset = 0
                _dedup_buffer: list[str] = []

                async for event in self._generate_stream_with_retry(context):
                    # Check cancel between every stream event — allows immediate
                    # cancellation instead of waiting for the full stream to finish.
                    if loop_ctx.cancel_event.is_set():
                        logger.info("Streaming cancelled mid-generation at round %d", _round_idx)
                        if room_id:
                            await tool_coalescer.close()
                        yield LoopEndMarker(reason="cancelled", rounds=_round_idx)
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
                            thinking_published = await self._close_thinking_window(
                                coalescer, room_id, thinking_parts, _round_idx, thinking_published
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
                    elif isinstance(event, StreamToolCallDelta):
                        # Composing a tool call's arguments ends the reasoning
                        # window exactly as the first text delta does: the model
                        # has stopped thinking and started producing. Without
                        # this a round that reasons and then calls a tool with
                        # no text leaves THINKING_START open for the whole
                        # composition.
                        if thinking_started and thinking_parts and room_id:
                            thinking_started = False
                            thinking_published = await self._close_thinking_window(
                                coalescer, room_id, thinking_parts, _round_idx, thinking_published
                            )
                        if room_id:
                            await tool_coalescer.add(
                                event.index,
                                event.id,
                                event.name,
                                len(event.arguments_delta),
                            )
                    elif isinstance(event, StreamToolCall):
                        tool_calls.append(event)
                        # External tools: fire hooks and yield persistence markers
                        if self._tool_handler is None and self._external_tool_handler is not None:
                            handler = self._external_tool_handler
                            # Extract result from arguments if embedded by proxy
                            args = dict(event.arguments)
                            provider_already_executed = "_result" in args
                            tool_result = args.pop("_result", None)
                            tool_is_error = args.pop("_is_error", False)

                            # Yield start marker for store persistence
                            yield ToolCallStartMarker(
                                tool_name=event.name,
                                tool_id=event.id,
                                arguments=args,
                            )
                            if room_id:
                                await self._publish_tool_event(
                                    EphemeralEventType.TOOL_CALL_START,
                                    room_id,
                                    [event.model_copy(update={"arguments": args})],
                                    _round_idx,
                                )

                            t0_ext = time.monotonic()
                            effective_result = tool_result or ""
                            if not provider_already_executed:
                                # Some external transports expose a pending call
                                # through the stream. Only those calls can still
                                # be gated or rewritten. A proxy that embeds
                                # ``_result`` has already performed the side
                                # effect; firing BEFORE_TOOL_USE then would give
                                # a dangerous, retroactive illusion of control.
                                decision = await handler.process_tool_call(
                                    event.name,
                                    args,
                                    tool_call_id=event.id,
                                    room_id=room_id,
                                )
                                if not decision.approved:
                                    effective_result = json.dumps(
                                        {
                                            "error": decision.reason
                                            or f"Tool '{event.name}' was denied"
                                        }
                                    )
                                    tool_is_error = True
                                else:
                                    if decision.modified_input is not None:
                                        args = decision.modified_input
                                    if decision.result is not None:
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
                            if room_id:
                                await self._publish_tool_event(
                                    EphemeralEventType.TOOL_CALL_END,
                                    room_id,
                                    [
                                        AIToolResultPart(
                                            tool_call_id=event.id,
                                            name=event.name,
                                            result=effective_result,
                                        )
                                    ],
                                    _round_idx,
                                    duration_ms=ext_duration_ms,
                                )
                    elif isinstance(event, StreamDone):
                        round_finish_reason = event.finish_reason
                        if event.usage:
                            round_in = event.usage.get("input_tokens", 0)
                            round_out = event.usage.get("output_tokens", 0)
                            _accumulate_usage(_total_usage, event.usage)
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
                    await self._close_thinking_window(
                        coalescer, room_id, thinking_parts, _round_idx, thinking_published
                    )
                # The composition is over for this round, whichever way the
                # round now ends — including the exits below that never reach
                # TOOL_CALL_START.
                if room_id:
                    await tool_coalescer.close()

                if not tool_calls:
                    # Final answer round. If it produced no text *after* a tool
                    # round, the model skipped verbalizing the result — re-prompt
                    # once (bounded) for the final answer instead of ending empty.
                    final_text = "".join(text_parts)
                    if self._try_empty_retry(
                        context,
                        loop_ctx,
                        state,
                        had_tool_round=_saw_tool_call_any,
                        final_text=final_text,
                        finish_reason=round_finish_reason,
                    ):
                        continue
                    # The one exit that is both the happy path and a silent
                    # failure, which is why it is the one that has to be named.
                    reason: LoopEndReason = final_round_reason(
                        had_tool_round=_saw_tool_call_any,
                        final_text=final_text,
                        finish_reason=round_finish_reason,
                        deadline_exceeded=state.deadline_exceeded(),
                        force_stopped=loop_ctx.force_stop,
                    )
                    yield LoopEndMarker(reason=reason, rounds=_round_idx)
                    return

                _saw_tool_call_any = True

                # Calls without a local handler are owned by an external
                # provider. An ExternalToolHandler observes them inline above.
                # Without one, dispatch the correct lifecycle hook directly:
                # BEFORE only for a still-pending call, ON_TOOL_CALL when the
                # provider embedded ``_result`` after executing it.
                if self._tool_handler is None:
                    if self._external_tool_handler is None:
                        for tc in tool_calls:
                            external_args = dict(tc.arguments)
                            provider_already_executed = "_result" in external_args
                            external_result = external_args.pop("_result", None)
                            external_args.pop("_is_error", None)
                            external_event = ToolCallEvent(
                                channel_id=self.channel_id,
                                channel_type=ChannelType.AI,
                                tool_call_id=tc.id,
                                name=tc.name,
                                arguments=external_args,
                                result=(
                                    external_result
                                    if isinstance(external_result, (str, list))
                                    else json.dumps(external_result)
                                    if external_result is not None
                                    else None
                                ),
                                room_id=room_id,
                            )
                            if provider_already_executed and self._tool_call_hook is not None:
                                await self._tool_call_hook(external_event)
                            elif (
                                not provider_already_executed
                                and self._before_tool_call_hook is not None
                            ):
                                await self._before_tool_call_hook(
                                    ToolCallEvent(
                                        channel_id=self.channel_id,
                                        channel_type=ChannelType.AI,
                                        tool_call_id=tc.id,
                                        name=tc.name,
                                        arguments=external_args,
                                        result=None,
                                        room_id=room_id,
                                    )
                                )

                    # External tools were handled inline during streaming.
                    # Persistence markers were yielded alongside hook callbacks.
                    # The turn's tool work is the provider's, already done, so
                    # this exit is a completion — and it still names itself:
                    # "every exit yields a marker" has no external-tools carve-out.
                    yield LoopEndMarker(reason="completed", rounds=_round_idx)
                    return

                if _round_idx >= self._max_tool_rounds:
                    logger.warning(
                        "Streaming tool loop reached max_tool_rounds=%d",
                        self._max_tool_rounds,
                    )
                    yield LoopEndMarker(reason="max_rounds", rounds=_round_idx)
                    return

                if state.deadline_exceeded():
                    logger.warning(
                        "Streaming tool loop timeout after %d rounds (%.0fs)",
                        _round_idx,
                        self._tool_loop_timeout_seconds,
                    )
                    yield LoopEndMarker(reason="timeout", rounds=_round_idx)
                    return

                state.warn_if_needed(_round_idx)

                tool_calls = self._cap_round_tool_calls(tool_calls, state.log_label)

                logger.info(
                    "Streaming tool round %d: %d call(s)",
                    _round_idx + 1,
                    len(tool_calls),
                )

                accumulated_text = "".join(text_parts)
                parts = self._build_assistant_parts(
                    "".join(thinking_parts),
                    thinking_signature,
                    accumulated_text,
                    tool_calls,
                )
                if accumulated_text:
                    _dedup_prefix = accumulated_text
                context.messages.append(AIMessage(role="assistant", content=parts))

                # Yield start markers for each tool call (persistence boundary)
                for tc in tool_calls:
                    yield ToolCallStartMarker(
                        tool_name=tc.name,
                        tool_id=tc.id,
                        arguments=tc.arguments,
                    )
                result_parts, duration_ms, executed_arguments = await self._execute_round_tools(
                    context,
                    tool_calls,
                    telemetry,
                    room_id,
                    _round_idx,
                    parent_span_id=span_id,
                )

                # Yield end markers with results (persistence boundary)
                for tc, rp in zip(tool_calls, result_parts, strict=False):
                    result_val = getattr(rp, "result", None)
                    is_error = isinstance(result_val, str) and result_val.startswith(
                        "Error executing tool"
                    )
                    yield ToolCallEndMarker(
                        tool_name=tc.name,
                        tool_id=tc.id,
                        arguments=executed_arguments.get(tc.id, tc.arguments),
                        result=result_val,
                        status="failed" if is_error else "completed",
                        duration_ms=duration_ms,
                        error=result_val if is_error else None,
                        structured_content=getattr(rp, "structured_content", None),
                    )
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
                    yield LoopEndMarker(reason="cancelled", rounds=_round_idx)
                    return

            # The for loop ran out of indices without returning — only possible
            # when an empty-retry consumed the final one. The budget is spent;
            # name it rather than letting the stream just end.
            yield LoopEndMarker(reason="max_rounds", rounds=self._max_tool_rounds)
        except Exception as exc:
            _span_errored = True
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            raise
        finally:
            if not _span_errored:
                usage_attrs: dict[str, Any] = {}
                if _total_usage.get("input_tokens") or _total_usage.get("output_tokens"):
                    usage_attrs[Attr.LLM_INPUT_TOKENS] = _total_usage.get("input_tokens", 0)
                    usage_attrs[Attr.LLM_OUTPUT_TOKENS] = _total_usage.get("output_tokens", 0)
                telemetry.end_span(span_id, attributes=usage_attrs)

            if self._after_response_hook and not _span_errored:
                try:
                    await self._after_response_hook(
                        AIResponseEvent(
                            channel_id=self.channel_id,
                            response_content="".join(_accumulated_text),
                            room_id=room_id,
                            # The zero defaults keep the two counters always
                            # present, as consumers of this event have read them.
                            usage={
                                "input_tokens": 0,
                                "output_tokens": 0,
                                **_total_usage,
                            },
                            latency_ms=int((time.monotonic() - _t0_stream) * 1000),
                            streaming=True,
                        )
                    )
                except Exception:
                    logger.debug("After-response hook failed (streaming)", exc_info=True)

            self._active_loops.pop(loop_ctx.loop_id, None)
            _current_loop_ctx.set(None)

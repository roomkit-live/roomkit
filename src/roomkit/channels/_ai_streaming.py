"""AIChannel mixin for the tool loop every turn runs, whatever its provider
streams and whether it carries tools (RFC §6.4)."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from roomkit.channels._ai_coalescers import (
    _TextCoalescer,
    _ThinkingCoalescer,
    _ToolCallDeltaCoalescer,
)
from roomkit.channels._ai_loop_rules import (
    AIToolLoopRulesMixin,
    _ToolLoopState,
    final_round_reason,
    interrupts_turn,
    require_schema_answer,
    turn_span_status,
    warn_refused,
)
from roomkit.channels._ai_stream_external_tools import _ExternalStreamTools
from roomkit.channels._ai_stream_round import _StreamRound, _StreamRoundState
from roomkit.channels._ai_tools import call_end_marker
from roomkit.channels._mark_copies import compile_mark_patterns, without_mark_copies
from roomkit.channels._note_blocks import add_turn_note
from roomkit.channels._turn_notes import carries_turn_images
from roomkit.core._turn_scope import current_turn_id
from roomkit.core.task_utils import shielded
from roomkit.models.channel import ChannelOutput
from roomkit.models.event import RoomEvent
from roomkit.models.streaming import (
    LoopEndMarker,
    LoopEndReason,
    SegmentBreakMarker,
    StreamDelta,
    ToolCallEndMarker,
    ToolCallStartMarker,
    UsageMarker,
    reports_counters,
)
from roomkit.models.tool_call import AIResponseEvent, ToolRoundEvent, response_transcript
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIToolResultPart,
    ProviderError,
    StreamToolCall,
)
from roomkit.providers.ai.tool_intent import owning_tools, take_intent
from roomkit.realtime.base import EphemeralEventType
from roomkit.telemetry.base import Attr, SpanKind, TelemetryProvider, span_error
from roomkit.telemetry.context import get_current_span
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import AfterToolRoundHook, BeforeGenerationHook
    from roomkit.channels._author_images import RefusedImages
    from roomkit.channels._tool_usage import ToolUsageMemory
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.tool_call import AfterResponseCallback, ToolCallObserver
    from roomkit.providers.ai.base import AIProvider
    from roomkit.tools.external import ExternalToolHandler


if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


@dataclass
class _StreamTurnState:
    """Invocation-owned state; each round contributes its delivered fragments."""

    loop_ctx: _ToolLoopContext
    telemetry: TelemetryProvider
    span_id: str
    room_id: str | None
    # The limits the turn runs under, as its end marker states them.
    max_rounds: int
    timeout_seconds: float | None
    # What the turn's last generation ran on: the channel's provider until
    # its fallback serves one.
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    # What the turn's last generation used, ``None`` once one failed, and
    # whether every generation reported its counters (RFC §6.4).
    last_usage: dict[str, Any] | None = field(default_factory=dict)
    usage_complete: bool = True
    segments: list[list[str]] = field(default_factory=list)
    tool_calls_count: int = 0
    tool_rounds_count: int = 0
    reason: LoopEndReason = "completed"
    started_at: float = field(default_factory=time.monotonic)
    dedup_prefix: str = ""
    saw_tool_call: bool = False
    # Each round's reasoning, for the turn's ON_AI_RESPONSE.
    thinking: list[str] = field(default_factory=list)
    # The provider error that interrupted the turn after a round (reason
    # ``error``): the turn reaches its end on it, then it is raised (RFC §6.4).
    error: Exception | None = None

    def count_round(self, calls: list[Any]) -> None:
        """Count one round of the channel's calls toward the turn's span."""
        self.tool_calls_count += len(calls)
        self.tool_rounds_count += 1

    def count_generation(self, state: _StreamRoundState, served_by: AIProvider) -> None:
        """Take what a generation that returned leaves the turn's record."""
        self.last_usage = dict(state.usage)
        self.usage_complete = self.usage_complete and reports_counters(state.usage)
        self.model = served_by.model_name
        self.provider = served_by.name

    def generation_failed(self) -> None:
        """A generation failed: the turn's last one reported nothing."""
        self.last_usage = None
        self.usage_complete = False

    def end(self, reason: LoopEndReason) -> LoopEndMarker:
        """End the loop on *reason*: the marker the consumer reads it from,
        with the tool rounds that ran and the limits the turn ran under. The
        loop reads its steering queue no more (RFC §21.3)."""
        self.reason = reason
        self.loop_ctx.steering_closed = True
        budget = self.loop_ctx.turn_budget
        return LoopEndMarker(
            reason=reason,
            rounds=self.tool_rounds_count,
            usage=dict(self.usage),
            max_rounds=self.max_rounds,
            timeout_seconds=self.timeout_seconds,
            budget_tokens=budget.tokens if budget is not None else None,
            budget_usd=budget.usd if budget is not None else None,
            last_usage=None if self.last_usage is None else dict(self.last_usage),
            usage_complete=self.usage_complete,
            model=self.model,
            provider=self.provider,
        )


def _turn_span_attributes(turn: _StreamTurnState) -> dict[str, Any]:
    """What the turn's rounds used, whether or not the turn reached its end."""
    attributes: dict[str, Any] = {Attr.LLM_TOOL_COUNT: turn.tool_calls_count}
    if turn.usage.get("input_tokens") or turn.usage.get("output_tokens"):
        attributes[Attr.LLM_INPUT_TOKENS] = turn.usage.get("input_tokens", 0)
        attributes[Attr.LLM_OUTPUT_TOKENS] = turn.usage.get("output_tokens", 0)
    return attributes


def _end_unfinished_turn(turn: _StreamTurnState, exc: BaseException) -> None:
    """End the span of a turn whose loop did not reach its end (RFC §6.4).

    A raise is an error. A close at a yield (a barge-in, a transport that
    stopped reading, a consumer that refused the answer) or a cancelled task
    is a cancellation. Such a turn reports nothing, so its span is where what
    its rounds used stays on record.
    """
    attributes = _turn_span_attributes(turn)
    if isinstance(exc, Exception):
        turn.telemetry.end_span(
            turn.span_id,
            status="error",
            error_message=span_error(turn.telemetry, exc),
            attributes=attributes,
        )
    else:
        turn.telemetry.end_span(turn.span_id, status="cancelled", attributes=attributes)


def _round_stop_reason(
    state: _StreamRoundState, loop_ctx: _ToolLoopContext
) -> LoopEndReason | None:
    """Why the loop stops after a round, if it does; a stop that came after the
    model's last event counts as one that came during it."""
    if state.cancelled or loop_ctx.cancel_event.is_set():
        return "cancelled"
    return "force_stopped" if loop_ctx.force_stop else None


async def _unrun_call_ends(calls: list[Any]) -> AsyncGenerator[StreamDelta, None]:
    """A failed end for each announced call a stop kept from running (RFC §21.3)."""
    for call in calls:
        yield ToolCallEndMarker(
            tool_name=call.name,
            tool_id=call.id,
            arguments=call.arguments,
            status="failed",
            error="cancelled",
            outcome="cancelled",
            intent=call.intent,
        )


def _answered_end(
    context: AIContext, turn: _StreamTurnState, reason: LoopEndReason
) -> LoopEndMarker:
    """The loop's end on *reason*, a constrained turn that ends without its
    answer failing first, inside the loop: whoever reads the loop (a room's
    turn, a reasoning backend) sees the same end (see
    :func:`require_schema_answer`)."""
    marker = turn.end(reason)
    require_schema_answer(context, reason)
    return marker


class AIStreamingMixin(AIToolLoopRulesMixin):
    """Streaming AI response generation with tool loop and deduplication.

    What it calls on the other mixins is declared once, in
    :class:`~roomkit.channels._ai_contract._AIChannelContract`, which it
    derives from for the type checker only.
    """

    _provider: AIProvider
    _fallback_provider: AIProvider | None
    _max_tool_rounds: int
    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _max_empty_retries: int
    _thinking_coalesce_ms: float
    _thinking_coalesce_chars: int
    _publish_text_deltas: bool
    _tool_call_intents: bool
    _text_coalesce_ms: float
    _text_coalesce_chars: int
    _active_loops: dict[str, _ToolLoopContext]
    _after_response_hook: AfterResponseCallback | None
    _before_generation_hook: BeforeGenerationHook | None
    _after_tool_round_hook: AfterToolRoundHook | None
    _tool_report_hook: ToolCallObserver | None
    _tool_observer_hook: ToolCallObserver | None
    _external_tool_handler: ExternalToolHandler | None
    _tool_usage: ToolUsageMemory
    _refused_images: RefusedImages
    channel_id: str

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
        thinking_chunks: list[str],
        round_idx: int,
        *,
        published: int,
    ) -> int:
        """Flush the buffered reasoning, publish ``THINKING_END``, return the offset.

        The window closes whenever the model stops reasoning and starts
        producing — a text delta, a tool call's first fragment, or the end of
        the stream — and on every abnormal exit of a round: a cancelled turn,
        a provider that died mid-reasoning, a consumer that stopped reading.
        One place for every call site, so a new exit cannot close a window
        differently from the others.

        Whatever closed it, ``THINKING_END`` carries the block reasoned so
        far. The subscriber has been reading that block in deltas, so an
        empty payload on an abnormal exit would be a second contract to
        learn for the one case where the block is already at hand; and the
        deltas the coalescer still holds go out ahead of the close instead
        of dying with the round.

        A round can open several windows (reason, answer, reason again), and
        each ``THINKING_END`` must carry its own block. ``published`` is how
        many of ``thinking_chunks`` earlier windows already sent; the caller
        keeps the returned value and hands it back at the next close. The list
        itself is never truncated — the tool loop replays it whole into the
        assistant message it sends back to the model.
        """
        await coalescer.flush()
        await self._publish_thinking_event(
            EphemeralEventType.THINKING_END,
            room_id,
            "".join(thinking_chunks[published:]),
            round_idx,
        )
        return len(thinking_chunks)

    def _new_text_coalescer(self, room_id: str | None, round_idx: int) -> _TextCoalescer | None:
        """The round's ``TEXT_DELTA`` window, or ``None`` when the channel
        publishes no answer text: the host did not ask for it, and
        ``BEFORE_BROADCAST`` does not see that stream (RFC §6.4)."""
        if not self._publish_text_deltas:
            return None
        return _TextCoalescer(
            self._publish_text_event,
            room_id,
            round_idx,
            flush_ms=self._text_coalesce_ms,
            flush_chars=self._text_coalesce_chars,
        )

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

    async def _start_streaming_tool_response(
        self,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
        *,
        notes: tuple[str, ...] = (),
    ) -> ChannelOutput:
        """Return a streaming response that handles tool calls between rounds.

        *notes* join the turn's notes before ``BEFORE_AI_GENERATION``, which sees
        them (a speak policy's decision, RFC §6.4).
        """
        ai_context = await self._build_context(event, binding, context)
        if notes:
            messages = list(ai_context.messages)
            for block in notes:
                messages = add_turn_note(messages, block)
            ai_context = ai_context.model_copy(update={"messages": messages})
        ai_context, blocked = await self._fire_before_generation_hook(ai_context, event)
        if blocked:
            return ChannelOutput.empty()
        # The generator below executes when the CONSUMER iterates the
        # stream — by then handle_event has reset the loop contextvar, so
        # the parent ctx (participant role, room, the toolset stamped by
        # _build_context) must be captured NOW and passed explicitly. So is
        # the span the turn answers under (the broadcast's), which the
        # consumer no longer runs in.
        return ChannelOutput(
            responded=True,
            response_stream=self._run_streaming_tool_loop(
                ai_context,
                parent_loop_ctx=_current_loop_ctx.get(),
                parent_span_id=get_current_span(),
            ),
            response_metadata=ai_context.response_metadata,
        )

    def _record_stream_usage(
        self, total: dict[str, int], rules: _ToolLoopState, usage: dict[str, Any]
    ) -> None:
        """Record a generation's usage: into the turn's total, against its
        budget, and as the input/output metrics."""
        rules.count(total, usage)
        telemetry = self._telemetry_provider
        for counter in ("input_tokens", "output_tokens"):
            telemetry.record_metric(
                f"roomkit.llm.{counter}",
                float(usage.get(counter, 0)),
                unit="tokens",
                attributes={"channel_id": self.channel_id},
            )

    def _new_turn_state(
        self, loop_ctx: _ToolLoopContext, room_id: str | None, parent_span_id: str | None
    ) -> _StreamTurnState:
        """The turn's state, its ``llm.generate`` span started, and the limits
        it runs under: the ones the loop enforces and its end marker states."""
        telemetry = self._telemetry_provider
        span_id = telemetry.start_span(
            SpanKind.LLM_GENERATE,
            "llm.generate",
            parent_id=parent_span_id or get_current_span(),
            room_id=room_id,
            channel_id=self.channel_id,
            attributes={
                Attr.PROVIDER: type(self._provider).__name__,
                Attr.LLM_STREAMING: True,
            },
        )
        return _StreamTurnState(
            loop_ctx,
            telemetry,
            span_id,
            room_id,
            max_rounds=self._max_tool_rounds,
            # Unset or zero: no deadline.
            timeout_seconds=self._tool_loop_timeout_seconds or None,
            model=self._provider.model_name,
            provider=self._provider.name,
        )

    @asynccontextmanager
    async def _streaming_tool_turn(
        self,
        context: AIContext,
        parent_loop_ctx: _ToolLoopContext | None,
        parent_span_id: str | None = None,
    ) -> AsyncGenerator[_StreamTurnState, None]:
        """Own the invocation context, activity registration and telemetry span."""
        # This body runs in the CONSUMER's context, which may hold a loop
        # context of its own (a handler draining a child channel's stream):
        # that is the value to put back when the turn ends, by value rather
        # than by token, since the turn may end in yet another context.
        enclosing_ctx = _current_loop_ctx.get()
        parent = parent_loop_ctx if parent_loop_ctx is not None else enclosing_ctx
        room = context.room.room if context.room else None
        room_id = room.id if room is not None else None
        loop_ctx = _ToolLoopContext.for_loop(parent, room_id, room=room)
        loop_ctx.turn_id = current_turn_id()
        loop_ctx.channel_id = self.channel_id
        _current_loop_ctx.set(loop_ctx)
        self._active_loops[loop_ctx.loop_id] = loop_ctx
        try:
            turn = self._new_turn_state(loop_ctx, room_id, parent_span_id)
            try:
                yield turn
            except BaseException as exc:
                await self._end_raised_turn(turn, exc)
                raise
            await self._finish_streaming_tool_turn(turn)
        finally:
            # Finalization may itself be cancelled while publishing a hook.
            self._active_loops.pop(loop_ctx.loop_id, None)
            try:
                await shielded(self._report_unreported_calls(loop_ctx))
            finally:
                loop_ctx.ended.set()
                loop_ctx.running = False
                _current_loop_ctx.set(enclosing_ctx)

    async def _end_raised_turn(self, turn: _StreamTurnState, exc: BaseException) -> None:
        """End a turn left by an exception: reported when the provider interrupted
        it after a round, since it reached its end on that error (RFC §6.4)."""
        if exc is turn.error:
            await self._finish_streaming_tool_turn(turn)
        else:
            _end_unfinished_turn(turn, exc)

    async def _finish_streaming_tool_turn(self, turn: _StreamTurnState) -> None:
        """Report the delivered transcript and the counters accumulated by this turn."""
        turn.telemetry.end_span(
            turn.span_id,
            status=turn_span_status(turn.reason),
            error_message=None if turn.error is None else span_error(turn.telemetry, turn.error),
            attributes=_turn_span_attributes(turn),
        )
        if self._after_response_hook:
            try:
                segments, transcript = response_transcript("".join(text) for text in turn.segments)
                await self._after_response_hook(
                    AIResponseEvent(
                        channel_id=self.channel_id,
                        response_content=transcript,
                        segments=segments,
                        room_id=turn.room_id,
                        tool_calls_count=turn.tool_calls_count,
                        round_count=turn.tool_rounds_count,
                        loop_end_reason=turn.reason,
                        declared_tools=list(turn.loop_ctx.declared_tools.values()),
                        thinking="\n\n".join(turn.thinking),
                        usage={"input_tokens": 0, "output_tokens": 0, **turn.usage},
                        latency_ms=int((time.monotonic() - turn.started_at) * 1000),
                        streaming=True,
                    )
                )
            except Exception:
                logger.debug("After-response hook failed (streaming)", exc_info=True)

    async def _stream_local_tool_round(
        self,
        context: AIContext,
        state: _StreamRoundState,
        turn: _StreamTurnState,
        index: int,
    ) -> AsyncGenerator[StreamDelta, None]:
        """Persist an assistant's calls and surround execution with lifecycle markers."""
        calls = self._cap_round_tool_calls(state.tool_calls, "Streaming tool loop")
        logger.info("Streaming tool round %d: %d call(s)", index + 1, len(calls))
        if state.text:
            turn.dedup_prefix = state.text
        # The provider's own calls of the round go back with the channel's,
        # each with its result, so the model reads every call it made.
        context.messages.append(
            AIMessage(
                role="assistant",
                content=state.transcript.parts([*state.provider_calls, *calls]),
            )
        )
        markers = [
            ToolCallStartMarker(
                tool_name=call.name, tool_id=call.id, arguments=call.arguments, intent=call.intent
            )
            for call in calls
        ]
        # Announced before the first start goes out: whatever cuts the round
        # from here on, the turn's end reports each call no report claimed.
        _announce(turn.loop_ctx, calls, markers)
        for marker in markers:
            yield marker
        # A stop that came while the calls were announced: none of them runs,
        # and the loop ends cancelled at its next check (RFC §21.3).
        ends = (
            _unrun_call_ends(calls)
            if turn.loop_ctx.cancel_event.is_set()
            else self._run_announced_calls(
                context, calls, markers, turn, index, state.provider_results
            )
        )
        async with aclosing(ends) as deltas:
            async for delta in deltas:
                yield delta

    async def _run_announced_calls(
        self,
        context: AIContext,
        calls: list[Any],
        markers: list[ToolCallStartMarker],
        turn: _StreamTurnState,
        index: int,
        answered: list[AIToolResultPart],
    ) -> AsyncGenerator[StreamDelta, None]:
        """Execute a round's announced calls, yield each call's end marker
        (set on its start marker as it finished, with the arguments it ran
        with), then hand the round to AFTER_TOOL_ROUND; *answered* are the
        round's calls the provider served, read beside them."""
        results, duration_ms = await self._execute_round_tools(
            context,
            calls,
            turn.telemetry,
            turn.room_id,
            index,
            parent_span_id=turn.span_id,
            answered=answered,
        )
        turn.count_round(calls)
        for call, marker, result in zip(calls, markers, results, strict=False):
            # Each call's end was set on its start as it finished; one is
            # built here only if it was not, so every start gets its end.
            if marker.ended is not None:
                yield marker.ended
            else:
                yield call_end_marker(call, marker, result, duration_ms)
        if turn.room_id:
            await self._publish_tool_event(
                EphemeralEventType.TOOL_CALL_END,
                turn.room_id,
                results,
                index,
                duration_ms=duration_ms,
                for_calls=calls,
            )
        await self._after_tool_round(context, calls, results, answered, turn, index)

    async def _after_tool_round(
        self,
        context: AIContext,
        calls: list[Any],
        results: list[AIToolResultPart],
        answered: list[AIToolResultPart],
        turn: _StreamTurnState,
        index: int,
    ) -> None:
        """Hand the round the channel just ran to AFTER_TOOL_ROUND and apply
        what its hooks asked: withdrawals for the rest of the turn, messages
        the next round reads after the results (RFC §6.4)."""
        hook = self._after_tool_round_hook
        if hook is None:
            return
        event = ToolRoundEvent(
            channel_id=self.channel_id,
            room_id=turn.room_id,
            round_index=index,
            calls=list(calls),
            results=list(results),
            answered=list(answered),
            tools=self._hook_toolset(turn.loop_ctx),
        )
        await hook(event)
        if event.withdrawn:
            turn.loop_ctx.withdraw(event.withdrawn)
        # What a hook adds after a round may quote the round's results: a copy
        # of a runtime mark in it is replaced, as in a message steering
        # injects (RFC §6.4).
        context.messages.extend(
            AIMessage(role="user", content=without_mark_copies(text)) for text in event.messages
        )

    async def _stream_generation(
        self, round_: _StreamRound, context: AIContext, turn: _StreamTurnState, index: int
    ) -> AsyncGenerator[StreamDelta, None]:
        """One round's generation; a provider error once a round ran ends the turn.

        The rounds already reached the room, so the turn reaches its end on
        the error, its ON_AI_RESPONSE fired, and the error then reaches the
        consumer (RFC §6.4). A turn that fails before any round is the error
        itself. Either way the stream's consumer logs it, once: the loop
        raises and logs nothing.
        """
        try:
            async with aclosing(
                round_.stream(
                    self._generate_stream_with_retry(
                        context,
                        received=round_.state.note_received,
                        on_fallback=round_.state.note_fallback,
                    )
                )
            ) as deltas:
                async for delta in deltas:
                    yield delta
        except Exception as exc:
            turn.generation_failed()
            # A provider fails whatever it raises (RMK-1199); only its own
            # error can say it refused the content, before it handed anything.
            first = index == 0 and not round_.state.received
            if first and isinstance(exc, ProviderError) and refuses_content(exc):
                self._forget_sent_images(turn, context)
            if not interrupts_turn(exc, after_round=turn.saw_tool_call):
                raise
            turn.error = exc
            yield _answered_end(context, turn, "error")
            raise
        if round_.state.thinking:
            turn.thinking.append(round_.state.thinking)
        turn.count_generation(round_.state, self._served_by(round_.state))
        yield UsageMarker(usage=dict(round_.state.usage), model=turn.model, provider=turn.provider)

    def _served_by(self, state: _StreamRoundState) -> AIProvider:
        """The provider that served a generation."""
        if state.by_fallback and self._fallback_provider is not None:
            return self._fallback_provider
        return self._provider

    def _forget_sent_images(self, turn: _StreamTurnState, context: AIContext) -> None:
        """Send no more the images a request whose content the provider refused
        carried: one the provider refused would fail every later turn while it
        stays in the window (RFC §6.4). The room's replayed tool images are
        replayed no more, its authors' images read as refused; a request that
        carried none changes nothing."""
        if carries_turn_images(context.messages):
            self._tool_usage.forget_images(turn.room_id)
        images = turn.loop_ctx.author_images
        if images is not None and images.sent and turn.room_id is not None:
            self._refused_images.add(turn.room_id, images.sent)

    async def _run_streaming_tool_loop(
        self,
        context: AIContext,
        *,
        parent_loop_ctx: _ToolLoopContext | None = None,
        parent_span_id: str | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """Orchestrate generation, termination decisions and local tool rounds."""
        # A loop driven without a built context (a realtime reasoning turn)
        # cleans what steering injects with patterns not compiled yet.
        await compile_mark_patterns()
        async with self._streaming_tool_turn(context, parent_loop_ctx, parent_span_id) as turn:
            loop_ctx = turn.loop_ctx
            external = self._external_stream_tools(turn)
            context, cancelled = self._drain_steering_queue(context, loop_ctx)
            if cancelled:
                yield turn.end("cancelled")
                return
            rules = self._new_loop_state(
                "Streaming tool loop", turn.timeout_seconds, loop_ctx.turn_budget
            )

            for index in range(self._max_tool_rounds + 1):
                if loop_ctx.cancel_event.is_set():
                    yield turn.end("cancelled")
                    return
                context = self._prepare_round_context(context, loop_ctx, rules, index)
                round_ = self._new_stream_round(turn, rules, index, external, context)
                turn.segments.append(round_.state.reported)
                # What this round declares: its tools' own schemas, which the
                # request shows with the intent property when asked (RFC §6.4).
                self._record_declared_tools(loop_ctx, context.tools)
                async with aclosing(
                    self._stream_generation(round_, context, turn, index)
                ) as deltas:
                    async for delta in deltas:
                        yield delta
                outcome = self._round_outcome(round_.state, turn, context, rules, index)
                if outcome == "retry":
                    yield SegmentBreakMarker()
                    continue
                if outcome == "completed" and index < self._max_tool_rounds:
                    context, again = self._steered_after_answer(context, round_.state, loop_ctx)
                    if again:
                        yield SegmentBreakMarker()
                        continue
                if outcome is not None:
                    yield _answered_end(context, turn, outcome)
                    return

                rules.warn_if_needed(turn.tool_rounds_count)
                async with aclosing(
                    self._stream_local_tool_round(context, round_.state, turn, index)
                ) as deltas:
                    async for delta in deltas:
                        yield delta
                context, cancelled = self._drain_steering_queue(context, loop_ctx)
                if cancelled:
                    yield turn.end("cancelled")
                    return

            # An empty-response retry can consume the final generation slot.
            yield _answered_end(context, turn, "max_rounds")

    def _steered_after_answer(
        self, context: AIContext, state: _StreamRoundState, loop_ctx: _ToolLoopContext
    ) -> tuple[AIContext, bool]:
        """Whether a message steered in while the model answered asks it again
        (RFC §21.3): the answer joins the conversation, then the message, and
        the same turn goes on, a round more. With none, the loop reads its
        queue no more, nothing awaited in between, so a message that comes
        after reaches no loop that would not read it."""
        if loop_ctx.steering_queue.empty() or state.provider_calls:
            loop_ctx.steering_closed = True
            return context, False
        answer = AIMessage(role="assistant", content=state.transcript.parts([]))
        answered = [*context.messages, answer]
        steered, cancelled = self._drain_steering_queue(
            context.model_copy(update={"messages": answered}), loop_ctx
        )
        # The drain appends what it injects to the copy's messages.
        if cancelled or len(steered.messages) == len(context.messages) + 1:
            loop_ctx.steering_closed = True
            return context, False
        return steered, True

    def _external_stream_tools(self, turn: _StreamTurnState) -> _ExternalStreamTools:
        """The turn's routing of the calls its provider serves."""
        return _ExternalStreamTools(
            channel_id=self.channel_id,
            room_id=turn.room_id,
            loop_ctx=turn.loop_ctx,
            publish=self._publish_tool_event,
            serves_locally=partial(self._serves_locally, turn.loop_ctx),
            bound=self._bound_provider_result,
            handler=self._external_tool_handler,
            report=self._tool_report_hook,
            observe=self._tool_observer_hook,
        )

    def _new_stream_round(
        self,
        turn: _StreamTurnState,
        rules: _ToolLoopState,
        index: int,
        external: _ExternalStreamTools,
        context: AIContext,
    ) -> _StreamRound:
        """Begin a generation round on *context*: its consumer, wired to this
        channel's windows and hooks, with the ids the previous round held freed."""
        # A provider's id names a call within its round only: one under it
        # in this round is a new call (RFC §12.4).
        turn.loop_ctx.calls.next_round()
        return _StreamRound(
            index=index,
            room_id=turn.room_id,
            cancel_event=turn.loop_ctx.cancel_event,
            thinking_coalescer=self._new_thinking_coalescer(turn.room_id, index),
            text_coalescer=self._new_text_coalescer(turn.room_id, index),
            new_composition=partial(self._new_tool_call_coalescer, turn.room_id, index),
            publish_thinking=self._publish_thinking_event,
            close_thinking=self._close_thinking_window,
            record_usage=partial(self._record_stream_usage, turn.usage, rules),
            prefix=turn.dedup_prefix,
            external_tools=external,
            take_intent=self._intent_taker(context, turn.loop_ctx),
        )

    def _intent_taker(
        self, context: AIContext, loop_ctx: _ToolLoopContext
    ) -> Callable[[StreamToolCall], StreamToolCall] | None:
        """What takes a call's intent out of its arguments, with
        ``tool_call_intents`` on: a tool of the round or of the turn's
        catalogue whose own schema declares the property keeps it (RFC §6.4)."""
        if not self._tool_call_intents:
            return None
        owners = owning_tools(context.tools, loop_ctx.all_context_tools)
        return partial(take_intent, owners=owners)

    def _round_outcome(
        self,
        state: _StreamRoundState,
        turn: _StreamTurnState,
        context: AIContext,
        rules: _ToolLoopState,
        index: int,
    ) -> LoopEndReason | Literal["retry"] | None:
        """What a generation leaves the loop: the turn's end, another try at an
        empty answer, or ``None`` for its local calls to run."""
        loop_ctx = turn.loop_ctx
        stop = _round_stop_reason(state, loop_ctx)
        if stop is not None:
            return stop
        if state.provider_calls and not state.tool_calls:
            # The provider ran its calls itself, or its handler decided them:
            # the provider's own loop goes on, not this one (RFC §9.3).
            turn.saw_tool_call = True
            return "completed"
        if not state.tool_calls:
            warn_refused(rules.log_label, state.finish_reason, state.refusal)
            again = self._try_round_again(
                context,
                loop_ctx,
                rules,
                had_tool_round=turn.saw_tool_call,
                final_text=state.text,
                finish_reason=state.finish_reason,
            )
            if again == "retry":
                return "retry"
            return final_round_reason(
                had_tool_round=turn.saw_tool_call,
                final_text=state.text,
                finish_reason=state.finish_reason,
                limit=rules.limit_passed(),
                force_stopped=loop_ctx.force_stop,
                unfinished=again == "unfinished",
            )
        turn.saw_tool_call = True
        if index >= self._max_tool_rounds:
            logger.warning("Streaming tool loop reached max_tool_rounds=%d", self._max_tool_rounds)
            return "max_rounds"
        return rules.limit_reached(turn.tool_rounds_count)

    def _serves_locally(self, loop_ctx: _ToolLoopContext, name: str) -> bool:
        """Whether the turn has a tool of the channel's own under *name*: a call
        to it is the channel's to serve, never an external handler's. A tool
        withdrawn for the turn (by BEFORE_AI_GENERATION or AFTER_TOOL_ROUND)
        stays the channel's, whose gate refuses it (RFC §6.4)."""
        if name in loop_ctx.withdrawn_tools:
            return True
        known = loop_ctx.all_context_tools
        if known is not None and name in {tool.name for tool in known}:
            return True
        return name in self._served_tool_names(loop_ctx.room_id)


def _announce(
    loop_ctx: _ToolLoopContext, calls: list[Any], markers: list[ToolCallStartMarker]
) -> None:
    """Announce a round's calls to the turn, each with its start marker.

    Two calls under one id in the round are two calls: the first holds the
    id, the second is announced a duplicate, refused at its gate (RFC §12.4),
    and each keeps its own record.
    """
    for call, marker in zip(calls, markers, strict=True):
        loop_ctx.calls.announce(call, marker=marker)


_NOT_CONTENT_STATUSES = frozenset({401, 402, 403, 404, 429})


def refuses_content(error: ProviderError) -> bool:
    """Whether *error* may be the provider refusing what a request carried: not
    retried, and no authentication, payment, permission, missing model or rate
    limit (RFC §6.4). Only the turn's first request, before anything streamed, can
    say so: a later round, or a stream under way, shows the content was taken."""
    return not error.retryable and error.status_code not in _NOT_CONTENT_STATUSES

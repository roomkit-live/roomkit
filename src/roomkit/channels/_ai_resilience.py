"""AIChannel mixin for retry, re-routing, and context overflow recovery."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_policy import declared_for
from roomkit.channels._ai_router import RouteRequest
from roomkit.channels._compaction import (
    compaction_cut,
    summary_text,
    with_compacted_history,
    with_results_stored,
)
from roomkit.channels._turn_budget import warn_if_priced_otherwise
from roomkit.channels._user_text import with_leading_text
from roomkit.models.channel import RetryPolicy
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AITextPart,
    ProviderError,
    StreamEvent,
    StreamThinkingDelta,
    StreamToolCallDelta,
    is_context_overflow_message,
)
from roomkit.providers.ai.response_schema import hold_until_checked
from roomkit.providers.ai.tool_intent import presented_tools
from roomkit.providers.utils import _aclose_stream

if TYPE_CHECKING:
    from roomkit.channels._ai_router import AIRouter
    from roomkit.channels._tool_eviction import ToolEviction
    from roomkit.memory.base import MemoryProvider
    from roomkit.providers.ai.base import AIMessage


if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


def _gains(summarized: list[AIMessage], rounds: list[AIMessage], kept: list[AIMessage]) -> bool:
    """Whether a compaction frees anything: messages to summarize, or a
    result stored."""
    return bool(summarized) or any(new is not old for new, old in zip(rounds, kept, strict=False))


@dataclass(frozen=True)
class _StreamRetryBoundary:
    """Separate two provider attempts of one model round.

    Composition deltas and reasoning are not persisted and therefore remain
    retryable, but they are visible: on the realtime bus, and reasoning to the
    channel rendering the stream. The streaming consumer uses this marker to
    close the abandoned composition and reset its counters before the
    replacement attempt starts; ``abandoned`` when the attempt handed nothing
    else, so its reasoning goes with it (RFC §6.4).
    """

    abandoned: bool = False


class _Shown:
    """What one attempt of a generation showed before it failed.

    A composition delta and a piece of reasoning are projections: neither is
    delivered as text nor persisted, so an attempt that showed only those can
    be replayed, by a retry, a compacted replay or another provider, without
    duplicating anything the room holds (RFC §6.4). Counting a composition
    delta as handed would cost every tool round its retry budget, since a
    call's fragments arrive long before the call; counting reasoning would
    refuse a retry to every model that reasons before it answers.
    """

    def __init__(self) -> None:
        self.emitted = False
        self._composition = False
        self._reasoning = False

    def note(self, event: StreamEvent) -> None:
        """Take in one event the attempt yielded."""
        if isinstance(event, StreamToolCallDelta):
            self._composition = True
        elif isinstance(event, StreamThinkingDelta):
            self._reasoning = True
        else:
            self.emitted = True

    def boundary(self) -> _StreamRetryBoundary | None:
        """The marker that closes what the failed attempt showed: its
        composition, and its reasoning when it handed nothing else."""
        abandoned = self._reasoning and not self.emitted
        if not (self._composition or abandoned):
            return None
        return _StreamRetryBoundary(abandoned=abandoned)


async def _noted(
    events: AsyncIterator[StreamEvent], received: Callable[[], None]
) -> AsyncIterator[StreamEvent]:
    """*events* as they come, *received* told of each before anything holds it."""
    try:
        async for event in events:
            received()
            yield event
    finally:
        await _aclose_stream(events)


class AIResilienceMixin(_AIChannelContract):
    """Retry logic, streaming retry, context overflow detection, and compaction.

    What it calls on the other mixins is declared once, in
    :class:`~roomkit.channels._ai_contract._AIChannelContract`, which it
    derives from for the type checker only.
    """

    _retry_policy: RetryPolicy | None
    _router: AIRouter | None
    _eviction: ToolEviction
    _memory: MemoryProvider
    _tool_call_intents: bool
    _tool_call_intents_required: bool
    _tool_call_intents_only: frozenset[str] | None
    channel_id: str

    async def _generate_stream_with_retry(
        self,
        context: AIContext,
        *,
        received: Callable[[], None] | None = None,
    ) -> AsyncIterator[StreamEvent | _StreamRetryBoundary]:
        """Stream on the turn's provider with compaction and retry, then, once
        its retries are spent, on the provider the channel's router hands the
        generation to, if it names one (RFC §6.4).

        This wrapper is the one layer positioned to know two facts, and both
        gate every recovery below:

        * **Whether anything already left for the consumer.** A stream that
          has yielded text, a call or its end cannot be re-entered — by retry,
          compaction or another provider alike — without duplicating
          delivered output in the room and in the persisted message, so a
          mid-stream failure propagates as-is. Reasoning and a call's
          fragments are shown, never held by the room: a stream that yielded
          only those is re-entered, after a boundary that closes what it
          showed (RFC §6.4).
        * **Whether the refusal is a context overflow.** Replaying the same
          context is a deterministic refusal, so neither the retry budget nor
          the router get it; the compacted replay runs first, once per call,
          and mutates ``context.messages`` in place so a tool loop builds its
          next rounds on the compacted history.
        """
        provider = self._turn_provider()
        policy = self._retry_policy or RetryPolicy(max_retries=0)
        last_error: ProviderError | None = None
        compacted = False

        attempt = 0
        while attempt <= policy.max_retries:
            shown = _Shown()
            # A provider the turn moved to that cannot hold a tool unseen
            # receives what the turn made callable declared plainly (RFC §6.4).
            stream = self._structured_stream(provider, declared_for(provider, context), received)
            try:
                async for event in stream:
                    shown.note(event)
                    yield event
                return  # Stream completed successfully
            except ProviderError as exc:
                if (boundary := shown.boundary()) is not None:
                    # Close this visible attempt before a retry or another
                    # provider starts, or before its failure reaches the caller.
                    yield boundary
                if shown.emitted:
                    raise
                if not compacted and self._is_context_overflow(exc):
                    # One compacted replay, then ordinary retry semantics: an
                    # error that only *sounded* like an overflow keeps its
                    # retry budget and its re-routing.
                    logger.warning("Context overflow on stream. Compacting and replaying.")
                    compacted = True
                    await self._compact_context(context)
                    continue
                last_error = exc
                if not exc.retryable:
                    raise
                if attempt >= policy.max_retries:
                    break
                delay = min(
                    policy.base_delay_seconds * (policy.exponential_base**attempt),
                    policy.max_delay_seconds,
                )
                logger.warning(
                    "Stream error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    policy.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1
            except Exception:
                if (boundary := shown.boundary()) is not None:
                    yield boundary
                raise
            finally:
                await _aclose_stream(stream)

        # Only reachable when nothing was emitted.
        if last_error is None:
            return
        other = await self._rerouted(provider, last_error)
        if other is None:
            raise last_error
        rerouted = self._rerouted_stream(other, context, last_error, received)
        async with aclosing(rerouted) as events:
            async for event in events:
                yield event
        # It served the generation: the turn's later rounds stay on it.
        self._get_loop_ctx().provider = other

    async def _rerouted(self, provider: AIProvider, error: ProviderError) -> AIProvider | None:
        """The provider the router hands a generation to once *provider*
        spent its retries on *error*; ``None`` without a router, or when it
        names none (RFC §6.4)."""
        if self._router is None:
            return None
        loop_ctx = self._get_loop_ctx()
        asked = loop_ctx.route_request or RouteRequest("turn", "")
        request = replace(asked, reason="retry", previous=provider, failed=error)
        other = await self._router.route(request)
        if other is not None:
            self._telemetry_to(other)
            warn_if_priced_otherwise(loop_ctx.turn_budget, other)
        return other

    async def _rerouted_stream(
        self,
        provider: AIProvider,
        context: AIContext,
        error: ProviderError,
        received: Callable[[], None] | None = None,
    ) -> AsyncGenerator[StreamEvent | _StreamRetryBoundary, None]:
        """*provider*'s stream, the one attempt the router handed it once the
        turn's provider spent its retries on *error* with nothing delivered.

        One that fails before it emits leaves *error* as the turn's, its own
        failure riding as the cause: it was a second chance, not the turn's
        provider.
        """
        logger.warning("Retries spent (%s); the router hands the generation over.", error)
        stream = self._structured_stream(provider, declared_for(provider, context), received)
        shown = _Shown()
        try:
            async for event in stream:
                shown.note(event)
                yield event
        except ProviderError as other_exc:
            if (boundary := shown.boundary()) is not None:
                yield boundary
            if shown.emitted:
                raise
            logger.error("The provider the router handed the generation to failed: %s", other_exc)
            raise error from other_exc
        except Exception:
            if (boundary := shown.boundary()) is not None:
                yield boundary
            raise
        finally:
            await _aclose_stream(stream)

    def _structured_stream(
        self,
        provider: AIProvider,
        context: AIContext,
        received: Callable[[], None] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """The provider's structured stream on *context* as the model is shown
        it, its text held back until checked when the turn is constrained to a
        response schema: the room must never receive an answer the check then
        refuses (RFC §6.7). *received* is told of each event the provider
        hands, held or not: its answer began (RFC §6.4)."""
        stream = provider.generate_structured_stream(self._shown_to_model(context))
        if received is not None:
            stream = _noted(stream, received)
        if context.response_schema is None:
            return stream
        return hold_until_checked(stream)

    def _shown_to_model(self, context: AIContext) -> AIContext:
        """*context* as the request shows it: with ``tool_call_intents`` on,
        each tool's schema, or those ``tool_call_intents_only`` names, with
        the reserved intent property (RFC §6.4), listed as required with
        ``tool_call_intents_required``. A copy: the
        turn's own tools, which its calls are checked against, stay as their
        tools declared them; the messages are the same list."""
        if not self._tool_call_intents or not context.tools:
            return context
        shown = presented_tools(
            context.tools,
            required=self._tool_call_intents_required,
            only=self._tool_call_intents_only,
        )
        return context.model_copy(update={"tools": shown})

    @staticmethod
    def _is_context_overflow(exc: ProviderError) -> bool:
        """Check if a provider error indicates context window overflow."""
        # A structural classification is believed in both directions; the
        # prose decides only when nobody classified. Letting wording override
        # an explicit "no" is how a rate limit whose message resembles an
        # overflow once lost its retry budget to a pointless compaction.
        if exc.context_overflow is not None:
            return exc.context_overflow
        return is_context_overflow_message(str(exc))

    async def _compact_context(self, context: AIContext) -> AIContext:
        """Emergency compaction: make room while keeping the turn's input whole.

        Summarizes the history before the turn's input, and stores the long
        results of the turn's older tool rounds for re-reading (RFC §6.4).

        Mutates ``context.messages`` in place and returns the same context.
        In-place is load-bearing, and owned here so no caller can forget it:
        the round-context chain passes shallow copies that share this list,
        so a returned replacement a caller had to assign would be silently
        dropped by the first copy in between — while the shared list reaches
        every holder, and the compacted history is what later rounds build on.
        """
        messages = context.messages
        if len(messages) <= 4:
            raise ProviderError(
                "Context too large but cannot compact further (<=4 messages)",
                retryable=False,
            )
        loop_ctx = self._get_loop_ctx()
        # A discussion's turn keeps the message it answers, wherever it reads.
        kept = loop_ctx.answered_input or loop_ctx.turn_input
        summarized, shortened = compaction_cut(messages, kept)
        rounds = messages[summarized:shortened]
        kept = [*with_results_stored(rounds, self._eviction), *messages[shortened:]]
        summary = summary_text(messages[:summarized])
        if not _gains(messages[:summarized], rounds, kept):
            raise ProviderError(
                "Context too large but nothing left to compact before the turn's input",
                retryable=False,
            )
        self._show_summarized_references(messages[:summarized])
        compacted = with_leading_text(summary, kept)
        if kept and kept[0] is loop_ctx.turn_input:
            loop_ctx.turn_input = compacted[0]
        if kept and kept[0] is loop_ctx.answered_input:
            loop_ctx.answered_input = compacted[0]
        context.messages[:] = compacted
        return context

    async def _compact_after_round(
        self, context: AIContext, room_id: str | None, usage: dict[str, Any] | None
    ) -> None:
        """Between two tool rounds, hand the memory the history before the
        turn's input and what the round's generation measured, and put in
        place the history it returns (RFC §6.4, §20.1), as
        :func:`with_compacted_history` places it. A memory that returns
        ``None`` changes nothing, and one that fails is logged. Mutates
        ``context.messages`` in place, as :meth:`_compact_context` does.
        """
        loop_ctx = self._get_loop_ctx()
        kept = loop_ctx.answered_input or loop_ctx.turn_input
        at = next((i for i, message in enumerate(context.messages) if message is kept), None)
        if at is None or room_id is None or usage is None:
            return
        history = context.messages[:at]
        compacted = await self._compacted_history(room_id, history, usage)
        if compacted is None:
            return
        self._show_summarized_references([m for m in history if m not in compacted])
        messages = with_compacted_history(context.messages, at, compacted, self._eviction)
        moved = messages[len(messages) - (len(context.messages) - at)]
        if loop_ctx.turn_input is kept:
            loop_ctx.turn_input = moved
        if loop_ctx.answered_input is kept:
            loop_ctx.answered_input = moved
        context.messages[:] = messages

    async def _compacted_history(
        self, room_id: str, history: list[AIMessage], usage: dict[str, Any]
    ) -> list[AIMessage] | None:
        """What the memory returns for *history* between two tool rounds;
        ``None`` when it fails, logged."""
        try:
            return await self._memory.compact(
                room_id, list(history), usage=dict(usage), channel_id=self.channel_id
            )
        except Exception:
            logger.warning("Memory compaction between two tool rounds failed", exc_info=True)
            return None

    def _maybe_truncate_result(
        self, result: str | list[AITextPart | AIImagePart], tool_call_id: str = ""
    ) -> str | list[AITextPart | AIImagePart]:
        """Delegate to ToolEviction for large result handling.

        A multimodal result (a list of parts, e.g. a tool that returned a
        screenshot) has its text evicted and its images kept.
        """
        if isinstance(result, str):
            return str(self._eviction.maybe_evict(result, tool_call_id))
        if isinstance(result, list):
            return self._eviction.maybe_evict_parts(result, tool_call_id)
        return result

"""AIChannel mixin for retry, fallback, and context overflow recovery."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import TYPE_CHECKING

from roomkit.channels._ai_policy import declared_for
from roomkit.channels._compaction import compaction_cut, summary_text, with_results_stored
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
    StreamToolCallDelta,
    is_context_overflow_message,
)
from roomkit.providers.ai.response_schema import hold_until_checked
from roomkit.providers.utils import _aclose_stream

if TYPE_CHECKING:
    from roomkit.channels._tool_eviction import ToolEviction
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


class _StreamRetryBoundary:
    """Separate two provider attempts of one model round.

    Composition deltas are not persisted and therefore remain retryable, but
    they are visible on the realtime bus. The streaming consumer uses this
    marker to close the abandoned composition and reset its counters before
    the replacement attempt starts.
    """


class AIResilienceMixin(_AIChannelContract):
    """Retry logic, streaming retry, context overflow detection, and compaction.

    What it calls on the other mixins is declared once, in
    :class:`~roomkit.channels._ai_contract._AIChannelContract`, which it
    derives from for the type checker only.
    """

    _retry_policy: RetryPolicy | None
    _provider: AIProvider
    _fallback_provider: AIProvider | None
    _eviction: ToolEviction

    async def _generate_stream_with_retry(
        self, context: AIContext
    ) -> AsyncIterator[StreamEvent | _StreamRetryBoundary]:
        """Stream with compaction, retry and optional fallback.

        This wrapper is the one layer positioned to know two facts, and both
        gate every recovery below:

        * **Whether anything already left for the consumer.** A stream that
          has yielded cannot be re-entered — by retry, compaction or fallback
          alike — without duplicating delivered output in the room and in the
          persisted message, so a mid-stream failure propagates as-is.
        * **Whether the refusal is a context overflow.** Replaying the same
          context is a deterministic refusal, so neither the retry budget nor
          the fallback provider get it; the compacted replay runs first, once
          per call, and mutates ``context.messages`` in place so a tool loop
          builds its next rounds on the compacted history.
        """
        policy = self._retry_policy or RetryPolicy(max_retries=0)
        last_error: ProviderError | None = None
        compacted = False

        attempt = 0
        while attempt <= policy.max_retries:
            emitted = False
            projected_composition = False
            stream = self._structured_stream(self._provider, context)
            try:
                async for event in stream:
                    # A composition delta is a projection: it is neither
                    # delivered as text nor persisted, so a stream that has
                    # only produced those can still be replayed without
                    # duplicating anything — which is the whole test this
                    # flag exists to make. Arming it on one would silently
                    # cost every tool round its retry budget, since a call's
                    # fragments now arrive long before the call itself.
                    if not isinstance(event, StreamToolCallDelta):
                        emitted = True
                    else:
                        projected_composition = True
                    yield event
                return  # Stream completed successfully
            except ProviderError as exc:
                if projected_composition:
                    # Close this visible attempt before a retry/fallback starts
                    # or before its terminal failure reaches the caller.
                    yield _StreamRetryBoundary()
                if emitted:
                    raise
                if not compacted and self._is_context_overflow(exc):
                    # One compacted replay, then ordinary retry semantics: an
                    # error that only *sounded* like an overflow keeps its
                    # retry budget and its fallback.
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
                if projected_composition:
                    yield _StreamRetryBoundary()
                raise
            finally:
                await _aclose_stream(stream)

        # Fallback — only reachable when nothing was emitted.
        if self._fallback_provider and last_error:
            async with aclosing(self._fallback_stream(context, last_error)) as events:
                async for event in events:
                    yield event
            return

        if last_error:
            raise last_error

    async def _fallback_stream(
        self, context: AIContext, primary_error: ProviderError
    ) -> AsyncGenerator[StreamEvent | _StreamRetryBoundary, None]:
        """The fallback provider's stream, once the primary's retries are spent
        and nothing reached the consumer.

        A fallback that fails before it emits leaves the primary's error as
        the turn's, its own failure riding as the cause: the fallback was a
        second chance, not the turn's provider.
        """
        logger.warning("Trying fallback provider for stream.")
        fallback = self._fallback_provider
        assert fallback is not None  # noqa: S101 — the caller checked
        stream = self._structured_stream(fallback, declared_for(fallback, context))
        emitted = False
        projected_composition = False
        try:
            async for event in stream:
                if isinstance(event, StreamToolCallDelta):
                    projected_composition = True
                else:
                    emitted = True
                yield event
        except ProviderError as fallback_exc:
            if projected_composition:
                yield _StreamRetryBoundary()
            if emitted:
                raise
            logger.error("Fallback provider also failed: %s", fallback_exc)
            raise primary_error from fallback_exc
        except Exception:
            if projected_composition:
                yield _StreamRetryBoundary()
            raise
        finally:
            await _aclose_stream(stream)

    @staticmethod
    def _structured_stream(provider: AIProvider, context: AIContext) -> AsyncIterator[StreamEvent]:
        """The provider's structured stream, its text held back until checked
        when the turn is constrained to a response schema: the room must never
        receive an answer the check then refuses (RFC §6.7)."""
        stream = provider.generate_structured_stream(context)
        if context.response_schema is None:
            return stream
        return hold_until_checked(stream)

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
        context.messages[:] = compacted
        return context

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

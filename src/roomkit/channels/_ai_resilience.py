"""AIChannel mixin for retry, fallback, and context overflow recovery."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from roomkit.models.channel import RetryPolicy
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    ProviderError,
    StreamEvent,
    is_context_overflow_message,
)

if TYPE_CHECKING:
    from roomkit.channels._tool_eviction import ToolEviction

logger = logging.getLogger("roomkit.channels.ai")


@runtime_checkable
class ResilienceHost(Protocol):
    """Contract: capabilities a host class must provide for AIResilienceMixin.

    Attributes provided by the host's ``__init__``:
        _retry_policy: Retry configuration (max retries, backoff).
        _provider: Primary AI provider for generation.
        _fallback_provider: Optional fallback when primary exhausts retries.
        _eviction: Tool result eviction / truncation strategy.
    """

    _retry_policy: RetryPolicy | None
    _provider: AIProvider
    _fallback_provider: AIProvider | None
    _eviction: ToolEviction


class AIResilienceMixin:
    """Retry logic, streaming retry, context overflow detection, and compaction.

    Host contract: :class:`ResilienceHost`.
    """

    _retry_policy: RetryPolicy | None
    _provider: AIProvider
    _fallback_provider: AIProvider | None
    _eviction: ToolEviction

    async def _generate_with_retry(self, context: AIContext) -> AIResponse:
        """Call provider.generate() with compaction, retry and optional fallback.

        A context overflow is handled before the retry budget and the
        fallback provider see it: replaying the same context is a
        deterministic refusal, so the one recovery with a chance is the
        compacted replay — once per call. The compaction mutates
        ``context.messages`` in place, so a caller running a tool loop
        builds its next rounds on the compacted history.
        """
        policy = self._retry_policy or RetryPolicy(max_retries=0)
        last_error: ProviderError | None = None
        compacted = False

        provider = self._provider
        attempt = 0
        while attempt <= policy.max_retries:
            try:
                return await provider.generate(context)
            except ProviderError as exc:
                if self._is_context_overflow(exc):
                    if compacted:
                        raise
                    logger.warning("Context overflow. Compacting and replaying.")
                    compacted = True
                    context.messages[:] = (await self._compact_context(context)).messages
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
                    "Provider error (attempt %d/%d, status=%s): %s. Retrying in %.1fs",
                    attempt + 1,
                    policy.max_retries,
                    exc.status_code,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1

        # All retries exhausted — try fallback provider
        if self._fallback_provider and last_error:
            logger.warning(
                "Primary provider failed after %d attempts. Trying fallback.",
                policy.max_retries + 1,
            )
            try:
                return await self._fallback_provider.generate(context)
            except ProviderError as fallback_exc:
                logger.error("Fallback provider also failed: %s", fallback_exc)
                raise last_error from fallback_exc

        if last_error:
            raise last_error
        raise RuntimeError("_generate_with_retry completed without result or exception")

    async def _generate_stream_with_retry(self, context: AIContext) -> AsyncIterator[StreamEvent]:
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
            try:
                async for event in self._provider.generate_structured_stream(context):
                    emitted = True
                    yield event
                return  # Stream completed successfully
            except ProviderError as exc:
                if emitted:
                    raise
                if self._is_context_overflow(exc):
                    if compacted:
                        raise
                    logger.warning("Context overflow on stream. Compacting and replaying.")
                    compacted = True
                    context.messages[:] = (await self._compact_context(context)).messages
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

        # Fallback — only reachable when nothing was emitted.
        if self._fallback_provider and last_error:
            logger.warning("Trying fallback provider for stream.")
            async for event in self._fallback_provider.generate_structured_stream(context):
                yield event
            return

        if last_error:
            raise last_error

    @staticmethod
    def _is_context_overflow(exc: ProviderError) -> bool:
        """Check if a provider error indicates context window overflow."""
        # The typed fact outranks the prose: an envelope that classified the
        # failure structurally may have rewrapped the provider's wording into
        # something no phrase can match.
        return exc.context_overflow or is_context_overflow_message(str(exc))

    async def _compact_context(self, context: AIContext) -> AIContext:
        """Emergency compaction: summarize the first half of messages."""
        messages = context.messages
        if len(messages) <= 4:
            raise ProviderError(
                "Context too large but cannot compact further (<=4 messages)",
                retryable=False,
            )

        split = len(messages) // 2
        # Never split an assistant/tool-result pair: a ``tool`` message whose
        # assistant tool-call turn fell into the summarized half is an orphan
        # the provider rejects with a 400 — the compaction meant to recover an
        # overflow would then kill the turn outright. Tool results always
        # directly follow their assistant message here, so advancing the split
        # past them keeps every pair whole (both summarized together).
        while split < len(messages) and messages[split].role == "tool":
            split += 1
        old_messages = messages[:split]
        recent_messages = messages[split:]

        # Build a quick summary of old messages
        summary_parts: list[str] = []
        for msg in old_messages:
            role = msg.role
            if isinstance(msg.content, str):
                text = msg.content[:500]
            elif isinstance(msg.content, list):
                text = " ".join(
                    p.text[:200] if hasattr(p, "text") else f"[{p.type}]"  # ty: ignore[not-subscriptable]
                    for p in msg.content
                )[:500]
            else:
                text = str(msg.content)[:500]
            summary_parts.append(f"[{role}]: {text}")

        summary_text = "\n".join(summary_parts)
        summary_msg = AIMessage(
            role="user",
            content=(f"[Context compacted — earlier conversation summary]\n{summary_text}"),
        )

        return context.model_copy(update={"messages": [summary_msg] + recent_messages})

    @staticmethod
    def _extract_accumulated_text(messages: list[AIMessage]) -> str:
        """Extract accumulated assistant text from message history."""
        parts: list[str] = []
        for msg in messages:
            if msg.role != "assistant":
                continue
            if isinstance(msg.content, str):
                parts.append(msg.content)
            elif isinstance(msg.content, list):
                for p in msg.content:
                    if isinstance(p, AITextPart) and p.text:
                        parts.append(p.text)
        return "\n".join(parts)

    def _maybe_truncate_result(
        self, result: str | list[AITextPart | AIImagePart], tool_call_id: str = ""
    ) -> str | list[AITextPart | AIImagePart]:
        """Delegate to ToolEviction for large (string) result handling.

        Multimodal results (a list of parts — e.g. a tool that returned an
        image) are passed through untouched; only string results are evicted.
        """
        if not isinstance(result, str):
            return result
        return str(self._eviction.maybe_evict(result, tool_call_id))

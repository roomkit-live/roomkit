"""AIChannel mixin for retry, fallback, and context overflow recovery."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from roomkit.models.channel import RetryPolicy
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    ProviderError,
    StreamEvent,
)

logger = logging.getLogger("roomkit.channels.ai")


class AIResilienceMixin:
    """Retry logic, streaming retry, context overflow detection, and compaction."""

    _retry_policy: RetryPolicy | None
    _provider: AIProvider
    _fallback_provider: AIProvider | None

    async def _generate_with_retry(self, context: AIContext) -> AIResponse:
        """Call provider.generate() with retry and optional fallback."""
        policy = self._retry_policy or RetryPolicy(max_retries=0)
        last_error: ProviderError | None = None

        provider = self._provider
        for attempt in range(policy.max_retries + 1):
            try:
                return await provider.generate(context)
            except ProviderError as exc:
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
        """Stream with retry on provider errors."""
        policy = self._retry_policy or RetryPolicy(max_retries=0)
        last_error: ProviderError | None = None

        for attempt in range(policy.max_retries + 1):
            try:
                async for event in self._provider.generate_structured_stream(context):
                    yield event
                return  # Stream completed successfully
            except ProviderError as exc:
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

        # Fallback
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
        msg = str(exc).lower()
        return any(
            phrase in msg
            for phrase in [
                "context length exceeded",
                "maximum context length",
                "token limit",
                "too many tokens",
                "request too large",
                "prompt is too long",
            ]
        )

    async def _compact_context(self, context: AIContext) -> AIContext:
        """Emergency compaction: summarize the first half of messages."""
        messages = context.messages
        if len(messages) <= 4:
            raise ProviderError(
                "Context too large but cannot compact further (<=4 messages)",
                retryable=False,
            )

        split = len(messages) // 2
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
                    p.text[:200] if hasattr(p, "text") else f"[{p.type}]" for p in msg.content
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

    def _maybe_truncate_result(self, result: str, tool_call_id: str = "") -> str:
        """Delegate to ToolEviction for large result handling."""
        return str(self._eviction.maybe_evict(result, tool_call_id))  # type: ignore[attr-defined]

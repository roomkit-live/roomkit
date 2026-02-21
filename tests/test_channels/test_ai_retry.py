"""Tests for AIChannel retry, fallback, and graceful degradation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import RetryPolicy
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    ProviderError,
    StreamDone,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider


class TestGenerateWithRetry:
    async def test_succeeds_first_try(self) -> None:
        """No retry when first attempt succeeds."""
        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider, retry_policy=RetryPolicy(max_retries=3))

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        response = await ch._generate_with_retry(context)

        assert response.content == "ok"
        assert len(provider.calls) == 1

    async def test_retries_on_retryable_error(self) -> None:
        """Retries up to max_retries on retryable errors then succeeds."""
        call_count = 0

        async def flaky_generate(context: AIContext) -> AIResponse:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ProviderError("rate limited", retryable=True, status_code=429)
            return AIResponse(content="recovered")

        provider = MockAIProvider()
        provider.generate = flaky_generate  # type: ignore[assignment]
        ch = AIChannel(
            "ai1",
            provider=provider,
            retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=0.01),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        response = await ch._generate_with_retry(context)

        assert response.content == "recovered"
        assert call_count == 3  # 2 failures + 1 success

    async def test_non_retryable_error_raises_immediately(self) -> None:
        """Non-retryable errors are raised without retry."""
        call_count = 0

        async def auth_fail(context: AIContext) -> AIResponse:
            nonlocal call_count
            call_count += 1
            raise ProviderError("invalid api key", retryable=False, status_code=401)

        provider = MockAIProvider()
        provider.generate = auth_fail  # type: ignore[assignment]
        ch = AIChannel(
            "ai1",
            provider=provider,
            retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=0.01),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        with pytest.raises(ProviderError, match="invalid api key"):
            await ch._generate_with_retry(context)

        assert call_count == 1  # No retry

    async def test_all_retries_exhausted_raises(self) -> None:
        """Raises last error when all retries are exhausted (no fallback)."""

        async def always_fail(context: AIContext) -> AIResponse:
            raise ProviderError("overloaded", retryable=True, status_code=529)

        provider = MockAIProvider()
        provider.generate = always_fail  # type: ignore[assignment]
        ch = AIChannel(
            "ai1",
            provider=provider,
            retry_policy=RetryPolicy(max_retries=2, base_delay_seconds=0.01),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        with pytest.raises(ProviderError, match="overloaded"):
            await ch._generate_with_retry(context)

    async def test_backoff_delays_are_exponential(self) -> None:
        """Verify delays follow exponential backoff pattern."""
        delays: list[float] = []

        async def tracking_sleep(seconds: float) -> None:
            delays.append(seconds)
            # Don't actually sleep in tests

        async def always_fail(context: AIContext) -> AIResponse:
            raise ProviderError("overloaded", retryable=True, status_code=503)

        provider = MockAIProvider()
        provider.generate = always_fail  # type: ignore[assignment]
        ch = AIChannel(
            "ai1",
            provider=provider,
            retry_policy=RetryPolicy(
                max_retries=3,
                base_delay_seconds=1.0,
                max_delay_seconds=10.0,
                exponential_base=2.0,
            ),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        with (
            patch("roomkit.channels.ai.asyncio.sleep", tracking_sleep),
            pytest.raises(ProviderError),
        ):
            await ch._generate_with_retry(context)

        assert len(delays) == 3  # 3 retries
        assert delays[0] == pytest.approx(1.0)  # 1.0 * 2^0
        assert delays[1] == pytest.approx(2.0)  # 1.0 * 2^1
        assert delays[2] == pytest.approx(4.0)  # 1.0 * 2^2

    async def test_no_retry_when_policy_is_none(self) -> None:
        """Without retry policy, errors are raised immediately."""

        async def fail(context: AIContext) -> AIResponse:
            raise ProviderError("error", retryable=True, status_code=503)

        provider = MockAIProvider()
        provider.generate = fail  # type: ignore[assignment]
        ch = AIChannel("ai1", provider=provider)  # No retry_policy

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        with pytest.raises(ProviderError):
            await ch._generate_with_retry(context)


class TestFallbackProvider:
    async def test_fallback_triggers_after_retry_exhaustion(self) -> None:
        """Fallback provider is used when primary exhausts all retries."""

        async def primary_fail(context: AIContext) -> AIResponse:
            raise ProviderError("overloaded", retryable=True, status_code=529)

        provider = MockAIProvider()
        provider.generate = primary_fail  # type: ignore[assignment]
        fallback = MockAIProvider(responses=["fallback response"])

        ch = AIChannel(
            "ai1",
            provider=provider,
            fallback_provider=fallback,
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.01),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        response = await ch._generate_with_retry(context)

        assert response.content == "fallback response"
        assert len(fallback.calls) == 1

    async def test_fallback_failure_raises_original_error(self) -> None:
        """When fallback also fails, the original error is raised."""

        async def primary_fail(context: AIContext) -> AIResponse:
            raise ProviderError("primary error", retryable=True, status_code=503)

        async def fallback_fail(context: AIContext) -> AIResponse:
            raise ProviderError("fallback error", retryable=False, status_code=500)

        provider = MockAIProvider()
        provider.generate = primary_fail  # type: ignore[assignment]
        fallback = MockAIProvider()
        fallback.generate = fallback_fail  # type: ignore[assignment]

        ch = AIChannel(
            "ai1",
            provider=provider,
            fallback_provider=fallback,
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.01),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        with pytest.raises(ProviderError, match="primary error") as exc_info:
            await ch._generate_with_retry(context)

        # Fallback error should be chained
        assert exc_info.value.__cause__ is not None

    async def test_no_fallback_on_first_try_success(self) -> None:
        """Fallback is not called when primary succeeds."""
        provider = MockAIProvider(responses=["primary ok"])
        fallback = MockAIProvider(responses=["fallback"])

        ch = AIChannel(
            "ai1",
            provider=provider,
            fallback_provider=fallback,
            retry_policy=RetryPolicy(max_retries=3),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        response = await ch._generate_with_retry(context)

        assert response.content == "primary ok"
        assert len(fallback.calls) == 0

    async def test_no_fallback_on_non_retryable_error(self) -> None:
        """Fallback is not used for non-retryable errors (they raise immediately)."""

        async def auth_fail(context: AIContext) -> AIResponse:
            raise ProviderError("bad key", retryable=False, status_code=401)

        provider = MockAIProvider()
        provider.generate = auth_fail  # type: ignore[assignment]
        fallback = MockAIProvider(responses=["fallback"])

        ch = AIChannel(
            "ai1",
            provider=provider,
            fallback_provider=fallback,
            retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=0.01),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        with pytest.raises(ProviderError, match="bad key"):
            await ch._generate_with_retry(context)

        assert len(fallback.calls) == 0


class TestStreamRetry:
    async def test_stream_succeeds_first_try(self) -> None:
        """Streaming completes without retry on success."""
        provider = MockAIProvider(responses=["stream ok"], streaming=True)
        ch = AIChannel(
            "ai1",
            provider=provider,
            retry_policy=RetryPolicy(max_retries=3),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        events = []
        async for event in ch._generate_stream_with_retry(context):
            events.append(event)

        text_events = [e for e in events if isinstance(e, StreamTextDelta)]
        assert len(text_events) == 1
        assert text_events[0].text == "stream ok"

    async def test_stream_retries_on_retryable_error(self) -> None:
        """Stream retries on retryable error then succeeds."""
        call_count = 0

        async def flaky_stream(context: AIContext):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError("rate limited", retryable=True, status_code=429)
            yield StreamTextDelta(text="recovered")
            yield StreamDone(finish_reason="stop")

        provider = MockAIProvider(streaming=True)
        provider.generate_structured_stream = flaky_stream  # type: ignore[assignment]
        ch = AIChannel(
            "ai1",
            provider=provider,
            retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=0.01),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        events = []
        async for event in ch._generate_stream_with_retry(context):
            events.append(event)

        text_events = [e for e in events if isinstance(e, StreamTextDelta)]
        assert len(text_events) == 1
        assert text_events[0].text == "recovered"

    async def test_stream_fallback_on_exhaustion(self) -> None:
        """Stream falls back to fallback provider after exhausting retries."""

        async def always_fail_stream(context: AIContext):
            raise ProviderError("overloaded", retryable=True, status_code=529)
            yield  # pragma: no cover — make it an async generator

        provider = MockAIProvider(streaming=True)
        provider.generate_structured_stream = always_fail_stream  # type: ignore[assignment]
        fallback = MockAIProvider(responses=["fallback stream"], streaming=True)

        ch = AIChannel(
            "ai1",
            provider=provider,
            fallback_provider=fallback,
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.01),
        )

        context = AIContext(messages=[AIMessage(role="user", content="hi")])
        events = []
        async for event in ch._generate_stream_with_retry(context):
            events.append(event)

        text_events = [e for e in events if isinstance(e, StreamTextDelta)]
        assert len(text_events) == 1
        assert text_events[0].text == "fallback stream"

"""Token bucket behavior at fractional delivery rates."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from roomkit.core.rate_limiter import TokenBucketRateLimiter
from roomkit.models.channel import RateLimit


@pytest.mark.parametrize(
    "limit",
    [RateLimit(max_per_second=0.5), RateLimit(max_per_minute=30), RateLimit(max_per_hour=1800)],
)
def test_fractional_rate_refills_without_allowing_a_burst(limit: RateLimit) -> None:
    limiter = TokenBucketRateLimiter()
    with patch("roomkit.core.rate_limiter.time.monotonic") as clock:
        clock.return_value = 0.0
        assert limiter.acquire("sms", limit)
        assert not limiter.acquire("sms", limit)
        clock.return_value = 1.0
        assert not limiter.acquire("sms", limit)
        clock.return_value = 2.0
        assert limiter.acquire("sms", limit)
        assert not limiter.acquire("sms", limit)
        clock.return_value = 3600.0
        assert limiter.acquire("sms", limit)
        assert not limiter.acquire("sms", limit)


async def test_wait_resumes_after_fractional_rate_refills() -> None:
    limiter = TokenBucketRateLimiter()
    limit = RateLimit(max_per_minute=30)
    with patch("roomkit.core.rate_limiter.time.monotonic") as clock:
        clock.return_value = 0.0
        assert limiter.acquire("sms", limit)

        async def advance(seconds: float) -> None:
            clock.return_value += seconds

        with patch("roomkit.core.rate_limiter.asyncio.sleep", side_effect=advance) as sleep:
            await limiter.wait("sms", limit)
        sleep.assert_awaited_once_with(2.0)
        assert not limiter.acquire("sms", limit)

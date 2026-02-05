"""Token-bucket rate limiter for channel delivery."""

from __future__ import annotations

import asyncio
import time

from roomkit.models.channel import RateLimit


class TokenBucketRateLimiter:
    """Per-channel token bucket rate limiter.

    Uses ``RateLimit.max_per_second`` for the bucket refill rate.
    Falls back to ``max_per_minute / 60`` or ``max_per_hour / 3600``.

    **Concurrency note:** This implementation relies on the CPython single-threaded
    asyncio model. The ``acquire`` method reads and writes ``_buckets`` with no
    ``await`` between the check and the set, making the token update atomic within
    a single event-loop iteration. The ``wait`` method uses ``asyncio.sleep``
    between acquire attempts, which is safe because each iteration re-reads the
    bucket state after yielding control.
    """

    def __init__(self) -> None:
        # channel_id -> (tokens, last_refill_time)
        self._buckets: dict[str, tuple[float, float]] = {}

    def _rate_per_second(self, rate_limit: RateLimit) -> float:
        if rate_limit.max_per_second is not None:
            return rate_limit.max_per_second
        if rate_limit.max_per_minute is not None:
            return rate_limit.max_per_minute / 60.0
        if rate_limit.max_per_hour is not None:
            return rate_limit.max_per_hour / 3600.0
        return float("inf")

    def _refill(self, channel_id: str, rate: float) -> float:
        """Refill tokens and return current count."""
        now = time.monotonic()
        tokens, last_refill = self._buckets.get(channel_id, (rate, now))
        elapsed = now - last_refill
        tokens = min(tokens + elapsed * rate, rate)  # cap at rate (burst = 1s)
        self._buckets[channel_id] = (tokens, now)
        return tokens

    def acquire(self, channel_id: str, rate_limit: RateLimit) -> bool:
        """Try to acquire a token. Returns True if allowed, False if rate limited."""
        rate = self._rate_per_second(rate_limit)
        if rate == float("inf"):
            return True

        tokens = self._refill(channel_id, rate)
        if tokens >= 1.0:
            self._buckets[channel_id] = (tokens - 1.0, time.monotonic())
            return True
        return False

    async def wait(self, channel_id: str, rate_limit: RateLimit) -> None:
        """Wait until a token is available (queue instead of drop)."""
        rate = self._rate_per_second(rate_limit)
        if rate == float("inf"):
            return

        while not self.acquire(channel_id, rate_limit):
            # Wait for one token to refill
            await asyncio.sleep(1.0 / rate)

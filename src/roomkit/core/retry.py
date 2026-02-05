"""Retry with exponential backoff for delivery operations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from roomkit.models.channel import RetryPolicy

logger = logging.getLogger("roomkit.retry")

__all__ = ["RetryPolicy", "retry_with_backoff"]


async def retry_with_backoff[T](
    fn: Callable[..., Coroutine[Any, Any, T]],
    policy: RetryPolicy,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute *fn* with exponential backoff retry.

    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + policy.max_retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= policy.max_retries:
                break
            delay = min(
                policy.base_delay_seconds * (policy.exponential_base**attempt),
                policy.max_delay_seconds,
            )
            logger.warning(
                "Attempt %d/%d failed, retrying in %.1fs",
                attempt + 1,
                policy.max_retries + 1,
                delay,
                extra={"attempt": attempt + 1, "delay": delay},
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc

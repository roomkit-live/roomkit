"""Retry with exponential backoff for delivery operations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from roomkit.models.channel import RetryPolicy

logger = logging.getLogger("roomkit.retry")

__all__ = ["RetryPolicy", "retry_with_backoff"]


def _is_retryable(exc: Exception, policy: RetryPolicy) -> bool:
    """Whether *exc* may be retried under *policy* (RFC §13.2).

    Two independent answers, and the error's own comes first. A provider that
    marks its exception ``retryable=False`` — a 400, a rejected recipient, a
    revoked credential — is believed whatever the policy says: replaying it
    spends the full backoff to reach the same refusal. Only an error that says
    nothing about itself falls through to the policy's list.
    """
    declared = getattr(exc, "retryable", None)
    if isinstance(declared, bool):
        return declared
    if policy.retryable_errors is None:
        return True
    names = {type(exc).__name__, *(base.__name__ for base in type(exc).__mro__)}
    return bool(names & set(policy.retryable_errors))


async def retry_with_backoff[T](
    fn: Callable[..., Coroutine[Any, Any, T]],
    policy: RetryPolicy,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute *fn* with exponential backoff retry.

    Only a retryable failure is retried (RFC §13.2). Raises the last exception
    once retries are exhausted, or immediately for one that is not retryable.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + policy.max_retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc, policy):
                logger.debug(
                    "Not retrying %s: the error is not retryable",
                    type(exc).__name__,
                    extra={"error_type": type(exc).__name__},
                )
                break
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

    if last_exc is None:
        raise RuntimeError("retry_with_backoff completed without result or exception")
    raise last_exc

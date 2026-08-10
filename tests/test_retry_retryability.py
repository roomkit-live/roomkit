"""Retry only what can succeed on a second try (RFC §13.2).

"Only errors marked as retryable = true SHOULD trigger retries." The loop
caught bare `Exception`, so a permanent 4xx — a rejected recipient, a revoked
credential — was replayed through the full backoff to reach the same refusal,
and the caller waited for it.
"""

from __future__ import annotations

import pytest

from roomkit.core.retry import retry_with_backoff
from roomkit.models.channel import RetryPolicy

FAST = RetryPolicy(max_retries=3, base_delay_seconds=0.001, max_delay_seconds=0.002)


class _PermanentError(Exception):
    retryable = False


class _TransientError(Exception):
    retryable = True


class _UnmarkedError(Exception):
    """Says nothing about itself."""


def _counting(exc: Exception):  # noqa: ANN202
    calls = {"n": 0}

    async def fn() -> str:
        calls["n"] += 1
        raise exc

    return fn, calls


class TestTheErrorDecidesFirst:
    async def test_a_permanent_error_is_not_replayed(self) -> None:
        fn, calls = _counting(_PermanentError("recipient rejected"))

        with pytest.raises(_PermanentError):
            await retry_with_backoff(fn, FAST)

        assert calls["n"] == 1

    async def test_a_transient_error_exhausts_the_budget(self) -> None:
        fn, calls = _counting(_TransientError("503"))

        with pytest.raises(_TransientError):
            await retry_with_backoff(fn, FAST)

        assert calls["n"] == 1 + FAST.max_retries

    async def test_the_error_outranks_the_policy_list(self) -> None:
        """A provider that says "do not retry me" is believed whatever the
        policy names — replaying it only spends the backoff."""
        policy = FAST.model_copy(update={"retryable_errors": ["_PermanentError"]})
        fn, calls = _counting(_PermanentError("still permanent"))

        with pytest.raises(_PermanentError):
            await retry_with_backoff(fn, policy)

        assert calls["n"] == 1


class TestThePolicyListNarrows:
    async def test_an_unmarked_error_is_retried_by_default(self) -> None:
        """No list, and the error says nothing: the previous behaviour."""
        fn, calls = _counting(_UnmarkedError("who knows"))

        with pytest.raises(_UnmarkedError):
            await retry_with_backoff(fn, FAST)

        assert calls["n"] == 1 + FAST.max_retries

    async def test_a_named_type_is_retried(self) -> None:
        policy = FAST.model_copy(update={"retryable_errors": ["_UnmarkedError"]})
        fn, calls = _counting(_UnmarkedError("named"))

        with pytest.raises(_UnmarkedError):
            await retry_with_backoff(fn, policy)

        assert calls["n"] == 1 + policy.max_retries

    async def test_an_unnamed_type_fails_on_its_first_attempt(self) -> None:
        policy = FAST.model_copy(update={"retryable_errors": ["TimeoutError"]})
        fn, calls = _counting(_UnmarkedError("not named"))

        with pytest.raises(_UnmarkedError):
            await retry_with_backoff(fn, policy)

        assert calls["n"] == 1

    async def test_a_base_class_name_matches(self) -> None:
        """Naming OSError covers ConnectionError, as a reader expects."""
        policy = FAST.model_copy(update={"retryable_errors": ["OSError"]})
        fn, calls = _counting(ConnectionError("reset"))

        with pytest.raises(ConnectionError):
            await retry_with_backoff(fn, policy)

        assert calls["n"] == 1 + policy.max_retries


class TestSuccess:
    async def test_a_recovering_call_returns_its_value(self) -> None:
        calls = {"n": 0}

        async def fn() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise _TransientError("not yet")
            return "ok"

        assert await retry_with_backoff(fn, FAST) == "ok"
        assert calls["n"] == 3

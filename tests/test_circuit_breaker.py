"""Tests for CircuitBreaker."""

from __future__ import annotations

import time
from unittest.mock import patch

from roomkit.core.circuit_breaker import CircuitBreaker


class TestCircuitBreakerBasics:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open
        assert cb.allow_request()

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open
        assert not cb.allow_request()

    def test_resets_on_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.is_closed
        assert cb.allow_request()

    def test_manual_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        cb.reset()
        assert cb.is_closed


class TestHalfOpenProbe:
    def test_half_open_allows_single_probe(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open or cb.is_half_open

        # With recovery_timeout=0, it transitions to half-open immediately
        # First call should be allowed (probe)
        assert cb.allow_request() is True
        # Second call should be blocked (probe already sent)
        assert cb.allow_request() is False
        # Third call too
        assert cb.allow_request() is False

    def test_half_open_probe_resets_on_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()

        # Probe allowed
        assert cb.allow_request() is True
        # Record success — resets to closed
        cb.record_success()
        assert cb.is_closed
        # Requests flow normally again
        assert cb.allow_request() is True
        assert cb.allow_request() is True

    def test_half_open_probe_resets_on_failure(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()

        # Probe allowed
        assert cb.allow_request() is True
        # Probe fails — goes back to open
        cb.record_failure()
        # Now with recovery_timeout=0 it's half-open again, probe flag was reset
        # So one more probe should be allowed
        assert cb.allow_request() is True
        assert cb.allow_request() is False

    def test_half_open_after_recovery_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        assert cb.allow_request() is False

        # Simulate time passing beyond recovery timeout
        with patch.object(time, "monotonic", return_value=time.monotonic() + 61.0):
            assert cb.is_half_open
            assert cb.allow_request() is True
            assert cb.allow_request() is False

    def test_late_refusal_can_only_release_its_own_probe(self) -> None:
        transitions: list[str] = []
        cb = CircuitBreaker(1, 0, on_state_change=transitions.append)
        allowed, ordinary = cb._acquire_request()
        assert allowed
        cb.record_failure()
        _, earlier = cb._acquire_request()
        cb.record_failure()
        _, active = cb._acquire_request()
        assert earlier is not None and active is not None
        for stale in (ordinary, earlier):
            cb._release_probe(stale)
            assert not cb.allow_request()
        cb._release_probe(active)
        assert cb._failure_count == 2
        assert cb.is_half_open and cb.allow_request()
        assert transitions == ["open"]
        cb.record_success()
        assert transitions == ["open", "closed"]

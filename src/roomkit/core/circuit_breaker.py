"""Circuit breaker for provider fault isolation."""

from __future__ import annotations

import time


class CircuitBreaker:
    """Simple circuit breaker (closed → open → half-open → closed).

    * **Closed** — requests flow normally.
    * **Open** — all requests fail immediately (after *failure_threshold* consecutive failures).
    * **Half-open** — after *recovery_timeout* seconds, allow one probe request.

    **Concurrency note:** This implementation relies on the CPython single-threaded
    asyncio model. State mutations in ``allow_request``, ``record_success``, and
    ``record_failure`` contain no ``await`` between the check and the set, making
    them atomic within a single event-loop iteration. No additional locking is
    required as long as all callers run in the same event loop.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count: int = 0
        self._opened_at: float | None = None
        self._half_open_probe_sent: bool = False

    # -- State queries --

    @property
    def is_open(self) -> bool:
        """True when the breaker is fully open (not half-open)."""
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        return elapsed < self._recovery_timeout

    @property
    def is_half_open(self) -> bool:
        """True when recovery timeout has elapsed and a probe is allowed."""
        if self._opened_at is None:
            return False
        return (time.monotonic() - self._opened_at) >= self._recovery_timeout

    @property
    def is_closed(self) -> bool:
        return self._opened_at is None

    def allow_request(self) -> bool:
        """Return True if a request should be attempted."""
        if self.is_closed:
            return True
        if self.is_half_open and not self._half_open_probe_sent:
            self._half_open_probe_sent = True
            return True
        return False

    # -- Recording --

    def record_success(self) -> None:
        """Record a successful call — resets the breaker to closed."""
        self._failure_count = 0
        self._opened_at = None
        self._half_open_probe_sent = False

    def record_failure(self) -> None:
        """Record a failed call — may trip the breaker open."""
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._opened_at = time.monotonic()
            self._half_open_probe_sent = False

    # -- Reset --

    def reset(self) -> None:
        """Manually close the breaker."""
        self._failure_count = 0
        self._opened_at = None
        self._half_open_probe_sent = False

"""A media clock the test sets, for roomkit's private clock seam (``roomkit._clock``).

Installed with ``_clock.installed(clock)``, it is what the timed media paths
read: ``now`` is the reading, and what is waited on it is recorded and ends at
once.
"""

from __future__ import annotations

from collections.abc import Awaitable


class SetClock:
    """A clock that reads what the test sets and records what is waited on it."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.slept: list[float] = []
        self.waited: list[float | None] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    async def wait_for[T](self, awaitable: Awaitable[T], timeout: float | None) -> T:
        self.waited.append(timeout)
        return await awaitable

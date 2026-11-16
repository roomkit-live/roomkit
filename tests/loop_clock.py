"""The running loop's clock, held still by the test.

A full garbage collection stops a worker's loop for up to about half a
second on a loaded runner (RMK-1714): longer than most budgets under test.
A deadline the library keeps on the loop's clock, held, is spent by what
the test moves, never by how the worker is scheduled: the shape of a Rust
test on tokio's paused clock.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest


class HeldClock:
    """The running loop's clock, held still from :meth:`hold` until
    :meth:`release`: real time passing does not move it, only :meth:`past`
    does."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop = asyncio.get_running_loop()
        self._real = loop.time
        self._held: float | None = None
        self._offset = 0.0
        monkeypatch.setattr(loop, "time", self)

    def __call__(self) -> float:
        if self._held is None:
            return self._real() + self._offset
        return self._held

    def hold(self) -> None:
        self._held = self()

    async def first(self, *tasks: asyncio.Future[Any], within: float = 5.0) -> None:
        """Wait for the first of *tasks*, *within* seconds of real time at
        most: while the clock is held, a wait on a loop timer never ends."""
        stop = threading.Event()
        bound = asyncio.ensure_future(asyncio.to_thread(stop.wait, within))
        try:
            await asyncio.wait({*tasks, bound}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop.set()
            await bound

    async def past(self, seconds: float) -> None:
        """Move the held clock *seconds* on; return once the loop has run
        what fell due by then."""
        assert self._held is not None
        # A hair past: a deadline the library set at T0 + 0.2 in one sum
        # must not land an ulp after a clock moved to T0 + 0.1 + 0.1, which
        # a large monotonic clock (weeks of uptime) rounds below it.
        self._held += seconds + 1e-6
        loop = asyncio.get_running_loop()
        due = loop.create_future()
        loop.call_at(self._held, lambda: due.cancelled() or due.set_result(None))
        await due

    def require(self, condition: bool, message: str) -> None:
        """Assert *condition* on the held clock, released first when it
        fails: a test failing held would hang in its cleanup, whose loop
        timers never fall due."""
        if not condition:
            self.release()
            raise AssertionError(message)

    def release(self) -> None:
        assert self._held is not None
        self._offset = self._held - self._real()
        self._held = None

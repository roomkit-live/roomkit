"""The virtual clock (README, Setup ``virtual_clock`` and ``clock.advance``).

A scenario that sets ``virtual_clock`` runs with roomkit's private clock seam
(``roomkit._clock``) on a clock only ``clock.advance`` moves: the media paths
that read it measure and wait out their durations on it, so a scenario passes a
10 s grace or a 5 s budget without waiting for it. The runner's own bounds (a
call's, a gate's, the idle wait's) stay on real time.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections.abc import Awaitable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness

# How long ``waiters`` waits for the waits it names (README, ``clock.advance``).
WAITERS_BOUND_S = 5.0
# The most event-loop rounds an advance gives the work a timer resumed.
_READY_ROUNDS = 1000


class VirtualClock:
    """A monotonic clock that moves only when advanced.

    It starts at the real clock's reading, so that a reading of 0.0, which the
    timed paths take for "never", does not come up. Each wait on it is a timer
    with its due time. Advancing ends, in the order they fall due, every timer
    due by the new time, each at its due time, the clock reading that time;
    the work a timer resumes runs until it waits again before the next one
    ends, so a wait it starts within the advance ends in it too.
    """

    def __init__(self) -> None:
        self._now = time.monotonic()
        self._timers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._order = itertools.count()
        # The task behind each pending timer: what ``voice.settle`` counts as
        # waiting on the clock rather than running.
        self._sleepers: dict[asyncio.Task[Any], asyncio.Future[None]] = {}

    def monotonic(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._timers, (self._now + seconds, next(self._order), future))
        task = asyncio.current_task()
        if task is not None:
            self._sleepers[task] = future
        try:
            await future
        finally:
            if task is not None and self._sleepers.get(task) is future:
                del self._sleepers[task]
            future.cancel()

    async def wait_for[T](self, awaitable: Awaitable[T], timeout: float | None) -> T:
        if timeout is None:
            return await awaitable
        inner = asyncio.ensure_future(awaitable)
        timer = asyncio.ensure_future(self.sleep(timeout))
        try:
            await asyncio.wait({inner, timer}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            inner.cancel()
            timer.cancel()
            raise
        if inner.done():
            timer.cancel()
            return inner.result()
        inner.cancel()
        with suppress(asyncio.CancelledError):
            await inner
        raise TimeoutError

    def pending(self) -> int:
        """How many waits on the clock have not ended."""
        return sum(1 for _, _, future in self._timers if not future.done())

    def waits(self, task: asyncio.Task[Any]) -> bool:
        """Whether *task* is waiting on the clock."""
        future = self._sleepers.get(task)
        return future is not None and not future.done()

    async def advance(self, ms: float) -> None:
        """Move the clock *ms* forward, ending every wait due by then, in order."""
        target = self._now + ms / 1000
        while self._timers and self._timers[0][0] <= target:
            due, _, future = heapq.heappop(self._timers)
            if future.done():
                continue
            self._now = max(self._now, due)
            future.set_result(None)
            await _run_ready()
        self._now = target
        await _run_ready()


async def _run_ready() -> None:
    """Until the event loop has nothing ready to run: the work a timer resumed
    has run to its next wait, on the clock or anything else (a gate, a thread,
    a queue). Bounded, for work that never stops yielding."""
    loop = asyncio.get_running_loop()
    for _ in range(_READY_ROUNDS):
        await asyncio.sleep(0)
        if not loop._ready:  # ty: ignore[unresolved-attribute]
            return


async def _advance(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    clock = harness.clock
    if clock is None:
        raise AssertionError("clock.advance needs setup.virtual_clock")
    waiters = spec.get("waiters", 0)
    try:
        async with asyncio.timeout(WAITERS_BOUND_S):
            while clock.pending() < waiters:
                await asyncio.sleep(0.001)
    except TimeoutError:
        raise AssertionError(
            f"{waiters} waits never pended on the clock within {WAITERS_BOUND_S} s"
        ) from None
    await clock.advance(spec["ms"])


CLOCK_CALLS = {"clock.advance": _advance}
CLOCK_ARGUMENTS = {"clock.advance": {"ms", "waiters"}}

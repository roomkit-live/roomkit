"""The clock roomkit's timed media paths read (private).

The RFC's media durations (the interruption strategy's waits, the echo guard's
grace, a turn's wait, the video bridge's keyframe interval and wait, a
conference recording's flush budget, a voice session's audio level throttle)
are measured and waited out through this module rather than through ``time``
and ``asyncio`` directly. The default is the real clock, so nothing changes for
an integrator; a test installs a clock of its own with :func:`installed` to
drive those paths without waiting real time.

Only the paths that read :func:`monotonic`, :func:`sleep` and :func:`wait_for`
follow an installed clock. Work handed to another thread (an inbound DSP
thread, a recorder's writer) still runs on real time: the clock decides when a
wait on the event loop ends, never how long a thread takes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Generator
from contextlib import contextmanager
from typing import Protocol


class Clock(Protocol):
    """What a clock gives the timed paths: a reading and two waits."""

    def monotonic(self) -> float:
        """Seconds on a clock that never goes back, as ``time.monotonic``."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Return once *seconds* passed on this clock, as ``asyncio.sleep``."""
        ...

    async def wait_for[T](self, awaitable: Awaitable[T], timeout: float | None) -> T:
        """Await *awaitable* for at most *timeout* seconds on this clock, as
        ``asyncio.wait_for``: past it, it is cancelled and TimeoutError raised."""
        ...


class _RealClock:
    """The default: ``time.monotonic`` and asyncio's own waits."""

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def wait_for[T](self, awaitable: Awaitable[T], timeout: float | None) -> T:
        return await asyncio.wait_for(awaitable, timeout)


_current: Clock = _RealClock()


def monotonic() -> float:
    """The installed clock's reading, ``time.monotonic()`` by default."""
    return _current.monotonic()


async def sleep(seconds: float) -> None:
    """Wait *seconds* on the installed clock."""
    await _current.sleep(seconds)


async def wait_for[T](awaitable: Awaitable[T], timeout: float | None) -> T:
    """``asyncio.wait_for`` on the installed clock."""
    return await _current.wait_for(awaitable, timeout)


@contextmanager
def installed(clock: Clock) -> Generator[Clock]:
    """Have the timed paths read *clock* until the block ends, then the one before."""
    global _current
    previous = _current
    _current = clock
    try:
        yield clock
    finally:
        _current = previous

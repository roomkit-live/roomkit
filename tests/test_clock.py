"""roomkit's private clock seam (``roomkit._clock``).

The timed media paths read their clock through it: the real one unless a test
installs another, and the real one again once the test's block ends.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable

import pytest

from roomkit import _clock


class _StoppedClock:
    """A clock that reads one time and records what is waited on it."""

    def __init__(self, now: float) -> None:
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


async def _answer() -> str:
    return "answered"


def test_the_real_clock_is_read_by_default() -> None:
    before = time.monotonic()
    reading = _clock.monotonic()
    assert before <= reading <= time.monotonic()


async def test_an_installed_clock_is_read_and_waited_on_until_its_block_ends() -> None:
    stopped = _StoppedClock(now=42.0)
    with _clock.installed(stopped):
        assert _clock.monotonic() == 42.0
        await _clock.sleep(5.0)
        assert await _clock.wait_for(_answer(), 7.0) == "answered"
    assert stopped.slept == [5.0]
    assert stopped.waited == [7.0]
    assert _clock.monotonic() != 42.0


def test_the_clock_before_comes_back_when_the_block_fails() -> None:
    with pytest.raises(RuntimeError), _clock.installed(_StoppedClock(now=42.0)):
        raise RuntimeError("the test failed")
    assert _clock.monotonic() != 42.0


async def test_a_wait_for_on_the_real_clock_times_out_as_asyncio_s() -> None:
    never: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    with pytest.raises(TimeoutError):
        await _clock.wait_for(never, 0.001)
    assert never.cancelled()

"""Fault and latency injection for MockConferenceBackend.

A backend that always succeeds, and succeeds instantly, proves only that the
happy path holds. What a conference actually has to survive is an SFU that
refuses the bot, a control call that times out mid-teardown, and a delivery slow
enough to make one track's latency into every track's latency — none of which a
test can produce by asking the mock nicely.

This is the table that answers those asks. It is deliberately not a second mock:
RFC section 12.10.11 requires *a* Mock implementation, and two of them drift.

Kept out of ``mock.py`` because injecting faults and scripting SFU events are
different jobs, and the mock already does the second one at length.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

ErrorSpec = BaseException | type[BaseException] | Callable[[], BaseException]
"""What a caller may hand ``fail()``: an instance, a class, or a factory."""


@dataclass
class _Fault:
    """A scripted failure and how many calls it still applies to."""

    error: ErrorSpec
    remaining: int | None
    """Calls left before the fault retires. ``None`` never retires."""


class MockFaults:
    """Per-operation failures and delays for a mock backend.

    Operations are named: backend methods (``join_as_bot``, ``leave``, ...) can
    both fail and be slowed; callback emissions (``track_audio``,
    ``participant_joined``, ...) can only be slowed, since a backend's emission
    loop swallows what its subscribers raise and a failure there would be
    invisible by construction.

    Example::

        faults = MockFaults(methods={"leave"}, emissions={"track_audio"})
        faults.fail("leave", TimeoutError, times=1)   # first teardown only
        faults.delay("track_audio", 0.05)             # slow delivery
    """

    def __init__(self, *, methods: frozenset[str], emissions: frozenset[str]) -> None:
        self._methods = methods
        self._emissions = emissions
        self._known = methods | emissions
        self._faults: dict[str, _Fault] = {}
        self._delays: dict[str, float] = {}

    def fail(
        self,
        operation: str,
        error: ErrorSpec | None = None,
        *,
        times: int | None = None,
    ) -> None:
        """Make ``operation`` raise.

        Args:
            operation: Backend method to fail.
            error: Exception instance, class, or factory. Defaults to a
                ``RuntimeError`` naming the operation.
            times: How many calls to fail. ``None`` fails every call; ``1``
                fails the first and lets the rest through, which is how a
                retry or a second teardown gets tested.
        """
        if operation in self._emissions:
            raise ValueError(
                f"{operation!r} is an emission, and emissions cannot be made to fail: "
                "a backend logs what its subscribers raise and carries on, so the "
                "failure would never be observable. Raise from the callback instead."
            )
        self._check(operation, self._methods, "method")
        if times is not None and times < 1:
            raise ValueError(f"times must be at least 1, got {times}")
        if error is None:
            error = RuntimeError(f"mock backend failed: {operation}")
        self._faults[operation] = _Fault(error=error, remaining=times)

    def delay(self, operation: str, seconds: float) -> None:
        """Make ``operation`` take ``seconds`` before doing its work.

        Applies to backend methods and to callback emissions alike. A delayed
        emission is what a slow media path looks like from outside.
        """
        self._check(operation, self._known, "operation")
        if seconds < 0:
            raise ValueError(f"seconds must not be negative, got {seconds}")
        self._delays[operation] = seconds

    def clear(self, operation: str | None = None) -> None:
        """Drop the injected behaviour, for one operation or for all of them."""
        if operation is None:
            self._faults.clear()
            self._delays.clear()
            return
        self._check(operation, self._known, "operation")
        self._faults.pop(operation, None)
        self._delays.pop(operation, None)

    async def apply(self, operation: str) -> None:
        """Sleep, then raise, as ``operation`` was configured to.

        In that order: a call that fails slowly is what a timeout looks like,
        and a call that failed instantly would never let a test hold the window
        open around it.
        """
        if (seconds := self._delays.get(operation)) is not None and seconds > 0:
            await asyncio.sleep(seconds)
        if (fault := self._faults.get(operation)) is None:
            return
        if fault.remaining is not None:
            fault.remaining -= 1
            if fault.remaining <= 0:
                del self._faults[operation]
        raise _build(fault.error)

    def _check(self, operation: str, known: frozenset[str], noun: str) -> None:
        """Refuse a name the backend has never heard of.

        A lever that silently does nothing is worse than no lever: the test
        passes believing it injected something, and the defect it was written
        for stays hidden.
        """
        if operation not in known:
            raise ValueError(f"unknown {noun} {operation!r}. Known: {', '.join(sorted(known))}")


def _build(error: ErrorSpec) -> BaseException:
    """Turn an instance, a class, or a factory into an exception to raise.

    A class and a factory are the same case: both are called to produce the
    exception. Only an instance is used as it stands.
    """
    if isinstance(error, BaseException):
        return error
    return error()

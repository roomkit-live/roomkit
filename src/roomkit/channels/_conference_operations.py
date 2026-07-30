"""Which of the channel's resources each in-flight operation is using.

Every close defect this channel has had was one shape: a resource — the
backend, the shared pipeline, the recognizer, the synthesizer — was closed
while an operation the channel had admitted was still inside it. Each fix
grew its own tracker: a set of teardown tasks for the backend, a set of
abandoned lanes for the pipeline, a per-room lock for the joins. The trackers
were each right and their *union* was the invariant, which is why every new
kind of operation reopened the race.

This module is that union, held in one place: an operation takes a lease on
the resources it uses for exactly as long as it uses them, and a resource is
closed only once no lease on it remains. The registry does not order the
shutdown — ConferenceChannel.close() still decides what happens when — it
answers the one question every step of it has to agree on: *is anything still
using this?*

Leases are deliberately synchronous to acquire and release. Taking one must
not await: an await is a cancellation point, and inserting new ones into the
paths this protects would open the very windows it exists to close.

The recorder is the one resource not counted here: its manager already keeps
a ledger of the calls running inside it — on worker threads, which counters
on the event loop cannot see ending — and refuses to release the provider
under them. One ledger per resource; the recorder's is in
``_conference_recording.py``.

See RFC section 12.10.4.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from enum import Enum


class ConferenceResource(Enum):
    """A resource the channel owns or fronts, closable exactly once."""

    BACKEND = "backend"
    PIPELINE = "pipeline"
    STT = "stt"
    TTS = "tts"


class ConferenceResourceClosedError(RuntimeError):
    """An operation asked to use a resource the shutdown has already closed.

    Raised instead of letting the call through, because the call would be a
    use-after-close: the backend it would reach is gone, and the least wrong
    answer is a refusal that names the shutdown rather than whatever the
    closed resource happens to do.
    """


class OperationLease:
    """One operation's hold on the resources it is using.

    Released exactly once, however many times :meth:`release` is called —
    the paths that release are teardown paths, and teardown paths overlap.
    """

    def __init__(
        self,
        registry: ConferenceOperations,
        resources: tuple[ConferenceResource, ...],
        lease_id: int,
    ) -> None:
        self._registry = registry
        self._resources = resources
        self._id = lease_id
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._registry._release(self._resources, self._id)


class ConferenceOperations:
    """The channel-wide registry of operations and the resources they hold."""

    def __init__(self) -> None:
        self._held: dict[ConferenceResource, int] = dict.fromkeys(ConferenceResource, 0)
        self._holders: dict[ConferenceResource, dict[int, str]] = {
            resource: {} for resource in ConferenceResource
        }
        self._idle: dict[ConferenceResource, asyncio.Event] = {}
        for resource in ConferenceResource:
            event = asyncio.Event()
            event.set()
            self._idle[resource] = event
        self._closed: set[ConferenceResource] = set()
        self._lease_ids = 0

    # -------------------------------------------------------------------------
    # Taking and releasing leases
    # -------------------------------------------------------------------------

    def acquire(self, *resources: ConferenceResource, what: str) -> OperationLease:
        """Hold resources for one operation, refusing any that is closed.

        ``what`` names the operation in the close report when it outlives the
        shutdown's budget — "publishing bot audio", "lane for track t-1" —
        so the failure says what is still running, not just that something is.
        """
        for resource in resources:
            if resource in self._closed:
                raise ConferenceResourceClosedError(
                    f"The conference channel's {resource.value} is closed; refusing to start "
                    f"{what!r} against it. The channel was shut down while this operation "
                    "was on its way in."
                )
        self._lease_ids += 1
        lease_id = self._lease_ids
        for resource in resources:
            self._held[resource] += 1
            self._holders[resource][lease_id] = what
            self._idle[resource].clear()
        return OperationLease(self, resources, lease_id)

    @contextlib.contextmanager
    def use(self, *resources: ConferenceResource, what: str) -> Iterator[None]:
        """Hold resources for the duration of a block."""
        lease = self.acquire(*resources, what=what)
        try:
            yield
        finally:
            lease.release()

    def _release(self, resources: tuple[ConferenceResource, ...], lease_id: int) -> None:
        for resource in resources:
            self._held[resource] -= 1
            self._holders[resource].pop(lease_id, None)
            if self._held[resource] <= 0:
                self._held[resource] = 0
                self._idle[resource].set()

    # -------------------------------------------------------------------------
    # What the shutdown reads
    # -------------------------------------------------------------------------

    def in_use(self, resource: ConferenceResource) -> bool:
        """Whether any admitted operation still holds the resource."""
        return self._held[resource] > 0

    def holders(self, resource: ConferenceResource) -> list[str]:
        """What is still using a resource, by the names the leases gave."""
        return sorted(set(self._holders[resource].values()))

    def is_closed(self, resource: ConferenceResource) -> bool:
        return resource in self._closed

    def mark_closed(self, resource: ConferenceResource) -> None:
        """Record that a resource is gone, so no further lease can be taken."""
        self._closed.add(resource)

    async def wait_idle(self, *resources: ConferenceResource, timeout: float) -> bool:
        """Wait for resources to fall out of use, on one shared budget.

        Says whether they all did. Bounded, because what holds a lease past
        this point is an operation that already outlived its own budget — the
        shutdown reports it and retains the resource rather than waiting on it
        twice.
        """
        pending = [
            self._idle[resource] for resource in resources if not self._idle[resource].is_set()
        ]
        if not pending:
            return True
        waiters = [asyncio.ensure_future(event.wait()) for event in pending]
        try:
            _, unfinished = await asyncio.wait(waiters, timeout=timeout)
            return not unfinished
        finally:
            for waiter in waiters:
                waiter.cancel()

    async def when_idle(self, *resources: ConferenceResource) -> None:
        """Wait however long it takes for resources to fall out of use.

        The deferred-close wait: it runs on a background task the shutdown has
        already reported, where its lack of a bound costs the shutdown nothing.
        """
        for resource in resources:
            await self._idle[resource].wait()

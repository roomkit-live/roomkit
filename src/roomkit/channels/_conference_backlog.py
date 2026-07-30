"""The bounded backlog a conference track's work is handed through.

Two things in a conference channel accept media from the backend's emission
loop and do the slow part elsewhere: the lane that recognises speech
(:mod:`roomkit.channels._conference_lane`) and the writer that records the
track (:mod:`roomkit.channels._conference_recording_writer`). Both need the
same thing from the callback's point of view — take this and return — and the
same thing under overload: a bound, an eviction policy, and a count of what was
lost.

That is what this holds, once. The two owners differ in what they do with an
item and in how they stop; they do not differ in how they are fed, and a second
implementation of the queue would be a second place for the drop policy to
drift from what the RFC says the implementation documents.

Dropping the *oldest* is the policy, for both: it keeps the work close to live
at the cost of a gap, where dropping the newest would hold on to audio that is
already too old to act on. RFC sections 12.10.4 and 12.10.8 require the choice
to be documented and the loss to be exposed — this is where the counting
happens; the owners name what was lost in their own terms.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

_DROP_LOG_INTERVAL = 100
"""How often an owner is told it is dropping, in dropped items.

A backlog that is behind drops continuously, so notifying per item would bury
everything else in the process.
"""


class TrackBacklog[T]:
    """A bounded queue that drops its oldest item rather than block a caller.

    :meth:`submit` never blocks and never awaits: it runs where frames arrive,
    and the backend awaits its subscribers one after another, so anything
    slower there stalls delivery for every other track in the conference.

    The consumer takes items with :meth:`get`, and :meth:`take_ready` gives it
    everything else already queued so a batch can be processed in one step —
    which is what makes one hand-off to a worker thread carry more than a
    single 20 ms frame.

    ``on_overflow`` is called with the running total every
    ``_DROP_LOG_INTERVAL`` drops, starting with the first, so the owner logs
    the loss in the terms its own reader understands.
    """

    def __init__(self, *, maxsize: int, on_overflow: Callable[[int], None]) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._on_overflow = on_overflow
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """How many items were evicted because the consumer fell behind."""
        return self._dropped

    def submit(self, item: T) -> None:
        """Hand an item over. Never blocks, never awaits."""
        if self._queue.full():
            # Drop the oldest rather than the new one, so the work stays close
            # to live. Nothing runs between the check and the get: this is one
            # event loop, so the queue cannot empty underneath it. task_done()
            # keeps join()'s accounting straight for the item that will never
            # be processed.
            self._queue.get_nowait()
            self._queue.task_done()
            self._record_drop()
        self._queue.put_nowait(item)

    def _record_drop(self) -> None:
        """Count an evicted item, and tell the owner without flooding its log."""
        self._dropped += 1
        if self._dropped % _DROP_LOG_INTERVAL == 1:
            self._on_overflow(self._dropped)

    def discard(self, count: int) -> None:
        """Count items lost for a reason other than the bound.

        A write the recorder refused is loss the same way an eviction is, and
        an integrator reading a hole in a recording asks how much went missing
        rather than by which mechanism. It does not go through
        ``on_overflow``: the owner already knows why those were lost and says
        so itself, where the overflow message would blame a backlog that was
        never full.
        """
        self._dropped += count

    async def get(self) -> T:
        """Wait for the next item."""
        return await self._queue.get()

    def take_ready(self) -> list[T]:
        """Everything already queued, in order. Never waits."""
        ready: list[T] = []
        while not self._queue.empty():
            ready.append(self._queue.get_nowait())
        return ready

    def task_done(self, count: int = 1) -> None:
        """Report items as processed, so :meth:`join` can tell when it is idle."""
        for _ in range(count):
            self._queue.task_done()

    async def join(self) -> None:
        """Wait until everything submitted has been reported done."""
        await self._queue.join()

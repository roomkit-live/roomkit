"""The summary a summarizing memory carries from one turn to the next
(RFC §20.3): what it last summarized of a room, so that a later read starts
from it rather than from the window alone."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace

from roomkit.memory._summary import is_summary, summary_message
from roomkit.memory.base import MemoryResult
from roomkit.models.event import RoomEvent


@dataclass(frozen=True)
class SummaryChain:
    """The summary of a room's turns up to *through*, the event at *index*,
    and the labels its lines named participants by (*speakers*)."""

    summary: str
    through: str
    index: int
    speakers: tuple[str, ...] = ()

    def after(self, prior: SummaryChain | None) -> SummaryChain:
        """This chain, built on *prior*: the speakers both named."""
        if prior is None:
            return self
        return replace(self, speakers=tuple(sorted({*prior.speakers, *self.speakers})))

    def carried(self, result: MemoryResult) -> MemoryResult:
        """*result* read after the summary: its events past the ones the
        summary covers, and the summary's message in place of any summary
        *result* held, after its other messages."""
        messages = [message for message in result.messages if not is_summary(message)]
        return replace(
            result,
            messages=[*messages, summary_message(self.summary)],
            events=self._after(result.events),
            speakers=sorted({*result.speakers, *self.speakers}),
        )

    def _after(self, events: list[RoomEvent]) -> list[RoomEvent]:
        """*events* after the last one the summary covers: those after it
        where the window still holds it, else those indexed after it."""
        for at, event in enumerate(events):
            if event.id == self.through:
                return events[at + 1 :]
        return [event for event in events if event.index > self.index]


class SummaryChains:
    """The latest :class:`SummaryChain` of each room an agent reads, kept in
    the process like a cache: one a room has not used for *ttl* seconds is
    forgotten, and at most *limit* rooms are kept, the least recently used
    dropped first. A forgotten chain is rebuilt from the window."""

    def __init__(self, ttl: float, limit: int) -> None:
        self._ttl = ttl
        self._limit = limit
        self._chains: OrderedDict[tuple[str, str | None], tuple[float, SummaryChain]] = (
            OrderedDict()
        )

    def get(self, room_id: str, channel_id: str | None, now: float) -> SummaryChain | None:
        """The chain of *room_id* for *channel_id*'s agent, its use at *now*
        recorded; ``None`` when there is none or it expired."""
        key = (room_id, channel_id)
        entry = self._chains.get(key)
        if entry is None:
            return None
        used_at, chain = entry
        if now - used_at >= self._ttl:
            del self._chains[key]
            return None
        self._chains[key] = (now, chain)
        self._chains.move_to_end(key)
        return chain

    def put(self, room_id: str, channel_id: str | None, chain: SummaryChain, now: float) -> None:
        """Make *chain* the one *room_id* is read after."""
        key = (room_id, channel_id)
        self._chains[key] = (now, chain)
        self._chains.move_to_end(key)
        while len(self._chains) > self._limit:
            self._chains.popitem(last=False)

    def forget(self, room_id: str) -> None:
        """Forget every chain of *room_id*."""
        for key in [key for key in self._chains if key[0] == room_id]:
            del self._chains[key]

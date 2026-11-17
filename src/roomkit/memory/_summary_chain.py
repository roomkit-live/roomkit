"""The summary a summarizing memory carries from one turn to the next
(RFC §20.3): what it last summarized of a room, so that a later read starts
from it rather than from the window alone."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime

from roomkit.memory._summary import is_summary, summary_message
from roomkit.memory.base import MemoryResult
from roomkit.memory.token_estimator import extract_event_text
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent


@dataclass(frozen=True)
class SummaryChain:
    """The summary of a room's turns, for the room created at *room*: the
    turns it *covers*, each by its id and the fingerprint of what it said
    when summarized, and the labels its lines named participants by
    (*speakers*)."""

    summary: str
    room: datetime | None
    covers: tuple[tuple[str, str], ...]
    speakers: tuple[str, ...] = ()

    def holds(self, context: RoomContext) -> bool:
        """Whether the summary still stands for the room of *context*: the
        room it was built for, not one created again under its id, and each
        turn it covers still in the room's history as the channel reads it,
        saying what it said. A turn deleted, edited or whose content is
        withheld since (RFC §10.3), or gone out of the history, where the
        memory can no longer tell, ends it."""
        if context.room.created_at != self.room:
            return False
        said = {event.id: fingerprint(event) for event in context.recent_events}
        return all(said.get(event_id) == print_ for event_id, print_ in self.covers)

    def after(self, prior: SummaryChain | None) -> SummaryChain:
        """This chain, built on *prior*: the turns and speakers both cover."""
        if prior is None:
            return self
        return replace(
            self,
            covers=(*prior.covers, *self.covers),
            speakers=tuple(sorted({*prior.speakers, *self.speakers})),
        )

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
        """*events* after the last one the summary covers; all of them when
        it covers none, which then all came after it."""
        covered = {event_id for event_id, _ in self.covers}
        last = max((at for at, event in enumerate(events) if event.id in covered), default=-1)
        return events[last + 1 :]


def chain_for(
    summary: str,
    summarized: list[RoomEvent],
    speakers: list[str],
    context: RoomContext,
) -> SummaryChain | None:
    """The chain for *summary* of the *summarized* turns, each fingerprinted
    as the room's history in *context* holds it; ``None`` when one is not
    there, a turn the memory could never check again."""
    held = {event.id: event for event in context.recent_events}
    if any(event.id not in held for event in summarized):
        return None
    covers = tuple((event.id, fingerprint(held[event.id])) for event in summarized)
    return SummaryChain(summary, context.room.created_at, covers, tuple(speakers))


def fingerprint(event: RoomEvent) -> str:
    """What *event* says as the channel reads it, digested: its text, which
    is empty for a deleted event whose content is withheld (RFC §10.3)."""
    text = extract_event_text(event)
    return hashlib.sha256(text.encode()).hexdigest()


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

    def drop(self, room_id: str, channel_id: str | None) -> None:
        """Forget the chain of *room_id* for *channel_id*'s agent."""
        self._chains.pop((room_id, channel_id), None)

    def forget(self, room_id: str) -> None:
        """Forget every chain of *room_id*."""
        for key in [key for key in self._chains if key[0] == room_id]:
            del self._chains[key]

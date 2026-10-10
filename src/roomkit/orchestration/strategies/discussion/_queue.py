"""A room's speak queue: who is owed a turn, who gets the next one (RFC §19.7.5).

Pure state, no I/O: every process that serves the room reads and changes it
under the room lock and stores it with the room, the lease naming the one
process that gives the turns (rule 16). An instruction's text is never stored:
its entry names the process that holds it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ._names import same_name
from .models import SpeakQueue

KEPT_ASKS = 8
"""The asks an entry keeps: the turn reads only the latest person's and the
latest, so a flood of mentions does not grow the stored queue."""

KEPT_ASKERS = 16


class Ask(BaseModel):
    """An event that asked for an agent's turn."""

    event_id: str
    depth: int
    asker: str
    """Who asked: an agent's channel id or a person's label."""
    person: bool = False
    named: bool = False
    """A person named the agent: the one ask an agent that only listens takes."""


class Entry(BaseModel):
    """A turn an agent is owed: one for every event that asked for it."""

    agent: str
    asks: list[Ask] = Field(default_factory=list)
    asked_by: list[tuple[str, bool]] = Field(default_factory=list)
    """Who asked, and whether a person (else an agent): a person whose label
    reads as an agent's channel id is still a person."""
    front: bool = False
    seq: int = 0
    """Arrival order among front requests, served first come first served."""
    instruction: str | None = None
    """The instruction this turn takes as its input, by event id."""
    holder: str | None = None
    """The process holding the instruction's text (rule 16)."""
    handed: bool = False
    """The lease was handed to that process for this turn once."""
    regenerate: bool = False
    """The turn regenerates the answer to the event it was asked by."""
    depth_recorded: bool = False

    @property
    def own_turn(self) -> bool:
        """A turn of its own, never merged: an instruction's or a regenerated
        answer's, given even while the discussion waits or the agent listens."""
        return self.instruction is not None or self.regenerate

    def add(self, ask: Ask) -> None:
        """Merge *ask* into the turn, keeping only what the turn reads."""
        self.asks.append(ask)
        asker = (ask.asker, ask.person)
        if ask.asker and asker not in self.asked_by:
            self.asked_by = [*self.asked_by, asker][-KEPT_ASKERS:]
        if len(self.asks) > KEPT_ASKS:
            latest_person = next((a for a in reversed(self.asks) if a.person), None)
            kept = self.asks[-KEPT_ASKS:]
            if latest_person is not None and latest_person not in kept:
                kept = [latest_person, *kept[1:]]
            self.asks = kept

    def answered(self) -> Ask | None:
        """The event the turn answers: the latest person's message among the
        events that asked for it, else the latest of them."""
        people = [a for a in self.asks if a.person]
        if people:
            return people[-1]
        return self.asks[-1] if self.asks else None

    def askers(self) -> list[tuple[str, bool]]:
        """Who asked for the turn, and whether a person, in order."""
        return list(self.asked_by)


@dataclass(frozen=True)
class Pick:
    """The next turn, if any, and what was set aside on the way: the turns the
    depth limit stopped (each recorded once, RFC §8.3) and the turns of their
    own past it, which do not wait (an instruction is dropped, a regenerated
    answer recorded and dropped)."""

    entry: Entry | None
    stopped: list[Entry] = field(default_factory=list)
    dropped: list[Entry] = field(default_factory=list)


class SpeakQueueState(BaseModel):
    """The stored speak queue of one room."""

    entries: list[Entry] = Field(default_factory=list)
    listening: list[str] = Field(default_factory=list)
    asked: list[tuple[str, str]] = Field(default_factory=list)
    over: bool = False
    turns_given: int = 0
    last_speaker: str | None = None
    seq: int = 0
    waiting: bool = False
    people_spoke: bool = False
    """A person's message reached the room: it has someone to wait for."""
    speaking: str | None = None
    """The running turn of the process holding the lease (rule 16)."""
    lease_holder: str | None = None
    lease_expires: float = 0.0
    version: int = 0
    """Counts the changes of the queue, for the lease holder to notice another
    process's."""

    # -- Asking --

    def ask(self, agent: str, ask: Ask, *, front: bool = False) -> None:
        """Queue *agent* for a turn *ask* asked for, or merge it into the turn it
        is owed: a front request (a person's) moves it to the front."""
        entry = next((e for e in self.entries if e.agent == agent and not e.own_turn), None)
        if entry is None:
            entry = Entry(agent=agent)
            self.entries.append(entry)
        entry.add(ask)
        entry.depth_recorded = False
        if front and not entry.front:
            entry.front = True
            entry.seq = self._next_seq()

    def queue_instruction(
        self, agent: str, instruction_id: str, depth: int, *, holder: str | None = None
    ) -> None:
        """An instruction addressed to *agent*: a turn of its own, at the front,
        taking the instruction as its input, whose text *holder* keeps."""
        entry = Entry(
            agent=agent,
            front=True,
            seq=self._next_seq(),
            instruction=instruction_id,
            holder=holder,
        )
        entry.add(Ask(event_id=instruction_id, depth=depth, asker=""))
        self.entries.append(entry)

    def queue_regeneration(self, agent: str, ask: Ask) -> bool:
        """A regenerated answer to the event *ask* names: a turn of its own,
        at the front, once per agent and event; whether it was queued."""
        if any(
            e.regenerate and e.agent == agent and e.asks[0].event_id == ask.event_id
            for e in self.entries
        ):
            return False
        entry = Entry(agent=agent, front=True, seq=self._next_seq(), regenerate=True)
        entry.add(ask)
        self.entries.append(entry)
        return True

    def record_asked(self, agent: str, person: str) -> None:
        if not any(a == agent and same_name(p, person) for a, p in self.asked):
            self.asked.append((agent, person))

    def asking(self, person: str | None) -> list[str]:
        """The agents that asked *person* (every person when None), in order."""
        return list(
            dict.fromkeys(a for a, p in self.asked if person is None or same_name(p, person))
        )

    def clear_asked(self, person: str | None, *, agents: list[str] | None = None) -> None:
        """Clear what was asked of *person* (every person when None), only of
        *agents* when given."""
        self.asked = [
            (a, p)
            for a, p in self.asked
            if not ((person is None or same_name(p, person)) and (agents is None or a in agents))
        ]

    # -- Giving turns --

    def ordered(self) -> list[Entry]:
        fronts = sorted((e for e in self.entries if e.front), key=lambda e: e.seq)
        return [*fronts, *(e for e in self.entries if not e.front)]

    def next_turn(self, max_depth: int) -> Pick:
        """The turn to give next, if any: the first entry that can take it, the
        agent that just spoke only when no other can (RFC §19.7.5 rule 7) or
        for a turn of its own. While the discussion waits for a person, only a
        turn of its own."""
        pick = Pick(None)
        runnable: list[Entry] = []
        for entry in self.ordered():
            if not self._may_take(entry) or (self.waiting and not entry.own_turn):
                continue
            answered = entry.answered()
            if answered is not None and answered.depth + 1 >= max_depth:
                self._set_aside(entry, pick)
                continue
            runnable.append(entry)
        first = next(
            (e for e in runnable if e.own_turn or e.agent != self.last_speaker),
            runnable[0] if runnable else None,
        )
        return Pick(first, pick.stopped, pick.dropped)

    def _set_aside(self, entry: Entry, pick: Pick) -> None:
        if entry.own_turn:
            self.entries.remove(entry)
            pick.dropped.append(entry)
        elif not entry.depth_recorded:
            entry.depth_recorded = True
            pick.stopped.append(entry)

    def _may_take(self, entry: Entry) -> bool:
        if entry.own_turn or entry.agent not in self.listening:
            return True
        # An agent that only listens answers a person who names it (§6.4).
        return any(a.person and a.named for a in entry.asks)

    def take(self, entry: Entry) -> None:
        self.entries.remove(entry)
        self.speaking = entry.agent
        self.turns_given += 1

    def ended(self, agent: str) -> None:
        if self.speaking == agent:
            self.speaking = None
        self.last_speaker = agent

    def person_wrote(self) -> bool:
        """A person's message: the discussion no longer waits (rule 10);
        whether it was waiting."""
        self.people_spoke = True
        was, self.waiting = self.waiting, False
        return was

    def owes_a_person(self, *, has_people: bool) -> bool:
        """Whether only a person can move the room on: the room has someone to
        wait for, and an agent asked one, an agent of the queue only listens,
        or the depth limit stopped a turn."""
        if not (has_people or self.people_spoke):
            return False
        if self.asked:
            return True
        return any(
            (e.agent in self.listening and not e.own_turn) or e.depth_recorded
            for e in self.entries
        )

    def drop_instructions(self) -> list[Entry]:
        """Remove the instruction turns, whose text a stopped process held;
        the turns removed."""
        dropped = [e for e in self.entries if e.instruction is not None]
        self.entries = [e for e in self.entries if e.instruction is None]
        return dropped

    def drop(self) -> list[Entry]:
        """Empty the queue; the instruction turns it dropped."""
        dropped = [e for e in self.entries if e.instruction is not None]
        self.entries = []
        return dropped

    def view(self) -> SpeakQueue:
        return SpeakQueue(
            speaking=self.speaking,
            queue=tuple(dict.fromkeys(e.agent for e in self.ordered())),
            listening=frozenset(self.listening),
            asked=tuple(self.asked),
            waiting=self.waiting,
            over=self.over,
        )

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

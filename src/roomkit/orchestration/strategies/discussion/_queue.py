"""A room's speak queue: who is owed a turn, who gets the next one (RFC §19.7.5).

Pure state, no I/O: the strategy reads and changes it under the room lock and
stores it with the room. Who speaks now, and an instruction's text, live in the
process that gives the turns and are never stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .models import SpeakQueue


class Ask(BaseModel):
    """An event that asked for an agent's turn."""

    event_id: str
    depth: int
    asker: str
    """Who asked: an agent's channel id or a person's name."""
    person: bool = False


class Entry(BaseModel):
    """A turn an agent is owed: one for every event that asked for it."""

    agent: str
    asks: list[Ask] = Field(default_factory=list)
    front: bool = False
    seq: int = 0
    """Arrival order among front requests, served first come first served."""
    instruction: str | None = None
    """The instruction this turn takes as its input, by event id."""
    regenerate: bool = False
    """The turn regenerates the answer to the event it was asked by."""
    depth_recorded: bool = False

    @property
    def own_turn(self) -> bool:
        """A turn of its own, never merged: an instruction's or a regenerated
        answer's, given even while the discussion waits or the agent listens."""
        return self.instruction is not None or self.regenerate

    def answered(self) -> Ask | None:
        """The event the turn answers: the latest person's message among the
        events that asked for it, else the latest of them."""
        people = [a for a in self.asks if a.person]
        if people:
            return people[-1]
        return self.asks[-1] if self.asks else None

    def askers(self) -> list[str]:
        return list(dict.fromkeys(a.asker for a in self.asks))


@dataclass(frozen=True)
class Pick:
    """The next turn: the entry, the event it answers, and the turns the depth
    limit stopped on the way (each recorded once, RFC §8.3)."""

    entry: Entry | None
    stopped: list[Entry] = field(default_factory=list)


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
    speaking: str | None = Field(default=None, exclude=True)
    """Who speaks belongs to the process that runs the turn (rule 16)."""

    # -- Asking --

    def ask(self, agent: str, ask: Ask, *, front: bool = False) -> None:
        """Queue *agent* for a turn *ask* asked for, or merge it into the turn it
        is owed: a front request (a person's) moves it to the front."""
        entry = next((e for e in self.entries if e.agent == agent and not e.own_turn), None)
        if entry is None:
            entry = Entry(agent=agent)
            self.entries.append(entry)
        entry.asks.append(ask)
        entry.depth_recorded = False
        if front and not entry.front:
            entry.front = True
            entry.seq = self._next_seq()

    def queue_instruction(self, agent: str, instruction_id: str, depth: int) -> None:
        """An instruction addressed to *agent*: a turn of its own, at the front,
        taking the instruction as its input."""
        self.entries.append(
            Entry(
                agent=agent,
                asks=[Ask(event_id=instruction_id, depth=depth, asker="")],
                front=True,
                seq=self._next_seq(),
                instruction=instruction_id,
            )
        )

    def queue_regeneration(self, agent: str, ask: Ask) -> None:
        """A regenerated answer to the event *ask* names: a turn of its own,
        at the front."""
        self.entries.append(
            Entry(agent=agent, asks=[ask], front=True, seq=self._next_seq(), regenerate=True)
        )

    def record_asked(self, agent: str, person: str) -> None:
        if (agent, person) not in self.asked:
            self.asked.append((agent, person))

    def asking(self, person: str | None) -> list[str]:
        """The agents that asked *person* (every person when None), in order."""
        return list(dict.fromkeys(a for a, p in self.asked if person is None or p == person))

    def clear_asked(self, person: str | None, *, agents: list[str] | None = None) -> None:
        """Clear what was asked of *person* (every person when None), only of
        *agents* when given."""
        self.asked = [
            (a, p)
            for a, p in self.asked
            if not ((person is None or p == person) and (agents is None or a in agents))
        ]

    # -- Giving turns --

    def ordered(self) -> list[Entry]:
        fronts = sorted((e for e in self.entries if e.front), key=lambda e: e.seq)
        return [*fronts, *(e for e in self.entries if not e.front)]

    def next_turn(self, max_depth: int) -> Pick:
        """The turn to give next, if any: the first entry that can take it, the
        agent that just spoke only when no other can (RFC §19.7.5 rule 7).
        While the discussion waits for a person, only a turn of its own."""
        stopped: list[Entry] = []
        runnable: list[Entry] = []
        for entry in self.ordered():
            if not self._may_take(entry) or (self.waiting and not entry.own_turn):
                continue
            answered = entry.answered()
            if answered is not None and answered.depth + 1 >= max_depth:
                if not entry.depth_recorded:
                    entry.depth_recorded = True
                    stopped.append(entry)
                continue
            runnable.append(entry)
        if not runnable:
            return Pick(None, stopped)
        first = next((e for e in runnable if e.agent != self.last_speaker), runnable[0])
        return Pick(first, stopped)

    def _may_take(self, entry: Entry) -> bool:
        if entry.own_turn:
            return True
        if entry.agent not in self.listening:
            return True
        # An agent that only listens answers a person who names it (§6.4).
        return any(a.person for a in entry.asks)

    def take(self, entry: Entry) -> None:
        self.entries.remove(entry)
        self.speaking = entry.agent
        self.turns_given += 1

    def ended(self, agent: str) -> None:
        if self.speaking == agent:
            self.speaking = None
        self.last_speaker = agent

    def person_wrote(self) -> None:
        """A person's message: the discussion no longer waits (rule 10)."""
        self.waiting = False

    def owes_a_person(self) -> bool:
        """Whether only a person can move the room on: an agent asked one, an
        agent of the queue only listens, or the depth limit stopped a turn."""
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

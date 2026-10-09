"""Staying quiet when asked: a state of the room, not a judgment remade every turn
(RFC §6.4).

A request to stay quiet or only listen from now on, re-judged from the recent turns
on every turn, faded as they passed and was lost once it left them (measured on a
live session: 0.63, then 0.33 two turns later). :class:`ListeningRooms` keeps it per
room; the classifier reads it with every turn and is asked, instead, whether the turn
puts a question to the agent and whether it lets the agent talk again.
"""

from __future__ import annotations

from dataclasses import dataclass

from roomkit.classifiers.base import YesNoQuestion

LISTEN_REQUEST = YesNoQuestion(
    "Does `last_turn` ask the assistant to stop talking, to stay quiet or to only listen "
    "from now on, until told otherwise? Asking it not to answer this one turn only, or "
    "talking about someone else being quiet, does not count."
)
"""Asked while the room is open: does the turn put it in the listening state?"""

ASKED_ME = YesNoQuestion(
    "The assistant was asked to only listen (`agent.listening_only`). Does `last_turn` "
    "put a question or a request to the assistant itself, expecting it to answer now? "
    "Thinking aloud, talking to someone else, or a question about the topic that is not "
    "put to the assistant does not count."
)
"""Asked while the room listens: is the turn a question for the agent, to answer once?"""

LIFT = YesNoQuestion(
    "The assistant was asked to only listen (`agent.listening_only`). Does `last_turn` "
    "tell the assistant it may talk or intervene again on its own, from now on? A "
    "question or a request put to it is not, by itself, letting it talk again: it "
    "answers that one and goes on listening."
)
"""Asked while the room listens: does the turn end the listening state?"""

DIRECT_ADDRESS = 2.0
"""The directness from which a question breaks the silence of a listening room: the
agent addressed indirectly (2) or directly (3), its name or "you". Below it, a
question said aside and one put to the agent read alike ("What is the base URL?" was
taken as put to the agent at 0.91, "What are we talking about?" at 0.26)."""

ASKED_LIMIT = 200
"""Characters of the request the classifier reads with every turn."""


@dataclass(frozen=True)
class Listening:
    """A room the agent was asked to only listen in."""

    asked: str
    """The request, as it was said, bounded."""


class ListeningRooms:
    """The rooms in the listening state, in memory, by room id: not stored, so a
    restart starts every room open, as it starts every thought empty."""

    def __init__(self) -> None:
        self._rooms: dict[str, Listening] = {}

    def of(self, room_id: str) -> Listening | None:
        """The listening state of *room_id*; ``None`` when the room is open."""
        return self._rooms.get(room_id)

    def start(self, room_id: str, asked: str) -> None:
        """Put *room_id* in the listening state, *asked* being the request."""
        self._rooms[room_id] = Listening(asked[:ASKED_LIMIT])

    def stop(self, room_id: str) -> None:
        """Open *room_id* again."""
        self._rooms.pop(room_id, None)

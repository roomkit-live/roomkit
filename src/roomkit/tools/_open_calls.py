"""A channel's calls waiting for their end rows, as its stored rows are read
back (RFC §6.4)."""

from __future__ import annotations

from collections import deque


class OpenCalls[T]:
    """The calls waiting for their end rows, first in, first out per turn and id.

    A round of a turn is over at that turn's first row that is no start,
    whatever another turn wrote among its starts. A call an earlier round left
    without its end stays so: the next end row under its id is a later call's.
    """

    def __init__(self) -> None:
        self._calls: dict[tuple[str | None, str], deque[tuple[int, T]]] = {}
        self._rounds: dict[str | None, int] = {}

    def hold(self, turn: str | None, tool_id: str, call: T) -> None:
        """*call* waits behind its round's calls under *tool_id*."""
        number = self._rounds.get(turn, 0)
        calls = self._calls.setdefault((turn, tool_id), deque())
        if calls and calls[-1][0] != number:
            calls.clear()
        calls.append((number, call))

    def round_over(self, turn: str | None) -> None:
        """A row of *turn* that is no start was read: its round is over."""
        self._rounds[turn] = self._rounds.get(turn, 0) + 1

    def take(self, turn: str | None, tool_id: str) -> T | None:
        """The call an end row of *turn* under *tool_id* closes, if any."""
        calls = self._calls.get((turn, tool_id))
        return calls.popleft()[1] if calls else None

"""A scripted dispatch policy for tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from .base import DispatchDecision, DispatchPolicy, DispatchTurn


class MockDispatchPolicy(DispatchPolicy):
    """Decides from a script: each call takes the next decision, given as a
    :class:`DispatchDecision` or as the agents' channel ids, and the last one
    repeats. With no script, the message asks what it asks with no policy:
    the candidates that asked its author, else every candidate. *delay*
    holds each decision back, *error* is raised instead. Records every turn.
    """

    def __init__(
        self,
        decisions: Sequence[DispatchDecision | Sequence[str]] = (),
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._decisions = [
            d if isinstance(d, DispatchDecision) else DispatchDecision(tuple(d), "scripted")
            for d in decisions
        ]
        self._delay = delay
        self._error = error
        self.turns: list[DispatchTurn] = []

    async def decide(self, turn: DispatchTurn) -> DispatchDecision:
        self.turns.append(turn)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        if not self._decisions:
            undecided = turn.asked or tuple(c.channel_id for c in turn.candidates)
            return DispatchDecision(tuple(undecided), "as with no policy")
        return self._decisions[min(len(self.turns), len(self._decisions)) - 1]

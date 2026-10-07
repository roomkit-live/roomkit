"""A scripted speak policy and thinker for tests."""

from __future__ import annotations

import asyncio

from roomkit.providers.ai.base import AIContext
from roomkit.speaking.base import SpeakDecision, SpeakMode, SpeakPolicy, SpeakTurn
from roomkit.speaking.thinker import Thinker
from roomkit.speaking.thought import Thought


class MockSpeakPolicy(SpeakPolicy):
    """Returns scripted decisions in order (the last one repeats) after *delay*
    seconds, or raises *error*; records the turns it was asked about."""

    def __init__(
        self,
        decisions: list[SpeakDecision | SpeakMode] | None = None,
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._decisions = [
            d if isinstance(d, SpeakDecision) else SpeakDecision(d) for d in decisions or ["speak"]
        ]
        self._delay = delay
        self._error = error
        self.turns: list[SpeakTurn] = []

    async def decide(self, turn: SpeakTurn) -> SpeakDecision:
        self.turns.append(turn)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._decisions[min(len(self.turns), len(self._decisions)) - 1]


class MockThinker(Thinker):
    """Returns scripted thoughts in order (the last one repeats) after *delay*
    seconds, or raises *error*; without a script it keeps the previous thought.
    Records each call's previous thought and context."""

    def __init__(
        self,
        thoughts: list[Thought] | None = None,
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._thoughts = list(thoughts or [])
        self._delay = delay
        self._error = error
        self.calls: list[tuple[Thought, AIContext]] = []

    async def think(self, previous: Thought, context: AIContext) -> Thought:
        self.calls.append((previous, context))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        if not self._thoughts:
            return previous
        return self._thoughts[min(len(self.calls), len(self._thoughts)) - 1]

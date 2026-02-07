"""Mock turn detector for testing."""

from __future__ import annotations

from roomkit.voice.pipeline.turn.base import TurnContext, TurnDecision, TurnDetector


class MockTurnDetector(TurnDetector):
    """Mock turn detector that returns a preconfigured sequence of decisions."""

    def __init__(self, decisions: list[TurnDecision] | None = None) -> None:
        self._decisions = decisions or []
        self._index = 0
        self.evaluations: list[TurnContext] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockTurnDetector"

    def evaluate(self, context: TurnContext) -> TurnDecision:
        self.evaluations.append(context)
        if self._index < len(self._decisions):
            decision = self._decisions[self._index]
            self._index += 1
            return decision
        return TurnDecision(is_complete=True, confidence=1.0)

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True

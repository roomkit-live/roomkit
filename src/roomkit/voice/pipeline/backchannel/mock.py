"""Mock backchannel detector for testing."""

from __future__ import annotations

from roomkit.voice.pipeline.backchannel.base import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)


class MockBackchannelDetector(BackchannelDetector):
    """Mock backchannel detector that returns a preconfigured sequence of decisions."""

    def __init__(self, decisions: list[BackchannelDecision] | None = None) -> None:
        self._decisions = decisions or []
        self._index = 0
        self.evaluations: list[BackchannelContext] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockBackchannelDetector"

    def classify(self, context: BackchannelContext) -> BackchannelDecision:
        self.evaluations.append(context)
        if self._index < len(self._decisions):
            decision = self._decisions[self._index]
            self._index += 1
            return decision
        return BackchannelDecision(is_backchannel=False, confidence=1.0)

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True

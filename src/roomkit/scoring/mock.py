"""Mock scorer for testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roomkit.scoring.base import ConversationScorer, Score


@dataclass
class _ScoreCall:
    response_content: str
    query: str
    room_id: str
    channel_id: str


class MockScorer(ConversationScorer):
    """Returns configured scores and records calls.

    Example::

        scorer = MockScorer(scores=[Score(value=0.9, dimension="relevance")])
        results = await scorer.score(
            response_content="Hello!", query="Hi", room_id="r1", channel_id="ai"
        )
        assert len(scorer.calls) == 1
    """

    def __init__(self, scores: list[Score] | None = None) -> None:
        self._scores = scores or [Score(value=1.0, dimension="default")]
        self.calls: list[_ScoreCall] = []

    @property
    def name(self) -> str:
        return "MockScorer"

    async def score(
        self,
        *,
        response_content: str,
        query: str,
        room_id: str,
        channel_id: str,
        usage: dict[str, Any] | None = None,
        thinking: str = "",
    ) -> list[Score]:
        self.calls.append(
            _ScoreCall(
                response_content=response_content,
                query=query,
                room_id=room_id,
                channel_id=channel_id,
            )
        )
        return list(self._scores)

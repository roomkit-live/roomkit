"""Mock knowledge source for testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roomkit.knowledge.base import KnowledgeResult, KnowledgeSource


@dataclass
class _SearchCall:
    query: str
    room_id: str | None
    limit: int


@dataclass
class _IndexCall:
    content: str
    metadata: dict[str, Any] | None


class MockKnowledgeSource(KnowledgeSource):
    """Records calls and returns configured results.

    Example::

        source = MockKnowledgeSource(
            results=[KnowledgeResult(content="Paris is the capital", score=0.95)],
        )
        results = await source.search("capital of France")
        assert len(source.search_calls) == 1
    """

    def __init__(self, results: list[KnowledgeResult] | None = None) -> None:
        self._results = results or []
        self.search_calls: list[_SearchCall] = []
        self.index_calls: list[_IndexCall] = []
        self.closed: bool = False

    @property
    def name(self) -> str:
        return "MockKnowledgeSource"

    async def search(
        self,
        query: str,
        *,
        room_id: str | None = None,
        limit: int = 5,
    ) -> list[KnowledgeResult]:
        self.search_calls.append(_SearchCall(query=query, room_id=room_id, limit=limit))
        return list(self._results[:limit])

    async def index(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.index_calls.append(_IndexCall(content=content, metadata=metadata))

    async def close(self) -> None:
        self.closed = True

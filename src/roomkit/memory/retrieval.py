"""Retrieval-augmented memory provider.

Enriches AI context with knowledge from pluggable sources (vector stores,
search engines, document indexes) while preserving the inner provider's
conversation history.
"""

from __future__ import annotations

import asyncio
import logging

from roomkit.knowledge.base import KnowledgeResult, KnowledgeSource
from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.models.context import RoomContext
from roomkit.models.event import CompositeContent, RoomEvent, TextContent
from roomkit.providers.ai.base import AIMessage

logger = logging.getLogger("roomkit.memory.retrieval")


class RetrievalMemory(MemoryProvider):
    """Wraps an inner provider and enriches context with knowledge sources.

    On ``retrieve``, queries all configured knowledge sources concurrently,
    merges results by score, and prepends a context message with relevant
    knowledge before the inner provider's messages.

    On ``ingest``, forwards to the inner provider and indexes text content
    in all knowledge sources.

    Parameters:
        sources: Knowledge sources to query on each retrieval.
        inner: The wrapped memory provider for conversation history.
        max_results: Maximum knowledge results to include in context.
        min_query_length: Minimum query length to trigger search.
    """

    def __init__(
        self,
        sources: list[KnowledgeSource],
        inner: MemoryProvider,
        *,
        max_results: int = 5,
        min_query_length: int = 3,
    ) -> None:
        self._sources = sources
        self._inner = inner
        self._max_results = max_results
        self._min_query_length = min_query_length

    @property
    def name(self) -> str:
        return f"RetrievalMemory({self._inner.name})"

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        inner_result = await self._inner.retrieve(
            room_id, current_event, context, channel_id=channel_id
        )
        if not self._sources:
            return inner_result

        query = self._extract_query(current_event)
        if not query or len(query) < self._min_query_length:
            return inner_result

        # Search all sources concurrently, tolerating individual failures
        raw_results = await asyncio.gather(
            *[s.search(query, room_id=room_id, limit=self._max_results) for s in self._sources],
            return_exceptions=True,
        )

        all_results: list[KnowledgeResult] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Knowledge source %s failed: %s",
                    self._sources[i].name,
                    result,
                )
                continue
            all_results.extend(result)

        if not all_results:
            return inner_result

        # Deduplicate by content, keep highest score
        seen: dict[str, KnowledgeResult] = {}
        for r in all_results:
            existing = seen.get(r.content)
            if existing is None or r.score > existing.score:
                seen[r.content] = r
        merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)[: self._max_results]

        knowledge_msg = AIMessage(role="user", content=self._format_results(merged))
        return MemoryResult(
            messages=[knowledge_msg] + inner_result.messages,
            events=inner_result.events,
        )

    async def ingest(
        self,
        room_id: str,
        event: RoomEvent,
        *,
        channel_id: str | None = None,
    ) -> None:
        await self._inner.ingest(room_id, event, channel_id=channel_id)
        text = self._extract_query(event)
        if not text:
            return
        results = await asyncio.gather(
            *[
                s.index(text, metadata={"room_id": room_id, "event_id": event.id})
                for s in self._sources
            ],
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Knowledge source %s index failed: %s",
                    self._sources[i].name,
                    result,
                )

    async def clear(self, room_id: str) -> None:
        await self._inner.clear(room_id)

    async def close(self) -> None:
        await self._inner.close()
        results = await asyncio.gather(
            *[s.close() for s in self._sources],
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Knowledge source %s close failed: %s",
                    self._sources[i].name,
                    result,
                )

    @staticmethod
    def _extract_query(event: RoomEvent) -> str:
        """Extract text from event content for search queries."""
        content = event.content
        if isinstance(content, TextContent):
            return content.body
        if isinstance(content, CompositeContent):
            texts = [p.body for p in content.parts if isinstance(p, TextContent)]
            return " ".join(texts)
        return ""

    @staticmethod
    def _format_results(results: list[KnowledgeResult]) -> str:
        """Format knowledge results as a context message."""
        lines = ["[Relevant context from knowledge sources]"]
        for r in results:
            source_prefix = f"[{r.source}] " if r.source else ""
            lines.append(f"- {source_prefix}{r.content}")
        return "\n".join(lines)

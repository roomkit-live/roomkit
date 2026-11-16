"""Retrieval-augmented memory provider.

Enriches AI context with knowledge from pluggable sources (vector stores,
search engines, document indexes) while preserving the inner provider's
conversation history.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from roomkit._media import content_words
from roomkit._text import json_text_line
from roomkit.channels._speaker import committed_label
from roomkit.knowledge.base import KnowledgeResult, KnowledgeSource
from roomkit.memory._wrapper import _MemoryWrapper
from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.tools.fence import fence

logger = logging.getLogger("roomkit.memory.retrieval")


class RetrievalMemory(_MemoryWrapper):
    """Wraps an inner provider and enriches context with knowledge sources.

    On ``retrieve``, queries all configured knowledge sources concurrently,
    merges results by score, and returns the passages as the turn's note
    (``MemoryResult.notes``), each in a ``<knowledge>`` block: they ride the
    turn's input, not the history a provider caches (RFC §20.2).

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
        super().__init__(inner)
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

        # For this turn alone: with the turn's notes, not ahead of the history
        # a provider caches (RFC §20.2).
        return replace(inner_result, notes=[*inner_result.notes, self._format_results(merged)])

    async def ingest(
        self,
        room_id: str,
        event: RoomEvent,
        *,
        channel_id: str | None = None,
    ) -> None:
        await super().ingest(room_id, event, channel_id=channel_id)
        text = self._extract_query(event)
        if not text:
            return
        metadata = {"room_id": room_id, "event_id": event.id, "speaker": committed_label(event)}
        results = await asyncio.gather(
            *[s.index(text, metadata=dict(metadata)) for s in self._sources],
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Knowledge source %s index failed: %s",
                    self._sources[i].name,
                    result,
                )

    async def close(self) -> None:
        await super().close()
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
        """The author's words in *event*, a media's caption included, as the
        search query: never a file name or a type (RFC §6.6)."""
        return content_words(event.content) or ""

    @staticmethod
    def _format_results(results: list[KnowledgeResult]) -> str:
        """The retrieved passages as the turn's note, each set apart as data."""
        lines = ["Relevant context from knowledge sources:"]
        for r in results:
            lines.append(fence("knowledge", _passage(r)))
        return "\n".join(lines)


def _passage(result: KnowledgeResult) -> str:
    """*result* as the note reads it: a passage taken from a turn
    (``event_id`` in its metadata) as a line of the conversation, its text a
    JSON string after the label of its author the runtime set when indexing
    it, so a name its author typed reads as their words (RFC §6.4); one
    indexed with no label quoted alone; a document after its source."""
    if "event_id" not in result.metadata:
        prefix = f"[{result.source}] " if result.source else ""
        return f"{prefix}{result.content}"
    text = json_text_line(result.content)
    speaker = result.metadata.get("speaker")
    return f"{speaker}: {text}" if isinstance(speaker, str) and speaker else text

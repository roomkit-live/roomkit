"""The scripted memory of a scenario's scripted agent.

roomkit-specs ``conformance/README.md`` ("Scripted agents") defines it: the
room's latest ``max_events`` events as a sliding window reads them, with the
messages, notes and speakers the scenario gives, wrapped in the memories the
scenario names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from roomkit.knowledge import KnowledgeResult, KnowledgeSource
from roomkit.memory import (
    BudgetAwareMemory,
    CompactingMemory,
    MemoryProvider,
    MemoryResult,
    RetrievalMemory,
    SlidingWindowMemory,
    SummarizingMemory,
)
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIMessage, AIProvider, AITextPart
from roomkit.tools import current_response_metadata

Compacted = Callable[[str, list[AIMessage], dict[str, int]], None]
"""What a scripted memory tells of each compaction it is asked: the room, the
history and the usage it was handed."""


def _built(message: Mapping[str, Any]) -> AIMessage:
    """A message the scenario says the memory built: its text, or its text
    parts in order."""
    if "parts" in message:
        parts = [AITextPart(text=text) for text in message["parts"]]
        return AIMessage(role=message["role"], content=parts)
    return AIMessage(role=message["role"], content=message["text"])


class ScriptedMemory(MemoryProvider):
    """The room's latest events, as :class:`SlidingWindowMemory` reads them,
    and what the scenario says the memory built, retrieved and names; each
    event it is handed goes to *ingested* (``expect.ingested``). Each
    compaction between two tool rounds answers the scenario's next
    ``compacts`` item, and goes to *compacted* (``expect.compacted``)."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        max_events: int,
        ingested: Callable[[str, RoomEvent], None] | None = None,
        compacted: Compacted | None = None,
    ) -> None:
        self._ingested = ingested
        self._compacted = compacted
        self._compacts = list(spec.get("compacts", []))
        self._window = SlidingWindowMemory(max_events=spec.get("max_events", max_events))
        self._messages = [_built(message) for message in spec.get("messages", [])]
        self._notes = list(spec.get("notes", []))
        self._speakers = list(spec.get("speakers", []))
        self._writes_record = dict(spec.get("writes_record", {}))

    @property
    def recent_events_window(self) -> int:
        return self._window.recent_events_window

    async def ingest(
        self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
    ) -> None:
        if self._ingested is not None:
            self._ingested(room_id, event)

    async def compact(
        self,
        room_id: str,
        history: list[AIMessage],
        *,
        usage: dict[str, int],
        channel_id: str | None = None,
    ) -> list[AIMessage] | None:
        if self._compacted is not None:
            self._compacted(room_id, history, dict(usage))
        answer = self._compacts.pop(0) if self._compacts else None
        if answer is None:
            return None
        if "fails" in answer:
            raise RuntimeError(answer["fails"])
        built = [AIMessage(role=m["role"], content=m["text"]) for m in answer.get("messages", [])]
        keep = answer.get("keep", 0)
        return [*built, *(history[-keep:] if keep else [])]

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        record = current_response_metadata()
        if record is not None:
            record.update(self._writes_record)
        window = await self._window.retrieve(
            room_id, current_event, context, channel_id=channel_id
        )
        return MemoryResult(
            messages=list(self._messages),
            events=window.events,
            notes=list(self._notes),
            speakers=list(self._speakers),
        )


def scripted_memory(
    spec: Mapping[str, Any],
    max_events: int,
    summarizer: AIProvider | None,
    ingested: Callable[[str, RoomEvent], None] | None = None,
    compacted: Compacted | None = None,
) -> MemoryProvider:
    """The memory *spec* describes, its wrappers from the innermost; *max_events*
    is its window by default, the agent's ``max_context_events``, *summarizer*
    the model a compacting or summarizing memory summarizes with,
    *ingested* told each event the memory is handed, and *compacted* each
    compaction it is asked."""
    memory: MemoryProvider = ScriptedMemory(spec, max_events, ingested, compacted)
    if (budget := spec.get("budget_aware")) is not None:
        memory = BudgetAwareMemory(
            memory,
            max_context_tokens=budget["max_context_tokens"],
            safety_margin_ratio=budget.get("safety_margin_ratio", 0.15),
            min_events=budget.get("min_events", 3),
            reserved_tokens=budget.get("reserved_tokens", 0),
        )
    if (compacting := spec.get("compacting")) is not None:
        memory = CompactingMemory(
            memory,
            _summarizer(summarizer),
            compacting["max_context_tokens"],
            summary_ratio=compacting.get("summary_ratio", 0.10),
            safety_margin_ratio=compacting.get("safety_margin_ratio", 0.15),
            min_events=compacting.get("min_events", 5),
        )
    if (summarizing := spec.get("summarizing")) is not None:
        memory = SummarizingMemory(
            memory,
            _summarizer(summarizer),
            summarizing["max_context_tokens"],
            tier1_ratio=summarizing.get("tier1_ratio", 0.50),
            tier2_ratio=summarizing.get("tier2_ratio", 0.85),
            truncate_chars=summarizing.get("truncate_chars", 2000),
            summary_max_tokens=summarizing.get("summary_max_tokens", 1000),
            min_events=summarizing.get("min_events", 5),
        )
    if (retrieval := spec.get("retrieval")) is not None:
        memory = RetrievalMemory(
            [ScriptedSource(source) for source in retrieval["sources"]],
            memory,
            max_results=retrieval.get("max_results", 5),
            min_query_length=retrieval.get("min_query_length", 3),
        )
    return memory


class ScriptedSource(KnowledgeSource):
    """A knowledge source answering every search with its results, at most
    the search's limit, or failing with its ``fails``."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self._name = spec["name"]
        self._results = [
            KnowledgeResult(
                content=result["content"],
                score=float(result.get("score", 0.0)),
                source=result.get("source", ""),
            )
            for result in spec.get("results", [])
        ]
        self._fails = spec.get("fails")

    @property
    def name(self) -> str:
        return self._name

    async def search(
        self, query: str, *, room_id: str | None = None, limit: int = 5
    ) -> list[KnowledgeResult]:
        if self._fails is not None:
            raise RuntimeError(self._fails)
        return self._results[:limit]


def _summarizer(summarizer: AIProvider | None) -> AIProvider:
    if summarizer is None:
        raise ValueError("a compacting or summarizing memory needs its summarizer")
    return summarizer

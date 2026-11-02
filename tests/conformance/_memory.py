"""The scripted memory of a scenario's scripted agent.

roomkit-specs ``conformance/README.md`` ("Scripted agents") defines it: the
room's latest ``max_events`` events as a sliding window reads them, with the
messages, notes and speakers the scenario gives, wrapped in the memories the
scenario names.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from roomkit.memory import (
    BudgetAwareMemory,
    CompactingMemory,
    MemoryProvider,
    MemoryResult,
    SlidingWindowMemory,
    SummarizingMemory,
)
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIMessage, AIProvider


class ScriptedMemory(MemoryProvider):
    """The room's latest events, as :class:`SlidingWindowMemory` reads them,
    and what the scenario says the memory built, retrieved and names."""

    def __init__(self, spec: Mapping[str, Any], max_events: int) -> None:
        self._window = SlidingWindowMemory(max_events=spec.get("max_events", max_events))
        self._messages = [
            AIMessage(role=message["role"], content=message["text"])
            for message in spec.get("messages", [])
        ]
        self._notes = list(spec.get("notes", []))
        self._speakers = list(spec.get("speakers", []))

    @property
    def recent_events_window(self) -> int:
        return self._window.recent_events_window

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
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
    spec: Mapping[str, Any], max_events: int, summarizer: AIProvider | None
) -> MemoryProvider:
    """The memory *spec* describes, its wrappers from the innermost; *max_events*
    is its window by default, the agent's ``max_context_events``, and
    *summarizer* the model a compacting or summarizing memory summarizes with."""
    memory: MemoryProvider = ScriptedMemory(spec, max_events)
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
    return memory


def _summarizer(summarizer: AIProvider | None) -> AIProvider:
    if summarizer is None:
        raise ValueError("a compacting or summarizing memory needs its summarizer")
    return summarizer

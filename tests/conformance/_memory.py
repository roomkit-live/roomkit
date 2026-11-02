"""The scripted memory of a scenario's scripted agent.

roomkit-specs ``conformance/README.md`` ("Scripted agents") defines it: the
room's latest ``max_events`` events as a sliding window reads them, with the
messages, notes and speakers the scenario gives, wrapped in the memories the
scenario names.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from roomkit.memory import BudgetAwareMemory, MemoryProvider, MemoryResult, SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIMessage


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


def scripted_memory(spec: Mapping[str, Any], max_events: int) -> MemoryProvider:
    """The memory *spec* describes; *max_events* is its window by default,
    the agent's ``max_context_events``."""
    memory: MemoryProvider = ScriptedMemory(spec, max_events)
    if (budget := spec.get("budget_aware")) is not None:
        memory = BudgetAwareMemory(
            memory,
            max_context_tokens=budget["max_context_tokens"],
            safety_margin_ratio=budget.get("safety_margin_ratio", 0.15),
            min_events=budget.get("min_events", 3),
            reserved_tokens=budget.get("reserved_tokens", 0),
        )
    return memory

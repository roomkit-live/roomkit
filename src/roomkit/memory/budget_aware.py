"""Token-budget-aware memory provider that trims old events to fit context limits."""

from __future__ import annotations

import logging

from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.token_estimator import estimate_tokens
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent, TextContent

logger = logging.getLogger("roomkit.memory.budget_aware")


class BudgetAwareMemory(MemoryProvider):
    """Wraps a MemoryProvider and trims results to fit within a token budget.

    The budget is calculated as::

        max_context_tokens * (1 - safety_margin_ratio)

    Events are trimmed from the oldest, preserving the most recent conversation.
    """

    def __init__(
        self,
        inner: MemoryProvider,
        max_context_tokens: int,
        safety_margin_ratio: float = 0.15,
        min_events: int = 3,
    ) -> None:
        self._inner = inner
        self._max_context_tokens = max_context_tokens
        self._safety_margin_ratio = safety_margin_ratio
        self._min_events = min_events

    @property
    def name(self) -> str:
        return f"BudgetAwareMemory({self._inner.name})"

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
        budget = int(self._max_context_tokens * (1 - self._safety_margin_ratio))
        trimmed_events = self._trim_events_to_budget(inner_result.events, budget)
        return MemoryResult(
            messages=inner_result.messages,
            events=trimmed_events,
        )

    def _trim_events_to_budget(
        self, events: list[RoomEvent], budget: int
    ) -> list[RoomEvent]:
        if not events:
            return events

        # Estimate tokens per event
        event_costs = []
        for e in events:
            text = e.content.body if isinstance(e.content, TextContent) else str(e.content)
            event_costs.append(estimate_tokens(text))

        # Keep from most recent, working backward
        total = 0
        keep_from = 0
        for i in range(len(events) - 1, -1, -1):
            if total + event_costs[i] > budget and (len(events) - i) > self._min_events:
                keep_from = i + 1
                break
            total += event_costs[i]

        if keep_from > 0:
            logger.info(
                "Trimmed %d oldest events to fit context budget (%d tokens kept of %d budget)",
                keep_from,
                total,
                budget,
            )

        return events[keep_from:]

    async def close(self) -> None:
        await self._inner.close()

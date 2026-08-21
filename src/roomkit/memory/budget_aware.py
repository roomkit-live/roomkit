"""Token-budget-aware memory provider that trims old events to fit context limits."""

from __future__ import annotations

import logging

from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.token_estimator import estimate_event_tokens, history_budget
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent

logger = logging.getLogger("roomkit.memory.budget_aware")


class BudgetAwareMemory(MemoryProvider):
    """Wraps a MemoryProvider and trims results to fit within a token budget.

    The history's budget is what is left of the window once everything else in
    it is paid for — see :func:`roomkit.memory.token_estimator.history_budget`,
    which owns the arithmetic. ``reserved_tokens`` is the caller's declaration
    of the non-history prompt it built (system prompt, tool schemas); the
    injected ``messages`` this wrapper passes through are measured here.

    Events are trimmed from the oldest, preserving the most recent conversation.

    The one way the result can still exceed the budget is ``min_events``: the
    most recent few events are kept even when they don't fit, because returning
    an empty history is worse than returning one that needs compaction.
    """

    def __init__(
        self,
        inner: MemoryProvider,
        max_context_tokens: int,
        safety_margin_ratio: float = 0.15,
        min_events: int = 3,
        reserved_tokens: int = 0,
    ) -> None:
        self._inner = inner
        self._max_context_tokens = max_context_tokens
        self._safety_margin_ratio = safety_margin_ratio
        self._min_events = min_events
        # System prompt + tool schemas: part of the same window, invisible
        # from here. A caller that cannot name its own footprint passes 0 and
        # gets a budget that only accounts for what this wrapper can measure.
        self._reserved_tokens = reserved_tokens

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
        budget = history_budget(
            max_context_tokens=self._max_context_tokens,
            reserved_tokens=self._reserved_tokens,
            messages=inner_result.messages,
            safety_margin_ratio=self._safety_margin_ratio,
        )
        trimmed_events = self._trim_events_to_budget(inner_result.events, budget)
        return MemoryResult(
            messages=inner_result.messages,
            events=trimmed_events,
        )

    def _trim_events_to_budget(self, events: list[RoomEvent], budget: int) -> list[RoomEvent]:
        if not events:
            return events

        event_costs = [estimate_event_tokens(e) for e in events]

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

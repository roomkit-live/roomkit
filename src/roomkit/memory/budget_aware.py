"""Token-budget-aware memory provider that trims old events to fit context limits."""

from __future__ import annotations

import logging
from dataclasses import replace

from roomkit.memory._turn_budget import history_event_costs, turn_history_budget
from roomkit.memory._wrapper import _MemoryWrapper
from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.token_estimator import estimate_event_tokens
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import ProviderError

logger = logging.getLogger("roomkit.memory.budget_aware")


class BudgetAwareMemory(_MemoryWrapper):
    """Wraps a MemoryProvider and trims results to fit within a token budget.

    The history's budget is what is left of the window once everything else in
    it is paid for — see :func:`roomkit.memory.token_estimator.history_budget`,
    which owns the arithmetic. Read for an AI channel's turn, the rest of the
    turn is what the channel measured (:func:`current_turn_footprint`, RFC
    §20): its input is reserved, and its reply budget when larger than the
    safety margin. ``reserved_tokens`` is a floor on that input (system prompt,
    tool schemas, the channel's notes), for a caller that knows a part the
    channel cannot; it is not added to the measure. The injected ``messages``,
    the memory's notes and the current turn this wrapper passes through are
    measured here. A current turn that alone exceeds a known window raises a
    non-retryable ``ProviderError(context_overflow=True)`` before retrieval or
    generation: trimming history cannot make that message fit.

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
        super().__init__(inner)
        self._max_context_tokens = max_context_tokens
        self._safety_margin_ratio = safety_margin_ratio
        self._min_events = min_events
        # A floor on the turn's measured input (``current_turn_footprint()``,
        # RFC §20): the larger of the two is reserved, never their sum.
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
        current_tokens = estimate_event_tokens(current_event, text_only=True)
        if self._max_context_tokens > 0 and current_tokens > self._max_context_tokens:
            raise ProviderError(
                f"The current message alone exceeds the model's context window "
                f"(~{current_tokens} tokens, window {self._max_context_tokens}). "
                "Shorten the message or choose a model with a larger context window.",
                context_overflow=True,
            )
        inner_result = await self._inner.retrieve(
            room_id, current_event, context, channel_id=channel_id
        )
        budget = turn_history_budget(
            max_context_tokens=self._max_context_tokens,
            safety_margin_ratio=self._safety_margin_ratio,
            current_event=current_event,
            reserved_tokens=self._reserved_tokens,
            carried=inner_result,
        )
        trimmed_events = self._trim_events_to_budget(inner_result.events, budget)
        return replace(inner_result, events=trimmed_events)

    def _trim_events_to_budget(self, events: list[RoomEvent], budget: int) -> list[RoomEvent]:
        if not events:
            return events

        # Each event with what the history replays with it: a turn is kept or
        # cut whole, its rounds of tool calls with it (RFC §20).
        event_costs = history_event_costs(events)

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

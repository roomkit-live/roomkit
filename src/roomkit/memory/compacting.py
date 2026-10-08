"""Compacting memory provider that summarizes old events to preserve context."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from dataclasses import replace

from roomkit.memory._summary import SummaryLines, summary_message
from roomkit.memory._wrapper import _MemoryWrapper
from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.token_estimator import (
    estimate_event_tokens,
    estimate_message_tokens,
    estimate_notes_tokens,
)
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIContext, AIMessage, AIProvider

logger = logging.getLogger("roomkit.memory.compacting")

_MAX_CACHE_ENTRIES = 200


class CompactingMemory(_MemoryWrapper):
    """Extends budget-aware trimming with summarization of removed events.

    When events exceed the token budget, older events are summarized using
    a lightweight AI provider (e.g. Haiku) and injected as a pre-built
    message at the start of the conversation.
    """

    def __init__(
        self,
        inner: MemoryProvider,
        provider: AIProvider,
        max_context_tokens: int,
        summary_ratio: float = 0.10,
        safety_margin_ratio: float = 0.15,
        min_events: int = 5,
        summary_cache_ttl_seconds: float = 300.0,
    ) -> None:
        super().__init__(inner)
        self._provider = provider
        self._max_context_tokens = max_context_tokens
        self._summary_ratio = summary_ratio
        self._safety_margin_ratio = safety_margin_ratio
        self._min_events = min_events
        self._cache_ttl = summary_cache_ttl_seconds
        self._summary_cache: OrderedDict[tuple[str, str | None], tuple[float, str, str]] = (
            OrderedDict()
        )
        self._cache_generation = 0

    @property
    def name(self) -> str:
        return f"CompactingMemory({self._inner.name})"

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
        # What the inner provider carries besides its events (its messages,
        # and the notes that ride the turn's input) is not compacted: it is
        # paid for first.
        carried = sum(estimate_message_tokens(m) for m in inner_result.messages)
        carried += estimate_notes_tokens(inner_result.notes)
        budget = int(self._max_context_tokens * (1 - self._safety_margin_ratio)) - carried

        events = inner_result.events
        # An image costs what the provider bills for its pixels, never its
        # URL's or its base64's length (RMK-589).
        event_costs = [estimate_event_tokens(e) for e in events]
        total_cost = sum(event_costs)

        if total_cost <= budget:
            return inner_result

        # Split: find how many recent events fit in (budget - summary_budget)
        summary_budget = int(self._max_context_tokens * self._summary_ratio)
        events_budget = budget - summary_budget

        keep_from = 0
        running = 0
        for i in range(len(events) - 1, -1, -1):
            if running + event_costs[i] > events_budget and (len(events) - i) > self._min_events:
                keep_from = i + 1
                break
            running += event_costs[i]

        if keep_from == 0:
            return inner_result  # Nothing to compact

        trimmed_events = events[:keep_from]
        kept_events = events[keep_from:]

        # Summarize trimmed events
        lines = SummaryLines(context, current_event, channel_id)
        summary = await self._get_or_create_summary(room_id, trimmed_events, lines, channel_id)

        return replace(
            inner_result,
            messages=[*inner_result.messages, summary_message(summary)],
            events=kept_events,
        )

    async def _get_or_create_summary(
        self,
        room_id: str,
        events: list[RoomEvent],
        lines: SummaryLines,
        channel_id: str | None = None,
    ) -> str:
        # Generate summary
        event_texts = lines(events)

        prompt = (
            "Summarize this conversation concisely. Focus on: decisions made, "
            "key findings, tool results, errors encountered, and current task state. "
            "Be specific about file names, error messages, and action outcomes.\n\n"
            + "\n".join(event_texts)
        )

        key = (room_id, channel_id)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        now = asyncio.get_running_loop().time()
        cached = self._summary_cache.get(key)
        if cached is not None:
            cached_ts, cached_digest, cached_summary = cached
            if now - cached_ts < self._cache_ttl and cached_digest == digest:
                self._summary_cache.move_to_end(key)
                return cached_summary
        generation = self._cache_generation

        try:
            response = await self._provider.generate(
                AIContext(
                    messages=[AIMessage(role="user", content=prompt)],
                    system_prompt="You are a conversation summarizer. Be concise and factual.",
                    temperature=0.0,
                    max_tokens=1000,
                )
            )
            summary = response.content.strip() or "[Summary generation failed]"
        except Exception as exc:
            logger.warning("Failed to generate summary: %s", exc)
            summary = f"[Earlier conversation with {len(events)} messages — summary unavailable]"

        # A clear/close during generation must not resurrect an erased summary.
        if generation == self._cache_generation:
            self._summary_cache[key] = (now, digest, summary)
            self._summary_cache.move_to_end(key)
            while len(self._summary_cache) > _MAX_CACHE_ENTRIES:
                self._summary_cache.popitem(last=False)
        return summary

    async def clear(self, room_id: str) -> None:
        self._cache_generation += 1
        for key in list(self._summary_cache):
            if key[0] == room_id:
                del self._summary_cache[key]
        await super().clear(room_id)

    async def close(self) -> None:
        self._cache_generation += 1
        self._summary_cache.clear()
        await super().close()
        await self._provider.close()

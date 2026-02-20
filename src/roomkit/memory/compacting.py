"""Compacting memory provider that summarizes old events to preserve context."""

from __future__ import annotations

import asyncio
import logging

from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.token_estimator import estimate_tokens
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, AIMessage, AIProvider

logger = logging.getLogger("roomkit.memory.compacting")


class CompactingMemory(MemoryProvider):
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
        self._inner = inner
        self._provider = provider
        self._max_context_tokens = max_context_tokens
        self._summary_ratio = summary_ratio
        self._safety_margin_ratio = safety_margin_ratio
        self._min_events = min_events
        self._cache_ttl = summary_cache_ttl_seconds
        self._summary_cache: dict[str, tuple[float, str]] = {}

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
        budget = int(self._max_context_tokens * (1 - self._safety_margin_ratio))

        events = inner_result.events
        event_costs = [self._estimate_event_tokens(e) for e in events]
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
        summary = await self._get_or_create_summary(room_id, trimmed_events)

        summary_message = AIMessage(
            role="user",
            content=f"[Conversation summary — earlier messages compacted]\n{summary}",
        )

        return MemoryResult(
            messages=inner_result.messages + [summary_message],
            events=kept_events,
        )

    @staticmethod
    def _estimate_event_tokens(event: RoomEvent) -> int:
        text = event.content.body if isinstance(event.content, TextContent) else str(event.content)
        return estimate_tokens(text)

    async def _get_or_create_summary(
        self, room_id: str, events: list[RoomEvent]
    ) -> str:
        # Check cache
        now = asyncio.get_event_loop().time()
        if room_id in self._summary_cache:
            cached_ts, cached_summary = self._summary_cache[room_id]
            if now - cached_ts < self._cache_ttl:
                return cached_summary

        # Generate summary
        event_texts: list[str] = []
        for e in events:
            role = (
                "assistant"
                if e.source and e.source.channel_type == ChannelType.AI
                else "user"
            )
            text = (
                e.content.body if isinstance(e.content, TextContent) else str(e.content)
            )
            event_texts.append(f"[{role}]: {text[:2000]}")

        prompt = (
            "Summarize this conversation concisely. Focus on: decisions made, "
            "key findings, tool results, errors encountered, and current task state. "
            "Be specific about file names, error messages, and action outcomes.\n\n"
            + "\n".join(event_texts)
        )

        try:
            response = await self._provider.generate(
                AIContext(
                    messages=[AIMessage(role="user", content=prompt)],
                    system_prompt="You are a conversation summarizer. Be concise and factual.",
                    temperature=0.0,
                    max_tokens=1000,
                )
            )
            summary = response.content or "[Summary generation failed]"
        except Exception as exc:
            logger.warning("Failed to generate summary: %s", exc)
            summary = (
                f"[Earlier conversation with {len(events)} messages — summary unavailable]"
            )

        self._summary_cache[room_id] = (now, summary)
        return summary

    async def close(self) -> None:
        await self._inner.close()
        await self._provider.close()

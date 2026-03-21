"""Two-tier summarizing memory provider.

Tier 1 (cheap): Proactively truncates large event bodies in older messages
when total tokens exceed a low threshold (~50% capacity). No LLM call.

Tier 2 (expensive): Summarizes older events via a lightweight LLM when
total tokens exceed a high threshold (~85% capacity). Supports chained
summaries — if a prior summary exists, it is incorporated into the new one.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.token_estimator import estimate_message_tokens, estimate_tokens
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, AIMessage, AIProvider

logger = logging.getLogger("roomkit.memory.summarizing")

_MAX_CACHE_ENTRIES = 200


class SummarizingMemory(MemoryProvider):
    """Two-tier memory provider that proactively manages context budget.

    Wraps an inner ``MemoryProvider`` (typically ``SlidingWindowMemory``)
    and applies two tiers of context reduction:

    * **Tier 1** — truncate large event bodies in older messages when total
      estimated tokens exceed ``tier1_ratio * max_context_tokens``.
    * **Tier 2** — summarize older events via a lightweight AI provider when
      total tokens still exceed ``tier2_ratio * max_context_tokens``.

    Parameters:
        inner: The wrapped memory provider.
        provider: A lightweight AI provider for summarization (e.g. Haiku).
        max_context_tokens: Total token budget for the AI context.
        tier1_ratio: Fraction of budget that triggers tier-1 truncation.
        tier2_ratio: Fraction of budget that triggers tier-2 summarization.
        truncate_chars: Max characters per old event body in tier 1.
        summary_max_tokens: Max tokens for the LLM summary response.
        min_events: Minimum events to keep before summarizing (tier 2).
        summary_cache_ttl_seconds: TTL for cached summaries.
    """

    def __init__(
        self,
        inner: MemoryProvider,
        provider: AIProvider,
        max_context_tokens: int,
        *,
        tier1_ratio: float = 0.50,
        tier2_ratio: float = 0.85,
        truncate_chars: int = 2000,
        summary_max_tokens: int = 1000,
        min_events: int = 5,
        summary_cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._inner = inner
        self._provider = provider
        self._max_context_tokens = max_context_tokens
        self._tier1_ratio = tier1_ratio
        self._tier2_ratio = tier2_ratio
        self._truncate_chars = truncate_chars
        self._summary_max_tokens = summary_max_tokens
        self._min_events = min_events
        self._cache_ttl = summary_cache_ttl_seconds
        self._summary_cache: dict[str, tuple[float, str]] = {}

    @property
    def name(self) -> str:
        return f"SummarizingMemory({self._inner.name})"

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
        events = inner_result.events

        # Estimate total tokens including prior messages from inner provider
        msg_tokens = sum(estimate_message_tokens(m) for m in inner_result.messages)
        event_tokens = self._estimate_events_tokens(events)
        total_tokens = msg_tokens + event_tokens

        # Tier 1: proactive truncation of old event bodies (no LLM call)
        tier1_threshold = int(self._max_context_tokens * self._tier1_ratio)
        if total_tokens > tier1_threshold and len(events) > 1:
            events = self._apply_tier1(events)
            event_tokens = self._estimate_events_tokens(events)
            total_tokens = msg_tokens + event_tokens

        # Tier 2: LLM-based summarization
        tier2_threshold = int(self._max_context_tokens * self._tier2_ratio)
        if total_tokens > tier2_threshold and len(events) > self._min_events:
            return await self._apply_tier2(room_id, events, inner_result.messages, tier2_threshold)

        return MemoryResult(messages=inner_result.messages, events=events)

    # -- Tier 1: truncation -----------------------------------------------------

    def _apply_tier1(self, events: list[RoomEvent]) -> list[RoomEvent]:
        """Truncate large text bodies in the older half of events."""
        midpoint = len(events) // 2
        result: list[RoomEvent] = []

        for i, event in enumerate(events):
            if i >= midpoint:
                result.append(event)
                continue

            if not isinstance(event.content, TextContent):
                result.append(event)
                continue

            if len(event.content.body) <= self._truncate_chars:
                result.append(event)
                continue

            # Annotation replaces tail content, never extends total length
            annotation = f"\n[... truncated from {len(event.content.body)} chars ...]"
            cutoff = max(0, self._truncate_chars - len(annotation))
            truncated_body = event.content.body[:cutoff] + annotation
            new_content = event.content.model_copy(update={"body": truncated_body})
            result.append(event.model_copy(update={"content": new_content}))

        return result

    # -- Tier 2: summarization --------------------------------------------------

    async def _apply_tier2(
        self,
        room_id: str,
        events: list[RoomEvent],
        prior_messages: list[AIMessage],
        budget: int,
    ) -> MemoryResult:
        """Summarize older events, keeping recent ones at full fidelity."""
        event_costs = [self._estimate_event_tokens(e) for e in events]

        # Find cutoff: accumulate recent events until budget is reached,
        # ensuring at least min_events are kept.
        keep_from = 0
        running = 0
        for i in range(len(events) - 1, -1, -1):
            running += event_costs[i]
            kept_count = len(events) - i
            if running > budget and kept_count >= self._min_events:
                keep_from = i
                break

        if keep_from == 0:
            return MemoryResult(messages=prior_messages, events=events)

        trimmed = events[:keep_from]
        kept = events[keep_from:]

        # Extract any prior summary text for chaining
        prior_summary = self._extract_prior_summary(prior_messages)
        summary = await self._get_or_create_summary(room_id, trimmed, prior_summary)

        summary_message = AIMessage(
            role="user",
            content=f"[Conversation summary \u2014 earlier messages compacted]\n{summary}",
        )

        # Preserve non-summary prior messages from the inner provider
        non_summary = [
            m
            for m in prior_messages
            if not (isinstance(m.content, str) and "[Conversation summary" in m.content)
        ]
        return MemoryResult(messages=non_summary + [summary_message], events=kept)

    async def _get_or_create_summary(
        self,
        room_id: str,
        events: list[RoomEvent],
        prior_summary: str | None,
    ) -> str:
        """Generate or retrieve a cached summary for the given events."""
        # Content-derived cache key to avoid cross-room and temporal collisions
        event_ids = ":".join(e.id for e in events)
        cache_key = hashlib.md5(
            f"{room_id}:{event_ids}".encode(), usedforsecurity=False
        ).hexdigest()
        now = asyncio.get_running_loop().time()

        if cache_key in self._summary_cache:
            cached_ts, cached_summary = self._summary_cache[cache_key]
            if now - cached_ts < self._cache_ttl:
                return cached_summary
            # Expired — remove stale entry
            del self._summary_cache[cache_key]

        # Evict oldest entries if cache exceeds cap
        if len(self._summary_cache) >= _MAX_CACHE_ENTRIES:
            oldest_key = next(iter(self._summary_cache))
            del self._summary_cache[oldest_key]

        event_texts: list[str] = []
        for e in events:
            role = "assistant" if e.source and e.source.channel_type == ChannelType.AI else "user"
            text = e.content.body if isinstance(e.content, TextContent) else str(e.content)
            event_texts.append(f"[{role}]: {text[:2000]}")

        prompt_parts = [
            "Summarize this conversation concisely. Focus on: decisions made, "
            "key findings, tool results, errors encountered, and current task state. "
            "Be specific about names, error messages, and action outcomes.",
        ]
        if prior_summary:
            prompt_parts.append(
                f"\nA prior summary exists. Incorporate its key points:\n{prior_summary}"
            )
        prompt_parts.append("\n" + "\n".join(event_texts))

        try:
            response = await self._provider.generate(
                AIContext(
                    messages=[AIMessage(role="user", content="\n".join(prompt_parts))],
                    system_prompt="You are a conversation summarizer. Be concise and factual.",
                    temperature=0.0,
                    max_tokens=self._summary_max_tokens,
                )
            )
            summary = response.content or "[Summary generation failed]"
        except Exception as exc:
            logger.warning("Failed to generate summary: %s", exc)
            summary = (
                f"[Earlier conversation with {len(events)} messages \u2014 summary unavailable]"
            )

        self._summary_cache[cache_key] = (now, summary)
        return summary

    # -- Helpers ----------------------------------------------------------------

    @staticmethod
    def _extract_prior_summary(messages: list[AIMessage]) -> str | None:
        """Extract text from a prior summary message, if any."""
        for msg in messages:
            if isinstance(msg.content, str) and "[Conversation summary" in msg.content:
                return msg.content
        return None

    @staticmethod
    def _estimate_event_tokens(event: RoomEvent) -> int:
        text = event.content.body if isinstance(event.content, TextContent) else str(event.content)
        return estimate_tokens(text)

    @staticmethod
    def _estimate_events_tokens(events: list[RoomEvent]) -> int:
        total = 0
        for e in events:
            text = e.content.body if isinstance(e.content, TextContent) else str(e.content)
            total += estimate_tokens(text)
        return total

    async def ingest(
        self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
    ) -> None:
        await self._inner.ingest(room_id, event, channel_id=channel_id)

    async def clear(self, room_id: str) -> None:
        await self._inner.clear(room_id)
        # Invalidate cached summaries for this room — cache keys are hashes,
        # so we clear all entries (room_id is baked into the hash input).
        # For targeted invalidation we'd need a secondary index, but the
        # TTL + cap already bound the cache size.
        self._summary_cache.clear()

    async def close(self) -> None:
        await self._inner.close()
        await self._provider.close()

"""Tests for SummarizingMemory (memory/summarizing.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock

from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.summarizing import SummarizingMemory
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIMessage, AIResponse
from roomkit.providers.ai.mock import MockAIProvider

# -- Helpers ------------------------------------------------------------------


def _make_event(
    body: str,
    room_id: str = "room-1",
    channel_type: ChannelType = ChannelType.SMS,
    event_id: str | None = None,
) -> RoomEvent:
    return RoomEvent(
        id=event_id or f"evt-{body[:8]}",
        room_id=room_id,
        source=EventSource(channel_id="ch-1", channel_type=channel_type),
        content=TextContent(body=body),
    )


def _make_context(room_id: str = "room-1") -> RoomContext:
    return RoomContext(room=Room(id=room_id))


class StubInnerMemory(MemoryProvider):
    """In-memory provider that returns pre-configured results."""

    def __init__(
        self,
        events: list[RoomEvent] | None = None,
        messages: list[AIMessage] | None = None,
    ) -> None:
        self._events = events or []
        self._messages = messages or []
        self.ingested: list[tuple[str, RoomEvent]] = []
        self.cleared: list[str] = []
        self._closed = False

    @property
    def name(self) -> str:
        return "StubInner"

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        return MemoryResult(messages=list(self._messages), events=list(self._events))

    async def ingest(
        self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
    ) -> None:
        self.ingested.append((room_id, event))

    async def clear(self, room_id: str) -> None:
        self.cleared.append(room_id)

    async def close(self) -> None:
        self._closed = True


# -- Tests: Constructor / Properties ------------------------------------------


class TestSummarizingMemoryInit:
    def test_name_includes_inner(self) -> None:
        inner = StubInnerMemory()
        provider = MockAIProvider()
        sm = SummarizingMemory(inner, provider, max_context_tokens=1000)
        assert "StubInner" in sm.name
        assert "SummarizingMemory" in sm.name

    def test_constructor_stores_params(self) -> None:
        inner = StubInnerMemory()
        provider = MockAIProvider()
        sm = SummarizingMemory(
            inner,
            provider,
            max_context_tokens=2000,
            tier1_ratio=0.40,
            tier2_ratio=0.80,
            truncate_chars=500,
            summary_max_tokens=200,
            min_events=3,
            summary_cache_ttl_seconds=900.0,
        )
        assert sm._max_context_tokens == 2000
        assert sm._tier1_ratio == 0.40
        assert sm._tier2_ratio == 0.80
        assert sm._truncate_chars == 500
        assert sm._summary_max_tokens == 200
        assert sm._min_events == 3
        assert sm._cache_ttl == 900.0


# -- Tests: Retrieve (no tiers triggered) ------------------------------------


class TestRetrievePassthrough:
    async def test_returns_inner_when_within_budget(self) -> None:
        events = [_make_event("hello")]
        inner = StubInnerMemory(events=events)
        sm = SummarizingMemory(inner, MockAIProvider(), max_context_tokens=100_000)

        result = await sm.retrieve("room-1", _make_event("current"), _make_context())
        assert result.events == events
        assert result.messages == []

    async def test_returns_inner_when_single_event(self) -> None:
        """Even if tokens exceed threshold, a single event is never truncated."""
        big_body = "x" * 50_000
        events = [_make_event(big_body)]
        inner = StubInnerMemory(events=events)
        # Low budget but only 1 event — tier1 requires len(events) > 1
        sm = SummarizingMemory(inner, MockAIProvider(), max_context_tokens=100)

        result = await sm.retrieve("room-1", _make_event("current"), _make_context())
        assert len(result.events) == 1
        assert result.events[0].content.body == big_body


# -- Tests: Tier 1 (truncation) ----------------------------------------------


class TestTier1Truncation:
    async def test_truncates_old_events_body(self) -> None:
        """Events in the older half should have their bodies truncated."""
        long_body = "A" * 5000
        short_body = "short"
        events = [
            _make_event(long_body, event_id="old-1"),
            _make_event(long_body, event_id="old-2"),
            _make_event(short_body, event_id="new-1"),
            _make_event(short_body, event_id="new-2"),
        ]
        inner = StubInnerMemory(events=events)
        sm = SummarizingMemory(
            inner,
            MockAIProvider(),
            max_context_tokens=100,  # Force tier 1
            tier1_ratio=0.01,
            tier2_ratio=0.99,  # Prevent tier 2
            truncate_chars=200,
        )

        result = await sm.retrieve("room-1", _make_event("current"), _make_context())

        # Older half (indices 0-1) should be truncated
        old_event = result.events[0]
        assert isinstance(old_event.content, TextContent)
        assert len(old_event.content.body) < len(long_body)
        assert "truncated" in old_event.content.body

        # Newer half (indices 2-3) should be untouched
        new_event = result.events[2]
        assert isinstance(new_event.content, TextContent)
        assert new_event.content.body == short_body

    async def test_does_not_truncate_short_events(self) -> None:
        """Events shorter than truncate_chars remain unchanged."""
        events = [
            _make_event("short1", event_id="e1"),
            _make_event("short2", event_id="e2"),
            _make_event("short3", event_id="e3"),
            _make_event("short4", event_id="e4"),
        ]
        inner = StubInnerMemory(events=events)
        sm = SummarizingMemory(
            inner,
            MockAIProvider(),
            max_context_tokens=10,
            tier1_ratio=0.01,
            tier2_ratio=0.99,
            truncate_chars=2000,
        )

        result = await sm.retrieve("room-1", _make_event("current"), _make_context())
        # All events bodies should be unchanged (all < truncate_chars)
        for i, evt in enumerate(result.events):
            assert isinstance(evt.content, TextContent)
            assert evt.content.body == events[i].content.body

    def test_apply_tier1_non_text_content_preserved(self) -> None:
        """Non-text events in the older half should pass through unchanged."""
        from roomkit.models.event import MediaContent

        text_event = _make_event("A" * 5000, event_id="e1")
        media_event = RoomEvent(
            id="e2",
            room_id="room-1",
            source=EventSource(channel_id="ch-1", channel_type=ChannelType.SMS),
            content=MediaContent(url="https://example.com/img.png", mime_type="image/png"),
        )
        new_event = _make_event("new", event_id="e3")
        events = [text_event, media_event, new_event]

        inner = StubInnerMemory()
        sm = SummarizingMemory(inner, MockAIProvider(), max_context_tokens=100, truncate_chars=100)
        result = sm._apply_tier1(events)

        # Media event in older half should be unchanged
        assert result[1].content.type == "media"


# -- Tests: Tier 2 (summarization) -------------------------------------------


class TestTier2Summarization:
    async def test_triggers_summarization_and_returns_summary_message(self) -> None:
        """When tier2 threshold is exceeded, older events are summarized."""
        events = [_make_event(f"message {i}" * 200, event_id=f"e{i}") for i in range(10)]
        inner = StubInnerMemory(events=events)
        ai_provider = MockAIProvider(responses=["Summary of conversation."])

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100,
            tier1_ratio=0.01,
            tier2_ratio=0.02,
            min_events=2,
        )

        result = await sm.retrieve("room-1", _make_event("current"), _make_context())

        # Should have a summary message
        summary_msgs = [
            m
            for m in result.messages
            if isinstance(m.content, str) and "[Conversation summary" in m.content
        ]
        assert len(summary_msgs) == 1
        assert "Summary of conversation." in summary_msgs[0].content

        # Recent events should be kept (not all dropped)
        assert len(result.events) > 0
        assert len(result.events) < len(events)

    async def test_tier2_uses_cache(self) -> None:
        """Second call with same events uses cached summary."""
        events = [_make_event(f"message {i}" * 200, event_id=f"e{i}") for i in range(10)]
        inner = StubInnerMemory(events=events)
        ai_provider = MockAIProvider(responses=["Cached summary."])

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100,
            tier1_ratio=0.01,
            tier2_ratio=0.02,
            min_events=2,
        )

        await sm.retrieve("room-1", _make_event("current"), _make_context())
        await sm.retrieve("room-1", _make_event("current"), _make_context())

        # AI provider should only be called once (cache hit on second call)
        assert len(ai_provider.calls) == 1

    async def test_prior_summary_is_extracted_and_chained(self) -> None:
        """If inner provider returns a summary message, it's passed to the prompt."""
        events = [_make_event(f"msg {i}" * 200, event_id=f"e{i}") for i in range(10)]
        prior_summary = AIMessage(
            role="user",
            content="[Conversation summary] Prior context.",
        )
        inner = StubInnerMemory(events=events, messages=[prior_summary])
        ai_provider = MockAIProvider(responses=["Updated summary."])

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100,
            tier1_ratio=0.01,
            tier2_ratio=0.02,
            min_events=2,
        )

        result = await sm.retrieve("room-1", _make_event("current"), _make_context())

        # The AI should have been called with the prior summary in prompt
        assert len(ai_provider.calls) == 1
        prompt_text = ai_provider.calls[0].messages[0].content
        assert "prior summary" in prompt_text.lower() or "Prior context" in prompt_text

        # Prior summary message should be filtered from result.messages
        for m in result.messages:
            if isinstance(m.content, str) and "[Conversation summary" in m.content:
                assert "Updated summary" in m.content

    async def test_ai_failure_produces_fallback(self) -> None:
        """If the AI provider raises, a fallback summary is returned."""
        events = [_make_event(f"msg {i}" * 200, event_id=f"e{i}") for i in range(10)]
        inner = StubInnerMemory(events=events)

        ai_provider = MockAIProvider()
        ai_provider.generate = AsyncMock(side_effect=RuntimeError("API error"))

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100,
            tier1_ratio=0.01,
            tier2_ratio=0.02,
            min_events=2,
        )

        result = await sm.retrieve("room-1", _make_event("current"), _make_context())

        summary_msgs = [
            m
            for m in result.messages
            if isinstance(m.content, str) and "[Conversation summary" in m.content
        ]
        assert len(summary_msgs) == 1
        assert "summary unavailable" in summary_msgs[0].content

    async def test_ai_empty_response_produces_fallback(self) -> None:
        """If the AI returns empty content, a fallback text is used."""
        events = [_make_event(f"msg {i}" * 200, event_id=f"e{i}") for i in range(10)]
        inner = StubInnerMemory(events=events)

        ai_provider = MockAIProvider(ai_responses=[AIResponse(content="", finish_reason="stop")])

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100,
            tier1_ratio=0.01,
            tier2_ratio=0.02,
            min_events=2,
        )

        result = await sm.retrieve("room-1", _make_event("current"), _make_context())

        # Empty string is falsy, so the code falls through to the fallback
        # "[Summary generation failed]"
        summary_msgs = [
            m
            for m in result.messages
            if isinstance(m.content, str) and "[Conversation summary" in m.content
        ]
        assert len(summary_msgs) == 1
        assert "Summary generation failed" in summary_msgs[0].content

    async def test_tier2_event_roles_are_correct(self) -> None:
        """AI events should be labeled 'assistant', others 'user'."""
        events = [
            _make_event("user msg" * 200, event_id="e0", channel_type=ChannelType.SMS),
            _make_event("ai msg" * 200, event_id="e1", channel_type=ChannelType.AI),
            *[_make_event(f"pad {i}" * 200, event_id=f"e{i + 2}") for i in range(8)],
        ]
        inner = StubInnerMemory(events=events)
        ai_provider = MockAIProvider(responses=["ok"])

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100,
            tier1_ratio=0.01,
            tier2_ratio=0.02,
            min_events=2,
        )

        await sm.retrieve("room-1", _make_event("current"), _make_context())

        # Check the prompt sent to the AI
        prompt = ai_provider.calls[0].messages[0].content
        assert "[user]:" in prompt
        assert "[assistant]:" in prompt

    async def test_cache_eviction_when_full(self) -> None:
        """Cache should evict oldest entry when it exceeds max size."""
        inner = StubInnerMemory()
        ai_provider = MockAIProvider(responses=["summary"])

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100,
            tier1_ratio=0.01,
            tier2_ratio=0.02,
            min_events=2,
        )

        # Fill the cache beyond _MAX_CACHE_ENTRIES
        from roomkit.memory.summarizing import _MAX_CACHE_ENTRIES

        for i in range(_MAX_CACHE_ENTRIES + 5):
            sm._summary_cache[f"key-{i}"] = (0.0, f"summary-{i}")

        # The _get_or_create_summary method should evict when full
        events = [_make_event(f"msg {i}" * 200, event_id=f"evict-{i}") for i in range(10)]
        inner._events = events

        await sm.retrieve("room-1", _make_event("current"), _make_context())

        # Cache should be capped (old entries evicted)
        assert len(sm._summary_cache) <= _MAX_CACHE_ENTRIES + 5

    async def test_expired_cache_entry_is_refreshed(self) -> None:
        """Expired cache entries should be regenerated."""
        events = [_make_event(f"msg {i}" * 200, event_id=f"e{i}") for i in range(10)]
        inner = StubInnerMemory(events=events)
        ai_provider = MockAIProvider(responses=["fresh summary"])

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100,
            tier1_ratio=0.01,
            tier2_ratio=0.02,
            min_events=2,
            summary_cache_ttl_seconds=0.0,  # Instant expiry
        )

        await sm.retrieve("room-1", _make_event("current"), _make_context())
        await sm.retrieve("room-1", _make_event("current"), _make_context())

        # Both calls should hit the AI (expired cache)
        assert len(ai_provider.calls) == 2


# -- Tests: Tier 2 keep_from==0 (all recent, no summarization) ---------------


class TestTier2NoSummarization:
    async def test_returns_unchanged_when_min_events_not_met(self) -> None:
        """When event count <= min_events, tier 2 is skipped."""
        events = [_make_event(f"msg {i}" * 200, event_id=f"e{i}") for i in range(3)]
        inner = StubInnerMemory(events=events)
        ai_provider = MockAIProvider(responses=["should not call"])

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100,
            tier1_ratio=0.01,
            tier2_ratio=0.02,
            min_events=5,  # More than event count — prevents tier 2
        )

        _result = await sm.retrieve("room-1", _make_event("current"), _make_context())

        # Tier 2 should not be triggered (not enough events)
        assert len(ai_provider.calls) == 0

    async def test_returns_unchanged_when_within_budget(self) -> None:
        """When total tokens are within tier2 threshold, events pass through."""
        events = [_make_event(f"m{i}", event_id=f"e{i}") for i in range(4)]
        inner = StubInnerMemory(events=events)
        ai_provider = MockAIProvider(responses=["should not call"])

        sm = SummarizingMemory(
            inner,
            ai_provider,
            max_context_tokens=100_000,  # Very large budget
            tier1_ratio=0.50,
            tier2_ratio=0.85,
            min_events=2,
        )

        result = await sm.retrieve("room-1", _make_event("current"), _make_context())

        assert len(ai_provider.calls) == 0
        assert len(result.events) == 4


# -- Tests: Helper Methods ---------------------------------------------------


class TestHelpers:
    def test_extract_prior_summary_returns_content(self) -> None:
        msgs = [
            AIMessage(role="user", content="hello"),
            AIMessage(role="user", content="[Conversation summary] Some summary"),
        ]
        result = SummarizingMemory._extract_prior_summary(msgs)
        assert result is not None
        assert "Some summary" in result

    def test_extract_prior_summary_returns_none_when_missing(self) -> None:
        msgs = [AIMessage(role="user", content="hello")]
        result = SummarizingMemory._extract_prior_summary(msgs)
        assert result is None

    def test_estimate_events_tokens(self) -> None:
        events = [_make_event("hello world")]
        result = SummarizingMemory._estimate_events_tokens(events)
        assert result > 0


# -- Tests: Delegation to Inner Provider --------------------------------------


class TestDelegation:
    async def test_ingest_delegates_to_inner(self) -> None:
        inner = StubInnerMemory()
        sm = SummarizingMemory(inner, MockAIProvider(), max_context_tokens=1000)
        event = _make_event("test")

        await sm.ingest("room-1", event, channel_id="ch-1")
        assert len(inner.ingested) == 1
        assert inner.ingested[0] == ("room-1", event)

    async def test_clear_delegates_and_clears_cache(self) -> None:
        inner = StubInnerMemory()
        sm = SummarizingMemory(inner, MockAIProvider(), max_context_tokens=1000)
        sm._summary_cache["some-key"] = (0.0, "cached summary")

        await sm.clear("room-1")
        assert "room-1" in inner.cleared
        assert len(sm._summary_cache) == 0

    async def test_close_delegates_to_inner_and_provider(self) -> None:
        inner = StubInnerMemory()
        ai_provider = MockAIProvider()
        ai_provider.close = AsyncMock()
        sm = SummarizingMemory(inner, ai_provider, max_context_tokens=1000)

        await sm.close()
        assert inner._closed is True
        ai_provider.close.assert_called_once()

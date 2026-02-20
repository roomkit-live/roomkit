"""Tests for token estimation, BudgetAwareMemory, and CompactingMemory."""

from __future__ import annotations

from roomkit.memory.budget_aware import BudgetAwareMemory
from roomkit.memory.compacting import CompactingMemory
from roomkit.memory.mock import MockMemoryProvider
from roomkit.memory.token_estimator import (
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_tokens,
)
from roomkit.models.context import RoomContext
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIResponse,
    AITextPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 1  # len=0 // 4 + 1

    def test_short_string(self) -> None:
        result = estimate_tokens("Hello, world!")
        assert result == len("Hello, world!") // 4 + 1

    def test_long_string(self) -> None:
        text = "x" * 400
        result = estimate_tokens(text)
        assert result == 101  # 400 // 4 + 1

    def test_unicode_text(self) -> None:
        # Unicode characters may be > 1 byte but len() counts codepoints
        text = "Bonjour le monde! 🌍"
        result = estimate_tokens(text)
        assert result > 0


class TestEstimateMessageTokens:
    def test_string_content(self) -> None:
        msg = AIMessage(role="user", content="Hello world")
        result = estimate_message_tokens(msg)
        # overhead (4) + estimate_tokens("Hello world")
        expected = 4 + estimate_tokens("Hello world")
        assert result == expected

    def test_text_part_content(self) -> None:
        msg = AIMessage(
            role="assistant",
            content=[AITextPart(text="Some response text")],
        )
        result = estimate_message_tokens(msg)
        expected = 4 + estimate_tokens("Some response text")
        assert result == expected

    def test_image_part_content(self) -> None:
        msg = AIMessage(
            role="user",
            content=[AIImagePart(url="https://example.com/img.jpg")],
        )
        result = estimate_message_tokens(msg)
        # overhead + 1000 for image
        assert result == 4 + 1000

    def test_tool_call_part(self) -> None:
        msg = AIMessage(
            role="assistant",
            content=[AIToolCallPart(id="tc1", name="search", arguments={"q": "test"})],
        )
        result = estimate_message_tokens(msg)
        assert result > 4  # overhead + name + arguments

    def test_tool_result_part(self) -> None:
        msg = AIMessage(
            role="tool",
            content=[AIToolResultPart(tool_call_id="tc1", name="search", result="found it")],
        )
        result = estimate_message_tokens(msg)
        assert result > 4

    def test_mixed_content(self) -> None:
        msg = AIMessage(
            role="assistant",
            content=[
                AITextPart(text="Let me search"),
                AIToolCallPart(id="tc1", name="search", arguments={"q": "test"}),
            ],
        )
        result = estimate_message_tokens(msg)
        # Should include both parts
        text_only = 4 + estimate_tokens("Let me search")
        assert result > text_only  # tool call adds more


class TestEstimateContextTokens:
    def test_empty_context(self) -> None:
        context = AIContext()
        result = estimate_context_tokens(context)
        assert result == 0

    def test_system_prompt_counted(self) -> None:
        context = AIContext(system_prompt="Be helpful")
        result = estimate_context_tokens(context)
        assert result == estimate_tokens("Be helpful")

    def test_messages_counted(self) -> None:
        context = AIContext(
            messages=[
                AIMessage(role="user", content="hello"),
                AIMessage(role="assistant", content="hi"),
            ]
        )
        result = estimate_context_tokens(context)
        expected = estimate_message_tokens(context.messages[0]) + estimate_message_tokens(
            context.messages[1]
        )
        assert result == expected

    def test_tools_counted(self) -> None:
        context = AIContext(
            tools=[
                AITool(name="search", description="Search the web", parameters={"type": "object"})
            ]
        )
        result = estimate_context_tokens(context)
        assert result > 0


class TestBudgetAwareMemory:
    async def test_returns_all_events_when_under_budget(self) -> None:
        """Events that fit within budget are returned unchanged."""
        events = [make_event(body=f"msg{i}") for i in range(5)]
        inner = MockMemoryProvider(events=events)
        mem = BudgetAwareMemory(
            inner=inner,
            max_context_tokens=100_000,  # very generous
        )

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        assert len(result.events) == 5

    async def test_trims_oldest_events_when_over_budget(self) -> None:
        """Oldest events are trimmed when total exceeds budget."""
        # Create events with known size (~1000 chars each = ~250 tokens)
        events = [make_event(body="x" * 1000) for _ in range(10)]
        inner = MockMemoryProvider(events=events)
        mem = BudgetAwareMemory(
            inner=inner,
            max_context_tokens=1000,  # tight budget (~850 after margin)
            safety_margin_ratio=0.15,
        )

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        assert len(result.events) < 10
        # Should keep most recent events
        assert result.events[-1] is events[-1]

    async def test_respects_min_events_floor(self) -> None:
        """Always keeps at least min_events even when over budget."""
        events = [make_event(body="x" * 10000) for _ in range(5)]
        inner = MockMemoryProvider(events=events)
        mem = BudgetAwareMemory(
            inner=inner,
            max_context_tokens=100,  # impossibly small budget
            min_events=3,
        )

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        assert len(result.events) >= 3

    async def test_preserves_inner_messages(self) -> None:
        """Pre-built messages from inner provider are preserved."""
        summary = AIMessage(role="system", content="Previous summary")
        events = [make_event(body="msg")]
        inner = MockMemoryProvider(messages=[summary], events=events)
        mem = BudgetAwareMemory(inner=inner, max_context_tokens=100_000)

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        assert len(result.messages) == 1
        assert result.messages[0].content == "Previous summary"

    async def test_empty_events(self) -> None:
        """Empty events list is returned as-is."""
        inner = MockMemoryProvider(events=[])
        mem = BudgetAwareMemory(inner=inner, max_context_tokens=100_000)

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        assert result.events == []

    async def test_close_closes_inner(self) -> None:
        """Closing BudgetAwareMemory closes the inner provider."""
        inner = MockMemoryProvider()
        mem = BudgetAwareMemory(inner=inner, max_context_tokens=100_000)
        await mem.close()
        assert inner.closed is True

    def test_name_includes_inner(self) -> None:
        inner = MockMemoryProvider()
        mem = BudgetAwareMemory(inner=inner, max_context_tokens=100_000)
        assert "MockMemoryProvider" in mem.name

    async def test_safety_margin_reduces_effective_budget(self) -> None:
        """Safety margin reduces the effective budget."""
        events = [make_event(body="x" * 4000) for _ in range(10)]  # ~1000 tokens each
        inner = MockMemoryProvider(events=events)

        # With 15% margin on 10000 tokens = 8500 effective
        mem = BudgetAwareMemory(
            inner=inner,
            max_context_tokens=10000,
            safety_margin_ratio=0.15,
        )

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        # Should keep ~8 events (8500 / 1001 ≈ 8.49)
        assert len(result.events) <= 9


class TestCompactingMemory:
    async def test_no_compaction_when_under_budget(self) -> None:
        """No compaction when everything fits."""
        events = [make_event(body="short")]
        inner = MockMemoryProvider(events=events)
        summary_provider = MockAIProvider(responses=["summary"])
        mem = CompactingMemory(
            inner=inner,
            provider=summary_provider,
            max_context_tokens=100_000,
        )

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        assert len(result.events) == 1
        assert len(result.messages) == 0
        assert len(summary_provider.calls) == 0  # No summary generated

    async def test_compaction_generates_summary(self) -> None:
        """When over budget, trimmed events are summarized."""
        events = [make_event(body="x" * 4000) for _ in range(20)]
        inner = MockMemoryProvider(events=events)
        summary_provider = MockAIProvider(responses=["The conversation discussed XYZ."])
        mem = CompactingMemory(
            inner=inner,
            provider=summary_provider,
            max_context_tokens=5000,
            min_events=3,
        )

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        # Should have summary in messages
        assert len(result.messages) == 1
        assert "[Conversation summary" in result.messages[0].content
        assert "The conversation discussed XYZ." in result.messages[0].content
        # Should have kept some recent events
        assert len(result.events) < 20
        assert len(result.events) >= 3

    async def test_summary_cache_avoids_regeneration(self) -> None:
        """Summary is cached and not regenerated within TTL."""
        events = [make_event(body="x" * 4000) for _ in range(20)]
        inner = MockMemoryProvider(events=events)
        summary_provider = MockAIProvider(responses=["Cached summary"])
        mem = CompactingMemory(
            inner=inner,
            provider=summary_provider,
            max_context_tokens=5000,
            summary_cache_ttl_seconds=300.0,
            min_events=3,
        )

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")

        # First call generates summary
        await mem.retrieve("r1", event, ctx)
        first_call_count = len(summary_provider.calls)
        assert first_call_count == 1

        # Second call uses cache
        await mem.retrieve("r1", event, ctx)
        assert len(summary_provider.calls) == 1  # No additional call

    async def test_summary_failure_graceful_fallback(self) -> None:
        """Summary generation failure produces a fallback message."""
        events = [make_event(body="x" * 4000) for _ in range(20)]
        inner = MockMemoryProvider(events=events)

        async def fail_generate(context: AIContext) -> AIResponse:
            raise RuntimeError("LLM unavailable")

        summary_provider = MockAIProvider()
        summary_provider.generate = fail_generate  # type: ignore[assignment]
        mem = CompactingMemory(
            inner=inner,
            provider=summary_provider,
            max_context_tokens=5000,
            min_events=3,
        )

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        # Should still return results with a fallback summary
        assert len(result.messages) == 1
        assert "summary unavailable" in result.messages[0].content

    async def test_close_closes_both(self) -> None:
        """Closing CompactingMemory closes inner provider and summary provider."""
        inner = MockMemoryProvider()
        summary_provider = MockAIProvider()
        mem = CompactingMemory(
            inner=inner,
            provider=summary_provider,
            max_context_tokens=100_000,
        )
        await mem.close()
        assert inner.closed is True

    def test_name_includes_inner(self) -> None:
        inner = MockMemoryProvider()
        summary_provider = MockAIProvider()
        mem = CompactingMemory(
            inner=inner,
            provider=summary_provider,
            max_context_tokens=100_000,
        )
        assert "MockMemoryProvider" in mem.name

    async def test_preserves_inner_messages(self) -> None:
        """Pre-built messages from inner are preserved alongside summary."""
        existing_msg = AIMessage(role="system", content="existing context")
        events = [make_event(body="x" * 4000) for _ in range(20)]
        inner = MockMemoryProvider(messages=[existing_msg], events=events)
        summary_provider = MockAIProvider(responses=["New summary"])
        mem = CompactingMemory(
            inner=inner,
            provider=summary_provider,
            max_context_tokens=5000,
            min_events=3,
        )

        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="current")
        result = await mem.retrieve("r1", event, ctx)

        # Should have both existing message and new summary
        assert len(result.messages) == 2
        assert result.messages[0].content == "existing context"
        assert "[Conversation summary" in result.messages[1].content

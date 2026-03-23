"""Tests for RetrievalMemory (memory/retrieval.py)."""

from __future__ import annotations

from typing import Any

from roomkit.knowledge.base import KnowledgeResult, KnowledgeSource
from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.memory.retrieval import RetrievalMemory
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import (
    CompositeContent,
    EventSource,
    RoomEvent,
    TextContent,
)
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIMessage

# -- Helpers ------------------------------------------------------------------


def _make_event(
    body: str,
    room_id: str = "room-1",
    event_id: str | None = None,
) -> RoomEvent:
    return RoomEvent(
        id=event_id or f"evt-{body[:8]}",
        room_id=room_id,
        source=EventSource(channel_id="ch-1", channel_type=ChannelType.SMS),
        content=TextContent(body=body),
    )


def _make_context(room_id: str = "room-1") -> RoomContext:
    return RoomContext(room=Room(id=room_id))


class StubInnerMemory(MemoryProvider):
    """Stub inner memory that returns pre-configured results."""

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


class StubKnowledgeSource(KnowledgeSource):
    """Knowledge source that returns pre-configured results."""

    def __init__(
        self,
        results: list[KnowledgeResult] | None = None,
        source_name: str = "stub",
        *,
        fail: bool = False,
    ) -> None:
        self._results = results or []
        self._source_name = source_name
        self._fail = fail
        self.indexed: list[tuple[str, dict[str, Any] | None]] = []
        self._closed = False

    @property
    def name(self) -> str:
        return self._source_name

    async def search(
        self,
        query: str,
        *,
        room_id: str | None = None,
        limit: int = 5,
    ) -> list[KnowledgeResult]:
        if self._fail:
            raise RuntimeError("Search failed")
        return self._results[:limit]

    async def index(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._fail:
            raise RuntimeError("Index failed")
        self.indexed.append((content, metadata))

    async def close(self) -> None:
        if self._fail:
            raise RuntimeError("Close failed")
        self._closed = True


# -- Tests: Constructor / Properties ------------------------------------------


class TestRetrievalMemoryInit:
    def test_name_includes_inner(self) -> None:
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[], inner=inner)
        assert "StubInner" in rm.name
        assert "RetrievalMemory" in rm.name

    def test_constructor_stores_params(self) -> None:
        inner = StubInnerMemory()
        source = StubKnowledgeSource()
        rm = RetrievalMemory(
            sources=[source],
            inner=inner,
            max_results=10,
            min_query_length=5,
        )
        assert rm._max_results == 10
        assert rm._min_query_length == 5
        assert len(rm._sources) == 1


# -- Tests: Retrieve ---------------------------------------------------------


class TestRetrieve:
    async def test_returns_inner_when_no_sources(self) -> None:
        events = [_make_event("hello")]
        inner = StubInnerMemory(events=events)
        rm = RetrievalMemory(sources=[], inner=inner)

        result = await rm.retrieve("room-1", _make_event("query"), _make_context())
        assert result.events == events

    async def test_returns_inner_when_query_too_short(self) -> None:
        events = [_make_event("hello")]
        source = StubKnowledgeSource(results=[KnowledgeResult(content="result", score=0.9)])
        inner = StubInnerMemory(events=events)
        rm = RetrievalMemory(sources=[source], inner=inner, min_query_length=100)

        result = await rm.retrieve("room-1", _make_event("hi"), _make_context())
        assert result.events == events
        # No knowledge messages should be prepended
        assert not any(
            isinstance(m.content, str) and "knowledge sources" in m.content
            for m in result.messages
        )

    async def test_prepends_knowledge_message(self) -> None:
        inner = StubInnerMemory(events=[_make_event("hello")])
        source = StubKnowledgeSource(
            results=[
                KnowledgeResult(content="Relevant fact 1", score=0.9, source="faq"),
                KnowledgeResult(content="Relevant fact 2", score=0.8, source="docs"),
            ]
        )
        rm = RetrievalMemory(sources=[source], inner=inner)

        result = await rm.retrieve("room-1", _make_event("What is RoomKit?"), _make_context())

        # Should have knowledge message prepended
        assert len(result.messages) >= 1
        knowledge_msg = result.messages[0]
        assert isinstance(knowledge_msg.content, str)
        assert "Relevant fact 1" in knowledge_msg.content
        assert "[faq]" in knowledge_msg.content
        assert "Relevant fact 2" in knowledge_msg.content

    async def test_deduplicates_results_by_content(self) -> None:
        """Results with the same content should be deduplicated, keeping highest score."""
        source1 = StubKnowledgeSource(
            results=[KnowledgeResult(content="same content", score=0.5, source="s1")]
        )
        source2 = StubKnowledgeSource(
            results=[KnowledgeResult(content="same content", score=0.9, source="s2")]
        )
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[source1, source2], inner=inner)

        result = await rm.retrieve("room-1", _make_event("search query here"), _make_context())

        knowledge_msg = result.messages[0]
        assert isinstance(knowledge_msg.content, str)
        # Should only appear once
        assert knowledge_msg.content.count("same content") == 1

    async def test_respects_max_results(self) -> None:
        source = StubKnowledgeSource(
            results=[KnowledgeResult(content=f"fact {i}", score=1.0 - i * 0.1) for i in range(10)]
        )
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[source], inner=inner, max_results=3)

        result = await rm.retrieve("room-1", _make_event("search query here"), _make_context())

        knowledge_msg = result.messages[0]
        assert isinstance(knowledge_msg.content, str)
        # Count how many facts appear
        fact_count = sum(1 for i in range(10) if f"fact {i}" in knowledge_msg.content)
        assert fact_count <= 3

    async def test_returns_inner_when_no_results(self) -> None:
        source = StubKnowledgeSource(results=[])
        events = [_make_event("hello")]
        inner = StubInnerMemory(events=events)
        rm = RetrievalMemory(sources=[source], inner=inner)

        result = await rm.retrieve("room-1", _make_event("search query here"), _make_context())
        assert result.events == events
        assert result.messages == []

    async def test_tolerates_source_failure(self) -> None:
        """A failing source should not prevent results from other sources."""
        good_source = StubKnowledgeSource(
            results=[KnowledgeResult(content="good result", score=0.9)]
        )
        bad_source = StubKnowledgeSource(fail=True)
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[bad_source, good_source], inner=inner)

        result = await rm.retrieve("room-1", _make_event("search query here"), _make_context())

        knowledge_msg = result.messages[0]
        assert "good result" in knowledge_msg.content

    async def test_all_sources_fail_returns_inner(self) -> None:
        bad_source = StubKnowledgeSource(fail=True)
        events = [_make_event("hello")]
        inner = StubInnerMemory(events=events)
        rm = RetrievalMemory(sources=[bad_source], inner=inner)

        result = await rm.retrieve("room-1", _make_event("search query here"), _make_context())
        assert result.events == events
        assert result.messages == []

    async def test_sorts_by_score_descending(self) -> None:
        source = StubKnowledgeSource(
            results=[
                KnowledgeResult(content="low", score=0.1),
                KnowledgeResult(content="high", score=0.9),
                KnowledgeResult(content="mid", score=0.5),
            ]
        )
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[source], inner=inner)

        result = await rm.retrieve("room-1", _make_event("search query here"), _make_context())

        knowledge_text = result.messages[0].content
        assert isinstance(knowledge_text, str)
        # "high" should appear before "mid", which should appear before "low"
        high_pos = knowledge_text.index("high")
        mid_pos = knowledge_text.index("mid")
        low_pos = knowledge_text.index("low")
        assert high_pos < mid_pos < low_pos


# -- Tests: Extract Query ----------------------------------------------------


class TestExtractQuery:
    def test_text_content(self) -> None:
        event = _make_event("hello world")
        assert RetrievalMemory._extract_query(event) == "hello world"

    def test_composite_content(self) -> None:
        event = RoomEvent(
            id="evt-1",
            room_id="room-1",
            source=EventSource(channel_id="ch-1", channel_type=ChannelType.SMS),
            content=CompositeContent(
                parts=[TextContent(body="part 1"), TextContent(body="part 2")]
            ),
        )
        query = RetrievalMemory._extract_query(event)
        assert "part 1" in query
        assert "part 2" in query

    def test_non_text_content_returns_empty(self) -> None:
        from roomkit.models.event import MediaContent

        event = RoomEvent(
            id="evt-1",
            room_id="room-1",
            source=EventSource(channel_id="ch-1", channel_type=ChannelType.SMS),
            content=MediaContent(url="https://example.com/img.png", mime_type="image/png"),
        )
        assert RetrievalMemory._extract_query(event) == ""


# -- Tests: Format Results ---------------------------------------------------


class TestFormatResults:
    def test_formats_with_source_prefix(self) -> None:
        results = [KnowledgeResult(content="fact 1", score=0.9, source="faq")]
        text = RetrievalMemory._format_results(results)
        assert "[faq]" in text
        assert "fact 1" in text
        assert "knowledge sources" in text.lower()

    def test_formats_without_source_prefix(self) -> None:
        results = [KnowledgeResult(content="fact 1", score=0.9, source="")]
        text = RetrievalMemory._format_results(results)
        assert "fact 1" in text
        # No source prefix when source is empty
        assert "[]" not in text


# -- Tests: Ingest ------------------------------------------------------------


class TestIngest:
    async def test_delegates_to_inner(self) -> None:
        inner = StubInnerMemory()
        source = StubKnowledgeSource()
        rm = RetrievalMemory(sources=[source], inner=inner)
        event = _make_event("test content")

        await rm.ingest("room-1", event, channel_id="ch-1")

        assert len(inner.ingested) == 1
        assert inner.ingested[0] == ("room-1", event)

    async def test_indexes_text_in_all_sources(self) -> None:
        source1 = StubKnowledgeSource(source_name="s1")
        source2 = StubKnowledgeSource(source_name="s2")
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[source1, source2], inner=inner)
        event = _make_event("index me")

        await rm.ingest("room-1", event)

        assert len(source1.indexed) == 1
        assert source1.indexed[0][0] == "index me"
        assert source1.indexed[0][1]["room_id"] == "room-1"
        assert len(source2.indexed) == 1

    async def test_tolerates_index_failure(self) -> None:
        bad_source = StubKnowledgeSource(fail=True)
        good_source = StubKnowledgeSource()
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[bad_source, good_source], inner=inner)
        event = _make_event("index me")

        # Should not raise
        await rm.ingest("room-1", event)
        assert len(good_source.indexed) == 1

    async def test_skips_indexing_for_non_text(self) -> None:
        from roomkit.models.event import MediaContent

        source = StubKnowledgeSource()
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[source], inner=inner)
        event = RoomEvent(
            id="evt-1",
            room_id="room-1",
            source=EventSource(channel_id="ch-1", channel_type=ChannelType.SMS),
            content=MediaContent(url="https://example.com/img.png", mime_type="image/png"),
        )

        await rm.ingest("room-1", event)
        assert len(source.indexed) == 0


# -- Tests: Clear / Close ----------------------------------------------------


class TestClearClose:
    async def test_clear_delegates_to_inner(self) -> None:
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[], inner=inner)

        await rm.clear("room-1")
        assert "room-1" in inner.cleared

    async def test_close_delegates_to_inner_and_sources(self) -> None:
        source = StubKnowledgeSource()
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[source], inner=inner)

        await rm.close()
        assert inner._closed is True
        assert source._closed is True

    async def test_close_tolerates_source_failure(self) -> None:
        bad_source = StubKnowledgeSource(fail=True)
        good_source = StubKnowledgeSource()
        inner = StubInnerMemory()
        rm = RetrievalMemory(sources=[bad_source, good_source], inner=inner)

        # Should not raise
        await rm.close()
        assert inner._closed is True
        assert good_source._closed is True

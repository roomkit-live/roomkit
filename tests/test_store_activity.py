"""Tests for activity store features: ToolCallContent, EventFilter, PersistencePolicy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import (
    EventSource,
    RoomEvent,
    TextContent,
    ToolCallContent,
)
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter, PersistencePolicy
from roomkit.store.memory import InMemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def _source(channel_id: str = "ai-1") -> EventSource:
    return EventSource(channel_id=channel_id, channel_type=ChannelType.AI)


async def _seed_room(store: InMemoryStore, room_id: str = "room-1") -> str:
    await store.create_room(Room(id=room_id))
    return room_id


# ---------------------------------------------------------------------------
# ToolCallContent model tests
# ---------------------------------------------------------------------------


class TestToolCallContent:
    def test_pending_defaults(self) -> None:
        tc = ToolCallContent(tool_name="web_search", tool_id="call-1")
        assert tc.type == "tool_call"
        assert tc.status == "pending"
        assert tc.result is None
        assert tc.duration_ms is None
        assert tc.error is None

    def test_completed(self) -> None:
        tc = ToolCallContent(
            tool_name="web_fetch",
            tool_id="call-2",
            arguments={"url": "https://example.com"},
            result={"html": "<h1>hi</h1>"},
            status="completed",
            duration_ms=1200,
        )
        assert tc.status == "completed"
        assert tc.duration_ms == 1200

    def test_failed_with_error(self) -> None:
        tc = ToolCallContent(
            tool_name="broken_tool",
            tool_id="call-3",
            status="failed",
            error="Connection timeout",
        )
        assert tc.status == "failed"
        assert tc.error == "Connection timeout"

    def test_discriminator_roundtrip(self) -> None:
        """ToolCallContent survives JSON serialization in a RoomEvent."""
        event = RoomEvent(
            room_id="r1",
            source=_source(),
            type=EventType.TOOL_CALL_START,
            content=ToolCallContent(
                tool_name="search",
                tool_id="c1",
                arguments={"q": "test"},
            ),
        )
        data = event.model_dump_json()
        restored = RoomEvent.model_validate_json(data)
        assert isinstance(restored.content, ToolCallContent)
        assert restored.content.tool_name == "search"


# ---------------------------------------------------------------------------
# EventFilter model tests
# ---------------------------------------------------------------------------


class TestEventFilter:
    def test_empty_filter(self) -> None:
        ef = EventFilter()
        assert ef.event_types is None
        assert ef.exclude_types is None

    def test_time_range_validation(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValueError, match="after_time must be before before_time"):
            EventFilter(after_time=now, before_time=now - timedelta(hours=1))

    def test_equal_times_rejected(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValueError, match="after_time must be before before_time"):
            EventFilter(after_time=now, before_time=now)


# ---------------------------------------------------------------------------
# PersistencePolicy tests
# ---------------------------------------------------------------------------


class TestPersistencePolicy:
    def test_default_persists_all(self) -> None:
        policy = PersistencePolicy()
        assert policy.should_persist(EventType.MESSAGE) is True
        assert policy.should_persist(EventType.TOOL_CALL_START) is True
        assert policy.should_persist(EventType.TYPING) is True

    def test_persist_types_whitelist(self) -> None:
        policy = PersistencePolicy(
            persist_types={EventType.MESSAGE, EventType.TOOL_CALL_START, EventType.TOOL_CALL_END}
        )
        assert policy.should_persist(EventType.MESSAGE) is True
        assert policy.should_persist(EventType.TOOL_CALL_START) is True
        assert policy.should_persist(EventType.TYPING) is False
        assert policy.should_persist(EventType.PRESENCE) is False

    def test_exclude_types(self) -> None:
        policy = PersistencePolicy(exclude_types={EventType.TYPING, EventType.PRESENCE})
        assert policy.should_persist(EventType.MESSAGE) is True
        assert policy.should_persist(EventType.TYPING) is False
        assert policy.should_persist(EventType.PRESENCE) is False

    def test_exclude_overrides_persist(self) -> None:
        policy = PersistencePolicy(
            persist_types={EventType.MESSAGE, EventType.TYPING},
            exclude_types={EventType.TYPING},
        )
        assert policy.should_persist(EventType.MESSAGE) is True
        assert policy.should_persist(EventType.TYPING) is False


# ---------------------------------------------------------------------------
# InMemoryStore EventFilter integration tests
# ---------------------------------------------------------------------------


class TestEventFilterInMemoryStore:
    async def test_filter_by_event_type(self, store: InMemoryStore) -> None:
        room_id = await _seed_room(store)

        # Add mixed events
        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                type=EventType.MESSAGE,
                content=TextContent(body="Hello"),
            ),
        )
        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                type=EventType.TOOL_CALL_START,
                content=ToolCallContent(tool_name="search", tool_id="c1", arguments={"q": "test"}),
            ),
        )
        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                type=EventType.TOOL_CALL_END,
                content=ToolCallContent(
                    tool_name="search",
                    tool_id="c1",
                    result="found it",
                    status="completed",
                    duration_ms=500,
                ),
            ),
        )
        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                type=EventType.MESSAGE,
                content=TextContent(body="Done"),
            ),
        )

        # Filter messages only
        messages = await store.list_events(
            room_id,
            event_filter=EventFilter(event_types=[EventType.MESSAGE]),
        )
        assert len(messages) == 2
        assert all(e.type == EventType.MESSAGE for e in messages)

        # Filter tool events only
        tools = await store.list_events(
            room_id,
            event_filter=EventFilter(
                event_types=[EventType.TOOL_CALL_START, EventType.TOOL_CALL_END]
            ),
        )
        assert len(tools) == 2

        # Exclude tool events
        no_tools = await store.list_events(
            room_id,
            event_filter=EventFilter(
                exclude_types=[EventType.TOOL_CALL_START, EventType.TOOL_CALL_END]
            ),
        )
        assert len(no_tools) == 2
        assert all(e.type == EventType.MESSAGE for e in no_tools)

    async def test_filter_by_correlation_id(self, store: InMemoryStore) -> None:
        room_id = await _seed_room(store)

        # Two responses with different correlation IDs
        for corr_id, body in [("resp-1", "First"), ("resp-2", "Second")]:
            await store.add_event_auto_index(
                room_id,
                RoomEvent(
                    room_id=room_id,
                    source=_source(),
                    type=EventType.MESSAGE,
                    content=TextContent(body=body),
                    correlation_id=corr_id,
                ),
            )

        results = await store.list_events(
            room_id,
            event_filter=EventFilter(correlation_id="resp-1"),
        )
        assert len(results) == 1
        assert results[0].correlation_id == "resp-1"

    async def test_filter_by_source_channel(self, store: InMemoryStore) -> None:
        room_id = await _seed_room(store)

        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source("ai-1"),
                content=TextContent(body="from ai"),
            ),
        )
        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=EventSource(channel_id="ws-1", channel_type=ChannelType.WEBSOCKET),
                content=TextContent(body="from ws"),
            ),
        )

        ai_events = await store.list_events(
            room_id,
            event_filter=EventFilter(source_channel_id="ai-1"),
        )
        assert len(ai_events) == 1

        ws_events = await store.list_events(
            room_id,
            event_filter=EventFilter(source_channel_type=ChannelType.WEBSOCKET),
        )
        assert len(ws_events) == 1

    async def test_filter_by_time_range(self, store: InMemoryStore) -> None:
        room_id = await _seed_room(store)

        t1 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 1, 11, 0, 0, tzinfo=UTC)
        t3 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

        for t, body in [(t1, "morning"), (t2, "midday"), (t3, "afternoon")]:
            await store.add_event_auto_index(
                room_id,
                RoomEvent(
                    room_id=room_id,
                    source=_source(),
                    content=TextContent(body=body),
                    created_at=t,
                ),
            )

        results = await store.list_events(
            room_id,
            event_filter=EventFilter(
                after_time=t1,
                before_time=t3,
            ),
        )
        assert len(results) == 1
        assert results[0].content.body == "midday"  # type: ignore[union-attr]

    async def test_event_filter_visibility_overrides_param(self, store: InMemoryStore) -> None:
        room_id = await _seed_room(store)

        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                content=TextContent(body="public"),
                visibility="all",
            ),
        )
        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                content=TextContent(body="agents only"),
                visibility="agents",
            ),
        )

        # event_filter.visibility should take precedence over visibility_filter
        results = await store.list_events(
            room_id,
            visibility_filter="all",  # would select 1
            event_filter=EventFilter(visibility="agents"),  # overrides to select the other
        )
        assert len(results) == 1
        assert results[0].content.body == "agents only"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Convenience method tests
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    async def test_get_conversation_returns_messages_only(self, store: InMemoryStore) -> None:
        room_id = await _seed_room(store)

        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                type=EventType.MESSAGE,
                content=TextContent(body="text 1"),
                correlation_id="resp-1",
            ),
        )
        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                type=EventType.TOOL_CALL_START,
                content=ToolCallContent(tool_name="search", tool_id="c1"),
                correlation_id="resp-1",
            ),
        )
        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                type=EventType.MESSAGE,
                content=TextContent(body="text 2"),
                correlation_id="resp-1",
            ),
        )

        convo = await store.get_conversation(room_id)
        assert len(convo) == 2
        assert all(e.type == EventType.MESSAGE for e in convo)

    async def test_get_timeline_returns_all(self, store: InMemoryStore) -> None:
        room_id = await _seed_room(store)

        for event_type, content in [
            (EventType.MESSAGE, TextContent(body="hi")),
            (EventType.TOOL_CALL_START, ToolCallContent(tool_name="t", tool_id="c1")),
            (
                EventType.TOOL_CALL_END,
                ToolCallContent(tool_name="t", tool_id="c1", result="ok", status="completed"),
            ),
            (EventType.MESSAGE, TextContent(body="done")),
        ]:
            await store.add_event_auto_index(
                room_id,
                RoomEvent(
                    room_id=room_id,
                    source=_source(),
                    type=event_type,
                    content=content,
                ),
            )

        timeline = await store.get_timeline(room_id)
        assert len(timeline) == 4
        assert [e.type for e in timeline] == [
            EventType.MESSAGE,
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_END,
            EventType.MESSAGE,
        ]

    async def test_get_timeline_with_filter(self, store: InMemoryStore) -> None:
        room_id = await _seed_room(store)

        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                type=EventType.MESSAGE,
                content=TextContent(body="hi"),
                correlation_id="resp-1",
            ),
        )
        await store.add_event_auto_index(
            room_id,
            RoomEvent(
                room_id=room_id,
                source=_source(),
                type=EventType.MESSAGE,
                content=TextContent(body="bye"),
                correlation_id="resp-2",
            ),
        )

        timeline = await store.get_timeline(
            room_id,
            event_filter=EventFilter(correlation_id="resp-1"),
        )
        assert len(timeline) == 1

"""In-app message threading (flat two-level model).

Covers the framework pieces added for threaded conversations:

* ``EventFilter`` validation of the mutually-exclusive thread filters.
* Store thread queries (``top_level_only`` / ``parent_event_id``) and
  ``get_thread_summaries`` aggregation, on the in-memory store.
* Thread-root normalisation through the locked pipeline (reply-to-a-reply
  collapses to the root; a dangling parent drops to top level).
* AI response inheritance of the thread root, on both the non-streaming and
  streaming generation paths, so ``@`` replies land back in the thread.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.conftest import make_event
from tests.test_framework import SimpleChannel


def _body(event: RoomEvent) -> str | None:
    return getattr(event.content, "body", None)


# ---------------------------------------------------------------------------
# EventFilter validation
# ---------------------------------------------------------------------------


class TestEventFilterThreadValidation:
    def test_top_level_only_and_parent_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            EventFilter(top_level_only=True, parent_event_id="root1")

    def test_top_level_only_alone_is_valid(self) -> None:
        assert EventFilter(top_level_only=True).top_level_only is True

    def test_parent_event_id_alone_is_valid(self) -> None:
        assert EventFilter(parent_event_id="root1").parent_event_id == "root1"


# ---------------------------------------------------------------------------
# Store thread queries + summaries (in-memory store)
# ---------------------------------------------------------------------------


async def _room_with_thread() -> tuple[InMemoryStore, RoomEvent, list[RoomEvent]]:
    """A room with one root, two replies to it, and one standalone message."""
    store = InMemoryStore()
    await store.create_room(Room(id="r1"))
    root = make_event(room_id="r1", body="root")
    await store.add_event(root)
    await store.add_event(make_event(room_id="r1", body="standalone"))
    reply1 = make_event(
        room_id="r1",
        body="reply1",
        parent_event_id=root.id,
        created_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
    )
    reply2 = make_event(
        room_id="r1",
        body="reply2",
        parent_event_id=root.id,
        created_at=datetime(2026, 1, 1, 12, 0, 2, tzinfo=UTC),
    )
    await store.add_event(reply1)
    await store.add_event(reply2)
    return store, root, [reply1, reply2]


class TestStoreThreadQueries:
    async def test_top_level_only_excludes_replies(self) -> None:
        store, _root, _replies = await _room_with_thread()
        events = await store.list_events("r1", event_filter=EventFilter(top_level_only=True))
        assert {_body(e) for e in events} == {"root", "standalone"}

    async def test_parent_event_id_returns_only_that_thread(self) -> None:
        store, root, _replies = await _room_with_thread()
        events = await store.list_events("r1", event_filter=EventFilter(parent_event_id=root.id))
        assert {_body(e) for e in events} == {"reply1", "reply2"}
        assert all(e.parent_event_id == root.id for e in events)

    async def test_thread_summaries_counts_and_last_reply(self) -> None:
        store, root, replies = await _room_with_thread()
        summaries = await store.get_thread_summaries("r1", [root.id])
        assert set(summaries) == {root.id}
        summary = summaries[root.id]
        assert summary.reply_count == 2
        assert summary.last_reply_at == replies[-1].created_at

    async def test_thread_summaries_absent_for_root_without_replies(self) -> None:
        store = InMemoryStore()
        await store.create_room(Room(id="r1"))
        root = make_event(room_id="r1", body="lonely root")
        await store.add_event(root)
        assert await store.get_thread_summaries("r1", [root.id]) == {}

    async def test_parent_event_id_round_trips_through_store(self) -> None:
        store = InMemoryStore()
        await store.create_room(Room(id="r1"))
        root = make_event(room_id="r1", body="root")
        await store.add_event(root)
        reply = make_event(room_id="r1", body="reply", parent_event_id=root.id)
        await store.add_event(reply)
        fetched = await store.get_event(reply.id)
        assert fetched is not None
        assert fetched.parent_event_id == root.id


# ---------------------------------------------------------------------------
# Thread-root normalisation through the locked pipeline
# ---------------------------------------------------------------------------


class _StubTransport(Channel):
    channel_type = ChannelType.WEBSOCKET
    category = ChannelCategory.TRANSPORT

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def _kit_with_room() -> tuple[RoomKit, str]:
    kit = RoomKit()
    kit.register_channel(_StubTransport("ch1"))
    room = await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ch1")
    return kit, room.id


class TestThreadRootNormalization:
    async def test_root_message_has_no_parent(self) -> None:
        kit, room_id = await _kit_with_room()
        root = await kit.send_event(room_id, "ch1", TextContent(body="root"))
        assert root.parent_event_id is None

    async def test_reply_points_at_root(self) -> None:
        kit, room_id = await _kit_with_room()
        root = await kit.send_event(room_id, "ch1", TextContent(body="root"))
        reply = await kit.send_event(
            room_id, "ch1", TextContent(body="reply"), parent_event_id=root.id
        )
        assert reply.parent_event_id == root.id

    async def test_reply_to_a_reply_collapses_to_root(self) -> None:
        kit, room_id = await _kit_with_room()
        root = await kit.send_event(room_id, "ch1", TextContent(body="root"))
        reply = await kit.send_event(
            room_id, "ch1", TextContent(body="reply"), parent_event_id=root.id
        )
        nested = await kit.send_event(
            room_id, "ch1", TextContent(body="nested"), parent_event_id=reply.id
        )
        # Flat two-level: the nested reply threads under the ROOT, not the reply.
        assert nested.parent_event_id == root.id

    async def test_dangling_parent_drops_to_top_level(self) -> None:
        kit, room_id = await _kit_with_room()
        event = await kit.send_event(
            room_id, "ch1", TextContent(body="orphan"), parent_event_id="does-not-exist"
        )
        assert event.parent_event_id is None


# ---------------------------------------------------------------------------
# AI response inheritance of the thread root
# ---------------------------------------------------------------------------


def _ai_binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


class TestAIResponseInheritsThreadNonStreaming:
    async def test_reply_inherits_thread_root(self) -> None:
        ch = AIChannel("ai1", provider=MockAIProvider(responses=["in-thread answer"]))
        ctx = RoomContext(room=Room(id="r1"))
        trigger = make_event(body="hi", channel_id="ws1", parent_event_id="root-42")
        output = await ch.on_event(trigger, _ai_binding(), ctx)
        assert output.response_events
        assert all(e.parent_event_id == "root-42" for e in output.response_events)

    async def test_top_level_trigger_stays_top_level(self) -> None:
        ch = AIChannel("ai1", provider=MockAIProvider(responses=["top-level answer"]))
        ctx = RoomContext(room=Room(id="r1"))
        trigger = make_event(body="hi", channel_id="ws1")
        output = await ch.on_event(trigger, _ai_binding(), ctx)
        assert output.response_events
        assert all(e.parent_event_id is None for e in output.response_events)


class TestInboundCarriesThreadParent:
    async def test_bare_channel_inbound_is_threaded_centrally(self) -> None:
        # Channels build their own RoomEvent in handle_inbound and need NOT copy
        # parent_event_id themselves — the inbound pipeline applies it centrally.
        # So even a channel that ignores it (the bare SimpleChannel here, or the
        # in-app WebSocketChannel, whose handle_inbound drops it) still threads.
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        root = await kit.send_event("r1", "sms1", TextContent(body="root"))
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="u1",
                content=TextContent(body="reply"),
                parent_event_id=root.id,
            )
        )
        assert result.event is not None
        assert result.event.parent_event_id == root.id


class TestAIResponseInheritsThreadIntegration:
    async def _run_reply_turn(self, provider: MockAIProvider) -> tuple[RoomKit, str]:
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(AIChannel("ai1", provider=provider))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        root = await kit.send_event("r1", "sms1", TextContent(body="root"))
        # Attach the AI only after the root exists, so it responds to the
        # threaded follow-up alone — not to the root (which would reply at top
        # level and muddy the assertion).
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="u1",
                content=TextContent(body="follow-up"),
                parent_event_id=root.id,
            )
        )
        return kit, root.id

    async def test_non_streaming_ai_reply_lands_in_thread(self) -> None:
        kit, root_id = await self._run_reply_turn(MockAIProvider(responses=["answer"]))
        ai_events = [e for e in await kit.store.list_events("r1") if e.source.channel_id == "ai1"]
        assert ai_events
        assert all(e.parent_event_id == root_id for e in ai_events)

    async def test_streaming_ai_reply_lands_in_thread(self) -> None:
        provider = MockAIProvider(
            streaming=True,
            ai_responses=[AIResponse(content="streamed answer", finish_reason="stop")],
        )
        kit, root_id = await self._run_reply_turn(provider)
        ai_events = [e for e in await kit.store.list_events("r1") if e.source.channel_id == "ai1"]
        assert ai_events
        assert all(e.parent_event_id == root_id for e in ai_events)

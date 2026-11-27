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

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.core.event_router import CHAIN_DEPTH_LIMIT
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.conftest import make_event
from tests.test_framework import AILikeChannel, SimpleChannel


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
    root = make_event(room_id="r1", body="root", index=0)
    await store.add_event(root)
    await store.add_event(make_event(room_id="r1", body="standalone", index=1))
    reply1 = make_event(
        room_id="r1",
        body="reply1",
        parent_event_id=root.id,
        index=2,
        created_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
    )
    reply2 = make_event(
        room_id="r1",
        body="reply2",
        parent_event_id=root.id,
        index=3,
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
        root = make_event(room_id="r1", body="root", index=0)
        await store.add_event(root)
        reply = make_event(room_id="r1", body="reply", parent_event_id=root.id, index=1)
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


# ---------------------------------------------------------------------------
# Every path that commits an event roots its thread
# ---------------------------------------------------------------------------


class _AnswersInThread(AILikeChannel):
    """An agent whose answers, *count* of them, name *parent* as their
    thread parent."""

    def __init__(self, channel_id: str, parent: str | None, count: int = 1) -> None:
        super().__init__(channel_id, response="answer")
        self.parent = parent
        self.count = count

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        output = await super().on_event(event, binding, context)
        [answer] = output.response_events
        answers = [
            answer.model_copy(update={"parent_event_id": self.parent}) for _ in range(self.count)
        ]
        return output.model_copy(update={"response_events": answers})


async def _threaded_kit(**kwargs: Any) -> tuple[RoomKit, RoomEvent, RoomEvent]:
    """A room holding a thread: its root and a reply to it."""
    kit = RoomKit(**kwargs)
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    root = await kit.send_event("r1", "sms1", TextContent(body="root"))
    reply = await kit.send_event("r1", "sms1", TextContent(body="reply"), parent_event_id=root.id)
    return kit, root, reply


def _ask() -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="ask"))


def _past_the_budget(kit: RoomKit) -> int:
    """How many passes take a cascade past its reentry budget (RFC §8.3)."""
    return kit._max_chain_depth * 10 + 1


async def _answer(kit: RoomKit, parent: str, *, count: int = 1) -> None:
    kit.register_channel(_AnswersInThread("ai1", parent, count))
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await kit.process_inbound(_ask(), room_id="r1")


async def _answer_past_the_budget(kit: RoomKit, parent: str) -> None:
    await _answer(kit, parent, count=_past_the_budget(kit))


async def _inject(kit: RoomKit, parent: str) -> None:
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="inject")
    async def inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if _body(event) != "ask":
            return HookResult.allow()
        injected = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="sms1", channel_type="sms"),
            content=TextContent(body="injected"),
            parent_event_id=parent,
        )
        return HookResult(action="allow", injected_events=[InjectedEvent(event=injected)])

    await kit.process_inbound(_ask(), room_id="r1")


async def _send_past_the_budget(kit: RoomKit, parent: str) -> None:
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="send")
    async def send(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if _body(event) == "ask":
            for _ in range(_past_the_budget(kit)):
                await kit.send_event(
                    "r1", "sms1", TextContent(body="sent"), parent_event_id=parent
                )
        return HookResult.allow()

    await kit.process_inbound(_ask(), room_id="r1")


async def _commit_record(kit: RoomKit, parent: str) -> None:
    record = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="sms1", channel_type="sms"),
        content=TextContent(body="record"),
        parent_event_id=parent,
    )
    await kit.commit_event("r1", record)


def _past_cap(event: RoomEvent) -> bool:
    return event.blocked_by == "reentry_loop_cap"


async def _stored(kit: RoomKit) -> list[RoomEvent]:
    """Every event the room holds, its BLOCKED records included."""
    return await kit.store.list_events(
        "r1", limit=200, event_filter=EventFilter(include_blocked=True)
    )


async def _send(kit: RoomKit, parent: str) -> None:
    await kit.send_event("r1", "sms1", TextContent(body="answered"), parent_event_id=parent)


_PATHS: dict[str, tuple[Any, Any]] = {
    "send_event": (_send, lambda e: _body(e) == "answered"),
    "response": (_answer, lambda e: _body(e) == "answer"),
    "response-past-the-budget": (_answer_past_the_budget, _past_cap),
    "hook-injected-event": (_inject, lambda e: _body(e) == "injected"),
    "send_event-past-the-budget": (_send_past_the_budget, _past_cap),
    "commit_event": (_commit_record, lambda e: _body(e) == "record"),
}


class _HeldStream(MockAIProvider):
    """A streamed answer that waits for *go* once its turn has started."""

    def __init__(self) -> None:
        super().__init__(
            streaming=True, ai_responses=[AIResponse(content="streamed", finish_reason="stop")]
        )
        self.entered = asyncio.Event()
        self.go = asyncio.Event()

    async def _scripted_events(self, context: Any) -> AsyncIterator[Any]:
        self.entered.set()
        await self.go.wait()
        async for event in super()._scripted_events(context):
            yield event


class TestEveryPathRootsItsThread:
    @pytest.mark.parametrize("path", list(_PATHS))
    @pytest.mark.parametrize(
        "names", ["a-reply", "no-event-of-the-room", "a-reply-whose-root-is-gone"]
    )
    async def test_a_parent_is_rooted(self, path: str, names: str) -> None:
        """Whatever path commits it, its BLOCKED record past the reentry
        budget included, an event naming a reply as its parent names that
        reply's thread root, and one whose parent, once rooted, names no event
        of the room (none, or a root a hard delete removed and its reply
        outlived) is posted at the top level (RFC §10.1 step 8)."""
        kit, root, reply = await _threaded_kit(max_chain_depth=2)
        if names == "a-reply-whose-root-is-gone":
            await kit.delete_event("r1", root.id, cascade_replies=False)
        commit, stored_by = _PATHS[path]
        await commit(kit, "does-not-exist" if names == "no-event-of-the-room" else reply.id)

        [stored] = [e for e in await _stored(kit) if stored_by(e)]
        assert stored.parent_event_id == (root.id if names == "a-reply" else None)
        await kit.close()

    async def test_a_streamed_answer_to_a_reply_whose_root_is_gone_is_top_level(self) -> None:
        """The root of a thread is deleted without its replies, and an agent
        streams an answer to one of them: its rows take the reply's parent,
        which the room no longer holds, and are posted at the top level."""
        kit, root, reply = await _threaded_kit()
        provider = MockAIProvider(
            streaming=True, ai_responses=[AIResponse(content="streamed", finish_reason="stop")]
        )
        kit.register_channel(AIChannel("ai1", provider=provider))
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
        await kit.delete_event("r1", root.id, cascade_replies=False)

        await kit.regenerate_response("r1", trigger_id=reply.id)

        rows = [e for e in await _stored(kit) if e.source.channel_id == "ai1"]
        assert rows
        assert all(e.parent_event_id is None for e in rows)
        await kit.close()

    async def test_a_streamed_answer_whose_root_is_deleted_mid_stream_is_top_level(self) -> None:
        """A thread's root is hard-deleted while an agent streams its answer
        to a reply: each row is rooted at its commit, and one committed after
        the delete names no root the room no longer holds (RFC §10.1 step 8)."""
        kit, root, reply = await _threaded_kit()
        provider = _HeldStream()
        kit.register_channel(AIChannel("ai1", provider=provider))
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
        ask = InboundMessage(
            channel_id="sms1",
            sender_id="u1",
            content=TextContent(body="ask"),
            parent_event_id=reply.id,
        )
        turn = asyncio.create_task(kit.process_inbound(ask, room_id="r1"))
        await asyncio.wait_for(provider.entered.wait(), 5)
        await kit.delete_event("r1", root.id)
        provider.go.set()
        await asyncio.wait_for(turn, 5)

        rows = [e for e in await _stored(kit) if e.source.channel_id == "ai1"]
        assert rows
        assert [e.parent_event_id for e in rows] == [None] * len(rows)
        await kit.close()

    async def test_a_depth_limit_record_of_an_answer_whose_root_is_gone_is_top_level(
        self,
    ) -> None:
        """Two agents answer a reply; a hook deletes the thread's root, its
        replies kept, as one answer commits. The record the depth limit
        leaves for the other agent, not asked about that answer, takes the
        answer's parent, the removed root, and is posted at the top level
        (RFC §10.1 step 8, §8.3)."""
        kit, root, reply = await _threaded_kit(max_chain_depth=2)
        for agent in ("a1", "a2"):
            kit.register_channel(AIChannel(agent, provider=MockAIProvider(responses=["ok"] * 4)))
            await kit.attach_channel("r1", agent, category=ChannelCategory.INTELLIGENCE)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="forget")
        async def forget(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_id == "a1" and await kit.store.get_event(root.id):
                await kit.delete_event("r1", root.id, cascade_replies=False)
            return HookResult.allow()

        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="u1",
                content=TextContent(body="ask"),
                parent_event_id=reply.id,
            ),
            room_id="r1",
        )

        stored = await _stored(kit)
        [answer] = [e for e in stored if e.source.channel_id == "a1" and e.blocked_by is None]
        [record] = [e for e in stored if e.responds_to == answer.id]
        assert (record.blocked_by, record.parent_event_id) == (CHAIN_DEPTH_LIMIT, None)
        await kit.close()

    async def test_a_hard_delete_of_the_root_removes_the_whole_thread(self) -> None:
        """An agent's answer, a hook's injected event and a host's record,
        each naming a reply, are replies of the root: deleting the root
        removes them with it (RFC §10.3)."""
        kit, root, reply = await _threaded_kit()
        for commit in (_answer, _inject, _commit_record):
            await commit(kit, reply.id)

        await kit.delete_event("r1", root.id)

        left = {_body(e) for e in await _stored(kit)}
        assert left & {"root", "reply", "answer", "injected", "record"} == set()
        await kit.close()

    async def test_an_injected_events_hooks_read_it_rooted(self) -> None:
        """A hook injects an event naming a reply: the hooks of its own pass
        read it rooted, as the event and as the newest of their context
        (RFC §10.1 step 8, §9.4)."""
        kit, root, reply = await _threaded_kit()
        seen: list[tuple[str | None, str | None]] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="read")
        async def read(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) == "injected":
                seen.append((event.parent_event_id, ctx.recent_events[-1].parent_event_id))
            return HookResult.allow()

        await _inject(kit, reply.id)

        assert seen == [(root.id, root.id)]
        await kit.close()

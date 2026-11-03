"""A response meets the same gates on every path that commits it (RMK-344).

RFC §10.1: an agent's response re-enters the locked section (step 14), so its
BEFORE_BROADCAST hooks run (step 9) before its source's right to write is
checked (step 11). A muted or read-only agent's answer is stored BLOCKED with
what its hooks decided kept (§7.5 rules 2 and 3), a hook that blocks it names
the block, and a regenerated answer re-enters like any other: hooks, reentry
budget, and the caller's ``response_events``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import Access, ChannelCategory, ChannelType, EventStatus, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult
from roomkit.models.store_filter import EventFilter
from roomkit.models.task import Task
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.buffered_agent import BufferedAgent
from tests.test_framework import SimpleChannel


class _StreamingTransport(SimpleChannel):
    """A transport that takes the live text of a streamed answer."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.live: list[object] = []

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[object],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        async for chunk in text_stream:
            self.live.append(chunk)
        return ChannelOutput.empty()


class _Chatty(Channel):
    """An agent that answers with *count* messages at once."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str, count: int) -> None:
        super().__init__(channel_id)
        self._count = count

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(
            responded=True,
            response_events=[
                RoomEvent(
                    room_id=event.room_id,
                    source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
                    content=TextContent(body=f"part {n}"),
                    chain_depth=event.chain_depth + 1,
                )
                for n in range(self._count)
            ],
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


def _body(event: RoomEvent) -> str:
    return getattr(event.content, "body", "")


def _user_msg() -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="hi"))


class _Room:
    """A room with a streaming transport, one agent, and a BEFORE_BROADCAST
    hook that records each event of the *watched* channel and files a task
    for it; asked to, it blocks the event, or mutes the event's source and
    lets it through."""

    def __init__(
        self,
        *,
        watch: str = "ai1",
        block: bool = False,
        mutes: bool = False,
        max_chain_depth: int = 5,
    ) -> None:
        self.kit = RoomKit(max_chain_depth=max_chain_depth)
        self.transport = _StreamingTransport("sms1")
        self.seen: list[str] = []
        self.blocked: list[FrameworkEvent] = []
        self.kit.register_channel(self.transport)

        @self.kit.hook(HookTrigger.BEFORE_BROADCAST, name="moderation")
        async def moderation(event: RoomEvent, context: RoomContext) -> HookResult:
            if event.source.channel_id != watch:
                return HookResult.allow()
            self.seen.append(_body(event))
            task = Task(id=f"t{len(self.seen)}", room_id=event.room_id, title=_body(event))
            if block:
                return HookResult.block("moderated", tasks=[task])
            if mutes:
                await self.kit.mute(event.room_id, event.source.channel_id)
            return HookResult(action="allow", tasks=[task])

        @self.kit.on("event_blocked")
        async def on_blocked(event: FrameworkEvent) -> None:
            self.blocked.append(event)

    async def attach(
        self,
        agent: Channel,
        *,
        access: Access = Access.READ_WRITE,
        muted: bool = False,
        transport_access: Access = Access.READ_WRITE,
    ) -> None:
        self.kit.register_channel(agent)
        await self.kit.create_room(room_id="r1")
        await self.kit.attach_channel("r1", "sms1", access=transport_access)
        await self.kit.attach_channel(
            "r1",
            agent.channel_id,
            category=ChannelCategory.INTELLIGENCE,
            access=access,
            muted=muted,
        )

    async def rows(self, channel_id: str = "ai1") -> list[tuple[str, EventStatus, str | None]]:
        events = await self.kit.store.list_events(
            "r1", event_filter=EventFilter(include_blocked=True)
        )
        return [
            (_body(e), e.status, e.blocked_by) for e in events if e.source.channel_id == channel_id
        ]

    async def task_titles(self) -> list[str]:
        return [t.title for t in await self.kit.store.list_tasks("r1")]


def _agent(streaming: bool, *responses: str) -> Channel:
    """The answering agent: an ``AIChannel`` streams its answer, and an agent
    that answers at once drives the framework's buffered path."""
    if streaming:
        return AIChannel("ai1", provider=MockAIProvider(responses=list(responses), streaming=True))
    return BufferedAgent("ai1", *responses)


@pytest.mark.parametrize(
    ("access", "muted", "streaming", "blocked_by"),
    [
        (Access.READ_ONLY, False, False, "source_read_only"),
        (Access.READ_ONLY, False, True, "source_read_only"),
        (Access.READ_WRITE, True, False, "source_muted"),
    ],
    ids=["read-only", "read-only-streamed", "muted"],
)
async def test_a_non_writable_agent_answer_meets_its_hooks_first(
    access: Access, muted: bool, streaming: bool, blocked_by: str
) -> None:
    room = _Room()
    await room.attach(_agent(streaming, "A1"), access=access, muted=muted)

    await room.kit.process_inbound(_user_msg(), room_id="r1")

    # The hook saw the answer and its task is kept (§7.5 rule 3); the answer
    # is stored BLOCKED and reaches no channel, live or committed.
    assert room.seen == ["A1"]
    assert await room.rows() == [("A1", EventStatus.BLOCKED, blocked_by)]
    assert await room.task_titles() == ["A1"]
    assert [e for e in room.transport.delivered if e.source.channel_id == "ai1"] == []
    assert room.transport.live == []


async def test_a_writable_agent_stream_is_piped_live() -> None:
    """The control for the case above: the same transport takes a writable
    agent's streamed answer live."""
    room = _Room()
    await room.attach(_agent(True, "A1"))

    await room.kit.process_inbound(_user_msg(), room_id="r1")

    assert "".join(chunk for chunk in room.transport.live if isinstance(chunk, str)) == "A1"
    assert await room.rows() == [("A1", EventStatus.DELIVERED, None)]


async def test_a_muted_agent_stream_is_closed_unread() -> None:
    """RFC §7.5 rule 2: a muted source's stream MAY be closed unread, the
    reply never generated; nothing is stored and no hook sees it."""
    room = _Room()
    provider = MockAIProvider(responses=["A1"], streaming=True)
    await room.attach(AIChannel("ai1", provider=provider), muted=True)

    await room.kit.process_inbound(_user_msg(), room_id="r1")

    assert (room.seen, await room.rows(), provider.calls) == ([], [], [])


async def test_a_hook_that_blocks_an_inbound_names_the_block_before_the_right_to_write() -> None:
    """Step 9 runs before step 11: a source that cannot write and that a
    hook blocks is recorded as the hook's block."""
    room = _Room(watch="sms1", block=True)
    await room.attach(_agent(False, "A1"), transport_access=Access.READ_ONLY)

    await room.kit.process_inbound(_user_msg(), room_id="r1")

    assert await room.rows("sms1") == [("hi", EventStatus.BLOCKED, "moderation")]
    assert await room.task_titles() == ["hi"]


async def test_a_hook_that_blocks_an_answer_names_the_block_before_the_right_to_write(
    streaming: bool,
) -> None:
    room = _Room(block=True)
    await room.attach(_agent(streaming, "A1"), access=Access.READ_ONLY)

    await room.kit.process_inbound(_user_msg(), room_id="r1")

    assert await room.rows() == [("A1", EventStatus.BLOCKED, "moderation")]
    assert await room.task_titles() == ["A1"]


@pytest.mark.parametrize("path", ["inbound", "buffered", "streamed"])
async def test_a_hook_that_mutes_the_source_blocks_the_event_it_reads(path: str) -> None:
    """The right to write is read as the hooks left it: a moderation hook
    that mutes the source of the event it reads has that event stored
    BLOCKED ``source_muted``, not only the ones after it."""
    if path == "inbound":
        room = _Room(watch="sms1", mutes=True)
        await room.attach(_agent(False, "A1"))
        await room.kit.process_inbound(_user_msg(), room_id="r1")
        assert await room.rows("sms1") == [("hi", EventStatus.BLOCKED, "source_muted")]
        return

    room = _Room(mutes=True)
    await room.attach(_agent(path == "streamed", "A1"))

    await room.kit.process_inbound(_user_msg(), room_id="r1")

    assert await room.rows() == [("A1", EventStatus.BLOCKED, "source_muted")]
    assert [e for e in room.transport.delivered if e.source.channel_id == "ai1"] == []


@pytest.mark.parametrize(
    ("access", "muted", "streaming", "blocked_by"),
    [
        (Access.READ_ONLY, False, False, "source_read_only"),
        (Access.READ_ONLY, False, True, "source_read_only"),
        (Access.READ_WRITE, True, False, "source_muted"),
    ],
    ids=["read-only", "read-only-streamed", "muted"],
)
async def test_event_blocked_names_the_answer_source(
    access: Access, muted: bool, streaming: bool, blocked_by: str
) -> None:
    room = _Room()
    await room.attach(_agent(streaming, "A1"), access=access, muted=muted)

    await room.kit.process_inbound(_user_msg(), room_id="r1")

    assert [(e.channel_id, e.data["blocked_by"]) for e in room.blocked] == [("ai1", blocked_by)]


async def test_event_blocked_names_an_inbound_source() -> None:
    room = _Room(watch="sms1", block=True)
    await room.attach(_agent(False, "A1"))

    await room.kit.process_inbound(_user_msg(), room_id="r1")

    assert [(e.channel_id, e.data["blocked_by"]) for e in room.blocked] == [("sms1", "moderation")]


async def test_a_hook_error_on_an_answer_is_announced(streaming: bool) -> None:
    """A response's hook errors reach the framework as an inbound event's do."""
    kit = RoomKit()
    errors: list[FrameworkEvent] = []
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(_agent(streaming, "A1"))

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="flaky")
    async def flaky(event: RoomEvent, context: RoomContext) -> HookResult:
        if event.source.channel_id == "ai1":
            raise RuntimeError("hook down")
        return HookResult.allow()

    @kit.on("hook_error")
    async def on_error(event: FrameworkEvent) -> None:
        errors.append(event)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(_user_msg(), room_id="r1")

    assert len(errors) == 1


async def test_a_regenerated_answer_meets_its_hooks(streaming: bool) -> None:
    room = _Room()
    await room.attach(_agent(streaming, "A1", "A2"))
    await room.kit.process_inbound(_user_msg(), room_id="r1")

    result = await room.kit.regenerate_response("r1")

    assert result is not None
    assert room.seen == ["A1", "A2"]
    assert await room.task_titles() == ["A1", "A2"]
    assert [_body(e) for e in result.response_events] == ["A2"]


@pytest.mark.parametrize(
    ("access", "muted", "streaming", "blocked_by"),
    [
        (Access.READ_ONLY, False, False, "source_read_only"),
        (Access.READ_ONLY, False, True, "source_read_only"),
        (Access.READ_WRITE, True, False, "source_muted"),
    ],
    ids=["read-only", "read-only-streamed", "muted"],
)
async def test_a_non_writable_agent_regenerated_answer_is_blocked(
    access: Access, muted: bool, streaming: bool, blocked_by: str
) -> None:
    room = _Room()
    await room.attach(_agent(streaming, "A1", "A2"), access=access, muted=muted)
    await room.kit.process_inbound(_user_msg(), room_id="r1")

    result = await room.kit.regenerate_response("r1")

    assert result is not None and result.response_events == []
    assert room.seen == ["A1", "A2"]
    assert await room.rows() == [
        ("A1", EventStatus.BLOCKED, blocked_by),
        ("A2", EventStatus.BLOCKED, blocked_by),
    ]


async def test_a_regeneration_spends_the_reentry_budget() -> None:
    """The budget is ``max_chain_depth * 10`` responses per cascade: the
    answer past it is stored BLOCKED ``reentry_loop_cap``, a regenerated one
    included (RFC §8.3)."""
    room = _Room(max_chain_depth=2)
    await room.attach(_Chatty("ai1", count=21))
    await room.kit.process_inbound(_user_msg(), room_id="r1")

    await room.kit.regenerate_response("r1")

    capped = [row for row in await room.rows() if row[2] == "reentry_loop_cap"]
    assert len(capped) == 2


async def test_a_muted_agent_answers_count_against_the_reentry_budget() -> None:
    """Every response that re-enters counts (RFC §8.3), a muted source's
    included: past the budget its answer is capped, not stored muted."""
    room = _Room(max_chain_depth=2)
    await room.attach(_Chatty("ai1", count=21), muted=True)

    await room.kit.process_inbound(_user_msg(), room_id="r1")

    reasons = [row[2] for row in await room.rows()]
    assert reasons == ["source_muted"] * 20 + ["reentry_loop_cap"]


# -- A binding changed during a stream holds from its next row -------------

_LOOKUP = AITool(name="lookup", description="Look up", parameters={"type": "object"})


def _live(transport: _StreamingTransport) -> list[str]:
    """What the streaming transport read: each text delta, then each row by type."""
    return [
        chunk if isinstance(chunk, str) else chunk.type.value
        for chunk in transport.live
        if isinstance(chunk, str | RoomEvent)
    ]


@pytest.mark.parametrize(
    ("change", "live", "web"),
    [
        (
            lambda kit: kit.mute("r1", "ai1"),
            ["Let me check.", "message", "tool_call_start"],
            ["message", "tool_call_start"],
        ),
        (
            lambda kit: kit.set_access("r1", "ws", Access.WRITE_ONLY),
            ["Let me check.", "message", "tool_call_start"],
            ["message", "tool_call_start", "tool_call_end", "message"],
        ),
        (
            lambda kit: kit.set_access("r1", "web", Access.WRITE_ONLY),
            ["Let me check.", "message", "tool_call_start", "tool_call_end", "Done.", "message"],
            ["message", "tool_call_start"],
        ),
    ],
    ids=["agent-muted", "target-made-write-only", "other-made-write-only"],
)
async def test_a_binding_change_during_a_stream_holds_from_its_next_row(
    change: Any, live: list[str], web: list[str]
) -> None:
    """RFC §7.2, §7.5, §10.2: each streamed row is planned with the bindings
    read at its commit, and once the room no longer lets the streaming
    target read the response, the target reads nothing more of it."""
    kit = RoomKit()
    ws, other = _StreamingTransport("ws"), SimpleChannel("web")

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        await change(kit)
        return "found"

    call = AIToolCall(id="c1", name="lookup", arguments={})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="Let me check.", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="Done."),
        ],
        streaming=True,
    )
    agent = AIChannel("ai1", provider=provider, tool_handler=lookup, tools=[_LOOKUP])
    for channel in (ws, other, agent):
        kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "web")
    await kit.attach_channel("r1", "ws")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    message = InboundMessage(channel_id="web", sender_id="u1", content=TextContent(body="hi"))
    await kit.process_inbound(message, room_id="r1")

    assert _live(ws) == live
    assert [e.type.value for e in other.delivered if e.source.channel_id == "ai1"] == web
    await kit.close()

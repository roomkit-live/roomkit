"""Past the chain-depth limit no agent is asked, streamed or not (RMK-283, RMK-287).

RFC §8.3: when a response would reach ``max_chain_depth`` the
agent's ``on_event`` is not called, so no model is called and no tool runs.
One record stands in for the response: stored BLOCKED with ``blocked_by =
"event_chain_depth_limit"``, observed, and announced as
``chain_depth_exceeded``.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventStatus, EventType
from roomkit.models.event import EditContent, EventContent, TextContent, ToolCallContent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

_ANSWERS = [
    AIResponse(
        content="Calling.",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="t1", name="lookup", arguments={})],
    ),
    AIResponse(content="Answer.", finish_reason="stop"),
]


class _Voice(SimpleChannel):
    """A streaming transport that records every chunk it is handed live."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.live: list[Any] = []

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(self, text_stream, event, binding, context):  # type: ignore[no-untyped-def]
        async for chunk in text_stream:
            self.live.append(chunk)
        return ChannelOutput.empty()


async def _turn(
    streaming: bool, max_chain_depth: int, then: Any = None
) -> tuple[RoomKit, SimpleChannel, _Voice, list[str], MockAIProvider]:
    kit = RoomKit(max_chain_depth=max_chain_depth)
    sms, voice = SimpleChannel("sms1"), _Voice("voice1")
    ran: list[str] = []
    provider = MockAIProvider(streaming=streaming, ai_responses=list(_ANSWERS))

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        if then is not None:
            await then(kit)
        return "ok"

    kit.register_channel(sms)
    kit.register_channel(voice)
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=provider,
            tool_handler=lookup,
            tools=[AITool(name="lookup", description="d")],
        )
    )
    await kit.create_room(room_id="r1")
    for channel_id in ("sms1", "voice1"):
        await kit.attach_channel("r1", channel_id)
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return kit, sms, voice, ran, provider


async def _say(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )


async def _ai_rows(kit: RoomKit) -> list[Any]:
    events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
    return [e for e in events if e.source.channel_id == "ai1"]


async def test_at_the_limit_the_agent_is_not_asked(streaming: bool) -> None:
    kit, sms, voice, ran, provider = await _turn(streaming, max_chain_depth=1)
    announced: list[Any] = []

    @kit.on("chain_depth_exceeded")
    async def on_exceeded(event: Any) -> None:
        announced.append(event)

    await _say(kit)

    # No model call and no tool, in either mode.
    assert provider.calls == []
    assert ran == []
    # One record stands in for the response.
    [row] = await _ai_rows(kit)
    assert (row.type, row.status, row.blocked_by, row.chain_depth) == (
        EventType.MESSAGE,
        EventStatus.BLOCKED,
        "event_chain_depth_limit",
        1,
    )
    assert row.content.body == ""
    assert [(e.event_id, e.data) for e in announced] == [
        (row.id, {"chain_depth": 1, "max_chain_depth": 1})
    ]
    observations = await kit.store.list_observations("r1")
    assert [o.id for o in observations] == [f"obs_{row.id}"]
    assert sms.delivered == []
    assert voice.live == []


async def test_the_limit_holds_at_its_default(streaming: bool) -> None:
    """A trigger at depth 3 is answered at 4 and delivered; one at 4 would be
    answered at 5, the default limit, and the agent is not asked."""
    for trigger_depth, blocked in ((3, False), (4, True)):
        kit, _, _, ran, _ = await _turn(streaming, max_chain_depth=5)

        await kit.send_event("r1", "sms1", TextContent(body="go"), chain_depth=trigger_depth)

        rows = await _ai_rows(kit)
        assert rows
        assert {row.chain_depth for row in rows} == {trigger_depth + 1}
        assert all((row.status == EventStatus.BLOCKED) is blocked for row in rows)
        assert (ran == []) is blocked
        assert len(rows) == 1 if blocked else len(rows) > 1


@pytest.mark.parametrize("kind", [EventType.TOOL_CALL_START, EventType.SYSTEM, EventType.EDIT])
async def test_what_opens_no_turn_at_the_limit_leaves_no_record(
    streaming: bool, kind: EventType
) -> None:
    """No agent answers a tool-call row, a notice nor an edit (RFC §6.4,
    RMK-1257), so none is recorded as not asked: an edit's target, a
    message, leaves the one record."""
    kit, _, _, _, _ = await _turn(streaming, max_chain_depth=1)
    records = 0
    if kind == EventType.TOOL_CALL_START:
        content: EventContent = ToolCallContent(
            tool_name="lookup", tool_id="t9", arguments={}, status="pending"
        )
    elif kind == EventType.SYSTEM:
        content = TextContent(body="The order shipped")
    else:
        said = await kit.send_event("r1", "sms1", TextContent(body="Oops"), participant_id="alice")
        content = EditContent(target_event_id=said.id, new_content=TextContent(body="Oops, fixed"))
        records = 1

    await kit.send_event("r1", "sms1", content, event_type=kind, participant_id="alice")

    assert len(await _ai_rows(kit)) == records
    assert len(await kit.store.list_observations("r1")) == records


async def test_a_room_closed_mid_stream_takes_no_further_row(streaming: bool) -> None:
    """Below the limit too, a closed room refuses writes (RFC §5.1)."""

    async def close_the_room(kit: RoomKit) -> None:
        await kit.close_room("r1")

    kit, _, _, _, _ = await _turn(streaming, max_chain_depth=5, then=close_the_room)

    await _say(kit)

    kinds = [row.type for row in await _ai_rows(kit)]
    assert EventType.TOOL_CALL_END not in kinds
    assert kinds.count(EventType.MESSAGE) <= 1


async def test_below_the_limit_nothing_is_blocked(streaming: bool) -> None:
    kit, sms, _, _, _ = await _turn(streaming, max_chain_depth=5)

    await _say(kit)

    rows = await _ai_rows(kit)
    assert rows
    assert all(row.status != EventStatus.BLOCKED for row in rows)
    assert all(row.chain_depth == 1 for row in rows)
    assert sms.delivered

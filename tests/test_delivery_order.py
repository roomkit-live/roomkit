"""Per-room delivery order across the paths that commit their own events.

RFC §10.2: delivery sets execute in index order, one event's set completing
before the next event's begins. The delivery cursor (``Room.delivered_index``)
is what enforces it, so it must never vouch for an event that has not been
delivered — an event committed here and broadcast where the caller stands
would publish its index at commit time and release the lane to run the *next*
one while it is still undelivered.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import AIChannel, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.base import Channel
from roomkit.core.lanes import DeliveryCascade, DeliveryPlan, ExecEntry
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse

ROOM = "r-order"


class OrderChannel(Channel):
    """Transport that records delivery order and can stall on one body."""

    channel_type = ChannelType.SMS

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.order: list[str] = []
        self.gate: asyncio.Event | None = None
        self.gate_for: str | None = None

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        body = getattr(event.content, "body", "")
        if self.gate is not None and body == self.gate_for:
            await self.gate.wait()
        self.order.append(body)
        return ChannelOutput.empty()


async def _kit_with_agent() -> tuple[RoomKit, OrderChannel]:
    kit = RoomKit()
    transport = OrderChannel("t")
    kit.register_channel(transport)
    kit.register_channel(Agent("ai", greeting="greeting"))
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "t", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, transport


async def _cursor(kit: RoomKit) -> tuple[int, int]:
    room = await kit._store.get_room(ROOM)
    assert room is not None
    return room.delivered_index, room.latest_index


@pytest.mark.asyncio
async def test_committed_event_stays_off_the_cursor_until_it_is_delivered() -> None:
    kit, transport = await _kit_with_agent()
    transport.gate = asyncio.Event()
    transport.gate_for = "greeting"

    greet = asyncio.create_task(kit.send_greeting(ROOM, agent_id="ai"))
    await asyncio.sleep(0.1)

    delivered, latest = await _cursor(kit)
    assert transport.order == []  # still stalled in deliver()
    assert delivered < latest  # ...and the cursor says so

    transport.gate.set()
    await asyncio.wait_for(greet, timeout=3.0)
    assert transport.order == ["greeting"]
    delivered, latest = await _cursor(kit)
    assert delivered == latest  # the cursor advances only now
    await kit.close()


@pytest.mark.asyncio
async def test_a_later_index_does_not_overtake_a_stalled_inline_delivery() -> None:
    kit, transport = await _kit_with_agent()
    transport.gate = asyncio.Event()
    transport.gate_for = "greeting"
    binding = await kit._store.get_binding(ROOM, "t")
    assert binding is not None

    greet = asyncio.create_task(kit.send_greeting(ROOM, agent_id="ai"))
    await asyncio.sleep(0.1)
    delivered, latest = await _cursor(kit)

    # Whatever the room commits next — a concurrent turn, another worker's
    # plan — its delivery set may not run before the greeting's.
    later = RoomEvent(
        room_id=ROOM,
        source=EventSource(channel_id="ai", channel_type=ChannelType.AI),
        content=TextContent(body="later"),
    )
    cascade = DeliveryCascade(ROOM, reentry_budget=5)
    cascade.retain()
    kit._lanes.enqueue(
        ROOM,
        ExecEntry(
            plan=DeliveryPlan(
                event=later,
                source_binding=None,
                context=await kit._build_context(ROOM),
                targets=[binding],
                injected=True,
                fire_after_broadcast=False,
            ),
            cascade=cascade,
            index=latest + 1,
        ),
    )
    await asyncio.sleep(0.1)
    assert transport.order == []  # the lane is held behind the greeting

    transport.gate.set()
    await asyncio.wait_for(greet, timeout=3.0)
    await asyncio.wait_for(cascade.wait(), timeout=3.0)
    assert transport.order == ["greeting", "later"]
    await kit.close()


class _SegmentedAI(AIProvider):
    """Streams two text segments split by a tool call."""

    @property
    def model_name(self) -> str:
        return "mock-segmented"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:  # pragma: no cover
        return AIResponse(content="unused")

    async def generate_stream(self, context: AIContext) -> AsyncIterator[Any]:
        yield "first "
        yield "half"
        yield ToolCallStartMarker(tool_name="lookup", tool_id="t1", arguments={})
        yield ToolCallEndMarker(
            tool_name="lookup", tool_id="t1", arguments={}, result="ok", status="completed"
        )
        yield "second half"


@pytest.mark.asyncio
async def test_a_streamed_segment_holds_the_cursor_until_it_is_delivered() -> None:
    """The sharp case: a stream commits several indexes in a row.

    With the segments broadcast in one batch after the stream, every one of
    them was on the cursor — declared delivered — while the batch had not
    started. Stalling the first segment's transport delivery makes that
    visible: the cursor must stay behind it, and the segments must reach the
    transport in index order.
    """
    kit = RoomKit()
    transport = OrderChannel("t")
    transport.gate = asyncio.Event()
    transport.gate_for = "first half"
    kit.register_channel(transport)
    kit.register_channel(AIChannel("ai", provider=_SegmentedAI()))
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "t", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, "ai", category=ChannelCategory.INTELLIGENCE)

    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="t", sender_id="u1", content=TextContent(body="go"))
        )
    )
    await asyncio.sleep(0.2)  # the stream runs to its end; delivery is stalled

    delivered, latest = await _cursor(kit)
    assert "first half" not in transport.order
    assert delivered < latest, "the cursor vouched for segments nobody has received"

    transport.gate.set()
    await asyncio.wait_for(turn, timeout=3.0)

    assert [b for b in transport.order if b] == ["first half", "second half"]
    events = await kit._store.list_events(ROOM)
    indexes = {
        getattr(e.content, "body", ""): e.index for e in events if getattr(e.content, "body", "")
    }
    assert indexes["first half"] < indexes["second half"]

    delivered, latest = await _cursor(kit)
    assert delivered == latest, "the turn returned with an index still undelivered"
    await kit.close()

"""RFC §14.3 — the room counters never diverge from the timeline.

Every event that enters a room's timeline — the trigger message, each AI
reentry/tool response, chain-depth-blocked events, and hook-injected events —
commits atomically (index + status + counters in one store transaction). These
end-to-end tests assert the invariant the review flagged: an observer must never
see a stored event the counters do not reflect, counters that count an absent
event, or a ``latest_index`` pointing at an event that was never stored.
"""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventStatus, EventType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.store_filter import PersistencePolicy
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


async def _assert_counters_match_timeline(kit: RoomKit, room_id: str) -> list[RoomEvent]:
    events = await kit.store.list_events(room_id, limit=200)
    room = await kit.store.get_room(room_id)
    assert room is not None
    assert room.event_count == len(events), "event_count must equal the stored event count"
    assert room.latest_index == max(e.index for e in events), (
        "latest_index must equal the highest stored index, never a phantom"
    )
    # Indices are dense and unique — no gap, no duplicate.
    assert sorted(e.index for e in events) == list(range(len(events)))
    return events


async def test_ai_reentry_committed_delivered_and_counters_consistent() -> None:
    """An AI response (a reentry) is committed DELIVERED — not left PENDING —
    and the room counters reflect it (RFC §10.1 step 13). Reproduces the review
    case: event_count=4, latest_index=2, PENDING reentry at index 3."""
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws1"))
    kit.register_channel(AIChannel("ai1", provider=MockAIProvider(responses=["hi there"])))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ws1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body="hello"))
    )

    events = await _assert_counters_match_timeline(kit, "r1")
    ai_events = [e for e in events if e.source.channel_id == "ai1"]
    assert ai_events, "the AI response must be persisted"
    assert all(e.status is EventStatus.DELIVERED for e in ai_events), "reentry must be DELIVERED"
    await kit.close()


async def test_chain_depth_blocked_events_keep_counters_consistent() -> None:
    """Chain-depth-blocked reentries are committed (indexed + counted) so the
    counters stay consistent with a timeline that contains BLOCKED events too."""
    kit = RoomKit(max_chain_depth=2)
    kit.register_channel(WebSocketChannel("ws1"))
    kit.register_channel(AIChannel("ai1", provider=MockAIProvider(responses=["a"] * 10)))
    kit.register_channel(AIChannel("ai2", provider=MockAIProvider(responses=["b"] * 10)))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ws1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r1", "ai2", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body="go"))
    )

    events = await _assert_counters_match_timeline(kit, "r1")
    assert any(e.status is EventStatus.BLOCKED for e in events), "a chain-depth block is expected"
    await kit.close()


async def test_policy_excluded_event_does_not_advance_latest_index() -> None:
    """A PersistencePolicy that excludes an event type must not leave the room
    with a latest_index pointing at the unstored event (the event is delivered
    but never persisted, so it consumes no index)."""
    kit = RoomKit(persistence_policy=PersistencePolicy(exclude_types={EventType.MESSAGE}))
    kit.register_channel(SimpleChannel("src"))
    kit.register_channel(SimpleChannel("dst"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src")
    await kit.attach_channel("r1", "dst")

    result = await kit.process_inbound(
        InboundMessage(channel_id="src", sender_id="u1", content=TextContent(body="hi"))
    )
    assert not result.blocked  # delivered, just not persisted

    events = await _assert_counters_match_timeline(kit, "r1")
    assert all(e.type is not EventType.MESSAGE for e in events), "the message must not be stored"
    await kit.close()


async def test_streaming_segments_committed_delivered_and_consistent() -> None:
    """A streamed AI response persists its segment DELIVERED (not PENDING) via
    the atomic commit, keeping the counters consistent (the streaming path used
    to bypass commit_event)."""
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws1"))
    kit.register_channel(
        AIChannel("ai1", provider=MockAIProvider(responses=["streamed reply"], streaming=True))
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ws1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="ws1", sender_id="u1", content=TextContent(body="hello"))
    )

    events = await _assert_counters_match_timeline(kit, "r1")
    ai_messages = [
        e for e in events if e.source.channel_id == "ai1" and e.type is EventType.MESSAGE
    ]
    assert ai_messages, "the streamed AI message must be persisted"
    assert all(e.status is EventStatus.DELIVERED for e in ai_messages)
    await kit.close()

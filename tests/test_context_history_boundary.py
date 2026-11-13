"""A bounded read records what it omitted before visibility removes any rows."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from roomkit import Room, RoomContext, RoomKit
from roomkit.core.mixins.inbound_locked import committed_in
from roomkit.core.mixins.lane_execution import provisional
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel


class CountingStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.limits: list[int] = []

    async def get_conversation(self, room_id: str, **kwargs: Any) -> list[RoomEvent]:
        self.limits.append(kwargs.get("limit", 50))
        return await super().get_conversation(room_id, **kwargs)


async def _seed(kit: RoomKit, count: int) -> None:
    await kit.create_room(room_id="room")
    for index in range(count):
        await kit.store.add_event_auto_index(
            "room",
            RoomEvent(
                room_id="room",
                source=EventSource(channel_id="source", channel_type=ChannelType.SMS),
                content=TextContent(body=f"m{index}"),
            ),
        )


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4])
async def test_a_single_bounded_read_distinguishes_exactly_full_from_truncated(limit: int) -> None:
    store = CountingStore()
    async with RoomKit(store=store) as kit:
        await _seed(kit, 3)
        context = await kit._build_context("room", recent_limit=limit)
        assert [event.content.body for event in context.recent_events] == (
            [f"m{i}" for i in range(max(0, 3 - limit), 3)] if limit else []
        )
        assert context.history_start_index == (max(0, 3 - limit) if limit else None)
        assert store.limits == ([limit + 1] if limit else [])


async def test_a_carried_tail_keeps_or_advances_its_boundary_without_a_second_read() -> None:
    class Reader(SimpleChannel):
        @property
        def recent_events_window(self) -> int:
            return 3

    store = CountingStore()
    async with RoomKit(store=store) as kit:
        await _seed(kit, 3)
        kit.register_channel(Reader("reader"))
        await kit.attach_channel("room", "reader")
        full = await kit._build_context("room")
        store.limits.clear()
        same = await kit._build_context("room", carrying=full)
        narrow = await kit._build_context("room", recent_limit=1, carrying=full)
        assert same.history_start_index == 0
        assert narrow.history_start_index == 2
        assert store.limits == []


def test_the_optional_boundary_survives_serialization_and_rejects_negative_indices() -> None:
    context = RoomContext(room=Room(id="room"), history_start_index=7)
    assert RoomContext.model_validate_json(context.model_dump_json()).history_start_index == 7
    assert RoomContext.model_validate({"room": {"id": "room"}}).history_start_index is None
    with pytest.raises(ValidationError):
        RoomContext(room=Room(id="room"), history_start_index=-1)


@pytest.mark.parametrize("append", [committed_in, provisional])
def test_appending_past_the_framework_cap_advances_the_carried_boundary(append: Any) -> None:
    source = EventSource(channel_id="source", channel_type=ChannelType.SMS)
    events = [
        RoomEvent(room_id="room", index=index, source=source, content=TextContent(body=str(index)))
        for index in range(2001)
    ]
    context = RoomContext(room=Room(id="room"), recent_events=events[:2000], history_start_index=0)
    if append is committed_in:
        carried = append(context, events[-1])
    else:
        _, carried = append(events[-1], context)
    assert len(carried.recent_events) == 2000
    assert carried.history_start_index == 1
    assert carried.recent_events[0].index == 1
    assert context.history_start_index == 0

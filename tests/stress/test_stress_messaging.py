"""Stress lane: messaging-path store costs at realistic room counts.

Pins the two costs RMK-88 removed — the O(rooms) routing scan and the
deep-copy-per-event context build — so a regression that reintroduces
either shows up as numbers, not vibes. Budgets are lax multiples of the
measured values (23 µs routing, 80 µs context build at 5 000 rooms) so a
loaded runner never flakes; the printed actuals are the record.
"""

from __future__ import annotations

import time

import pytest

from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import Access, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore

pytestmark = pytest.mark.stress

N_ROOMS = 5000
N_EVENTS = 50


async def _populated_store() -> InMemoryStore:
    store = InMemoryStore()
    for i in range(N_ROOMS):
        rid = f"room-{i}"
        await store.create_room(Room(id=rid))
        await store.add_binding(
            ChannelBinding(
                channel_id="sms1",
                room_id=rid,
                channel_type=ChannelType.SMS,
                access=Access.READ_WRITE,
                participant_id=f"user-{i}",
            )
        )
    for n in range(N_EVENTS):
        await store.commit_event(
            "room-4999",
            RoomEvent(
                room_id="room-4999",
                type=EventType.MESSAGE,
                source=EventSource(channel_id="sms1", channel_type=ChannelType.SMS),
                content=TextContent(body=f"message {n} " + "x" * 200),
            ),
        )
    return store


class TestMessagingStoreCosts:
    async def test_routing_lookup_is_indexed_not_scanned(self) -> None:
        store = await _populated_store()
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            room = await store.find_latest_room("user-4999", channel_type="sms")
        per_call_us = (time.perf_counter() - t0) / reps * 1e6
        assert room is not None and room.id == "room-4999"
        print(f"find_latest_room @ {N_ROOMS} rooms: {per_call_us:.1f} us/call")
        # A scan measured ~1500 us here; the index ~23 us. 300 us catches a
        # reintroduced scan while shrugging off runner noise.
        assert per_call_us < 300, f"routing lookup degraded to {per_call_us:.0f} us"

    async def test_context_build_shares_events(self) -> None:
        store = await _populated_store()
        kit = RoomKit(store=store)
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            ctx = await kit._build_context("room-4999", recent_limit=N_EVENTS)
        per_call_us = (time.perf_counter() - t0) / reps * 1e6
        assert len(ctx.recent_events) == N_EVENTS
        print(f"_build_context({N_EVENTS}): {per_call_us:.1f} us/call")
        # Deep-copying 50 events measured ~1600 us; shared reads ~80 us.
        assert per_call_us < 600, f"context build degraded to {per_call_us:.0f} us"

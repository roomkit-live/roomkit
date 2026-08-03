"""Per-room recent-events window resolution.

``_build_context`` derives how many recent events to load from the bound
channels' declared ``recent_events_window``: a transport-only room (realtime
voice) loads the 50-event floor instead of the 2000-event ceiling, whose
deserialisation several times per turn stalls the audio loop. These tests pin
that derivation so the voice/text split can't silently regress to loading the
ceiling.
"""

from __future__ import annotations

from types import SimpleNamespace

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.core.mixins.helpers import (
    _RECENT_EVENTS_FLOOR,
    _RECENT_EVENTS_LIMIT,
    HelpersMixin,
)
from roomkit.memory.base import DEFAULT_RECENT_EVENTS_WINDOW, MemoryProvider
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent


def _resolve(channels: dict, channel_ids: list[str]) -> int:
    bindings = [SimpleNamespace(channel_id=cid) for cid in channel_ids]
    fake_self = SimpleNamespace(_channels=channels)
    return HelpersMixin._resolve_recent_events_limit(fake_self, bindings)


# ── Declared windows ──────────────────────────────────────────────


def test_sliding_window_declares_its_event_count() -> None:
    assert SlidingWindowMemory(max_events=30).recent_events_window == 30


def test_token_aware_provider_inherits_full_pool() -> None:
    # A provider that trims by budget (not count) doesn't override → full pool,
    # so the framework keeps loading the ceiling for it (no regression).
    class _Budgeted(MemoryProvider):
        async def retrieve(self, *a, **k):  # type: ignore[no-untyped-def]
            ...

    assert _Budgeted().recent_events_window == DEFAULT_RECENT_EVENTS_WINDOW


def test_channel_base_reads_no_history() -> None:
    # Transport channels (voice, WS) inherit the 0 default.
    class _Transport(Channel):
        async def handle_inbound(self, *a, **k):  # type: ignore[no-untyped-def]
            ...

        async def deliver(self, *a, **k):  # type: ignore[no-untyped-def]
            ...

    assert _Transport("ws:x").recent_events_window == 0


# ── Resolution ────────────────────────────────────────────────────


def test_transport_only_room_loads_floor() -> None:
    # No channel reads history (or none registered) → floor, not the ceiling.
    assert _resolve({}, ["voice", "ws"]) == _RECENT_EVENTS_FLOOR


def test_room_takes_largest_channel_window() -> None:
    channels = {
        "text": SimpleNamespace(recent_events_window=2000),
        "ws": SimpleNamespace(recent_events_window=0),
    }
    assert _resolve(channels, ["text", "ws"]) == 2000


def test_small_window_still_floored() -> None:
    channels = {"text": SimpleNamespace(recent_events_window=10)}
    assert _resolve(channels, ["text"]) == _RECENT_EVENTS_FLOOR


def test_window_capped_at_ceiling() -> None:
    channels = {"text": SimpleNamespace(recent_events_window=10_000_000)}
    assert _resolve(channels, ["text"]) == _RECENT_EVENTS_LIMIT


# ── What the window contains ──────────────────────────────────────


async def _seed(kit: RoomKit, room_id: str, count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        await kit.store.commit_event(
            room_id,
            RoomEvent(
                room_id=room_id,
                type=EventType.MESSAGE,
                source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
                content=TextContent(body=f"msg-{i}"),
            ),
        )


async def test_window_holds_the_rooms_tail() -> None:
    """A room longer than the window is represented by its tail (RMK-99).

    What every hook and every AI channel reads is the current conversation, not
    the room's opening — an agent that quotes the start of a long room and
    misses what was just said is this assertion failing.
    """
    kit = RoomKit()
    room = await kit.create_room()
    await _seed(kit, room.id, _RECENT_EVENTS_FLOOR + 10)

    context = await kit._build_context(room.id)

    assert len(context.recent_events) == _RECENT_EVENTS_FLOOR
    newest = f"msg-{_RECENT_EVENTS_FLOOR + 9}"
    assert context.recent_events[-1].content.body == newest  # type: ignore[union-attr]
    assert [e.index for e in context.recent_events] == sorted(
        e.index for e in context.recent_events
    )


async def test_window_advances_with_the_conversation() -> None:
    """Two contexts built either side of a new message differ by that message."""
    kit = RoomKit()
    room = await kit.create_room()
    await _seed(kit, room.id, _RECENT_EVENTS_FLOOR + 1)

    before = await kit._build_context(room.id)
    await _seed(kit, room.id, 1, start=_RECENT_EVENTS_FLOOR + 1)
    after = await kit._build_context(room.id)

    assert after.recent_events[-1].index == before.recent_events[-1].index + 1
    assert after.recent_events[0].index == before.recent_events[0].index + 1

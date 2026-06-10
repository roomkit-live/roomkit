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
from roomkit.core.mixins.helpers import (
    _RECENT_EVENTS_FLOOR,
    _RECENT_EVENTS_LIMIT,
    HelpersMixin,
)
from roomkit.memory.base import DEFAULT_RECENT_EVENTS_WINDOW, MemoryProvider
from roomkit.memory.sliding_window import SlidingWindowMemory


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

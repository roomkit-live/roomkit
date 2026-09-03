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
from typing import Any

from roomkit import HookExecution, HookResult, HookTrigger
from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.core.mixins.helpers import (
    _RECENT_EVENTS_FLOOR,
    _RECENT_EVENTS_LIMIT,
    HelpersMixin,
)
from roomkit.identity.base import IdentityResolver
from roomkit.memory.base import DEFAULT_RECENT_EVENTS_WINDOW, MemoryProvider
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType, IdentificationStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.identity import IdentityHookResult, IdentityResult
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


def _resolve(
    channels: dict,
    channel_ids: list[str],
    *,
    hooked: bool = True,
    identity: bool = False,
    reads_history: bool = False,
) -> int:
    bindings = [SimpleNamespace(channel_id=cid) for cid in channel_ids]
    engine = SimpleNamespace(has_hooks=lambda trigger=None: hooked)
    identity_hooks = {HookTrigger.ON_IDENTITY_UNKNOWN: [object()]} if identity else {}
    fake_self = SimpleNamespace(
        _channels=channels, _hook_engine=engine, _identity_hooks=identity_hooks
    )
    return HelpersMixin._resolve_recent_events_limit(
        fake_self, bindings, reads_history=reads_history
    )


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
    # No channel reads history (or none registered) → floor, not the ceiling,
    # while a hook is registered to read it.
    assert _resolve({}, ["voice", "ws"]) == _RECENT_EVENTS_FLOOR


def test_transport_only_room_without_a_hook_loads_nothing() -> None:
    # The floor exists for hooks; with none registered nothing would read it.
    assert _resolve({}, ["voice", "ws"], hooked=False) == 0


def test_a_declared_window_loads_without_a_hook() -> None:
    channels = {"text": SimpleNamespace(recent_events_window=10)}
    assert _resolve(channels, ["text"], hooked=False) == 10


def test_an_identity_hook_keeps_the_floor() -> None:
    # Identity hooks live in the framework's registry, not the engine's index.
    assert _resolve({}, ["ws"], hooked=False, identity=True) == _RECENT_EVENTS_FLOOR


def test_a_caller_that_reads_the_tail_keeps_the_floor() -> None:
    assert _resolve({}, ["ws"], hooked=False, reads_history=True) == _RECENT_EVENTS_FLOOR


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


def _hooked(kit: RoomKit) -> None:
    """Register one no-op hook: the floor is loaded for hooks, and only for them."""

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def observe(event: RoomEvent, ctx: RoomContext) -> None:
        return None


async def test_window_holds_the_rooms_tail() -> None:
    """A room longer than the window is represented by its tail (RMK-99).

    What every hook and every AI channel reads is the current conversation, not
    the room's opening — an agent that quotes the start of a long room and
    misses what was just said is this assertion failing.
    """
    kit = RoomKit()
    _hooked(kit)
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
    _hooked(kit)
    room = await kit.create_room()
    await _seed(kit, room.id, _RECENT_EVENTS_FLOOR + 1)

    before = await kit._build_context(room.id)
    await _seed(kit, room.id, 1, start=_RECENT_EVENTS_FLOOR + 1)
    after = await kit._build_context(room.id)

    assert after.recent_events[-1].index == before.recent_events[-1].index + 1
    assert after.recent_events[0].index == before.recent_events[0].index + 1


# ── The floor is for hooks ────────────────────────────────────────


async def test_no_hook_and_no_reader_skips_the_history_read() -> None:
    """A transport-only room with no hook loads no history — and runs no query.

    The floor was loaded for every message whether or not anything read it:
    a Postgres round trip and fifty models per message, for nobody (RMK-103).
    """
    kit = RoomKit()
    room = await kit.create_room()
    await _seed(kit, room.id, _RECENT_EVENTS_FLOOR + 10)
    reads: list[int] = []
    original = kit.store.get_conversation

    async def counting(room_id: str, **kwargs: Any) -> list[RoomEvent]:
        reads.append(kwargs.get("limit", -1))
        return await original(room_id, **kwargs)

    kit.store.get_conversation = counting  # type: ignore[method-assign]

    context = await kit._build_context(room.id)

    assert context.recent_events == []
    assert reads == []


async def test_a_registered_hook_still_gets_the_floor() -> None:
    """One hook, on any trigger, and the floor is back exactly as it was."""
    kit = RoomKit()
    room = await kit.create_room()
    await _seed(kit, room.id, _RECENT_EVENTS_FLOOR + 10)
    _hooked(kit)

    context = await kit._build_context(room.id)

    assert len(context.recent_events) == _RECENT_EVENTS_FLOOR
    assert context.recent_events[-1].content.body == f"msg-{_RECENT_EVENTS_FLOOR + 9}"  # type: ignore[union-attr]


async def test_a_hook_on_a_real_turn_reads_the_recent_conversation() -> None:
    """The card's own criterion: a registered hook receives its context.

    A BEFORE_BROADCAST hook on a transport-only room sees the floor's tail
    on a real inbound message, not the empty list the hook-less path gets.
    The room is filled through the pipeline too — sixty hook-less turns, each
    building its context without a history read — so the delivery lanes have
    seen every event and owe the last turn no gap wait.
    """
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "sms")

    def message(body: str) -> InboundMessage:
        return InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body=body))

    for i in range(_RECENT_EVENTS_FLOOR + 10):
        await kit.process_inbound(message(f"msg-{i}"))
    seen: list[list[RoomEvent]] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def gate(event: RoomEvent, ctx: RoomContext) -> HookResult:
        seen.append(list(ctx.recent_events))
        return HookResult.allow()

    await kit.process_inbound(message("hello"))

    assert len(seen) == 1
    assert len(seen[0]) >= _RECENT_EVENTS_FLOOR
    bodies = [e.content.body for e in seen[0]]  # type: ignore[union-attr]
    assert f"msg-{_RECENT_EVENTS_FLOOR + 9}" in bodies


async def test_an_identity_hook_reads_the_recent_conversation() -> None:
    """Identity hooks are hooks too, in a registry the engine's index never sees.

    A room whose only hook is an ``@kit.identity_hook`` still loads the floor:
    "this number said who it was three messages ago" is exactly the glance the
    floor exists for.
    """

    class _Unknown(IdentityResolver):
        async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
            return IdentityResult(status=IdentificationStatus.UNKNOWN)

    kit = RoomKit(identity_resolver=_Unknown())
    kit.register_channel(SimpleChannel("sms"))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "sms")

    def message(body: str) -> InboundMessage:
        return InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body=body))

    for i in range(_RECENT_EVENTS_FLOOR + 10):
        await kit.process_inbound(message(f"msg-{i}"))
    seen: list[int] = []

    @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
    async def who(
        event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
    ) -> IdentityHookResult:
        seen.append(len(ctx.recent_events))
        return IdentityHookResult.reject("not today")

    await kit.process_inbound(message("hello"))

    assert seen == [_RECENT_EVENTS_FLOOR]


async def test_regenerate_finds_its_trigger_without_a_hook() -> None:
    """``regenerate_response`` scans the tail for its trigger: it is a reader.

    An intelligence channel with a zero window and no hook in the process left
    it an empty tail and a ``None`` where a regenerated turn was due.
    """
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(
        AIChannel(
            "ai",
            provider=MockAIProvider(responses=["again"]),
            memory=SlidingWindowMemory(max_events=0),
        )
    )
    room = await kit.create_room()
    await kit.attach_channel(room.id, "sms")
    await kit.attach_channel(room.id, "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="hello"))
    )

    assert await kit.regenerate_response(room.id) is not None

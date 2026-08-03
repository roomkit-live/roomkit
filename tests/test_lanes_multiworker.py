"""Delivery lanes through the full pipeline, across simulated workers.

Two complete RoomKit instances share one InMemoryStore and one lock manager —
the lightest faithful model of a multi-process deployment (each worker has
its own channel objects and its own lanes; the store and the locks are the
shared substrate). What test_lanes.py proves at engine level, this file
proves through ``process_inbound``.
"""

from __future__ import annotations

import asyncio

from roomkit import RoomKit
from roomkit.core.locks import InMemoryLockManager, _held_rooms
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.store.memory import InMemoryStore
from tests.test_framework import AILikeChannel, SimpleChannel


class RecordingChannel(SimpleChannel):
    """A transport whose deliveries land in a shared, tagged log."""

    def __init__(self, channel_id: str, tag: str, log: list[tuple[str, int]]) -> None:
        super().__init__(channel_id)
        self._tag = tag
        self._log = log

    async def deliver(self, event, binding, context):  # type: ignore[override]
        self._log.append((self._tag, event.index))
        return await super().deliver(event, binding, context)


def make_worker(
    store: InMemoryStore,
    locks: InMemoryLockManager,
    tag: str,
    log: list[tuple[str, int]],
    **kit_kwargs,
) -> tuple[RoomKit, RecordingChannel]:
    kit = RoomKit(store=store, lock_manager=locks, **kit_kwargs)
    kit.register_channel(SimpleChannel("src-a"))
    kit.register_channel(SimpleChannel("src-b"))
    recorder = RecordingChannel("recorder", tag, log)
    kit.register_channel(recorder)
    return kit, recorder


async def test_two_kits_interleaved_deliveries_follow_index_order() -> None:
    store = InMemoryStore()
    locks = InMemoryLockManager()
    log: list[tuple[str, int]] = []
    kit_a, _ = make_worker(store, locks, "A", log)
    kit_b, _ = make_worker(store, locks, "B", log)
    await kit_a.create_room(room_id="r1")
    await kit_a.attach_channel("r1", "src-a")
    await kit_a.attach_channel("r1", "src-b")
    await kit_a.attach_channel("r1", "recorder")

    async def send(kit: RoomKit, channel_id: str, body: str) -> None:
        await kit.process_inbound(
            InboundMessage(channel_id=channel_id, sender_id="u", content=TextContent(body=body))
        )

    # Ten turns racing across both workers on the same room.
    await asyncio.gather(
        *[
            send(kit_a if i % 2 == 0 else kit_b, "src-a" if i % 2 == 0 else "src-b", f"m{i}")
            for i in range(10)
        ]
    )

    # Global delivery order across BOTH workers is exactly index order —
    # the shared cursor, not the lock, is what serializes execution.
    indexes = [idx for _, idx in log]
    assert indexes == sorted(indexes)
    assert len(indexes) == 10
    # Each worker executed only its own plans, and both did some.
    assert {tag for tag, _ in log} == {"A", "B"}
    room = await store.get_room("r1")
    assert room is not None
    assert room.delivered_index == room.latest_index == max(indexes)
    await kit_a.close()
    await kit_b.close()


async def test_crashed_worker_commit_is_skipped_and_room_recovers() -> None:
    """Worker A commits an event and dies before delivering it. Worker B's
    next delivery waits out the gap, skips the hole observably, and the room
    keeps flowing — bounded loss, no wedge."""
    store = InMemoryStore()
    locks = InMemoryLockManager()
    log: list[tuple[str, int]] = []
    kit_a, _ = make_worker(store, locks, "A", log)
    kit_b, _ = make_worker(store, locks, "B", log, delivery_gap_timeout=0.2)
    skipped: list[dict] = []

    @kit_b.on("delivery_skipped")
    async def capture(fe) -> None:
        skipped.append(fe.data)

    await kit_a.create_room(room_id="r1")
    await kit_a.attach_channel("r1", "src-a")
    await kit_a.attach_channel("r1", "src-b")
    await kit_a.attach_channel("r1", "recorder")

    # "Crash" A's delivery side: its lanes are sealed, so its next commit
    # is stored but never delivered — exactly a worker dying between the
    # commit and the POST.
    await kit_a._lanes.aclose()
    result = await kit_a.process_inbound(
        InboundMessage(channel_id="src-a", sender_id="u", content=TextContent(body="lost"))
    )
    assert result.event is not None  # committed…
    lost_index = result.event.index
    assert not any(idx == lost_index for _, idx in log)  # …but never delivered

    # B's next message must go through after the gap timeout.
    await kit_b.process_inbound(
        InboundMessage(channel_id="src-b", sender_id="u", content=TextContent(body="after"))
    )
    assert [tag for tag, _ in log] == ["B"]
    assert skipped == [{"from_index": lost_index, "to_index": lost_index}]
    room = await store.get_room("r1")
    assert room is not None
    assert room.delivered_index == room.latest_index
    await kit_a.close()
    await kit_b.close()


async def test_delivery_runs_outside_the_room_lock() -> None:
    """The point of the lanes: channel delivery no longer holds the room
    lock, so a second turn can enter the locked section while the first is
    still delivering."""
    kit = RoomKit()
    held_during_delivery: list[frozenset[str]] = []

    class ProbeChannel(SimpleChannel):
        async def deliver(self, event, binding, context):  # type: ignore[override]
            held_during_delivery.append(_held_rooms.get())
            return await super().deliver(event, binding, context)

    kit.register_channel(SimpleChannel("src"))
    kit.register_channel(ProbeChannel("probe"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src")
    await kit.attach_channel("r1", "probe")
    await kit.process_inbound(
        InboundMessage(channel_id="src", sender_id="u", content=TextContent(body="hi"))
    )
    assert held_during_delivery, "delivery ran"
    assert all("r1" not in held for held in held_during_delivery)
    await kit.close()


async def test_ai_chain_completes_before_return_and_cursor_catches_up() -> None:
    """The observable contract the whole suite leans on, stated explicitly:
    when process_inbound returns, the AI's response is committed AND its
    delivery set has executed; the cursor matches the timeline."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("src"))
    kit.register_channel(AILikeChannel("ai"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src")
    await kit.attach_channel("r1", "ai")
    result = await kit.process_inbound(
        InboundMessage(channel_id="src", sender_id="u", content=TextContent(body="hi"))
    )
    assert result.event is not None
    # Timeline: 2 system channel-attached events, the trigger, the AI reply.
    events = await kit.store.list_events("r1")
    assert [e.source.channel_id for e in events[-2:]] == ["src", "ai"]
    room = await kit.store.get_room("r1")
    assert room is not None
    assert room.delivered_index == room.latest_index == events[-1].index
    await kit.close()


async def test_send_event_from_sync_hook_commits_and_eventually_delivers() -> None:
    """A send_event issued from a BEFORE_BROADCAST hook runs under the room
    lock: its cascade wait must short-circuit (no deadlock), the event must
    be committed on return, and its delivery must still happen — in lane
    order, shortly after."""
    from roomkit.models.enums import HookTrigger
    from roomkit.models.hook import HookResult

    kit = RoomKit()
    src = SimpleChannel("src")
    kit.register_channel(src)
    fired = False

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="auto_reply")
    async def auto_reply(event, context) -> HookResult:
        nonlocal fired
        if not fired and getattr(event.content, "body", "") == "hi":
            fired = True
            sent = await kit.send_event(
                room_id="r1", channel_id="src", content=TextContent(body="from-hook")
            )
            assert sent.index is not None  # committed on return
        return HookResult(action="allow")

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src")
    kit.register_channel(SimpleChannel("other"))
    await kit.attach_channel("r1", "other")

    await kit.process_inbound(
        InboundMessage(channel_id="src", sender_id="u", content=TextContent(body="hi"))
    )
    events = await kit.store.list_events("r1")
    bodies = [getattr(e.content, "body", "") for e in events]
    assert "hi" in bodies and "from-hook" in bodies
    # The hook's event was enqueued detached; its delivery follows in lane
    # order shortly after the caller returns.
    other = kit.get_channel("other")
    for _ in range(100):
        if len(other.delivered) >= 2:
            break
        await asyncio.sleep(0.01)
    assert {getattr(e.content, "body", "") for e in other.delivered} == {"hi", "from-hook"}
    await kit.close()


def test_store_commit_event_stays_behind_the_choke_points() -> None:
    """Every framework commit must flow through the three gates
    (_persist_committed / _commit_indexed / _commit_to_lane) so each index
    reaches the delivery cursor exactly once. A direct store.commit_event
    anywhere else in core silently grows a permanent cursor hole."""
    from pathlib import Path

    core = Path(__file__).parent.parent / "src" / "roomkit" / "core"
    allowed = {"helpers.py": 2, "lane_execution.py": 1}
    offenders: list[str] = []
    for path in core.rglob("*.py"):
        count = path.read_text().count("store.commit_event(")
        if count == 0:
            continue
        if path.name not in allowed or count > allowed[path.name]:
            offenders.append(f"{path.relative_to(core)}: {count} call(s)")
    assert not offenders, (
        "store.commit_event outside the commit gates (helpers._persist_committed, "
        f"helpers._commit_indexed, lane_execution._commit_to_lane): {offenders}"
    )


def test_inline_broadcast_stays_behind_the_delivery_lane() -> None:
    """A committed event's delivery belongs to the room's lane (RFC §10.2).

    ``router.broadcast`` plans and executes where the caller stands, which
    preserves per-room order only while nothing else can deliver for that
    room — and a commit publishes its index on the cursor, which is exactly
    what releases the lane to run the next one. Committing and then
    broadcasting inline therefore inverts delivery order. The two survivors
    each escape that: a child room has a single deliverer and never enqueues
    a plan, and regenerate re-solicits an *already delivered* event without
    committing anything. Use ``_commit_and_deliver`` for anything else.
    """
    from pathlib import Path

    core = Path(__file__).parent.parent / "src" / "roomkit" / "core"
    allowed = {"_child_execution.py": 1, "regenerate.py": 1}
    offenders: list[str] = []
    for path in core.rglob("*.py"):
        count = path.read_text().count("router.broadcast(")
        if count == 0:
            continue
        if path.name not in allowed or count > allowed[path.name]:
            offenders.append(f"{path.relative_to(core)}: {count} call(s)")
    assert not offenders, (
        "inline router.broadcast after a commit inverts per-room delivery order — "
        f"use lane_execution._commit_and_deliver instead: {offenders}"
    )

"""Delivery-lane engine unit tests (RFC §10.1 steps 12-14, §10.2, §13.5).

The lane engine is exercised against the real InMemoryStore (the cursor CAS
is the contract under test) and a fake LaneHost that records executions.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.core.lanes import (
    DeliveryCascade,
    DeliveryPlan,
    ExecEntry,
    LaneConfig,
    RoomLaneRegistry,
    _active_lane_room,
)
from roomkit.core.locks import InMemoryLockManager, _held_rooms
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore

ROOM = "lane-room"


def make_plan(event_id: str) -> DeliveryPlan:
    """A minimal plan; the engine treats its payload opaquely."""
    event = RoomEvent(
        id=event_id,
        room_id=ROOM,
        source=EventSource(channel_id="src", channel_type=ChannelType.WEBSOCKET),
        content=TextContent(body=event_id),
    )
    return DeliveryPlan(
        event=event,
        source_binding=None,
        context=None,  # type: ignore[arg-type] — never dereferenced by the engine
        targets=[],
    )


class FakeHost:
    """Records what the engine asked of it."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self.executed: list[str] = []
        self.post_effects: list[str] = []
        self.reentered: list[str] = []
        self.framework_events: list[tuple[str, dict[str, Any] | None]] = []
        self.execute_gate: asyncio.Event | None = None
        self.execute_raises: Exception | None = None
        self.on_execute: Any = None
        self.on_reentry: Any = None

    async def _execute_plan(self, plan: DeliveryPlan) -> Any:
        if self.execute_gate is not None:
            await self.execute_gate.wait()
        if self.on_execute is not None:
            await self.on_execute(plan)
        if self.execute_raises is not None:
            raise self.execute_raises
        self.executed.append(plan.event.id)
        return {"plan": plan.event.id}

    async def _post_plan_effects(
        self, plan: DeliveryPlan, result: Any, cascade: DeliveryCascade
    ) -> None:
        self.post_effects.append(plan.event.id)

    async def _reentry_commit_pass(
        self, room_id: str, plan: DeliveryPlan, result: Any, cascade: DeliveryCascade
    ) -> None:
        self.reentered.append(plan.event.id)
        if self.on_reentry is not None:
            await self.on_reentry(room_id, plan, cascade)

    async def _emit_framework_event(
        self,
        event_type: str,
        room_id: str | None = None,
        channel_id: str | None = None,
        event_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.framework_events.append((event_type, data))


def fast_config(**overrides: Any) -> LaneConfig:
    cfg = {
        "gap_timeout": 10.0,
        "gap_backoff_initial": 0.01,
        "gap_backoff_max": 0.02,
        "idle_ttl": 5.0,
        "close_grace": 0.5,
    }
    cfg.update(overrides)
    return LaneConfig(**cfg)


async def make_rig(
    **config: Any,
) -> tuple[InMemoryStore, FakeHost, RoomLaneRegistry, InMemoryLockManager]:
    store = InMemoryStore()
    await store.create_room(Room(id=ROOM))
    host = FakeHost(store)
    locks = InMemoryLockManager()
    registry = RoomLaneRegistry(host, locks, fast_config(**config))
    return store, host, registry, locks


def enqueue_exec(
    registry: RoomLaneRegistry,
    cascade: DeliveryCascade,
    event_id: str,
    index: int | None,
    after_index: int = -1,
) -> None:
    cascade.retain()
    entry = ExecEntry(
        plan=make_plan(event_id), cascade=cascade, index=index, after_index=after_index
    )
    registry.enqueue(ROOM, entry)


async def cursor(store: InMemoryStore) -> int:
    room = await store.get_room(ROOM)
    assert room is not None
    return room.delivered_index


@pytest.mark.asyncio
async def test_plans_execute_in_index_order_despite_arrival_order() -> None:
    store, host, registry, _ = await make_rig()
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    # Arrival order 2, 0, 1 — index order must win.
    enqueue_exec(registry, cascade, "e2", 2)
    enqueue_exec(registry, cascade, "e0", 0)
    enqueue_exec(registry, cascade, "e1", 1)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert host.executed == ["e0", "e1", "e2"]
    assert await cursor(store) == 2
    # Post effects and the reentry pass ran for each plan, in order.
    assert host.post_effects == ["e0", "e1", "e2"]
    assert host.reentered == ["e0", "e1", "e2"]
    await registry.aclose()


@pytest.mark.asyncio
async def test_cursor_entries_advance_without_executing() -> None:
    store, host, registry, _ = await make_rig()
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e0", 0)
    registry.note_committed(ROOM, 1)  # a BLOCKED event: index, no delivery
    enqueue_exec(registry, cascade, "e2", 2)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert host.executed == ["e0", "e2"]
    assert await cursor(store) == 2
    await registry.aclose()


@pytest.mark.asyncio
async def test_unindexed_entries_run_at_their_anchor_without_moving_cursor() -> None:
    store, host, registry, _ = await make_rig()
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e0", 0)
    # Two persistence-policy-excluded events anchored behind index 0.
    enqueue_exec(registry, cascade, "u1", None, after_index=0)
    enqueue_exec(registry, cascade, "u2", None, after_index=0)
    enqueue_exec(registry, cascade, "e1", 1)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert host.executed == ["e0", "u1", "u2", "e1"]
    assert await cursor(store) == 1
    await registry.aclose()


@pytest.mark.asyncio
async def test_gap_blocks_then_resumes_when_another_worker_advances() -> None:
    store, host, registry, _ = await make_rig(gap_timeout=10.0)
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e1", 1)  # hole at index 0
    await asyncio.sleep(0.1)
    assert host.executed == []  # blocked on the hole, no skip yet
    # "Another worker" delivers index 0 and advances the shared cursor.
    assert await store.advance_delivered_index(ROOM, 0)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert host.executed == ["e1"]
    assert await cursor(store) == 1
    assert not any(name == "delivery_skipped" for name, _ in host.framework_events)
    await registry.aclose()


@pytest.mark.asyncio
async def test_gap_skips_after_timeout_and_emits_delivery_skipped() -> None:
    store, host, registry, _ = await make_rig(gap_timeout=0.05)
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e3", 3)  # holes at 0..2
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert host.executed == ["e3"]
    assert await cursor(store) == 3
    skipped = [data for name, data in host.framework_events if name == "delivery_skipped"]
    assert skipped == [{"from_index": 0, "to_index": 2}]
    await registry.aclose()


@pytest.mark.asyncio
async def test_skip_never_jumps_past_a_local_plan() -> None:
    store, host, registry, _ = await make_rig(gap_timeout=0.05)
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e1", 1)
    enqueue_exec(registry, cascade, "e3", 3)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    # Both local plans executed, in order — the skips only covered the holes
    # (0, then 2), never a locally-held index.
    assert host.executed == ["e1", "e3"]
    skipped = [data for name, data in host.framework_events if name == "delivery_skipped"]
    assert skipped == [
        {"from_index": 0, "to_index": 0},
        {"from_index": 2, "to_index": 2},
    ]
    await registry.aclose()


@pytest.mark.asyncio
async def test_stale_plan_is_dropped_when_cursor_already_passed_it() -> None:
    store, host, registry, _ = await make_rig()
    # Another worker skipped over this process's index 0 (declared it lost).
    assert await store.advance_delivered_index(ROOM, 1, force=True)
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e0", 0)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert host.executed == []
    assert cascade.cancelled == "delivery_stale"
    assert await cursor(store) == 1
    await registry.aclose()


@pytest.mark.asyncio
async def test_execution_failure_resolves_wait_and_records_error() -> None:
    store, host, registry, _ = await make_rig()
    boom = RuntimeError("provider exploded")
    host.execute_raises = boom
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e0", 0)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert cascade.error is boom
    # The cursor still advanced: a failed delivery set is complete, not a hole.
    assert await cursor(store) == 0
    await registry.aclose()


@pytest.mark.asyncio
async def test_close_aborts_pending_work_and_wakes_waiters() -> None:
    store, host, registry, _ = await make_rig(close_grace=0.05)
    host.execute_gate = asyncio.Event()  # never set: execution hangs
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e0", 0)
    enqueue_exec(registry, cascade, "e1", 1)
    await asyncio.sleep(0.05)
    await asyncio.wait_for(registry.aclose(), timeout=2.0)
    await asyncio.wait_for(cascade.wait(), timeout=1.0)
    assert cascade.cancelled is not None
    assert host.executed == []


@pytest.mark.asyncio
async def test_sealed_registry_drops_new_entries_without_hanging_waiters() -> None:
    store, host, registry, _ = await make_rig()
    await registry.aclose()
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e0", 0)
    await asyncio.wait_for(cascade.wait(), timeout=1.0)
    assert cascade.cancelled == "kit_closed"
    assert host.executed == []


@pytest.mark.asyncio
async def test_idle_lane_retires_and_is_recreated_on_demand() -> None:
    store, host, registry, _ = await make_rig(idle_ttl=0.05)
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e0", 0)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    await asyncio.sleep(0.2)
    assert registry._lanes == {}  # retired
    cascade2 = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade2, "e1", 1)
    await asyncio.wait_for(cascade2.wait(), timeout=2.0)
    assert host.executed == ["e0", "e1"]
    await registry.aclose()


@pytest.mark.asyncio
async def test_executor_runs_outside_the_enqueuers_lock_context() -> None:
    """The lane task must not inherit the enqueuer's held room locks.

    An inherited ``_held_rooms`` would let the reentry pass take the lock
    manager's reentrant fast path without actually holding the room lock.
    """
    store, host, registry, locks = await make_rig()
    seen: list[tuple[frozenset[str], str | None]] = []

    async def observe(plan: DeliveryPlan) -> None:
        seen.append((_held_rooms.get(), _active_lane_room.get()))

    host.on_execute = observe
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    async with locks.locked(ROOM):
        enqueue_exec(registry, cascade, "e0", 0)
        # The claim is a different key from the room lock, so the executor
        # delivers WHILE we still hold the room lock — the point of the lanes.
        await asyncio.sleep(0.1)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert seen, "plan executed while the enqueuer held the room lock"
    (held, lane_room) = seen[0]
    assert ROOM not in held  # no inherited room-lock membership
    assert lane_room == ROOM
    await registry.aclose()


@pytest.mark.asyncio
async def test_wait_short_circuits_inside_the_lane_executor() -> None:
    """cascade.wait() from inside a plan's execution must not self-deadlock."""
    store, host, registry, _ = await make_rig()
    cascade = DeliveryCascade(ROOM, reentry_budget=10)

    async def reenter(plan: DeliveryPlan) -> None:
        # A tool handler calling send_event would end up waiting on a cascade
        # of this very room; it must return immediately.
        await asyncio.wait_for(cascade.wait(), timeout=0.5)

    host.on_execute = reenter
    enqueue_exec(registry, cascade, "e0", 0)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert host.executed == ["e0"]
    await registry.aclose()


@pytest.mark.asyncio
async def test_wait_short_circuits_while_holding_the_room_lock() -> None:
    store, host, registry, locks = await make_rig()
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    cascade.retain()  # pending unit that will not resolve during the check
    async with locks.locked(ROOM):
        await asyncio.wait_for(cascade.wait(), timeout=0.5)  # returns, no deadlock
    cascade.release()
    await registry.aclose()


@pytest.mark.asyncio
async def test_reentry_pass_children_complete_before_wait_resolves() -> None:
    """Retain-child-before-release-parent: an AI chain resolves as one cascade."""
    store, host, registry, _ = await make_rig()
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    spawned = False

    async def spawn_child(room_id: str, plan: DeliveryPlan, cas: DeliveryCascade) -> None:
        nonlocal spawned
        if plan.event.id == "trigger" and not spawned:
            spawned = True
            # The reentry pass commits the response at the next index and
            # enqueues its plan behind the trigger — unit retained first.
            cas.retain()
            registry.enqueue(
                room_id,
                ExecEntry(plan=make_plan("response"), cascade=cas, index=1, after_index=-1),
            )

    host.on_reentry = spawn_child
    enqueue_exec(registry, cascade, "trigger", 0)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    assert host.executed == ["trigger", "response"]
    assert await cursor(store) == 1
    await registry.aclose()


@pytest.mark.asyncio
async def test_reentry_budget_exhausts() -> None:
    cascade = DeliveryCascade(ROOM, reentry_budget=2)
    assert cascade.consume_reentry_budget()
    assert cascade.consume_reentry_budget()
    assert not cascade.consume_reentry_budget()


@pytest.mark.asyncio
async def test_no_tasks_leak_after_close() -> None:
    store, host, registry, _ = await make_rig()
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    enqueue_exec(registry, cascade, "e0", 0)
    await asyncio.wait_for(cascade.wait(), timeout=2.0)
    await registry.aclose()
    lingering = [
        t
        for t in asyncio.all_tasks()
        if t.get_name().startswith("roomkit-delivery-lane-") and not t.done()
    ]
    assert lingering == []


@pytest.mark.asyncio
async def test_two_registries_one_store_preserve_global_index_order() -> None:
    """Two processes simulated: shared store + shared lock manager, one lane
    registry per 'process'. Each executes only its own plans; the shared
    cursor forces the global per-room order."""
    store = InMemoryStore()
    await store.create_room(Room(id=ROOM))
    locks = InMemoryLockManager()
    order: list[str] = []

    class OrderedHost(FakeHost):
        async def _execute_plan(self, plan: DeliveryPlan) -> Any:
            order.append(plan.event.id)
            await asyncio.sleep(0.01)
            return {}

    host_a, host_b = OrderedHost(store), OrderedHost(store)
    reg_a = RoomLaneRegistry(host_a, locks, fast_config())
    reg_b = RoomLaneRegistry(host_b, locks, fast_config())
    cas_a = DeliveryCascade(ROOM, reentry_budget=10)
    cas_b = DeliveryCascade(ROOM, reentry_budget=10)
    # Interleaved commits: even indexes on A, odd on B, enqueued out of order.
    for idx in (4, 0, 2):
        cas_a.retain()
        reg_a.enqueue(ROOM, ExecEntry(plan=make_plan(f"a{idx}"), cascade=cas_a, index=idx))
    for idx in (3, 1, 5):
        cas_b.retain()
        reg_b.enqueue(ROOM, ExecEntry(plan=make_plan(f"b{idx}"), cascade=cas_b, index=idx))
    await asyncio.wait_for(asyncio.gather(cas_a.wait(), cas_b.wait()), timeout=5.0)
    assert order == ["a0", "b1", "a2", "b3", "a4", "b5"]
    assert (await store.get_room(ROOM)).delivered_index == 5
    await reg_a.aclose()
    await reg_b.aclose()


@pytest.mark.asyncio
async def test_crashed_worker_hole_is_skipped_and_room_recovers() -> None:
    """Worker A commits index 0 but dies before delivering. Worker B's lane
    waits gap_timeout, skips the hole (observable), and delivers its own."""
    store = InMemoryStore()
    await store.create_room(Room(id=ROOM))
    locks = InMemoryLockManager()
    host_b = FakeHost(store)
    reg_b = RoomLaneRegistry(host_b, locks, fast_config(gap_timeout=0.1))
    # Worker A: committed index 0, then crashed — no lane entry anywhere.
    cas_b = DeliveryCascade(ROOM, reentry_budget=10)
    cas_b.retain()
    reg_b.enqueue(ROOM, ExecEntry(plan=make_plan("b1"), cascade=cas_b, index=1))
    await asyncio.wait_for(cas_b.wait(), timeout=2.0)
    assert host_b.executed == ["b1"]
    assert (await store.get_room(ROOM)).delivered_index == 1
    skipped = [d for n, d in host_b.framework_events if n == "delivery_skipped"]
    assert skipped == [{"from_index": 0, "to_index": 0}]
    await reg_b.aclose()


@pytest.mark.asyncio
async def test_claim_blocks_gap_clock_while_owner_is_executing() -> None:
    """While the hole's owner actively executes (claim held), a waiting lane
    blocks on claim acquisition instead of accumulating gap time — a slow
    delivery is never mistaken for a crash."""
    store = InMemoryStore()
    await store.create_room(Room(id=ROOM))
    locks = InMemoryLockManager()

    slow_gate = asyncio.Event()

    class SlowHost(FakeHost):
        async def _execute_plan(self, plan: DeliveryPlan) -> Any:
            if plan.event.id == "slow0":
                await slow_gate.wait()  # long provider round trip
            self.executed.append(plan.event.id)
            return {}

    host_a = SlowHost(store)
    host_b = FakeHost(store)
    # B's gap timeout is SHORTER than A's execution: without the claim
    # gating the clock, B would skip past the in-flight index 0.
    reg_a = RoomLaneRegistry(host_a, locks, fast_config())
    reg_b = RoomLaneRegistry(host_b, locks, fast_config(gap_timeout=0.05))
    cas_a = DeliveryCascade(ROOM, reentry_budget=10)
    cas_b = DeliveryCascade(ROOM, reentry_budget=10)
    cas_a.retain()
    reg_a.enqueue(ROOM, ExecEntry(plan=make_plan("slow0"), cascade=cas_a, index=0))
    await asyncio.sleep(0.05)  # A's executor is now inside execute, claim held
    cas_b.retain()
    reg_b.enqueue(ROOM, ExecEntry(plan=make_plan("b1"), cascade=cas_b, index=1))
    await asyncio.sleep(0.3)  # far beyond B's gap timeout
    slow_gate.set()
    await asyncio.wait_for(asyncio.gather(cas_a.wait(), cas_b.wait()), timeout=2.0)
    assert host_a.executed == ["slow0"]
    assert host_b.executed == ["b1"]
    assert not any(n == "delivery_skipped" for n, _ in host_b.framework_events)
    await reg_a.aclose()
    await reg_b.aclose()


@pytest.mark.asyncio
async def test_drain_completes_queued_work_before_close() -> None:
    store, host, registry, _ = await make_rig()
    cascade = DeliveryCascade(ROOM, reentry_budget=10)
    for i in range(5):
        enqueue_exec(registry, cascade, f"e{i}", i)
    await registry.aclose()  # bounded drain runs first
    assert host.executed == [f"e{i}" for i in range(5)]
    assert cascade.cancelled is None
    await asyncio.wait_for(cascade.wait(), timeout=0.5)

"""Tests for InMemoryLockManager."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest

from roomkit import RoomKit
from roomkit.core.locks import InMemoryLockManager, RoomLockManager, _has_room_lock, _held_rooms


class TestInMemoryLockManager:
    async def test_same_room_same_lock(self) -> None:
        mgr = InMemoryLockManager()
        lock1 = mgr._get_lock("r1")
        lock2 = mgr._get_lock("r1")
        assert lock1 is lock2

    async def test_different_rooms_different_locks(self) -> None:
        mgr = InMemoryLockManager()
        lock1 = mgr._get_lock("r1")
        lock2 = mgr._get_lock("r2")
        assert lock1 is not lock2

    async def test_serialization(self) -> None:
        """Mutual exclusion, asserted for real: overlap inside the critical
        section fails the test — a no-op lock would report max_concurrent == 5.
        """
        mgr = InMemoryLockManager()
        in_critical = 0
        max_concurrent = 0
        order: list[int] = []

        async def task(n: int) -> None:
            nonlocal in_critical, max_concurrent
            async with mgr.locked("r1"):
                in_critical += 1
                max_concurrent = max(max_concurrent, in_critical)
                order.append(n)
                # Yield inside the critical section so a broken lock lets the
                # other tasks in and the overlap is observed.
                await asyncio.sleep(0.001)
                in_critical -= 1

        await asyncio.gather(*(task(n) for n in range(5)))
        assert max_concurrent == 1
        assert len(order) == 5

    async def test_different_rooms_do_not_serialize(self) -> None:
        """The lock is per-room: two rooms' critical sections may overlap."""
        mgr = InMemoryLockManager()
        in_critical = 0
        max_concurrent = 0
        gate = asyncio.Event()

        async def task(room: str, first: bool) -> None:
            nonlocal in_critical, max_concurrent
            async with mgr.locked(room):
                in_critical += 1
                max_concurrent = max(max_concurrent, in_critical)
                if first:
                    await gate.wait()
                else:
                    gate.set()
                in_critical -= 1

        await asyncio.gather(task("r1", True), task("r2", False))
        assert max_concurrent == 2

    async def test_lru_eviction(self) -> None:
        mgr = InMemoryLockManager(max_locks=2)
        mgr._get_lock("r1")
        mgr._release_ref("r1")
        mgr._get_lock("r2")
        mgr._release_ref("r2")
        mgr._get_lock("r3")
        assert mgr.size == 2
        assert "r1" not in mgr._locks

    async def test_held_lock_not_evicted(self) -> None:
        mgr = InMemoryLockManager(max_locks=2)
        lock1 = mgr._get_lock("r1")
        await lock1.acquire()
        try:
            mgr._get_lock("r2")
            mgr._get_lock("r3")
            # r1 is held so cannot be evicted; size may exceed max
            assert "r1" in mgr._locks
        finally:
            lock1.release()

    async def test_reentrant_same_context(self) -> None:
        """Same context can re-acquire the same room lock without deadlocking."""
        mgr = InMemoryLockManager()
        async with mgr.locked("r1"):
            assert "r1" in _held_rooms.get()
            # Inner acquisition should succeed immediately (reentrant)
            async with mgr.locked("r1"):
                assert "r1" in _held_rooms.get()
            assert "r1" in _held_rooms.get()
        assert "r1" not in _held_rooms.get()

    async def test_reentrant_three_deep(self) -> None:
        mgr = InMemoryLockManager()
        async with mgr.locked("r1"):  # noqa: SIM117
            async with mgr.locked("r1"):  # noqa: SIM117
                async with mgr.locked("r1"):
                    assert "r1" in _held_rooms.get()
        assert "r1" not in _held_rooms.get()

    async def test_reentrant_different_rooms_independent(self) -> None:
        """Reentrancy tracking is per-room."""
        mgr = InMemoryLockManager()
        async with mgr.locked("r1"), mgr.locked("r2"):
            held = _held_rooms.get()
            assert "r1" in held
            assert "r2" in held

    async def test_reentrant_via_gather(self) -> None:
        """Child tasks from asyncio.gather inherit the held set via ContextVar."""
        mgr = InMemoryLockManager()
        acquired_in_child = False

        async def child() -> None:
            nonlocal acquired_in_child
            # This runs in a child task that inherited the ContextVar
            async with mgr.locked("r1"):
                acquired_in_child = True

        async with mgr.locked("r1"):
            await asyncio.gather(child())

        assert acquired_in_child


async def test_child_must_acquire_after_parent_releases() -> None:
    mgr = InMemoryLockManager()
    start = asyncio.Event()
    entered = asyncio.Event()

    async def child() -> None:
        await start.wait()
        async with mgr.locked("r1"):
            entered.set()

    async with mgr.locked("r1"):
        task = asyncio.create_task(child())
    try:
        async with mgr.locked("r1"):
            start.set()
            await asyncio.sleep(0)
            assert not entered.is_set()
    finally:
        await asyncio.wait_for(task, 1)
    assert entered.is_set()


async def test_reentrance_is_scoped_to_manager() -> None:
    first = InMemoryLockManager()
    second = InMemoryLockManager()
    async with first.locked("r1"), second.locked("r1"):
        assert first._locks["r1"].locked()
        assert second._locks["r1"].locked()


async def test_a_lock_of_another_kind_is_no_room_lock() -> None:
    """A kit's lock of another kind (a delivery claim) keyed like a room id is
    held in a map of its own and never reads as that room's lock."""
    rooms = InMemoryLockManager()
    async with rooms._key_space("delivery").locked("r1"):
        assert not _has_room_lock("r1")
        async with asyncio.timeout(1):
            async with rooms.locked("r1"):
                assert _has_room_lock("r1")


class SharedLockManager(RoomLockManager):
    """A lock manager of no in-memory kind, as one shared across processes:
    one lock per key, whatever the key."""

    def __init__(self) -> None:
        self.locks: dict[str, asyncio.Lock] = {}
        self.keys: list[str] = []

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        self.keys.append(room_id)
        async with self.locks.setdefault(room_id, asyncio.Lock()):
            yield


async def test_a_room_id_with_a_nul_character_is_refused() -> None:
    """A NUL character leads the keys of the kit's other locks in a shared
    manager (RFC §13.5): no room id holds one."""
    kit = RoomKit(lock_manager=SharedLockManager())
    with pytest.raises(ValueError, match="NUL"):
        await kit.create_room(room_id="\x00__delivery__:r1")
    await kit.close()


class MarkingLockManager(InMemoryLockManager):
    """A manager whose ``locked()`` is its own and marks its leases, as
    ``PostgresAdvisoryLockManager`` does: the kit holds every kind in it."""

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        async with super().locked(room_id):
            yield


@pytest.mark.parametrize("kind", ["room", "verification"])
@pytest.mark.parametrize("locks", [InMemoryLockManager, MarkingLockManager])
async def test_a_lock_held_and_asked_again_once_the_close_began_is_no_wait(
    kind: str, locks: type[InMemoryLockManager]
) -> None:
    """A call holding a lock that asks for it again once the kit's close
    stopped admission is not cut: a lock the call holds is no wait, whatever
    its kind and its manager (RFC §12.10.4)."""
    kit = RoomKit(lock_manager=locks())
    held = kit._lock_manager if kind == "room" else kit._verification_locks
    async with kit._door_runs.run(), held.locked("k"):
        kit._door_runs.closed = True
        async with held.locked("k"):
            pass
    kit._door_runs.closed = False
    await kit.close()

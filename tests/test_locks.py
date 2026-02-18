"""Tests for InMemoryLockManager."""

from __future__ import annotations

import asyncio

from roomkit.core.locks import InMemoryLockManager, _held_rooms


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
        mgr = InMemoryLockManager()
        order: list[int] = []

        async def task(n: int) -> None:
            async with mgr.locked("r1"):
                order.append(n)
                await asyncio.sleep(0.01)

        await asyncio.gather(task(1), task(2))
        assert len(order) == 2

    async def test_lru_eviction(self) -> None:
        mgr = InMemoryLockManager(max_locks=2)
        mgr._get_lock("r1")
        mgr._get_lock("r2")
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

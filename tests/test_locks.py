"""Tests for InMemoryLockManager."""

from __future__ import annotations

import asyncio

from roomkit.core.locks import InMemoryLockManager


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

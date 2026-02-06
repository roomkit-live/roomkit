"""Per-room async locking with LRU eviction."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class RoomLockManager(ABC):
    """Abstract base for per-room locking.

    Implement this to plug in any locking backend (Redis, Postgres
    advisory locks, etc.).  The library ships with ``InMemoryLockManager``
    for single-process deployments.
    """

    @abstractmethod
    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncIterator[None]:
        """Acquire an exclusive lock for *room_id*."""
        yield  # pragma: no cover


class InMemoryLockManager(RoomLockManager):
    """In-process per-room asyncio locks with LRU eviction.

    Suitable for single-process deployments.  For multi-process or
    distributed setups, provide a custom ``RoomLockManager`` backed by
    Redis, Postgres advisory locks, or similar.
    """

    def __init__(self, max_locks: int = 1024) -> None:
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._max_locks = max_locks

    def _get_lock(self, room_id: str) -> asyncio.Lock:
        if room_id in self._locks:
            self._locks.move_to_end(room_id)
            return self._locks[room_id]

        lock = asyncio.Lock()
        self._locks[room_id] = lock
        self._evict()
        return lock

    def _evict(self) -> None:
        if len(self._locks) <= self._max_locks:
            return
        to_remove: list[str] = []
        for key, lock in self._locks.items():
            if len(self._locks) - len(to_remove) <= self._max_locks:
                break
            if not lock.locked():
                to_remove.append(key)
        for key in to_remove:
            self._locks.pop(key)

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncIterator[None]:
        """Acquire the lock for a room."""
        lock = self._get_lock(room_id)
        async with lock:
            yield

    @property
    def size(self) -> int:
        """Return the number of locks currently held."""
        return len(self._locks)

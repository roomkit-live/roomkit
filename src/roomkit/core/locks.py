"""Per-room async locking with LRU eviction."""

from __future__ import annotations

import asyncio
import contextvars
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# ContextVar tracking which rooms the current execution context holds locks
# for.  asyncio.gather() copies the parent context to child tasks, so
# children see the parent's held set and can re-enter without deadlocking.
_held_rooms: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "_room_locks_held", default=frozenset()
)


class RoomLockManager(ABC):
    """Abstract base for per-room locking.

    Implement this to plug in any locking backend (Redis, Postgres
    advisory locks, etc.).  The library ships with ``InMemoryLockManager``
    for single-process deployments.

    Implementations should be **reentrant** within the same execution
    context (including child tasks spawned by ``asyncio.gather``): if
    a coroutine already holds the lock for a room and awaits code that
    tries to acquire the same room lock, the inner acquisition must
    succeed without deadlocking.  This is required because tool handlers
    (e.g. handoff) may update room state while the inbound pipeline
    already holds the room lock.
    """

    @abstractmethod
    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncIterator[None]:
        """Acquire an exclusive lock for *room_id*."""
        yield  # pragma: no cover


class InMemoryLockManager(RoomLockManager):
    """In-process per-room asyncio locks with LRU eviction.

    Reentrant within the same execution context: if the current context
    already holds the lock for a given room (including child tasks
    spawned by ``asyncio.gather``), ``locked()`` yields immediately
    instead of deadlocking.

    Suitable for single-process deployments.  For multi-process or
    distributed setups, provide a custom ``RoomLockManager`` backed by
    Redis, Postgres advisory locks, or similar.
    """

    def __init__(self, max_locks: int = 1024) -> None:
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._refcounts: dict[str, int] = {}
        self._max_locks = max_locks

    def _get_lock(self, room_id: str) -> asyncio.Lock:
        if room_id in self._locks:
            self._locks.move_to_end(room_id)
            self._refcounts[room_id] = self._refcounts.get(room_id, 0) + 1
            return self._locks[room_id]

        lock = asyncio.Lock()
        self._locks[room_id] = lock
        self._refcounts[room_id] = 1
        self._evict()
        return lock

    def _release_ref(self, room_id: str) -> None:
        """Decrement the reference count for a room lock."""
        count = self._refcounts.get(room_id, 0) - 1
        if count <= 0:
            self._refcounts.pop(room_id, None)
        else:
            self._refcounts[room_id] = count

    def _evict(self) -> None:
        if len(self._locks) <= self._max_locks:
            return
        to_remove: list[str] = []
        for key, lock in self._locks.items():
            if len(self._locks) - len(to_remove) <= self._max_locks:
                break
            if not lock.locked() and self._refcounts.get(key, 0) <= 0:
                to_remove.append(key)
        for key in to_remove:
            self._locks.pop(key)
            self._refcounts.pop(key, None)

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncIterator[None]:
        """Acquire the lock for a room (reentrant via ContextVar)."""
        held = _held_rooms.get()
        if room_id in held:
            # Reentrant: this execution context already holds the lock.
            yield
            return

        lock = self._get_lock(room_id)
        try:
            async with lock:
                token = _held_rooms.set(held | frozenset({room_id}))
                try:
                    yield
                finally:
                    _held_rooms.reset(token)
        finally:
            self._release_ref(room_id)

    @property
    def size(self) -> int:
        """Return the number of locks currently held."""
        return len(self._locks)

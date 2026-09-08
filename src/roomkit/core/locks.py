"""Per-room async locking with LRU eviction."""

from __future__ import annotations

import asyncio
import contextvars
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

# ContextVar tracking which rooms the current execution context holds locks
# for.  asyncio.gather() copies the parent context to child tasks, so
# children see the parent's held set and can re-enter without deadlocking.
_held_rooms: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "_room_locks_held", default=frozenset()
)


@dataclass
class _LockLease:
    manager: RoomLockManager
    room_id: str
    active: bool = True


_held_leases: contextvars.ContextVar[tuple[_LockLease, ...]] = contextvars.ContextVar(
    "_room_lock_leases", default=()
)


def _has_room_lock(room_id: str, manager: RoomLockManager | None = None) -> bool:
    """Inherited context grants reentrancy only while its acquisition is live."""
    return room_id in _held_rooms.get() and any(
        lease.active and lease.room_id == room_id and (manager is None or lease.manager is manager)
        for lease in _held_leases.get()
    )


@contextmanager
def _mark_room_locked(manager: RoomLockManager, room_id: str) -> Iterator[None]:
    lease = _LockLease(manager, room_id)
    rooms_token = _held_rooms.set(_held_rooms.get() | {room_id})
    leases_token = _held_leases.set((*_held_leases.get(), lease))
    try:
        yield
    finally:
        # Children keep copies of the context, but share this revocable lease.
        lease.active = False
        _held_leases.reset(leases_token)
        _held_rooms.reset(rooms_token)


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
        """Acquire an exclusive lock for *room_id*.

        Acquisition MUST be cancellation-safe: the inbound pipeline bounds the
        wait with ``process_timeout`` (RFC §13.6), so a caller that gives up
        queueing is cancelled here. An implementation that leaves the lock
        taken, or a reference held, on cancellation strands the room — every
        later event for it queues behind a lock nobody owns.
        """
        yield  # pragma: no cover

    async def close(self) -> None:
        """Release any resources held by the lock manager (e.g. a connection
        pool). Called by ``RoomKit.close()``. Default is a no-op; overrides
        MUST be idempotent."""
        return None


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
        self._mgr_lock = asyncio.Lock()

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
        if _has_room_lock(room_id, self):
            # Reentrant: this execution context already holds the lock.
            yield
            return

        async with self._mgr_lock:
            lock = self._get_lock(room_id)
        try:
            async with lock:
                with _mark_room_locked(self, room_id):
                    yield
        finally:
            async with self._mgr_lock:
                self._release_ref(room_id)

    @property
    def size(self) -> int:
        """Return the number of locks currently held."""
        return len(self._locks)

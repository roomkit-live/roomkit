"""Per-room async locking with LRU eviction."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("roomkit.locks")

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


@dataclass(eq=False, slots=True)
class _Call:
    running: bool = True


@dataclass(frozen=True, slots=True)
class _LentLease:
    """A lease lent to a call (a hook's body) while the call runs: a task the
    call spawned keeps a copy of the context, and finds the lease lent no
    more once the call is over, though the section holding the lock goes
    on."""

    lease: _LockLease | _LentLease
    call: _Call

    @property
    def manager(self) -> RoomLockManager:
        return self.lease.manager

    @property
    def room_id(self) -> str:
        return self.lease.room_id

    @property
    def active(self) -> bool:
        return self.call.running and self.lease.active


_held_leases: contextvars.ContextVar[tuple[_LockLease | _LentLease, ...]] = contextvars.ContextVar(
    "_room_lock_leases", default=()
)


def _has_room_lock(room_id: str, manager: RoomLockManager | None = None) -> bool:
    """Inherited context grants reentrancy only while its acquisition is live.

    Without *manager*, any room lock counts, never a lock of another kind (a
    delivery claim, a verification) whose key reads like a room id."""
    if isinstance(manager, ReleaseTolerantLockManager):
        manager = manager.inner
    return room_id in _held_rooms.get() and any(
        lease.active
        and lease.room_id == room_id
        and (lease.manager is manager if manager is not None else _locks_rooms(lease.manager))
        for lease in _held_leases.get()
    )


def _locks_rooms(manager: RoomLockManager) -> bool:
    """Whether *manager* holds room locks, not a kit's locks of another kind."""
    return not isinstance(manager, _KeySpace)


def _holds_a_room_lock(manager: RoomLockManager | None = None) -> bool:
    """Whether this context holds a live lock on any room of *manager*."""
    return any(_has_room_lock(room_id, manager) for room_id in _held_rooms.get())


def unlocked_context() -> contextvars.Context:
    """A copy of the current context that holds no room lock: a task started
    in it from under a lock asks for the lock as any later caller does, and
    waits for its holder, never entering as the holder's reentrant child."""
    context = contextvars.copy_context()
    context.run(_held_rooms.set, frozenset())
    context.run(_held_leases.set, ())
    return context


@contextmanager
def lent_while_it_runs() -> Generator[None, None, None]:
    """Run a call (a hook's body) with the room locks this context holds lent
    to it while it runs (RFC §9.5.1 Reentrance, §13.5): a task it spawns,
    awaited or not, acts for it under them while it runs; once it is over,
    that task holds none of them, though the section that holds the lock
    goes on, and its calls into the room are new arrivals.

    Such a call is an ordinary arrival, neither refused nor adopted: it waits
    for the lock the section holds. A later hook of the same pass that awaits
    the task therefore waits until its own timeout, which cancels the task
    it awaited unless it shields it."""
    leases = _held_leases.get()
    if not leases:
        yield
        return
    call = _Call()
    token = _held_leases.set(tuple(_LentLease(lease, call) for lease in leases))
    try:
        yield
    finally:
        call.running = False
        _held_leases.reset(token)


@contextmanager
def _mark_room_locked(manager: RoomLockManager, room_id: str) -> Generator[None, None, None]:
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
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
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

    def _key_space(self, kind: str) -> RoomLockManager:
        """The locks of *kind* the kit takes from this manager besides its
        rooms' (``"delivery"``: a room's delivery claim; ``"verification"``:
        an identity's verification). A manager shared across processes holds
        them among its rooms' keys, each key led by a NUL character, which
        no room id holds (``create_room`` refuses one, RFC §13.5)."""
        return _NulLedKeys(self)


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
        self._key_spaces: dict[str, _KeySpace] = {}

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
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
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

    def _key_space(self, kind: str) -> RoomLockManager:
        """A map of its own for the locks of *kind*, shared by every kit that
        shares this manager: no room id, whatever it reads, names one of
        their keys (RFC §13.5). A subclass that takes its locks its own way
        (``locked`` overridden) is asked for every kind, as a shared
        manager is."""
        if type(self).locked is not InMemoryLockManager.locked:
            return super()._key_space(kind)
        space = self._key_spaces.get(kind)
        if space is None:
            space = self._key_spaces[kind] = _KeySpace(self._max_locks)
        return space


class _KeySpace(InMemoryLockManager):
    """An in-memory manager's locks of one kind other than its rooms'."""


# What leads the key of a lock of another kind than a room's in a shared
# manager: no room id holds it.
_OTHER_KIND = "\x00"


class _NulLedKeys(RoomLockManager):
    """A shared manager's locks of a kind other than its rooms': the same
    manager, each key led by a NUL character (``\x00__delivery__:r1``,
    ``\x00verification:org-a:alice``)."""

    def __init__(self, inner: RoomLockManager) -> None:
        self.inner = inner

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        async with self.inner.locked(f"{_OTHER_KIND}{room_id}"):
            yield

    async def close(self) -> None:
        """The shared manager is closed as the room locks' manager."""
        return None


# The failed releases met by the operation that set the list (RFC §13.5).
_release_failures: contextvars.ContextVar[list[Exception] | None] = contextvars.ContextVar(
    "_room_lock_release_failures", default=None
)


@contextmanager
def collecting_release_failures() -> Generator[list[Exception], None, None]:
    """Collect the failed releases of the room locks taken within, so that
    the operation can report them (RFC §13.5)."""
    failures: list[Exception] = []
    token = _release_failures.set(failures)
    try:
        yield failures
    finally:
        _release_failures.reset(token)


def failed_release() -> Exception | None:
    """The first failed release the collecting operation met so far, if any."""
    failures = _release_failures.get()
    return failures[0] if failures else None


class ReleaseTolerantLockManager(RoomLockManager):
    """The kit's hold on a lock manager: a release that fails changes
    nothing the locked section decided or wrote (RFC §13.5).

    The failure is logged and handed to the operation collecting it, never
    raised: an event committed stays committed, a refusal stays a refusal.
    A failure met inside the section stays the one raised. What a failed
    release leaves behind is the wrapped manager's to recover; the next
    section asks it for the lock again.
    """

    def __init__(
        self,
        inner: RoomLockManager,
        waiting: Callable[[str], AbstractContextManager[None]] | None = None,
    ) -> None:
        self.inner = inner
        # Around each wait for a lock: the kit's close cuts a call waiting
        # for its room's lock at once (RFC §12.10.4).
        self._waiting = waiting or (lambda _room_id: nullcontext())

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        lock = self.inner.locked(room_id)
        # A lock this context holds already, of any kind, is no wait: asked
        # of the manager that marks it, under the key it holds.
        manager, key = self.inner, room_id
        if isinstance(manager, _NulLedKeys):
            manager, key = manager.inner, f"{_OTHER_KIND}{room_id}"
        held = _has_room_lock(key, manager)
        with nullcontext() if held else self._waiting(room_id):
            await lock.__aenter__()
        try:
            yield
        except BaseException as exc:
            try:
                await lock.__aexit__(type(exc), exc, exc.__traceback__)
            except Exception as release_error:
                # A cancellation met while releasing is the caller's, not a
                # failed release: it is raised, as on the normal path.
                if release_error is not exc:
                    _log_failed_release(room_id, release_error)
            raise
        try:
            await lock.__aexit__(None, None, None)
        except Exception as release_error:
            _log_failed_release(room_id, release_error)
            failures = _release_failures.get()
            if failures is not None:
                failures.append(release_error)

    async def close(self) -> None:
        await self.inner.close()

    async def _close_bounded(self) -> None:
        """The inner manager's close as the kit's close runs it: bounded when
        it offers that (RFC §12.10.4), else its own."""
        await close_bounded(self.inner)


async def close_bounded(resource: Any) -> None:
    """Close *resource* (a store, a lock manager) as the kit's close does:
    through its bounded close when it offers one, which gives its
    connections still in use a moment, then terminates them; else through
    its own ``close()``, on its own budget (RFC §12.10.4)."""
    bounded = getattr(resource, "_close_bounded", None)
    await (bounded() if bounded is not None else resource.close())


def _log_failed_release(room_id: str, error: BaseException) -> None:
    logger.warning("Releasing the lock of room %s failed: %r", room_id, error)

"""PostgreSQL advisory-lock RoomLockManager for multi-process deployments.

An in-memory ``RoomLockManager`` only serializes rooms within a single process.
When several RoomKit processes share one PostgreSQL store (a load-balanced
deployment), room processing must be serialized *across* processes — otherwise
two workers can interleave on the same room (RFC §8.1 / §13.5). This manager
uses PostgreSQL **session** advisory locks for that.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from roomkit.core.locks import RoomLockManager, _held_rooms


def _advisory_key(room_id: str) -> int:
    """Map a room id to a signed 64-bit key for ``pg_advisory_lock``.

    Uses a stable hash (not Python's salted ``hash()``). Distinct rooms may
    collide onto the same key — harmless: they merely serialize together.
    """
    digest = hashlib.blake2b(room_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


class PostgresAdvisoryLockManager(RoomLockManager):
    """Cross-process per-room locking via PostgreSQL session advisory locks.

    A session-level advisory lock is bound to the connection that took it, held
    for the whole ``locked()`` block. This manager therefore MUST use a
    connection pool **separate** from the store's query pool: a held lock
    connection would otherwise starve the query the locked block itself needs,
    deadlocking. Size ``max_size`` for the number of rooms processed
    concurrently.

    Reentrant within an execution context (mirrors ``InMemoryLockManager``), so
    a tool handler that re-enters the pipeline under the same room lock does not
    deadlock.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool: Any = None,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        try:
            import asyncpg as _asyncpg
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required for PostgresAdvisoryLockManager. "
                "Install it with: pip install roomkit[postgres]"
            ) from exc
        self._asyncpg = _asyncpg
        self._dsn = dsn
        self._pool = pool
        self._owns_pool = pool is None
        self._min_size = min_size
        self._max_size = max_size

    async def init(self) -> None:
        """Create the (separate) advisory-lock connection pool if needed."""
        if self._pool is None:
            self._pool = await self._asyncpg.create_pool(
                self._dsn, min_size=self._min_size, max_size=self._max_size
            )

    async def close(self) -> None:
        """Release the connection pool if we own it. Idempotent."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncIterator[None]:
        """Hold a cross-process exclusive lock for *room_id* (reentrant)."""
        held = _held_rooms.get()
        if room_id in held:
            # Reentrant: this execution context already holds the lock.
            yield
            return
        if self._pool is None:
            raise RuntimeError(
                "PostgresAdvisoryLockManager.init() must be called before use"
            )
        key = _advisory_key(room_id)
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT pg_advisory_lock($1)", key)
            token = _held_rooms.set(held | frozenset({room_id}))
            try:
                yield
            finally:
                _held_rooms.reset(token)
                await conn.execute("SELECT pg_advisory_unlock($1)", key)

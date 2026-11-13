"""Tests for PostgresAdvisoryLockManager, the UNIQUE index, and the
multiprocess-safety warning (H1b)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from roomkit.core.locks import InMemoryLockManager
from roomkit.store.postgres_lock import PostgresAdvisoryLockManager, _advisory_key
from roomkit.store.sqlite import SQLiteStore


class _FakeConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, sql: str, *args: object) -> None:
        self.calls.append((sql, args))


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self.closed = False

    def acquire(self):
        conn = self._conn

        @asynccontextmanager
        async def _cm():
            yield conn

        return _cm()

    async def close(self) -> None:
        self.closed = True


# ── Advisory key ────────────────────────────────────────────────


def test_advisory_key_is_stable_signed_int64() -> None:
    assert _advisory_key("room-1") == _advisory_key("room-1")
    assert -(2**63) <= _advisory_key("room-1") < 2**63
    assert _advisory_key("room-1") != _advisory_key("room-2")


# ── Locking ─────────────────────────────────────────────────────


async def test_locked_takes_and_releases_advisory_lock() -> None:
    conn = _FakeConn()
    mgr = PostgresAdvisoryLockManager(pool=_FakePool(conn))
    async with mgr.locked("room-1"):
        pass
    key = _advisory_key("room-1")
    assert ("SELECT pg_advisory_lock($1)", (key,)) in conn.calls
    assert ("SELECT pg_advisory_unlock($1)", (key,)) in conn.calls


async def test_locked_is_reentrant_for_same_room() -> None:
    conn = _FakeConn()
    mgr = PostgresAdvisoryLockManager(pool=_FakePool(conn))
    async with mgr.locked("room-1"), mgr.locked("room-1"):  # reentrant
        pass
    acquisitions = [c for c in conn.calls if "pg_advisory_lock" in c[0]]
    assert len(acquisitions) == 1


async def test_locked_raises_without_init() -> None:
    mgr = PostgresAdvisoryLockManager(dsn="postgres://x")
    with pytest.raises(RuntimeError, match="init"):
        async with mgr.locked("room-1"):
            pass


async def test_close_skips_external_pool() -> None:
    pool = _FakePool(_FakeConn())
    mgr = PostgresAdvisoryLockManager(pool=pool)  # external pool → not owned
    await mgr.close()
    assert pool.closed is False


# ── Multiprocess-safety warning ─────────────────────────────────


@pytest.mark.parametrize("lock_manager", [InMemoryLockManager, None], ids=["given", "default"])
@pytest.mark.parametrize("store", ["PostgresStore", "SQLiteStore"])
def test_warns_persistent_store_with_in_memory_lock(
    store: str,
    lock_manager: type[InMemoryLockManager] | None,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A store other processes may share, PostgreSQL's or an SQLite file,
    paired with an in-process lock, given or by default, is told at the
    kit's construction (RFC §13.5). No connection is opened."""
    from roomkit import RoomKit
    from roomkit.store.postgres import PostgresStore

    shared = (
        PostgresStore(pool=MagicMock())
        if store == "PostgresStore"
        else SQLiteStore(tmp_path / "rooms.db")
    )
    with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
        RoomKit(store=shared, lock_manager=lock_manager() if lock_manager else None)
    told = f"{store} is paired with InMemoryLockManager"
    assert [r for r in caplog.records if r.getMessage().startswith(told)] != []


def test_no_warning_for_in_memory_store(caplog: pytest.LogCaptureFixture) -> None:
    from roomkit import RoomKit

    with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
        RoomKit()  # default InMemoryStore + InMemoryLockManager
    assert not any("InMemoryLockManager" in r.message for r in caplog.records)


async def test_child_reacquires_after_parent_releases() -> None:
    import asyncio

    conn = _FakeConn()
    mgr = PostgresAdvisoryLockManager(pool=_FakePool(conn))
    start = asyncio.Event()

    async def child() -> None:
        await start.wait()
        async with mgr.locked("r1"):
            pass

    async with mgr.locked("r1"):
        task = asyncio.create_task(child())
    start.set()
    await asyncio.wait_for(task, 1)
    assert sum("pg_advisory_lock(" in sql for sql, _ in conn.calls) == 2


async def test_advisory_lock_is_not_bypassed_by_another_manager() -> None:
    conn = _FakeConn()
    mgr = PostgresAdvisoryLockManager(pool=_FakePool(conn))
    async with InMemoryLockManager().locked("r1"), mgr.locked("r1"):
        assert sum("pg_advisory_lock(" in sql for sql, _ in conn.calls) == 1

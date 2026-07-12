"""Tests for PostgresAdvisoryLockManager, the UNIQUE index, and the
multiprocess-safety warning (H1b)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from roomkit.core.locks import InMemoryLockManager
from roomkit.store.postgres_lock import PostgresAdvisoryLockManager, _advisory_key


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


# ── Schema: UNIQUE(room_id, index) ──────────────────────────────


def test_events_index_is_unique_in_schema() -> None:
    from roomkit.store import postgres_schema

    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_room_index" in postgres_schema.SCHEMA
    assert "Upgraded idx_events_room_index to UNIQUE" in postgres_schema.SCHEMA


# ── Multiprocess-safety warning ─────────────────────────────────


def test_warns_persistent_store_with_in_memory_lock(caplog: pytest.LogCaptureFixture) -> None:
    from roomkit import RoomKit
    from roomkit.store.postgres import PostgresStore

    pg = PostgresStore(pool=MagicMock())
    with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
        RoomKit(store=pg, lock_manager=InMemoryLockManager())
    assert any("InMemoryLockManager" in r.message for r in caplog.records)


def test_no_warning_for_in_memory_store(caplog: pytest.LogCaptureFixture) -> None:
    from roomkit import RoomKit

    with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
        RoomKit()  # default InMemoryStore + InMemoryLockManager
    assert not any("InMemoryLockManager" in r.message for r in caplog.records)

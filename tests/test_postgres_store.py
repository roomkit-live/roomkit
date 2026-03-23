"""Tests for PostgresStore (store/postgres.py)."""

from __future__ import annotations

import importlib
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import (
    ChannelType,
    EventType,
    RoomStatus,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.task import Observation, Task


def _build_mock_asyncpg() -> MagicMock:
    """Build a mock asyncpg module."""
    asyncpg = MagicMock()
    asyncpg.create_pool = AsyncMock()
    return asyncpg


def _make_mock_conn() -> AsyncMock:
    """Build a mock asyncpg connection with all query methods."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=None)
    # transaction() returns an async context manager
    conn.transaction = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    return conn


def _make_store_with_pool():
    """Create a PostgresStore with a mocked pool and connection.

    Returns (store, mock_conn) so tests can inspect SQL calls.
    """
    mock_asyncpg = _build_mock_asyncpg()
    mock_conn = _make_mock_conn()

    # pool.acquire() must be an async context manager yielding the connection
    mock_pool = MagicMock()

    @asynccontextmanager
    async def _acquire(timeout=5.0):
        yield mock_conn

    mock_pool.acquire = _acquire
    mock_pool.close = AsyncMock()

    with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
        importlib.invalidate_caches()
        mod = importlib.import_module("roomkit.store.postgres")
        importlib.reload(mod)
        store = mod.PostgresStore(pool=mock_pool)
    return store, mock_conn


def _make_room(room_id: str = "room-1") -> Room:
    return Room(id=room_id, organization_id="org-1", status=RoomStatus.ACTIVE)


def _make_event(room_id: str = "room-1", event_id: str = "evt-1") -> RoomEvent:
    return RoomEvent(
        id=event_id,
        room_id=room_id,
        type=EventType.MESSAGE,
        source=EventSource(channel_id="ch-1", channel_type=ChannelType.SMS),
        content=TextContent(body="hello"),
    )


def _make_binding(room_id: str = "room-1", channel_id: str = "ch-1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=ChannelType.SMS,
    )


def _make_participant(room_id: str = "room-1", pid: str = "p-1") -> Participant:
    return Participant(id=pid, room_id=room_id, channel_id="ch-1")


def _make_identity(identity_id: str = "id-1") -> Identity:
    return Identity(
        id=identity_id,
        display_name="Test User",
        channel_addresses={"sms": ["+1234567890"]},
    )


def _make_task(room_id: str = "room-1", task_id: str = "task-1") -> Task:
    return Task(id=task_id, room_id=room_id, title="Do something")


def _make_observation(room_id: str = "room-1", obs_id: str = "obs-1") -> Observation:
    return Observation(id=obs_id, room_id=room_id, channel_id="ch-1", content="Noticed something")


class TestPostgresStore:
    def test_constructor_with_dsn(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            store = mod.PostgresStore(dsn="postgres://localhost/test")
            assert store._dsn == "postgres://localhost/test"
            assert store._pool is None
            assert store._owns_pool is True

    def test_constructor_with_pool(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        mock_pool = MagicMock()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            store = mod.PostgresStore(pool=mock_pool)
            assert store._pool is mock_pool
            assert store._owns_pool is False

    def test_constructor_without_asyncpg_raises(self) -> None:
        # Replace asyncpg in sys.modules with a sentinel that triggers ImportError
        import builtins

        real_import = builtins.__import__

        def _block_asyncpg(name: str, *args: object, **kwargs: object) -> object:
            if name == "asyncpg":
                raise ImportError("No module named 'asyncpg'")
            return real_import(name, *args, **kwargs)

        saved = sys.modules.pop("asyncpg", None)
        try:
            with patch.object(builtins, "__import__", side_effect=_block_asyncpg):
                mod = importlib.import_module("roomkit.store.postgres")
                importlib.reload(mod)
                with pytest.raises(ImportError, match="asyncpg"):
                    mod.PostgresStore(dsn="postgres://localhost/test")
        finally:
            if saved is not None:
                sys.modules["asyncpg"] = saved

    async def test_close_releases_pool(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        mock_pool = AsyncMock()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            # _owns_pool=True so close() should release the pool
            store = mod.PostgresStore(dsn="postgres://localhost/test")
            store._pool = mock_pool
            store._owns_pool = True
            await store.close()
            mock_pool.close.assert_awaited_once()
            assert store._pool is None

    async def test_close_skips_external_pool(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        mock_pool = AsyncMock()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            store = mod.PostgresStore(pool=mock_pool)
            await store.close()
            # External pool should NOT be closed
            mock_pool.close.assert_not_awaited()

    def test_ensure_pool_raises_without_init(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            store = mod.PostgresStore(dsn="postgres://localhost/test")
            with pytest.raises(RuntimeError, match="init"):
                store._ensure_pool()

    # ── Init ────────────────────────────────────────────────────

    async def test_init_creates_pool_and_schema(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        mock_conn = _make_mock_conn()
        mock_pool = MagicMock()

        @asynccontextmanager
        async def _acquire(timeout=5.0):
            yield mock_conn

        mock_pool.acquire = _acquire
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            store = mod.PostgresStore(dsn="postgres://localhost/test")
            await store.init(min_size=1, max_size=5)

            mock_asyncpg.create_pool.assert_awaited_once()
            # Schema SQL was executed
            mock_conn.execute.assert_awaited_once()

    async def test_init_with_existing_pool_only_runs_schema(self) -> None:
        store, mock_conn = _make_store_with_pool()
        # init() should NOT create a new pool, just run schema
        await store.init()
        mock_conn.execute.assert_awaited_once()

    # ── Context manager ─────────────────────────────────────────

    async def test_async_context_manager(self) -> None:
        mock_asyncpg = _build_mock_asyncpg()
        mock_conn = _make_mock_conn()
        mock_pool = MagicMock()

        @asynccontextmanager
        async def _acquire(timeout=5.0):
            yield mock_conn

        mock_pool.acquire = _acquire
        mock_pool.close = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.store.postgres")
            importlib.reload(mod)
            async with mod.PostgresStore(dsn="postgres://localhost/test") as store:
                assert store._pool is not None
            # After exit, pool should be closed
            mock_pool.close.assert_awaited_once()

    # ── Room CRUD ───────────────────────────────────────────────

    async def test_create_room(self) -> None:
        store, mock_conn = _make_store_with_pool()
        room = _make_room()
        result = await store.create_room(room)
        assert result.id == room.id
        mock_conn.execute.assert_awaited_once()
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO rooms" in call_args[0][0]
        assert call_args[0][1] == room.id

    async def test_get_room_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        room = _make_room()
        mock_conn.fetchrow.return_value = {"data": room.model_dump_json()}
        result = await store.get_room("room-1")
        assert result is not None
        assert result.id == "room-1"

    async def test_get_room_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.get_room("nonexistent")
        assert result is None

    async def test_update_room(self) -> None:
        store, mock_conn = _make_store_with_pool()
        room = _make_room()
        result = await store.update_room(room)
        assert result.id == room.id
        call_args = mock_conn.execute.call_args
        assert "UPDATE rooms" in call_args[0][0]

    async def test_delete_room_success(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.execute.return_value = "DELETE 1"
        result = await store.delete_room("room-1")
        assert result is True

    async def test_delete_room_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.execute.return_value = "DELETE 0"
        result = await store.delete_room("nonexistent")
        assert result is False

    async def test_list_rooms(self) -> None:
        store, mock_conn = _make_store_with_pool()
        room = _make_room()
        mock_conn.fetch.return_value = [{"data": room.model_dump_json()}]
        result = await store.list_rooms(offset=0, limit=10)
        assert len(result) == 1
        assert result[0].id == "room-1"

    async def test_list_rooms_empty(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        result = await store.list_rooms()
        assert result == []

    # ── find_rooms ──────────────────────────────────────────────

    async def test_find_rooms_no_filters(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        result = await store.find_rooms()
        assert result == []
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "WHERE" not in sql or "WHERE " not in sql.split("ORDER")[0]

    async def test_find_rooms_with_org_id(self) -> None:
        store, mock_conn = _make_store_with_pool()
        room = _make_room()
        mock_conn.fetch.return_value = [{"data": room.model_dump_json()}]
        result = await store.find_rooms(organization_id="org-1")
        assert len(result) == 1
        call_args = mock_conn.fetch.call_args
        assert "organization_id" in call_args[0][0]

    async def test_find_rooms_with_status(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        await store.find_rooms(status="active")
        call_args = mock_conn.fetch.call_args
        assert "status" in call_args[0][0]

    async def test_find_rooms_with_metadata_filter(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        await store.find_rooms(metadata_filter={"key": "value"})
        call_args = mock_conn.fetch.call_args
        assert "metadata" in call_args[0][0]

    async def test_find_rooms_with_all_filters(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        await store.find_rooms(
            organization_id="org-1",
            status="active",
            metadata_filter={"tag": "vip"},
            limit=25,
            offset=5,
        )
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "organization_id" in sql
        assert "status" in sql
        assert "metadata" in sql

    async def test_find_rooms_with_enum_status(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        await store.find_rooms(status=RoomStatus.ACTIVE)
        call_args = mock_conn.fetch.call_args
        # The status value should have been extracted via .value
        assert "active" in call_args[0] or RoomStatus.ACTIVE in call_args[0]

    # ── find_latest_room ────────────────────────────────────────

    async def test_find_latest_room_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        room = _make_room()
        mock_conn.fetchrow.return_value = {"data": room.model_dump_json()}
        result = await store.find_latest_room("p-1")
        assert result is not None
        assert result.id == "room-1"

    async def test_find_latest_room_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.find_latest_room("p-1")
        assert result is None

    async def test_find_latest_room_with_channel_type_and_status(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        await store.find_latest_room("p-1", channel_type=ChannelType.SMS, status=RoomStatus.ACTIVE)
        call_args = mock_conn.fetchrow.call_args
        sql = call_args[0][0]
        assert "channel_type" in sql or "b.channel_type" in sql
        assert "r.status" in sql

    # ── find_room_id_by_channel ─────────────────────────────────

    async def test_find_room_id_by_channel_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = {"room_id": "room-1"}
        result = await store.find_room_id_by_channel("ch-1")
        assert result == "room-1"

    async def test_find_room_id_by_channel_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.find_room_id_by_channel("ch-missing")
        assert result is None

    async def test_find_room_id_by_channel_with_status(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = {"room_id": "room-1"}
        result = await store.find_room_id_by_channel("ch-1", status="active")
        assert result == "room-1"
        call_args = mock_conn.fetchrow.call_args
        assert "r.status" in call_args[0][0]

    async def test_find_room_id_by_channel_with_enum_status(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        await store.find_room_id_by_channel("ch-1", status=RoomStatus.ACTIVE)
        call_args = mock_conn.fetchrow.call_args
        assert "r.status" in call_args[0][0]

    # ── Event operations ────────────────────────────────────────

    async def test_add_event(self) -> None:
        store, mock_conn = _make_store_with_pool()
        event = _make_event()
        result = await store.add_event(event)
        assert result.id == event.id
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO events" in call_args[0][0]

    async def test_get_event_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        event = _make_event()
        mock_conn.fetchrow.return_value = {"data": event.model_dump_json()}
        result = await store.get_event("evt-1")
        assert result is not None
        assert result.id == "evt-1"

    async def test_get_event_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.get_event("nonexistent")
        assert result is None

    async def test_update_event(self) -> None:
        store, mock_conn = _make_store_with_pool()
        event = _make_event()
        result = await store.update_event(event)
        assert result.id == event.id
        call_args = mock_conn.execute.call_args
        assert "UPDATE events" in call_args[0][0]

    async def test_list_events_basic(self) -> None:
        store, mock_conn = _make_store_with_pool()
        event = _make_event()
        mock_conn.fetch.return_value = [{"data": event.model_dump_json()}]
        result = await store.list_events("room-1")
        assert len(result) == 1
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "OFFSET" in sql  # default pagination

    async def test_list_events_with_visibility_filter(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        await store.list_events("room-1", visibility_filter="all")
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "visibility" in sql

    async def test_list_events_with_after_index(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        await store.list_events("room-1", after_index=5)
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "index" in sql
        assert "OFFSET" not in sql  # cursor mode has no OFFSET

    async def test_list_events_with_before_index(self) -> None:
        store, mock_conn = _make_store_with_pool()
        event = _make_event()
        mock_conn.fetch.return_value = [{"data": event.model_dump_json()}]
        result = await store.list_events("room-1", before_index=10)
        assert len(result) == 1
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "DESC" in sql  # before_index uses descending order

    async def test_list_events_mutually_exclusive_cursors(self) -> None:
        store, _ = _make_store_with_pool()
        with pytest.raises(ValueError, match="mutually exclusive"):
            await store.list_events("room-1", after_index=5, before_index=10)

    async def test_check_idempotency_exists(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = {"?column?": 1}
        result = await store.check_idempotency("room-1", "key-1")
        assert result is True

    async def test_check_idempotency_not_exists(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.check_idempotency("room-1", "key-missing")
        assert result is False

    async def test_get_event_count(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = {"cnt": 42}
        result = await store.get_event_count("room-1")
        assert result == 42

    async def test_get_event_count_no_row(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.get_event_count("room-1")
        assert result == 0

    async def test_add_event_auto_index(self) -> None:
        store, mock_conn = _make_store_with_pool()
        event = _make_event()
        # First fetchrow: lock row (FOR UPDATE)
        # Second fetchrow: get next index
        mock_conn.fetchrow.side_effect = [
            {"id": "room-1"},  # SELECT FOR UPDATE
            {"next_idx": 3},  # COALESCE(MAX(...))
        ]
        result = await store.add_event_auto_index("room-1", event)
        assert result.index == 3
        # execute should have been called for the INSERT
        mock_conn.execute.assert_awaited_once()
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO events" in call_args[0][0]

    # ── Binding operations ──────────────────────────────────────

    async def test_add_binding(self) -> None:
        store, mock_conn = _make_store_with_pool()
        binding = _make_binding()
        result = await store.add_binding(binding)
        assert result.channel_id == "ch-1"
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO bindings" in call_args[0][0]
        assert "ON CONFLICT" in call_args[0][0]

    async def test_get_binding_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        binding = _make_binding()
        mock_conn.fetchrow.return_value = {"data": binding.model_dump_json()}
        result = await store.get_binding("room-1", "ch-1")
        assert result is not None
        assert result.channel_id == "ch-1"

    async def test_get_binding_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.get_binding("room-1", "ch-missing")
        assert result is None

    async def test_update_binding(self) -> None:
        store, mock_conn = _make_store_with_pool()
        binding = _make_binding()
        result = await store.update_binding(binding)
        assert result.channel_id == binding.channel_id
        call_args = mock_conn.execute.call_args
        assert "UPDATE bindings" in call_args[0][0]

    async def test_remove_binding_success(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.execute.return_value = "DELETE 1"
        result = await store.remove_binding("room-1", "ch-1")
        assert result is True

    async def test_remove_binding_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.execute.return_value = "DELETE 0"
        result = await store.remove_binding("room-1", "ch-missing")
        assert result is False

    async def test_list_bindings(self) -> None:
        store, mock_conn = _make_store_with_pool()
        binding = _make_binding()
        mock_conn.fetch.return_value = [{"data": binding.model_dump_json()}]
        result = await store.list_bindings("room-1")
        assert len(result) == 1
        assert result[0].channel_id == "ch-1"

    async def test_list_bindings_empty(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        result = await store.list_bindings("room-1")
        assert result == []

    # ── Participant operations ──────────────────────────────────

    async def test_add_participant(self) -> None:
        store, mock_conn = _make_store_with_pool()
        participant = _make_participant()
        result = await store.add_participant(participant)
        assert result.id == "p-1"
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO participants" in call_args[0][0]

    async def test_get_participant_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        participant = _make_participant()
        mock_conn.fetchrow.return_value = {"data": participant.model_dump_json()}
        result = await store.get_participant("room-1", "p-1")
        assert result is not None
        assert result.id == "p-1"

    async def test_get_participant_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.get_participant("room-1", "p-missing")
        assert result is None

    async def test_update_participant(self) -> None:
        store, mock_conn = _make_store_with_pool()
        participant = _make_participant()
        result = await store.update_participant(participant)
        assert result.id == "p-1"
        call_args = mock_conn.execute.call_args
        assert "UPDATE participants" in call_args[0][0]

    async def test_list_participants(self) -> None:
        store, mock_conn = _make_store_with_pool()
        participant = _make_participant()
        mock_conn.fetch.return_value = [{"data": participant.model_dump_json()}]
        result = await store.list_participants("room-1")
        assert len(result) == 1
        assert result[0].id == "p-1"

    # ── Read tracking ───────────────────────────────────────────

    async def test_mark_read(self) -> None:
        store, mock_conn = _make_store_with_pool()
        await store.mark_read("room-1", "ch-1", "evt-5")
        call_args = mock_conn.execute.call_args
        assert "read_markers" in call_args[0][0]

    async def test_mark_all_read_with_events(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = {"id": "evt-10"}
        await store.mark_all_read("room-1", "ch-1")
        # Should have called fetchrow (get latest event) + execute (upsert marker)
        assert mock_conn.fetchrow.await_count >= 1
        assert mock_conn.execute.await_count >= 1

    async def test_mark_all_read_no_events(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        await store.mark_all_read("room-1", "ch-1")
        # No execute call because no events exist
        mock_conn.execute.assert_not_awaited()

    async def test_get_unread_count_no_marker(self) -> None:
        store, mock_conn = _make_store_with_pool()
        # First fetchrow: no marker
        # Second fetchrow: total count
        mock_conn.fetchrow.side_effect = [None, {"cnt": 7}]
        result = await store.get_unread_count("room-1", "ch-1")
        assert result == 7

    async def test_get_unread_count_with_marker_and_index(self) -> None:
        store, mock_conn = _make_store_with_pool()
        # First fetchrow: marker exists
        # Second fetchrow: last_read_event index
        # Third fetchrow: unread count
        mock_conn.fetchrow.side_effect = [
            {"event_id": "evt-5"},
            {"idx": 5},
            {"cnt": 3},
        ]
        result = await store.get_unread_count("room-1", "ch-1")
        assert result == 3

    async def test_get_unread_count_marker_event_missing(self) -> None:
        store, mock_conn = _make_store_with_pool()
        # Marker exists but event has no index
        mock_conn.fetchrow.side_effect = [
            {"event_id": "evt-5"},
            None,
            {"cnt": 10},
        ]
        result = await store.get_unread_count("room-1", "ch-1")
        assert result == 10

    async def test_get_unread_count_marker_event_null_index(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.side_effect = [
            {"event_id": "evt-5"},
            {"idx": None},
            {"cnt": 8},
        ]
        result = await store.get_unread_count("room-1", "ch-1")
        assert result == 8

    # ── Identity operations ─────────────────────────────────────

    async def test_create_identity(self) -> None:
        store, mock_conn = _make_store_with_pool()
        identity = _make_identity()
        result = await store.create_identity(identity)
        assert result.id == "id-1"
        # Should have executed: INSERT identities + INSERT identity_addresses
        assert mock_conn.execute.await_count >= 2

    async def test_create_identity_no_addresses(self) -> None:
        store, mock_conn = _make_store_with_pool()
        identity = Identity(id="id-2", channel_addresses={})
        result = await store.create_identity(identity)
        assert result.id == "id-2"
        # Only the identities INSERT, no address inserts
        assert mock_conn.execute.await_count == 1

    async def test_get_identity_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        identity = _make_identity()
        mock_conn.fetchrow.return_value = {"data": identity.model_dump_json()}
        result = await store.get_identity("id-1")
        assert result is not None
        assert result.id == "id-1"

    async def test_get_identity_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.get_identity("id-missing")
        assert result is None

    async def test_resolve_identity_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        identity = _make_identity()
        mock_conn.fetchrow.return_value = {"data": identity.model_dump_json()}
        result = await store.resolve_identity("sms", "+1234567890")
        assert result is not None
        assert result.id == "id-1"

    async def test_resolve_identity_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.resolve_identity("sms", "+0000000000")
        assert result is None

    async def test_link_address_new_address(self) -> None:
        store, mock_conn = _make_store_with_pool()
        identity = _make_identity()
        mock_conn.fetchrow.return_value = {"data": identity.model_dump_json()}
        await store.link_address("id-1", "email", "user@example.com")
        # Should have: fetchrow (get identity) + execute (update identities) +
        # execute (upsert identity_addresses)
        assert mock_conn.execute.await_count >= 2

    async def test_link_address_existing_address(self) -> None:
        store, mock_conn = _make_store_with_pool()
        identity = _make_identity()
        mock_conn.fetchrow.return_value = {"data": identity.model_dump_json()}
        # Link an address that already exists in the identity
        await store.link_address("id-1", "sms", "+1234567890")
        # Should still upsert the address row, but NOT update identity data
        # (because the address is already in the list)
        # 1 execute for the INSERT INTO identity_addresses only
        assert mock_conn.execute.await_count >= 1

    async def test_link_address_identity_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        await store.link_address("id-missing", "sms", "+1111111111")
        # Should return early without executing anything
        mock_conn.execute.assert_not_awaited()

    # ── Task operations ─────────────────────────────────────────

    async def test_add_task(self) -> None:
        store, mock_conn = _make_store_with_pool()
        task = _make_task()
        result = await store.add_task(task)
        assert result.id == "task-1"
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO tasks" in call_args[0][0]

    async def test_get_task_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        task = _make_task()
        mock_conn.fetchrow.return_value = {"data": task.model_dump_json()}
        result = await store.get_task("task-1")
        assert result is not None
        assert result.id == "task-1"

    async def test_get_task_not_found(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetchrow.return_value = None
        result = await store.get_task("task-missing")
        assert result is None

    async def test_list_tasks_no_status(self) -> None:
        store, mock_conn = _make_store_with_pool()
        task = _make_task()
        mock_conn.fetch.return_value = [{"data": task.model_dump_json()}]
        result = await store.list_tasks("room-1")
        assert len(result) == 1

    async def test_list_tasks_with_status(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        await store.list_tasks("room-1", status="pending")
        call_args = mock_conn.fetch.call_args
        assert "status" in call_args[0][0]

    async def test_update_task(self) -> None:
        store, mock_conn = _make_store_with_pool()
        task = _make_task()
        result = await store.update_task(task)
        assert result.id == "task-1"
        call_args = mock_conn.execute.call_args
        assert "UPDATE tasks" in call_args[0][0]

    # ── Observation operations ──────────────────────────────────

    async def test_add_observation(self) -> None:
        store, mock_conn = _make_store_with_pool()
        obs = _make_observation()
        result = await store.add_observation(obs)
        assert result.id == "obs-1"
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO observations" in call_args[0][0]

    async def test_list_observations(self) -> None:
        store, mock_conn = _make_store_with_pool()
        obs = _make_observation()
        mock_conn.fetch.return_value = [{"data": obs.model_dump_json()}]
        result = await store.list_observations("room-1")
        assert len(result) == 1
        assert result[0].id == "obs-1"

    async def test_list_observations_empty(self) -> None:
        store, mock_conn = _make_store_with_pool()
        mock_conn.fetch.return_value = []
        result = await store.list_observations("room-1")
        assert result == []

    # ── Telemetry span ──────────────────────────────────────────

    async def test_query_span_on_success(self) -> None:
        store, _ = _make_store_with_pool()
        # _query_span should not raise on success
        with store._query_span("test_op", "test_table"):
            pass  # no exception

    async def test_query_span_on_error(self) -> None:
        store, _ = _make_store_with_pool()
        with (
            pytest.raises(ValueError, match="boom"),
            store._query_span("test_op", "test_table"),
        ):
            raise ValueError("boom")

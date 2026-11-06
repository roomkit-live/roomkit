"""Tests for PostgresStore.

These tests require a running PostgreSQL instance. Set POSTGRES_DSN to run them:

    POSTGRES_DSN=postgresql://user:pass@localhost/roomkit_test \
    pytest tests/test_store_postgres.py -v
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import struct
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from uuid import uuid4

import pytest

from roomkit import AddressLinkedError, EventNotFoundError, IdentityNotFoundError
from roomkit.core.mixins.verification import HELD_KEY
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType, EventType, RoomStatus
from roomkit.models.event import EditContent, TextContent
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.task import Observation, Task
from roomkit.store.base import marked_deleted
from tests.conftest import PoolCheckoutRecorder

POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(POSTGRES_DSN is None, reason="POSTGRES_DSN not set"),
    pytest.mark.asyncio,
]


@pytest.fixture
async def store():
    from roomkit.store.postgres import PostgresStore

    _index_counters.clear()
    s = PostgresStore(dsn=POSTGRES_DSN)
    await s.init(min_size=1, max_size=3)
    # Clean tables before each test
    async with s._pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE rooms, events, bindings, participants, "
            "identities, identity_addresses, tasks, observations, read_markers CASCADE"
        )
    yield s
    await s.close()


# Per-room monotonic index, mirroring what the framework assigns — so a test
# that adds several events via ``add_event`` gets distinct indices and does not
# trip the UNIQUE(room_id, index) constraint. Reset per test by the fixture.
_index_counters: dict[str, int] = {}


def _make_event(room_id: str = "r1", body: str = "hello", **kwargs):
    from roomkit.models.event import EventSource, RoomEvent, TextContent

    if "index" not in kwargs:
        kwargs["index"] = _index_counters.get(room_id, 0)
        _index_counters[room_id] = kwargs["index"] + 1
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
        content=TextContent(body=body),
        **kwargs,
    )


async def test_a_store_with_no_dsn_reads_the_pg_variables(monkeypatch) -> None:
    """HITL 78cd4240: with ``dsn=None`` the connection comes from the ``PG*`` environment
    variables, ``PGPASSWORD`` included, as asyncpg reads them."""
    from roomkit.store.postgres import PostgresStore

    assert POSTGRES_DSN is not None
    url = urlparse(POSTGRES_DSN)
    host = url.hostname or parse_qs(url.query).get("host", [""])[0]
    for name, value in {
        "PGHOST": host,
        "PGPORT": str(url.port or 5432),
        "PGUSER": unquote(url.username or ""),
        "PGPASSWORD": unquote(url.password or ""),
        "PGDATABASE": url.path.lstrip("/"),
    }.items():
        monkeypatch.setenv(name, value)
    store = PostgresStore()
    await store.init(min_size=1, max_size=1)
    try:
        assert await store.room_exists("nobody's") is False
        async with store._pool.acquire() as conn:
            assert await conn.fetchval("SELECT current_database()") == url.path.lstrip("/")
    finally:
        await store.close()


# asyncpg's defaults for what a connection string leaves out (HITL fa9dd934), each
# test named as its Rust twin in roomkit-rs crates/roomkit-postgres/tests/defaults.rs.
# A directory holding the server's Unix socket, whose connections ask no password
# where TCP asks one: those tests skip themselves without it.
POSTGRES_SOCKET_DIR = os.environ.get("POSTGRES_SOCKET_DIR")


def _connection_parts() -> dict[str, str]:
    """``POSTGRES_DSN``'s host, port, user, password, database and query."""
    assert POSTGRES_DSN is not None
    url = urlparse(POSTGRES_DSN)
    return {
        "host": url.hostname or parse_qs(url.query).get("host", [""])[0],
        "port": str(url.port or 5432),
        "user": unquote(url.username or ""),
        "password": unquote(url.password or ""),
        "database": url.path.lstrip("/"),
        "query": f"?{url.query}" if url.query else "",
    }


def _without_pg_variables(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """No ``PG*`` variable, and ``HOME`` a directory of the test's: no test reads the
    real ``~/.pgpass``."""
    for name in list(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name)
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))


async def _skip_unless_tcp_asks_a_password(parts: dict[str, str]) -> None:
    """Skips the test on a server that lets a TCP connection in with no password: a
    connection there would prove nothing of where a password came from."""
    import asyncpg

    try:
        conn = await asyncpg.connect(
            host=parts["host"],
            port=int(parts["port"]),
            user=parts["user"],
            database=parts["database"],
        )
    except asyncpg.InvalidPasswordError:
        return
    await conn.close()
    pytest.skip("the server asks no password over TCP")


def _dead_port() -> int:
    """A local TCP port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def _connects(dsn: str | None) -> bool:
    """Whether a store on *dsn* opens, and whether its connection is a Unix socket's."""
    from roomkit.store.postgres import PostgresStore

    store = PostgresStore(dsn)
    await store.init(min_size=1, max_size=1)
    try:
        assert await store.room_exists("nobody's") is False
        async with store._pool.acquire() as conn:
            return await conn.fetchval("SELECT inet_client_addr()") is None
    finally:
        await store.close()


@pytest.mark.parametrize("named", [False, True], ids=["no-host", "pghost"])
async def test_a_store_with_no_host_tries_the_local_sockets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, named: bool
) -> None:
    """With no host in the string nor ``PGHOST``, asyncpg tries the sockets of
    ``/run/postgresql``, ``/var/run/postgresql``, ``/tmp`` and ``/private/tmp`` before
    ``localhost``: the server's socket, linked into ``/tmp``, lets the store in with no
    password. ``PGHOST`` naming the socket's directory does too."""
    if POSTGRES_SOCKET_DIR is None:
        pytest.skip("POSTGRES_SOCKET_DIR not set")
    parts = _connection_parts()
    socket_file = Path(POSTGRES_SOCKET_DIR) / f".s.PGSQL.{parts['port']}"
    if not socket_file.exists():
        pytest.skip(f"no socket {socket_file}")
    _without_pg_variables(monkeypatch, tmp_path / "home")
    await _skip_unless_tcp_asks_a_password(parts)
    monkeypatch.setenv("PGPORT", parts["port"])
    link = Path("/tmp") / socket_file.name
    if named:
        monkeypatch.setenv("PGHOST", POSTGRES_SOCKET_DIR)
    elif link.exists() or link.is_symlink():
        pytest.skip(f"{link} is not this test's")
    else:
        link.symlink_to(socket_file)
    try:
        assert await _connects(f"postgresql://{quote(parts['user'])}@/{parts['database']}")
    finally:
        if not named:
            link.unlink()


@pytest.mark.parametrize("where", ["home", "PGPASSFILE", "passfile"])
async def test_a_store_with_no_password_reads_the_password_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, where: str
) -> None:
    """With no password in the string nor ``PGPASSWORD``, asyncpg takes it from the first
    line of ``~/.pgpass`` that matches the host, port, database and user, ``*`` matching
    any: or from the file ``PGPASSFILE`` or the string's ``passfile`` names."""
    parts = _connection_parts()
    _without_pg_variables(monkeypatch, tmp_path / "home")
    await _skip_unless_tcp_asks_a_password(parts)
    passfile = tmp_path / "home" / ".pgpass" if where == "home" else tmp_path / "passwords"
    passfile.write_text(
        "# a comment, then a blank line\n\n"
        f"elsewhere:{parts['port']}:*:{parts['user']}:wrong\n"
        f"{parts['host']}:{parts['port']}:*:{parts['user']}:{parts['password']}\n"
    )
    passfile.chmod(0o600)
    query = parts["query"]
    if where == "PGPASSFILE":
        monkeypatch.setenv("PGPASSFILE", str(passfile))
    elif where == "passfile":
        query = f"{query}{'&' if query else '?'}passfile={quote(str(passfile))}"
    dsn = (
        f"postgresql://{quote(parts['user'])}@{parts['host']}:{parts['port']}/{parts['database']}"
    )
    assert await _connects(f"{dsn}{query}") is False


async def test_a_password_file_others_can_read_is_not_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """asyncpg ignores a password file with any access for the group or others, as
    libpq does, with a warning: the connection goes without a password, and the server
    refuses it."""
    import asyncpg

    parts = _connection_parts()
    _without_pg_variables(monkeypatch, tmp_path / "home")
    await _skip_unless_tcp_asks_a_password(parts)
    passfile = tmp_path / "home" / ".pgpass"
    passfile.write_text(f"*:*:*:*:{parts['password']}\n")
    passfile.chmod(0o640)
    dsn = (
        f"postgresql://{quote(parts['user'])}@{parts['host']}:{parts['port']}/{parts['database']}"
    )
    with (
        pytest.warns(UserWarning, match="has group or world access"),
        pytest.raises(asyncpg.InvalidPasswordError),
    ):
        await _connects(f"{dsn}{parts['query']}")


async def test_a_short_password_file_line_before_a_match_fails_init(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HITL 8142aaf2: asyncpg cuts each line of the file into five fields and raises on
    one with fewer, met before the line that matches: ``init`` fails, a ``ValueError``."""
    parts = _connection_parts()
    _without_pg_variables(monkeypatch, tmp_path / "home")
    passfile = tmp_path / "home" / ".pgpass"
    passfile.write_text(f"short:line\n*:*:*:*:{parts['password']}\n")
    passfile.chmod(0o600)
    dsn = (
        f"postgresql://{quote(parts['user'])}@{parts['host']}:{parts['port']}/{parts['database']}"
    )
    with pytest.raises(ValueError):
        await _connects(f"{dsn}{parts['query']}")


async def test_an_empty_pgpassword_is_a_password(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """asyncpg takes an empty ``PGPASSWORD`` for the password, and reads no password file
    then: a matching ``~/.pgpass`` is not read, and the server refuses the empty one."""
    import asyncpg

    parts = _connection_parts()
    _without_pg_variables(monkeypatch, tmp_path / "home")
    await _skip_unless_tcp_asks_a_password(parts)
    passfile = tmp_path / "home" / ".pgpass"
    passfile.write_text(f"*:*:*:*:{parts['password']}\n")
    passfile.chmod(0o600)
    monkeypatch.setenv("PGPASSWORD", "")
    dsn = (
        f"postgresql://{quote(parts['user'])}@{parts['host']}:{parts['port']}/{parts['database']}"
    )
    with pytest.raises(asyncpg.InvalidPasswordError):
        await _connects(f"{dsn}{parts['query']}")


@pytest.mark.parametrize(
    "setting",
    [
        "connect_timeout=3",
        "keepalives=1",
        "keepalives_idle=30",
        "keepalives_interval=10",
        "keepalives_retries=3",
        "channel_binding=prefer",
        "load_balance_hosts=random",
    ],
)
async def test_a_parameter_asyncpg_does_not_read_goes_to_the_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, setting: str
) -> None:
    """HITL 22f38d15 (C): asyncpg reads none of these parameters and passes each to the
    server as a setting, which refuses the connection."""
    import asyncpg

    assert POSTGRES_DSN is not None
    _without_pg_variables(monkeypatch, tmp_path / "home")
    dsn = f"{POSTGRES_DSN}{'&' if '?' in POSTGRES_DSN else '?'}{setting}"
    with pytest.raises(asyncpg.UndefinedObjectError, match="unrecognized configuration parameter"):
        await _connects(dsn)


async def test_a_tcp_user_timeout_is_the_server_s_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HITL 22f38d15 (C): asyncpg passes ``tcp_user_timeout`` to the server, which takes it
    for its own setting of the session."""
    from roomkit.store.postgres import PostgresStore

    assert POSTGRES_DSN is not None
    _without_pg_variables(monkeypatch, tmp_path / "home")
    store = PostgresStore(
        f"{POSTGRES_DSN}{'&' if '?' in POSTGRES_DSN else '?'}tcp_user_timeout=1000"
    )
    await store.init(min_size=1, max_size=1)
    try:
        async with store._pool.acquire() as conn:
            assert await conn.fetchval("SHOW tcp_user_timeout") == "1000"
    finally:
        await store.close()


@pytest.mark.parametrize("form", ["netloc", "query", "environment"])
async def test_a_host_list_goes_past_a_dead_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, form: str
) -> None:
    """A host list, in the string or in ``PGHOST`` and ``PGPORT``, is tried in its
    order: a host nothing listens on is passed for the next."""
    parts = _connection_parts()
    _without_pg_variables(monkeypatch, tmp_path / "home")
    dead = _dead_port()
    auth = f"{quote(parts['user'])}:{quote(parts['password'])}"
    hosts = f"127.0.0.1,{parts['host']}"
    ports = f"{dead},{parts['port']}"
    query = parts["query"]
    if form == "netloc":
        dsn = f"postgresql://{auth}@127.0.0.1:{dead},{parts['host']}:{parts['port']}/"
    elif form == "query":
        query = f"{query}{'&' if query else '?'}host={hosts}&port={ports}"
        dsn = f"postgresql://{auth}@/"
    else:
        monkeypatch.setenv("PGHOST", hosts)
        monkeypatch.setenv("PGPORT", ports)
        dsn = f"postgresql://{auth}@/"
    assert await _connects(f"{dsn}{parts['database']}{query}") is False


@contextlib.asynccontextmanager
async def _fake_server(behaviour: str) -> AsyncIterator[int]:
    """The port of a local server that answers each connection as *behaviour* says:
    ``refuses-tls`` answers ``N`` to the TLS request, ``resets`` resets the connection
    once the client wrote to it."""

    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if behaviour == "refuses-tls":
            await reader.readexactly(8)
            writer.write(b"N")
            await writer.drain()
            await reader.read()
        else:
            await reader.read(1)
            writer.get_extra_info("socket").setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()
        await server.wait_closed()


async def test_a_host_refusing_tls_is_passed_for_the_next(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A host that refuses the TLS ``sslmode=require`` asks for is one asyncpg cannot
    reach: it tries the next, here the server's socket, which asks no TLS."""
    if POSTGRES_SOCKET_DIR is None:
        pytest.skip("POSTGRES_SOCKET_DIR not set")
    parts = _connection_parts()
    _without_pg_variables(monkeypatch, tmp_path / "home")
    async with _fake_server("refuses-tls") as fake:
        dsn = (
            f"postgresql://{quote(parts['user'])}@/{parts['database']}"
            f"?host=127.0.0.1,{POSTGRES_SOCKET_DIR}&port={fake},{parts['port']}"
            "&sslmode=require"
        )
        assert await _connects(dsn)


@pytest.mark.parametrize("first", ["refuses", "resets"])
async def test_a_refusal_on_a_host_ends_the_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first: str
) -> None:
    """asyncpg goes to the next host only when a host cannot be reached: a server that
    refuses the connection, or a connection lost once open, ends the list, though the
    server's socket next in it would let the store in with no password."""
    import asyncpg

    if POSTGRES_SOCKET_DIR is None:
        pytest.skip("POSTGRES_SOCKET_DIR not set")
    parts = _connection_parts()
    _without_pg_variables(monkeypatch, tmp_path / "home")
    async with contextlib.AsyncExitStack() as stack:
        if first == "refuses":
            await _skip_unless_tcp_asks_a_password(parts)
            host, port, error = parts["host"], parts["port"], asyncpg.InvalidPasswordError
        else:
            fake = await stack.enter_async_context(_fake_server("resets"))
            host, port = "127.0.0.1", f"{fake},{parts['port']}"
            error = asyncpg.ConnectionDoesNotExistError
        monkeypatch.setenv("PGHOST", f"{host},{POSTGRES_SOCKET_DIR}")
        monkeypatch.setenv("PGPORT", port)
        dsn = f"postgresql://{quote(parts['user'])}@/{parts['database']}?sslmode=disable"
        with pytest.raises(error):
            await _connects(dsn)


class TestRoomOperations:
    async def test_create_and_get(self, store) -> None:
        room = Room(id="r1")
        await store.create_room(room)
        result = await store.get_room("r1")
        assert result is not None
        assert result.id == "r1"

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_room("nope") is None

    async def test_update_room(self, store) -> None:
        room = Room(id="r1")
        await store.create_room(room)
        room.status = RoomStatus.CLOSED
        await store.update_room(room)
        result = await store.get_room("r1")
        assert result is not None
        assert result.status == RoomStatus.CLOSED

    async def test_an_empty_room_stores_no_latest_index(self, store) -> None:
        await store.create_room(Room(id="r1"))
        async with store._pool.acquire() as conn:
            stored = await conn.fetchval("SELECT latest_index FROM rooms WHERE id = 'r1'")
        assert stored is None

    async def test_init_clears_the_zero_an_empty_room_was_stored_with(self, store) -> None:
        """A deployment from before latest_index was nullable (RFC §5.1)."""
        async with store._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO rooms (id, event_count, latest_index) VALUES ('old', 0, 0)"
            )
            await conn.execute("ALTER TABLE rooms ALTER COLUMN latest_index SET NOT NULL")
            await conn.execute("ALTER TABLE rooms ALTER COLUMN latest_index SET DEFAULT 0")
        await store.init(min_size=1, max_size=3)
        async with store._pool.acquire() as conn:
            stored = await conn.fetchval("SELECT latest_index FROM rooms WHERE id = 'old'")
        assert stored is None

    async def test_init_stamps_the_visibility_of_events_stored_before(self, store) -> None:
        """An event stored ``all`` under a narrowed binding takes the
        binding's scope, once (RFC §7.3)."""
        await store.create_room(Room(id="r1"))
        await store.add_binding(
            ChannelBinding(
                room_id="r1", channel_id="ch1", channel_type=ChannelType.SMS, visibility="ws"
            )
        )
        await store.add_event(_make_event(body="under the binding"))
        await store.add_event(_make_event(body="its own scope", visibility="ai"))
        await store.add_event(
            _make_event(
                body="held",
                visibility="transport",
                metadata={HELD_KEY: {"visibility": "all", "identity_id": "i1"}},
            )
        )
        async with store._pool.acquire() as conn:
            await conn.execute("UPDATE events SET visibility = 'all' WHERE visibility = 'ws'")
            await conn.execute("DELETE FROM schema_version WHERE version >= 3")
        await store.init(min_size=1, max_size=3)
        events = await store.list_events("r1")
        assert {e.content.body: e.visibility for e in events} == {
            "under the binding": "ws",
            "its own scope": "ai",
            "held": "transport",
        }
        held = next(e for e in events if e.content.body == "held")
        assert held.metadata[HELD_KEY]["visibility"] == "ws"

    async def test_delivered_index_defaults_to_minus_one(self, store) -> None:
        await store.create_room(Room(id="r1"))
        room = await store.get_room("r1")
        assert room is not None
        assert room.delivered_index == -1

    async def test_advance_delivered_index_cas(self, store) -> None:
        await store.create_room(Room(id="r1"))
        assert await store.advance_delivered_index("r1", 0) is True
        assert await store.advance_delivered_index("r1", 1) is True
        assert await store.advance_delivered_index("r1", 3) is False  # gap
        assert await store.advance_delivered_index("r1", 1) is False  # regress
        room = await store.get_room("r1")
        assert room is not None
        assert room.delivered_index == 1

    async def test_advance_delivered_index_force_skips_gap(self, store) -> None:
        await store.create_room(Room(id="r1"))
        assert await store.advance_delivered_index("r1", 5, force=True) is True
        assert await store.advance_delivered_index("r1", 3, force=True) is False
        room = await store.get_room("r1")
        assert room is not None
        assert room.delivered_index == 5

    async def test_update_room_never_rewinds_delivered_index(self, store) -> None:
        """delivered_index is absent from update_room's column list."""
        await store.create_room(Room(id="r1"))
        stale = await store.get_room("r1")
        assert stale is not None
        await store.advance_delivered_index("r1", 0)
        await store.update_room(stale.model_copy(update={"status": RoomStatus.CLOSED}))
        room = await store.get_room("r1")
        assert room is not None
        assert room.status == RoomStatus.CLOSED
        assert room.delivered_index == 0

    async def test_patch_room_metadata_merges(self, store) -> None:
        await store.create_room(Room(id="r1", metadata={"a": 1, "keep": "x"}))
        result = await store.patch_room_metadata("r1", {"a": 2, "b": 3})
        assert result is not None
        assert result.metadata == {"a": 2, "keep": "x", "b": 3}
        stored = await store.get_room("r1")
        assert stored is not None
        assert stored.metadata == {"a": 2, "keep": "x", "b": 3}

    async def test_patch_room_metadata_unset(self, store) -> None:
        await store.create_room(Room(id="r1", metadata={"drop": 1, "keep": 2}))
        result = await store.patch_room_metadata("r1", {"new": 3}, unset=["drop"])
        assert result is not None
        assert result.metadata == {"keep": 2, "new": 3}

    async def test_patch_room_metadata_nonexistent(self, store) -> None:
        assert await store.patch_room_metadata("nope", {"a": 1}) is None

    async def test_patch_room_metadata_does_not_touch_counters(self, store) -> None:
        await store.create_room(Room(id="r1", metadata={"seed": True}))
        for i in range(3):
            await store.commit_event("r1", _make_event(body=f"m{i}"))
        await store.patch_room_metadata("r1", {"title": "t"})
        room = await store.get_room("r1")
        assert room is not None
        assert room.event_count == 3
        assert room.latest_index == 2
        assert room.metadata == {"seed": True, "title": "t"}

    async def test_patch_room_metadata_concurrent_with_commits(self, store) -> None:
        """Interleaved patches and commits across pool connections must not
        clobber each other — the read-modify-write regression this API exists
        to prevent. Every patched key and every committed event must survive.
        """
        await store.create_room(Room(id="r1"))
        patches = [store.patch_room_metadata("r1", {f"k{i}": i}) for i in range(10)]
        commits = [store.commit_event("r1", _make_event(body=f"m{i}")) for i in range(10)]
        await asyncio.gather(*patches, *commits)
        room = await store.get_room("r1")
        assert room is not None
        assert room.event_count == 10
        assert room.latest_index == 9
        assert {f"k{i}": i for i in range(10)}.items() <= room.metadata.items()

    async def test_delete_room(self, store) -> None:
        await store.create_room(Room(id="r1"))
        assert await store.delete_room("r1") is True
        assert await store.get_room("r1") is None

    async def test_delete_nonexistent(self, store) -> None:
        assert await store.delete_room("nope") is False

    async def test_list_rooms(self, store) -> None:
        for i in range(5):
            await store.create_room(Room(id=f"r{i}"))
        rooms = await store.list_rooms()
        assert len(rooms) == 5

    async def test_list_rooms_pagination(self, store) -> None:
        for i in range(10):
            await store.create_room(Room(id=f"r{i}"))
        page1 = await store.list_rooms(offset=0, limit=3)
        page2 = await store.list_rooms(offset=3, limit=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id


class TestEventOperations:
    async def test_add_and_get(self, store) -> None:
        await store.create_room(Room(id="r1"))
        event = _make_event(room_id="r1")
        await store.add_event(event)
        result = await store.get_event(event.id)
        assert result is not None
        assert result.id == event.id

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_event("nope") is None

    async def test_list_events(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(5):
            await store.add_event(_make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1")
        assert len(events) == 5

    async def test_list_events_pagination(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event(_make_event(room_id="r1", body=f"msg{i}"))
        page = await store.list_events("r1", offset=2, limit=3)
        assert len(page) == 3

    async def test_idempotency_check(self, store) -> None:
        await store.create_room(Room(id="r1"))
        event = _make_event(room_id="r1", idempotency_key="key1")
        await store.add_event(event)
        assert await store.check_idempotency("r1", "key1", channel_id="ch1") is True
        assert await store.check_idempotency("r1", "key2", channel_id="ch1") is False
        # A key is unique per room and per source channel (RFC §13.4).
        assert await store.check_idempotency("r1", "key1", channel_id="ch2") is False

    async def test_event_count(self, store) -> None:
        await store.create_room(Room(id="r1"))
        assert await store.get_event_count("r1") == 0
        await store.add_event(_make_event(room_id="r1"))
        assert await store.get_event_count("r1") == 1

    async def test_update_event_persists_source(self, store) -> None:
        from roomkit.models.enums import ChannelType as CT

        await store.create_room(Room(id="r1"))
        event = _make_event(room_id="r1")
        await store.add_event(event)
        patched = event.model_copy(
            update={
                "source": event.source.model_copy(
                    update={"channel_type": CT.SYSTEM, "participant_id": "task_result"}
                )
            }
        )
        await store.update_event(patched)
        stored = await store.get_event(event.id)
        assert stored is not None
        assert stored.source.channel_type is CT.SYSTEM
        assert stored.source.participant_id == "task_result"

    async def test_commit_after_deleting_latest_keeps_index_monotonic(self, store) -> None:
        await store.create_room(Room(id="r1"))
        first = await store.commit_event("r1", _make_event(room_id="r1", body="first"))
        second = await store.commit_event("r1", _make_event(room_id="r1", body="second"))

        await store.delete_event("r1", second.id)
        third = await store.commit_event("r1", _make_event(room_id="r1", body="third"))

        assert [first.index, second.index, third.index] == [0, 1, 2]
        assert [event.index for event in await store.list_events("r1")] == [0, 2]

    async def test_auto_index_after_deleting_latest_keeps_index_monotonic(self, store) -> None:
        await store.create_room(Room(id="r1"))
        first = await store.add_event_auto_index("r1", _make_event(room_id="r1"))
        second = await store.add_event_auto_index("r1", _make_event(room_id="r1"))

        await store.delete_event("r1", second.id)
        third = await store.add_event_auto_index("r1", _make_event(room_id="r1"))

        assert [first.index, second.index, third.index] == [0, 1, 2]


class TestBindingOperations:
    async def test_add_and_get(self, store) -> None:
        await store.create_room(Room(id="r1"))
        binding = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        await store.add_binding(binding)
        result = await store.get_binding("r1", "ch1")
        assert result is not None
        assert result.channel_id == "ch1"

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_binding("r1", "nope") is None

    async def test_update_binding(self, store) -> None:
        await store.create_room(Room(id="r1"))
        binding = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        await store.add_binding(binding)
        binding.muted = True
        await store.update_binding(binding)
        result = await store.get_binding("r1", "ch1")
        assert result is not None
        assert result.muted is True

    async def test_remove_binding(self, store) -> None:
        await store.create_room(Room(id="r1"))
        binding = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        await store.add_binding(binding)
        assert await store.remove_binding("r1", "ch1") is True
        assert await store.get_binding("r1", "ch1") is None

    async def test_remove_nonexistent(self, store) -> None:
        assert await store.remove_binding("r1", "nope") is False

    async def test_list_bindings(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(3):
            await store.add_binding(
                ChannelBinding(channel_id=f"ch{i}", room_id="r1", channel_type=ChannelType.SMS)
            )
        bindings = await store.list_bindings("r1")
        assert len(bindings) == 3


class TestParticipantOperations:
    async def test_add_and_get(self, store) -> None:
        await store.create_room(Room(id="r1"))
        p = Participant(id="p1", room_id="r1", channel_id="ch1")
        await store.add_participant(p)
        result = await store.get_participant("r1", "p1")
        assert result is not None
        assert result.id == "p1"

    async def test_list_participants(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(3):
            await store.add_participant(Participant(id=f"p{i}", room_id="r1", channel_id=f"ch{i}"))
        participants = await store.list_participants("r1")
        assert len(participants) == 3

    async def test_update_participant(self, store) -> None:
        await store.create_room(Room(id="r1"))
        p = Participant(id="p1", room_id="r1", channel_id="ch1")
        await store.add_participant(p)
        p.display_name = "Alice"
        await store.update_participant(p)
        result = await store.get_participant("r1", "p1")
        assert result is not None
        assert result.display_name == "Alice"

    async def test_connected_via_round_trips(self, store) -> None:
        """RFC §5.5: the channels a participant was reached through MUST persist.

        A list that only lived in the model would make the multi-channel record
        true for one process and empty for the next one to read it.
        """
        await store.create_room(Room(id="r1"))
        p = Participant(id="p1", room_id="r1", channel_id="ch1", connected_via=["ch1"])
        await store.add_participant(p)
        stored = await store.get_participant("r1", "p1")
        assert stored is not None
        assert stored.connected_via == ["ch1"]

        await store.update_participant(p.model_copy(update={"connected_via": ["ch1", "ch2"]}))
        result = await store.get_participant("r1", "p1")
        assert result is not None
        assert result.connected_via == ["ch1", "ch2"]
        # order is order of first sight, not set order
        assert result.connected_via[0] == "ch1"

    @pytest.mark.parametrize("candidates", [None, [], ["alice-id", "bob-id"]])
    async def test_candidates_round_trip(self, store, candidates: list[str] | None) -> None:
        """RFC §11.2: the identities an ambiguous sender could be persist, and
        none offered (NULL) is not none left ([])."""
        await store.create_room(Room(id="r1"))
        p = Participant(id="p1", room_id="r1", channel_id="ch1")
        await store.add_participant(p.model_copy(update={"candidates": candidates}))
        stored = await store.get_participant("r1", "p1")
        assert stored is not None
        assert stored.candidates == candidates

        await store.update_participant(p.model_copy(update={"candidates": ["carol-id"]}))
        stored = await store.get_participant("r1", "p1")
        assert stored is not None
        assert stored.candidates == ["carol-id"]

    async def test_connected_via_repairs_a_row_that_predates_it(self, store) -> None:
        """RFC §5.5: an old empty list reads with its primary included."""
        await store.create_room(Room(id="r1"))
        await store.add_participant(Participant(id="p1", room_id="r1", channel_id="ch1"))
        async with store._pool.acquire() as conn:
            await conn.execute(
                "UPDATE participants SET connected_via='{}' WHERE room_id='r1' AND id='p1'"
            )
        result = await store.get_participant("r1", "p1")
        assert result is not None
        assert result.connected_via == ["ch1"]

    async def test_init_backfills_connected_via_on_a_legacy_row(self, store) -> None:
        """The additive schema pass persists the primary on pre-column rows."""
        await store.create_room(Room(id="r1"))
        await store.add_participant(Participant(id="p1", room_id="r1", channel_id="ch1"))
        async with store._pool.acquire() as conn:
            await conn.execute(
                "UPDATE participants SET connected_via='{}' WHERE room_id='r1' AND id='p1'"
            )

        await store.init()

        async with store._pool.acquire() as conn:
            connected_via = await conn.fetchval(
                "SELECT connected_via FROM participants WHERE room_id='r1' AND id='p1'"
            )
        assert connected_via == ["ch1"]


class TestReadTracking:
    async def test_unread_count_initial(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for _ in range(3):
            await store.add_event(_make_event(room_id="r1"))
        count = await store.get_unread_count("r1", "ch1")
        assert count == 3

    async def test_mark_read(self, store) -> None:
        await store.create_room(Room(id="r1"))
        events = []
        for _ in range(3):
            e = _make_event(room_id="r1")
            await store.add_event(e)
            events.append(e)
        await store.mark_read("r1", "ch1", events[1].id)
        count = await store.get_unread_count("r1", "ch1")
        assert count == 1

    async def test_mark_all_read(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for _ in range(3):
            await store.add_event(_make_event(room_id="r1"))
        await store.mark_all_read("r1", "ch1")
        count = await store.get_unread_count("r1", "ch1")
        assert count == 0

    async def test_list_read_markers(self, store) -> None:
        await store.create_room(Room(id="r1"))
        events = []
        for _ in range(3):
            e = _make_event(room_id="r1")
            await store.add_event(e)
            events.append(e)
        await store.mark_read("r1", "ch1", events[2].id)
        await store.mark_read("r1", "ch2", events[0].id)
        markers = await store.list_read_markers("r1")
        assert set(markers) == {"ch1", "ch2"}
        # ch1 read further than ch2; a channel with no marker is absent
        assert markers["ch1"] > markers["ch2"]
        assert "ch3" not in markers


class TestIdentityOperations:
    async def test_create_and_get(self, store) -> None:
        identity = Identity(
            id="id1",
            display_name="Alice",
            channel_addresses={"sms": ["+15551234567"]},
        )
        await store.create_identity(identity)
        result = await store.get_identity("id1")
        assert result is not None
        assert result.display_name == "Alice"

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_identity("nope") is None

    async def test_resolve_by_address(self, store) -> None:
        identity = Identity(
            id="id1",
            channel_addresses={"sms": ["+15551234567"]},
        )
        await store.create_identity(identity)
        result = await store.resolve_identity("sms", "+15551234567")
        assert result is not None
        assert result.id == "id1"

    async def test_resolve_unknown_address(self, store) -> None:
        result = await store.resolve_identity("sms", "+10000000000")
        assert result is None

    async def test_link_address(self, store) -> None:
        identity = Identity(id="id1")
        await store.create_identity(identity)
        await store.link_address("id1", "email", "alice@example.com")

        result = await store.resolve_identity("email", "alice@example.com")
        assert result is not None
        assert result.id == "id1"

        fetched = await store.get_identity("id1")
        assert fetched is not None
        assert "alice@example.com" in fetched.channel_addresses.get("email", [])

    async def test_multiple_addresses_same_channel(self, store) -> None:
        identity = Identity(
            id="id1",
            channel_addresses={"sms": ["+15551111111", "+15552222222"]},
        )
        await store.create_identity(identity)
        r1 = await store.resolve_identity("sms", "+15551111111")
        r2 = await store.resolve_identity("sms", "+15552222222")
        assert r1 is not None and r1.id == "id1"
        assert r2 is not None and r2.id == "id1"


class TestIdentityDetails:
    async def test_what_an_identity_holds_beside_its_addresses_round_trips(self, store) -> None:
        """RFC §5.6: as the other stores keep them, through get and resolve."""
        identity = Identity(
            id="i1",
            phone="+15551234567",
            external_id="crm_42",
            external_ids={"crm": "42", "erp": "77"},
            channel_addresses={"sms": ["+15551234567"]},
        )
        await store.create_identity(identity)
        assert await store.get_identity("i1") == identity
        assert await store.resolve_identity("sms", "+15551234567") == identity

    async def test_an_identity_created_again_replaces_the_stored_one(self, store) -> None:
        await store.create_identity(Identity(id="i1", phone="+1555", external_id="crm_42"))
        renamed = Identity(id="i1", display_name="Alice B.")
        assert await store.create_identity(renamed) == renamed
        assert await store.get_identity("i1") == renamed

    async def test_a_creation_racing_another_organization_s_is_refused(self, store) -> None:
        """The refusal is the write's own: a creation that waits on another's
        uncommitted row reads it once that commits, and writes nothing."""
        import asyncpg

        creator = await asyncpg.connect(POSTGRES_DSN)
        try:
            async with creator.transaction():
                await creator.execute(
                    "INSERT INTO identities (id, organization_id) VALUES ('i1', 'acme')"
                )
                racing = asyncio.create_task(
                    store.create_identity(Identity(id="i1", organization_id="globex"))
                )
                await _until_blocked_by(creator)
            with pytest.raises(IdentityNotFoundError):
                await asyncio.wait_for(racing, 2)
        finally:
            await creator.close()
        assert await store.get_identity("i1") == Identity(id="i1", organization_id="acme")

    async def test_a_replacement_removes_a_row_of_another_scope_for_a_kept_address(
        self, store
    ) -> None:
        """An address it keeps, under its own organization: a row of another scope
        for the same address goes, not only stops resolving."""
        identity = Identity(id="i1", organization_id="acme", channel_addresses={"sms": ["+1555"]})
        await store.create_identity(identity)
        rows = "SELECT organization_id FROM identity_addresses WHERE identity_id = 'i1'"
        async with store._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO identity_addresses (channel_type, address, identity_id,"
                " organization_id) VALUES ('sms', '+1555', 'i1', '')"
            )
        await store.create_identity(identity)
        async with store._pool.acquire() as conn:
            scopes = [row["organization_id"] for row in await conn.fetch(rows)]
        assert scopes == ["acme"]
        assert await store.resolve_identity("sms", "+1555", organization_id="acme") == identity

    @pytest.mark.parametrize(
        ("held_in", "stored"), [("", "acme"), ("globex", "acme"), ("acme", None)]
    )
    async def test_a_row_of_another_scope_than_its_identity_resolves_for_nobody(
        self, store, held_in: str, stored: str | None
    ) -> None:
        """RFC §17.2: an identity is read in its own organization only, whatever a
        row left by an upgrade or written directly says."""
        await store.create_identity(Identity(id="i1", organization_id=stored))
        async with store._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO identity_addresses (channel_type, address, identity_id,"
                " organization_id) VALUES ('sms', '+1555', 'i1', $1)",
                held_in,
            )
        for scope in (None, "acme", "globex"):
            assert await store.resolve_identity("sms", "+1555", organization_id=scope) is None

    @pytest.mark.parametrize(
        ("held_in", "stored"), [("", "acme"), ("globex", "acme"), ("acme", None)]
    )
    @pytest.mark.parametrize("taking", ["create_identity", "link_address"])
    async def test_a_row_no_read_serves_leaves_its_address_free(
        self, store, held_in: str, stored: str | None, taking: str
    ) -> None:
        """RFC §5.6: an address is refused for what another identity of the organization
        holds. A row an older release left, whose identity is of another organization,
        resolves for nobody: the organization's own identity takes the address."""
        await store.create_identity(Identity(id="i1", organization_id=stored))
        async with store._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO identity_addresses (channel_type, address, identity_id,"
                " organization_id) VALUES ('sms', '+1555', 'i1', $1)",
                held_in,
            )
        scope = held_in or None
        if taking == "create_identity":
            await store.create_identity(
                Identity(id="dana", organization_id=scope, channel_addresses={"sms": ["+1555"]})
            )
        else:
            await store.create_identity(Identity(id="dana", organization_id=scope))
            await store.link_address("dana", "sms", "+1555", organization_id=scope)
        resolved = await store.resolve_identity("sms", "+1555", organization_id=scope)
        assert resolved is not None and resolved.id == "dana"

    async def test_a_creation_waiting_on_a_new_identity_s_address_is_refused(self, store) -> None:
        """RFC §5.6: the holder's organization is read once the first commits. Its
        identity is new to the waiting statement's snapshot, and is no row to take."""
        import asyncpg

        creator = await asyncpg.connect(POSTGRES_DSN)
        try:
            async with creator.transaction():
                await creator.execute(
                    "INSERT INTO identities (id, organization_id) VALUES ('i1', 'acme')"
                )
                await creator.execute(
                    "INSERT INTO identity_addresses (channel_type, address, identity_id,"
                    " organization_id) VALUES ('sms', '+1555', 'i1', 'acme')"
                )
                racing = asyncio.create_task(
                    store.create_identity(
                        Identity(
                            id="i2", organization_id="acme", channel_addresses={"sms": ["+1555"]}
                        )
                    )
                )
                await _until_blocked_by(creator)
            with pytest.raises(AddressLinkedError):
                await asyncio.wait_for(racing, 2)
        finally:
            await creator.close()
        resolved = await store.resolve_identity("sms", "+1555", organization_id="acme")
        assert resolved is not None and resolved.id == "i1"

    async def test_two_identities_taking_a_row_no_read_serves_leave_one_winner(
        self, store
    ) -> None:
        """RFC §5.6: the first holds the address from then on, and the second is refused."""
        await store.create_identity(Identity(id="i1", organization_id="globex"))
        for number in range(8):
            address = f"+1{number}"
            async with store._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO identity_addresses (channel_type, address, identity_id,"
                    " organization_id) VALUES ('sms', $1, 'i1', 'acme')",
                    address,
                )
            taken = await asyncio.gather(
                *(
                    store.create_identity(
                        Identity(
                            id=f"{name}{number}",
                            organization_id="acme",
                            channel_addresses={"sms": [address]},
                        )
                    )
                    for name in ("a", "b")
                ),
                return_exceptions=True,
            )
            won = [outcome for outcome in taken if isinstance(outcome, Identity)]
            assert len(won) == 1, taken
            assert isinstance(next(o for o in taken if o not in won), AddressLinkedError)
            resolved = await store.resolve_identity("sms", address, organization_id="acme")
            assert resolved is not None and resolved.id == won[0].id

    @pytest.mark.parametrize("keeping", [False, True])
    async def test_two_replacements_swapping_addresses_are_refused_without_a_deadlock(
        self, store, keeping: bool
    ) -> None:
        """Each new address filed before the old ones go, in one order whoever
        files them: neither waits on what the other holds, and each reads the
        refusal, not a deadlock, whether or not it keeps its own."""
        for number in range(8):
            one, two = f"i{number}a", f"i{number}b"
            first, second = f"+1{number}", f"+2{number}"
            await store.create_identity(Identity(id=one, channel_addresses={"sms": [first]}))
            await store.create_identity(Identity(id=two, channel_addresses={"sms": [second]}))
            taken_by_one = [first, second] if keeping else [second]
            taken_by_two = [second, first] if keeping else [first]
            swapped = await asyncio.gather(
                store.create_identity(Identity(id=one, channel_addresses={"sms": taken_by_one})),
                store.create_identity(Identity(id=two, channel_addresses={"sms": taken_by_two})),
                return_exceptions=True,
            )
            assert all(isinstance(outcome, AddressLinkedError) for outcome in swapped), swapped

    async def test_an_unlink_removes_the_identity_s_rows_in_every_scope(self, store) -> None:
        """RFC §5.6: a link revoked is revoked, a row an upgrade left in another
        scope with it, so that no later rewrite can restore it."""
        identity = Identity(id="i1", organization_id="acme", channel_addresses={"sms": ["+1"]})
        await store.create_identity(identity)
        async with store._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO identity_addresses (channel_type, address, identity_id,"
                " organization_id) VALUES ('sms', '+1', 'i1', '')"
            )
        await store.unlink_address("i1", "sms", "+1", organization_id="acme")
        async with store._pool.acquire() as conn:
            left = await conn.fetchval("SELECT count(*) FROM identity_addresses")
        assert left == 0


class TestTaskOperations:
    async def test_add_and_get(self, store) -> None:
        await store.create_room(Room(id="r1"))
        task = Task(id="t1", room_id="r1", title="Follow up")
        await store.add_task(task)
        result = await store.get_task("t1")
        assert result is not None
        assert result.title == "Follow up"

    async def test_get_nonexistent(self, store) -> None:
        assert await store.get_task("nope") is None

    async def test_list_tasks_by_room(self, store) -> None:
        await store.create_room(Room(id="r1"))
        await store.create_room(Room(id="r2"))
        await store.add_task(Task(id="t1", room_id="r1", title="Task 1"))
        await store.add_task(Task(id="t2", room_id="r1", title="Task 2"))
        await store.add_task(Task(id="t3", room_id="r2", title="Task 3"))
        tasks = await store.list_tasks("r1")
        assert len(tasks) == 2

    async def test_list_tasks_by_status(self, store) -> None:
        await store.create_room(Room(id="r1"))
        await store.add_task(Task(id="t1", room_id="r1", title="Pending", status="pending"))
        await store.add_task(Task(id="t2", room_id="r1", title="Done", status="completed"))
        pending = await store.list_tasks("r1", status="pending")
        assert len(pending) == 1
        assert pending[0].title == "Pending"

    async def test_update_task(self, store) -> None:
        await store.create_room(Room(id="r1"))
        task = Task(id="t1", room_id="r1", title="Follow up")
        await store.add_task(task)
        task.status = "completed"
        await store.update_task(task)
        result = await store.get_task("t1")
        assert result is not None
        assert result.status == "completed"


class TestObservationOperations:
    async def test_add_and_list(self, store) -> None:
        await store.create_room(Room(id="r1"))
        obs = Observation(
            id="obs1",
            room_id="r1",
            channel_id="ai1",
            content="sentiment positive",
            category="sentiment",
        )
        await store.add_observation(obs)
        observations = await store.list_observations("r1")
        assert len(observations) == 1
        assert observations[0].category == "sentiment"

    async def test_list_by_room(self, store) -> None:
        await store.create_room(Room(id="r1"))
        await store.create_room(Room(id="r2"))
        await store.add_observation(
            Observation(id="obs1", room_id="r1", channel_id="ai1", content="a")
        )
        await store.add_observation(
            Observation(id="obs2", room_id="r2", channel_id="ai1", content="b")
        )
        r1_obs = await store.list_observations("r1")
        r2_obs = await store.list_observations("r2")
        assert len(r1_obs) == 1
        assert len(r2_obs) == 1


class TestFindRooms:
    async def test_find_by_status(self, store) -> None:
        await store.create_room(Room(id="r1", status=RoomStatus.ACTIVE))
        await store.create_room(Room(id="r2", status=RoomStatus.CLOSED))
        await store.create_room(Room(id="r3", status=RoomStatus.ACTIVE))
        rooms = await store.find_rooms(status=RoomStatus.ACTIVE)
        assert len(rooms) == 2
        assert all(r.status == RoomStatus.ACTIVE for r in rooms)

    async def test_find_by_metadata(self, store) -> None:
        await store.create_room(Room(id="r1", metadata={"org": "acme", "tier": "premium"}))
        await store.create_room(Room(id="r2", metadata={"org": "globex", "tier": "basic"}))
        rooms = await store.find_rooms(metadata_filter={"org": "acme"})
        assert len(rooms) == 1
        assert rooms[0].id == "r1"

    async def test_find_no_match(self, store) -> None:
        await store.create_room(Room(id="r1"))
        rooms = await store.find_rooms(status=RoomStatus.CLOSED)
        assert len(rooms) == 0


class TestListEventsVisibilityFilter:
    async def test_visibility_filter(self, store) -> None:
        await store.create_room(Room(id="r1"))
        e1 = _make_event(room_id="r1", body="public", visibility="all")
        e2 = _make_event(room_id="r1", body="internal", visibility="internal")
        e3 = _make_event(room_id="r1", body="public2", visibility="all")
        await store.add_event(e1)
        await store.add_event(e2)
        await store.add_event(e3)

        all_events = await store.list_events("r1")
        assert len(all_events) == 3

        public_only = await store.list_events("r1", visibility_filter="all")
        assert len(public_only) == 2

        internal_only = await store.list_events("r1", visibility_filter="internal")
        assert len(internal_only) == 1


class TestCursorPagination:
    async def test_after_index(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(5):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", after_index=1, limit=50)
        assert len(events) == 3
        assert all(e.index > 1 for e in events)

    async def test_before_index(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(5):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", before_index=3, limit=50)
        assert len(events) == 3
        assert all(e.index < 3 for e in events)
        assert [e.index for e in events] == [0, 1, 2]

    async def test_before_index_returns_last_n(self, store) -> None:
        """before_index returns the *last* limit events before cursor."""
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", before_index=8, limit=3)
        assert len(events) == 3
        assert [e.index for e in events] == [5, 6, 7]

    async def test_after_index_with_limit(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", after_index=2, limit=3)
        assert len(events) == 3
        assert events[0].index == 3

    async def test_mutually_exclusive(self, store) -> None:
        await store.create_room(Room(id="r1"))
        with pytest.raises(ValueError, match="mutually exclusive"):
            await store.list_events("r1", after_index=0, before_index=5)

    async def test_cursor_with_visibility_filter(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(6):
            vis = "all" if i % 2 == 0 else "internal"
            await store.add_event_auto_index(
                "r1", _make_event(room_id="r1", body=f"msg{i}", visibility=vis)
            )
        events = await store.list_events("r1", after_index=0, visibility_filter="all")
        assert len(events) == 2
        assert all(e.visibility == "all" for e in events)

    async def test_empty_result(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(3):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))
        events = await store.list_events("r1", after_index=10)
        assert events == []


class TestConversationRecency:
    """RMK-99 — against a real database: SQL decides this ordering, not Python."""

    async def test_get_conversation_returns_the_tail(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))

        convo = await store.get_conversation("r1", limit=3)

        assert [e.index for e in convo] == [7, 8, 9]
        assert convo[-1].content.body == "msg9"

    async def test_get_conversation_with_cursor_reads_forward(self, store) -> None:
        await store.create_room(Room(id="r1"))
        for i in range(10):
            await store.add_event_auto_index("r1", _make_event(room_id="r1", body=f"msg{i}"))

        page = await store.get_conversation("r1", limit=3, after_index=5)

        assert [e.index for e in page] == [6, 7, 8]


class TestDeleteRoomCleanup:
    async def test_cascading_delete(self, store) -> None:
        """ON DELETE CASCADE removes all child rows."""
        await store.create_room(Room(id="r1"))
        await store.add_event(_make_event(room_id="r1", idempotency_key="k1"))
        await store.add_task(Task(id="t1", room_id="r1", title="T"))
        await store.add_observation(
            Observation(id="obs1", room_id="r1", channel_id="c1", content="x")
        )
        await store.add_binding(
            ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        )
        await store.add_participant(Participant(id="p1", room_id="r1", channel_id="ch1"))

        assert await store.delete_room("r1") is True

        assert await store.get_room("r1") is None
        assert await store.get_event_count("r1") == 0
        assert await store.list_tasks("r1") == []
        assert await store.list_observations("r1") == []
        assert await store.list_bindings("r1") == []
        assert await store.list_participants("r1") == []


@pytest.fixture
def checkouts(monkeypatch) -> Callable[[Any], PoolCheckoutRecorder]:
    """Return ``recorder_for(store)`` — real pool checkouts, per store instance."""
    from roomkit.store.postgres import PostgresStore

    recorders: dict[int, PoolCheckoutRecorder] = {}

    def recorder_for(s: Any) -> PoolCheckoutRecorder:
        if id(s) not in recorders:
            recorders[id(s)] = PoolCheckoutRecorder(s._pool)
        return recorders[id(s)]

    monkeypatch.setattr(PostgresStore, "_ensure_pool", recorder_for)
    return recorder_for


class TestConnectionTenure:
    """RMK-97 — ``connection()`` bounds how long a pooled connection is held.

    asyncpg resets a connection on release, so every checkout is a full extra
    round trip; on the inbound path that overhead is the same order as the
    reads it brackets.
    """

    async def test_the_block_pays_one_checkout_for_every_call(self, store, checkouts) -> None:
        await store.create_room(Room(id="r1"))
        recorder = checkouts(store)
        recorder.reset()

        async with store.connection():
            await store.get_room("r1")
            await store.list_bindings("r1")
            await store.list_participants("r1")

        assert recorder.total == 1

    async def test_the_same_calls_outside_pay_one_checkout_each(self, store, checkouts) -> None:
        await store.create_room(Room(id="r1"))
        recorder = checkouts(store)
        recorder.reset()

        await store.get_room("r1")
        await store.list_bindings("r1")
        await store.list_participants("r1")

        assert recorder.total == 3

    async def test_a_nested_block_joins_the_outer_one(self, store, checkouts) -> None:
        await store.create_room(Room(id="r1"))
        recorder = checkouts(store)
        recorder.reset()

        async with store.connection():
            await store.get_room("r1")
            async with store.connection():
                await store.list_bindings("r1")
            await store.list_participants("r1")

        assert recorder.total == 1

    async def test_the_default_load_room_context_rides_the_block(self, store, checkouts) -> None:
        """The ABC default composes three getters and brackets them itself.

        ``PostgresStore`` overrides the method, so call the default explicitly:
        this is what a third-party pooled backend inherits without writing any
        SQL of its own.
        """
        from roomkit.store.base import ConversationStore

        await store.create_room(Room(id="r1"))
        recorder = checkouts(store)
        recorder.reset()

        room, bindings, participants = await ConversationStore.load_room_context(store, "r1")

        assert room is not None
        assert (bindings, participants) == ([], [])
        assert recorder.total == 1

    async def test_a_failure_inside_the_block_returns_the_connection(self, store) -> None:
        await store.get_room("warm-the-pool")
        idle_before = store._pool.get_idle_size()

        with pytest.raises(RuntimeError, match="boom"):
            async with store.connection():
                await store.get_room("r1")
                raise RuntimeError("boom")

        assert store._pool.get_idle_size() == idle_before
        assert await store.get_room("r1") is None

    async def test_concurrent_blocks_do_not_share_one_connection(self, store) -> None:
        """Two tasks, two connections — a shared one would raise on the second
        query ("another operation is in progress")."""
        await store.create_room(Room(id="r1"))
        seen: list[int] = []

        async def read_twice() -> None:
            async with store.connection():
                seen.append(await store.get_event_count("r1"))
                await asyncio.sleep(0)  # hand over inside the block
                seen.append(await store.get_event_count("r1"))

        await asyncio.gather(read_twice(), read_twice())

        assert seen == [0, 0, 0, 0]

    async def test_another_store_is_not_lent_this_one_s_connection(self, store, checkouts) -> None:
        """The binding carries its owner: two stores may hold two pools onto
        two different databases, and lending across them would query the wrong
        one."""
        from roomkit.store.postgres import PostgresStore

        other = PostgresStore(dsn=POSTGRES_DSN)
        await other.init(min_size=1, max_size=2)
        try:
            await store.create_room(Room(id="r1"))
            recorder = checkouts(other)
            recorder.reset()

            async with store.connection():
                assert await other.get_room("r1") is not None

            assert recorder.total == 1
        finally:
            await other.close()


class TestContextManager:
    async def test_async_context_manager(self) -> None:
        from roomkit.store.postgres import PostgresStore

        store = PostgresStore(dsn=POSTGRES_DSN)
        async with store as s:
            room = Room(id="ctx-test")
            await s.create_room(room)
            result = await s.get_room("ctx-test")
            assert result is not None


class TestThreadQueries:
    """Flat two-level threading — the ``parent_event_id`` filters + summaries."""

    async def test_top_level_only_excludes_replies(self, store) -> None:
        from roomkit.models.store_filter import EventFilter

        await store.create_room(Room(id="r1"))
        root = _make_event(body="root")
        await store.add_event(root)
        await store.add_event(_make_event(body="standalone"))
        await store.add_event(_make_event(body="reply1", parent_event_id=root.id))
        await store.add_event(_make_event(body="reply2", parent_event_id=root.id))

        top = await store.list_events("r1", event_filter=EventFilter(top_level_only=True))
        assert {e.content.body for e in top} == {"root", "standalone"}

    async def test_parent_event_id_returns_only_that_thread(self, store) -> None:
        from roomkit.models.store_filter import EventFilter

        await store.create_room(Room(id="r1"))
        root = _make_event(body="root")
        await store.add_event(root)
        await store.add_event(_make_event(body="reply1", parent_event_id=root.id))
        await store.add_event(_make_event(body="reply2", parent_event_id=root.id))

        replies = await store.list_events("r1", event_filter=EventFilter(parent_event_id=root.id))
        assert {e.content.body for e in replies} == {"reply1", "reply2"}
        assert all(e.parent_event_id == root.id for e in replies)

    async def test_thread_summaries_counts_and_last_reply(self, store) -> None:
        from datetime import UTC, datetime

        await store.create_room(Room(id="r1"))
        root = _make_event(body="root")
        await store.add_event(root)
        await store.add_event(
            _make_event(
                body="r1",
                parent_event_id=root.id,
                created_at=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
            )
        )
        last = _make_event(
            body="r2",
            parent_event_id=root.id,
            created_at=datetime(2026, 1, 1, 12, 0, 2, tzinfo=UTC),
        )
        await store.add_event(last)

        summaries = await store.get_thread_summaries("r1", [root.id])
        assert set(summaries) == {root.id}
        assert summaries[root.id].reply_count == 2
        assert summaries[root.id].last_reply_at == last.created_at

    async def test_thread_summaries_empty_cases(self, store) -> None:
        await store.create_room(Room(id="r1"))
        root = _make_event(body="root")
        await store.add_event(root)
        assert await store.get_thread_summaries("r1", [root.id]) == {}
        assert await store.get_thread_summaries("r1", []) == {}


async def test_a_confidence_reads_as_written(store) -> None:
    await store.create_room(Room(id="r1"))
    written = Observation(id="o1", room_id="r1", channel_id="ch1", content="c", confidence=0.42)
    await store.add_observation(written)
    assert await store.list_observations("r1") == [written]


async def test_init_widens_the_confidence_of_a_store_from_before(store) -> None:
    """A value a REAL column holds reads as written once widened, through its
    shortest digits, not as the nearest double of the single."""
    await store.create_room(Room(id="r1"))
    written = Observation(id="o1", room_id="r1", channel_id="ch1", content="c", confidence=0.42)
    await store.add_observation(written)
    async with store._pool.acquire() as conn:
        await conn.execute("ALTER TABLE observations ALTER COLUMN confidence TYPE REAL")
    await store.init()
    await store.init()
    assert await store.list_observations("r1") == [written]


async def test_init_adds_what_an_identity_holds_to_a_table_without_it(store) -> None:
    await store.create_identity(Identity(id="i1", display_name="Alice"))
    async with store._pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE identities DROP COLUMN phone, DROP COLUMN external_id,"
            " DROP COLUMN external_ids"
        )
    await store.init()
    await store.init()
    assert await store.get_identity("i1") == Identity(id="i1", display_name="Alice")


async def test_init_moves_an_unscoped_address_to_its_identity_s_organization(store) -> None:
    """HITL 0f96a597, RFC §17.2: a row the upgrade left unscoped moves to the
    organization of its identity, once, if the identity still declares the address,
    so that its sender stays identified there; one whose organization holds the
    address already yields to that row. A link since revoked is not restored: its row
    goes, and the address is free (HITL 6928c418). An identity of no organization keeps
    its unscoped row, and a row of another organization is left alone, which no read
    serves."""
    alice = Identity(id="alice", organization_id="acme")
    bob = Identity(id="bob", organization_id="acme", channel_addresses={"sms": ["+2"]})
    carol = Identity(id="carol", channel_addresses={"sms": ["+3"]})
    for identity in (alice, bob, carol):
        await store.create_identity(identity)
    rows = "SELECT address, identity_id, organization_id FROM identity_addresses ORDER BY 1, 3"
    async with store._pool.acquire() as conn:
        await conn.execute(
            'UPDATE identities SET channel_addresses = \'{"sms": ["+1", "+2"]}\''
            " WHERE id = 'alice'"
        )
        await conn.executemany(
            "INSERT INTO identity_addresses (channel_type, address, identity_id,"
            " organization_id) VALUES ('sms', $1, $2, $3)",
            [
                ("+1", "alice", ""),
                ("+2", "alice", ""),
                ("+4", "alice", "globex"),
                ("+5", "alice", ""),
            ],
        )
        await conn.execute("DELETE FROM schema_version WHERE version >= 5")
    await store.init()
    await store.init()
    async with store._pool.acquire() as conn:
        stored = [tuple(row) for row in await conn.fetch(rows)]
    assert stored == [
        ("+1", "alice", "acme"),
        ("+2", "bob", "acme"),
        ("+3", "carol", ""),
        ("+4", "alice", "globex"),
    ]
    moved = await store.resolve_identity("sms", "+1", organization_id="acme")
    assert moved is not None and moved.id == "alice"
    assert await store.resolve_identity("sms", "+1") is None
    for scope in (None, "acme"):
        assert await store.resolve_identity("sms", "+5", organization_id=scope) is None
    dana = Identity(id="dana", channel_addresses={"sms": ["+5"]})
    await store.create_identity(dana)
    assert await store.resolve_identity("sms", "+5") == dana


async def test_the_rewrite_takes_only_a_list_for_a_declaration(store) -> None:
    """A declaration no API writes (a text, a map) declares nothing: the row goes,
    as on SQLite, and the rewrite does not fail."""
    await store.create_identity(Identity(id="alice", organization_id="acme"))
    async with store._pool.acquire() as conn:
        await conn.execute(
            "UPDATE identities SET channel_addresses ="
            ' \'{"sms": "+1", "mms": {"+2": true}}\' WHERE id = \'alice\''
        )
        await conn.executemany(
            "INSERT INTO identity_addresses (channel_type, address, identity_id,"
            " organization_id) VALUES ($1, $2, 'alice', '')",
            [("sms", "+1"), ("mms", "+2")],
        )
        await conn.execute("DELETE FROM schema_version WHERE version >= 5")
    await store.init()
    async with store._pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM identity_addresses") == 0


async def test_the_rewrite_waits_for_an_unlink_in_flight(store) -> None:
    """An older process's unlink, uncommitted, has removed the address from the
    declaration the rewrite reads: the rewrite waits, and restores nothing."""
    import asyncpg

    await store.create_identity(Identity(id="alice", organization_id="acme"))
    async with store._pool.acquire() as conn:
        await conn.execute(
            "UPDATE identities SET channel_addresses = '{\"sms\": [\"+1\"]}' WHERE id = 'alice'"
        )
        await conn.execute(
            "INSERT INTO identity_addresses (channel_type, address, identity_id,"
            " organization_id) VALUES ('sms', '+1', 'alice', '')"
        )
        await conn.execute("DELETE FROM schema_version WHERE version >= 5")
    unlinker = await asyncpg.connect(POSTGRES_DSN)
    try:
        async with unlinker.transaction():
            await unlinker.execute(
                "UPDATE identities SET channel_addresses = '{}' WHERE id = 'alice'"
            )
            rewrite = asyncio.create_task(store.init())
            await _until_blocked_by(unlinker)
        await asyncio.wait_for(rewrite, 2)
    finally:
        await unlinker.close()
    assert await store.resolve_identity("sms", "+1", organization_id="acme") is None


async def test_binding_policy_migration_preserves_old_rows(store) -> None:
    from roomkit.models.channel import RateLimit, RetryPolicy

    await store.create_room(Room(id="r1"))
    binding = ChannelBinding(room_id="r1", channel_id="sms", channel_type="sms")
    await store.add_binding(binding)
    async with store._pool.acquire() as conn:
        await conn.execute("ALTER TABLE bindings DROP COLUMN rate_limit, DROP COLUMN retry_policy")
    # Reinitializing the store upgrades the old schema, including repeated runs.
    await store.init()
    await store.init()
    assert await store.get_binding("r1", "sms") == binding
    changed = binding.model_copy(
        update={
            "rate_limit": RateLimit(max_per_minute=30),
            "retry_policy": RetryPolicy(max_retries=7),
        }
    )
    await store.update_binding(changed)
    assert await store.get_binding("r1", "sms") == changed


async def test_init_creates_each_index_it_lacks_and_leaves_the_others_be(store) -> None:
    """init() creates every index of its schema that is missing, as it
    defines it (a unique one unique, a partial one partial, the thread
    index composite), and finds an index present as it is."""
    from roomkit.store.postgres_schema import V1_TABLES

    tables = [*V1_TABLES, "verifications", "verification_requests", "voice_deliveries"]
    indexes = (
        "SELECT indexname, indexdef FROM pg_indexes"
        " WHERE schemaname = current_schema() AND tablename = ANY($1::text[])"
    )
    async with store._pool.acquire() as conn:
        before = {row["indexname"]: row["indexdef"] for row in await conn.fetch(indexes, tables)}
        for name in before:
            if name.startswith("idx_"):
                await conn.execute(f'DROP INDEX "{name}"')
    await store.init()
    await store.init()
    async with store._pool.acquire() as conn:
        after = {row["indexname"]: row["indexdef"] for row in await conn.fetch(indexes, tables)}
    assert after == before
    assert "(parent_event_id, index) WHERE" in after["idx_events_parent_index"]


async def test_the_legacy_parent_index_is_dropped_on_request_only(store) -> None:
    """init() never drops: the single-column idx_events_parent the composite
    supersedes stays until drop_legacy_parent_index() is asked, which a dry
    run only reports."""
    legacy = "SELECT 1 FROM pg_indexes WHERE schemaname = current_schema() AND indexname = $1"
    async with store._pool.acquire() as conn:
        await conn.execute("CREATE INDEX idx_events_parent ON events(parent_event_id)")
    try:
        await store.init()
        dry = await store.drop_legacy_parent_index()
        async with store._pool.acquire() as conn:
            kept = await conn.fetchval(legacy, "idx_events_parent")
        dropped = await store.drop_legacy_parent_index(dry_run=False)
        again = await store.drop_legacy_parent_index(dry_run=False)
        async with store._pool.acquire() as conn:
            gone = await conn.fetchval(legacy, "idx_events_parent") is None
    finally:
        async with store._pool.acquire() as conn:
            await conn.execute("DROP INDEX IF EXISTS idx_events_parent")
    assert (dry, dropped, again) == (
        {"action": "dry_run"},
        {"action": "dropped"},
        {"action": "noop"},
    )
    assert kept == 1 and gone


async def test_init_seeds_the_index_marks_of_a_store_from_before_them(store) -> None:
    """A store from before the per-room high-water marks gains them at its
    first init(), seeded from its rooms and events, so the next commit takes
    the next index."""
    await store.create_room(Room(id="r1"))
    await store.create_room(Room(id="r2"))
    for _ in range(3):
        await store.commit_event("r1", _make_event())
    async with store._pool.acquire() as conn:
        await conn.execute("DROP TABLE event_sequences")
    await store.init()

    committed = await store.commit_event("r1", _make_event())

    assert committed.index == 3
    async with store._pool.acquire() as conn:
        seeded = await conn.fetch("SELECT room_id, next_index FROM event_sequences")
    assert {(row["room_id"], row["next_index"]) for row in seeded} == {("r1", 4), ("r2", 0)}


async def test_a_store_from_before_group_bindings_upgrades_its_rows(store) -> None:
    """The additive ``is_group`` column (RFC §5.7): an existing binding reads as
    not a group, and a group binding round-trips once the column is there."""
    await store.create_room(Room(id="r1"))
    binding = ChannelBinding(room_id="r1", channel_id="sms", channel_type="sms")
    await store.add_binding(binding)
    async with store._pool.acquire() as conn:
        await conn.execute("ALTER TABLE bindings DROP COLUMN is_group")
    await store.init()
    await store.init()
    assert await store.get_binding("r1", "sms") == binding
    grouped = binding.model_copy(update={"group": True})
    await store.update_binding(grouped)
    assert await store.get_binding("r1", "sms") == grouped


class TestEventFromRow:
    """A row the host reads with a query of its own maps as the store maps it."""

    async def test_a_raw_row_is_the_event_get_event_reads(self, store) -> None:
        from roomkit.store.postgres import PostgresStore

        await store.create_room(Room(id="r1"))
        committed = await store.commit_event("r1", _make_event(body="from a page"))
        async with store._pool.acquire() as conn:
            # A host's page query: every column of events, plus one of its own.
            row = await conn.fetchrow(
                "SELECT events.*, 'page-1' AS page FROM events WHERE id = $1", committed.id
            )

        assert PostgresStore.event_from_row(row) == await store.get_event(committed.id)


# -- RFC §8.5: an answer names the event it answers ---------------------------


async def test_an_answer_keeps_and_finds_the_event_it_answers(store) -> None:
    from roomkit.models.store_filter import EventFilter

    await store.create_room(Room(id="r1"))
    asked = await store.add_event(_make_event(body="la météo ?"))
    answer = await store.add_event(_make_event(body="8 degrés", responds_to=asked.id))
    await store.add_event(_make_event(body="autre chose"))

    stored = await store.get_event(answer.id)
    answers = await store.list_events(
        "r1", offset=0, limit=10, event_filter=EventFilter(responds_to=asked.id)
    )
    assert stored is not None and stored.responds_to == asked.id
    assert [e.id for e in answers] == [answer.id]


@pytest.mark.parametrize("column", ["addressed_to", "responds_to", "delivery_results"])
async def test_init_adds_the_column_to_an_events_table_without_it(store, column: str) -> None:
    from roomkit.store.postgres import PostgresStore

    await store.create_room(Room(id="r1"))
    old = await store.add_event(_make_event(body="avant"))
    async with store._pool.acquire() as conn:
        await conn.execute("DROP INDEX IF EXISTS idx_events_room_responds_to")
        await conn.execute(f"ALTER TABLE events DROP COLUMN {column}")

    upgraded = PostgresStore(dsn=POSTGRES_DSN)
    await upgraded.init(min_size=1, max_size=2)
    try:
        kept = await upgraded.get_event(old.id)
        async with upgraded._pool.acquire() as conn:
            found = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns"
                " WHERE table_schema = current_schema()"
                " AND table_name = 'events' AND column_name = $1",
                column,
            )
    finally:
        await upgraded.close()
    assert found == 1
    assert kept is not None
    assert (kept.addressed_to, kept.responds_to, kept.delivery_results) == (None, None, {})


async def test_init_records_when_a_room_stored_paused_before_paused_at_paused(store) -> None:
    """RFC §5.1: a store from before ``paused_at`` gains the column, and a
    room stored paused takes its last modification as the instant it paused."""
    from roomkit.models.enums import RoomStatus
    from roomkit.store.postgres import PostgresStore

    paused = await store.create_room(Room(id="r1", status=RoomStatus.PAUSED))
    await store.create_room(Room(id="r2"))
    async with store._pool.acquire() as conn:
        await conn.execute("ALTER TABLE rooms DROP COLUMN paused_at")

    upgraded = PostgresStore(dsn=POSTGRES_DSN)
    await upgraded.init(min_size=1, max_size=2)
    try:
        was_paused = await upgraded.get_room("r1")
        active = await upgraded.get_room("r2")
    finally:
        await upgraded.close()
    assert was_paused is not None and active is not None
    assert was_paused.paused_at == paused.updated_at
    assert active.paused_at is None


async def test_init_moves_the_author_registers_to_the_private_state(store) -> None:
    """RMK-631: a store from before keeps a room's register of authors in
    its metadata, which clients may read; init() moves it to the room's
    private state, once, and drops a malformed one. A copy that is not what it moved
    (a process from before wrote it meanwhile) stays for the room's next
    turn to merge: dropped, its ranks would be given again."""
    from roomkit.store.postgres import PostgresStore

    register = {"version": 3, "names": {"a" * 16: "b" * 16 + ":1"}}
    later = {"version": 3, "names": {"a" * 16: "b" * 16 + ":1 " + "c" * 16 + ":2"}}
    await store.create_room(Room(id="r1", metadata={"topic": "x", "author_register": register}))
    await store.create_room(Room(id="r2", metadata={"author_register": "junk"}))
    await store.create_room(Room(id="r3", metadata={"author_register": later}))
    async with store._pool.acquire() as conn:
        await conn.execute("DELETE FROM schema_version WHERE version >= 4")
        await conn.execute(
            "INSERT INTO private_state (room_id, key, value)"
            " VALUES ('r3', 'author_register', $1::jsonb)",
            register,
        )

    upgraded = PostgresStore(dsn=POSTGRES_DSN)
    await upgraded.init(min_size=1, max_size=2)
    try:
        moved, kept = (
            await upgraded.get_private_state("r1", "author_register"),
            await upgraded.get_room("r1"),
        )
        junk, emptied = (
            await upgraded.get_private_state("r2", "author_register"),
            await upgraded.get_room("r2"),
        )
        held, pending = (
            await upgraded.get_private_state("r3", "author_register"),
            await upgraded.get_room("r3"),
        )
    finally:
        await upgraded.close()
    assert moved == register
    assert kept is not None and kept.metadata == {"topic": "x"}
    assert junk is None
    assert emptied is not None and emptied.metadata == {}
    assert held == register
    assert pending is not None and pending.metadata == {"author_register": later}


async def test_a_private_state_write_racing_the_room_s_deletion_writes_nothing(store) -> None:
    """RMK-631: a write of a room's private state while the room is being
    deleted waits for the deletion, then writes nothing, as for a room the
    store does not hold, rather than raise the foreign key's violation."""
    import asyncpg

    await store.create_room(Room(id="r1"))
    deleter = await asyncpg.connect(POSTGRES_DSN)
    try:
        async with deleter.transaction():
            await deleter.execute("DELETE FROM rooms WHERE id = 'r1'")
            write = asyncio.create_task(
                store.set_private_state("r1", "author_register", {"version": 3, "names": {}})
            )
            await _until_blocked_by(deleter)
        await asyncio.wait_for(write, 2)
    finally:
        await deleter.close()
    assert await store.get_private_state("r1", "author_register") is None


async def _until_blocked_by(holder: Any) -> None:
    """Return once a session waits on a lock *holder*'s transaction holds."""
    deadline = asyncio.get_running_loop().time() + 2
    blocked = "SELECT count(*) FROM pg_stat_activity WHERE $1 = ANY(pg_blocking_pids(pid))"
    async with asyncio.timeout_at(deadline):
        while not await holder.fetchval(blocked, holder.get_server_pid()):
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("adding", ["add_event", "add_event_auto_index"])
async def test_an_add_waiting_on_a_commit_holds_nothing_the_commit_needs(
    store, adding: str
) -> None:
    """A commit holds the room, then its sequence. An add takes them in that order
    too, so that it waits on the room with the sequence free: in the other order,
    each waits on what the other holds, and the server kills one (RMK-1365)."""
    import asyncpg

    await store.create_room(Room(id="r1"))
    await store.add_event_auto_index("r1", _make_event(room_id="r1"))
    committer = await asyncpg.connect(POSTGRES_DSN)
    try:
        async with committer.transaction():
            await committer.execute("SELECT 1 FROM rooms WHERE id = 'r1' FOR UPDATE")
            if adding == "add_event":
                added = store.add_event(_make_event(room_id="r1", index=7))
            else:
                added = store.add_event_auto_index("r1", _make_event(room_id="r1"))
            add = asyncio.create_task(added)
            await _until_blocked_by(committer)
            await committer.execute(
                "UPDATE event_sequences SET next_index = next_index + 1 WHERE room_id = 'r1'"
            )
        await asyncio.wait_for(add, 2)
    finally:
        await committer.close()


@pytest.mark.parametrize("adding", ["add_event", "add_event_auto_index"])
async def test_an_add_holds_its_room_so_that_no_stream_of_them_starves_a_commit(
    store, adding: str
) -> None:
    """HITL 0affbb3b: an add waiting on the sequence holds the room in a mode the next add
    cannot share, and that a foreign key can: ``FOR NO KEY UPDATE``. The stress test
    ``test_a_stream_of_adds_starves_nothing_on_its_room`` shows why."""
    import asyncpg

    await store.create_room(Room(id="r1"))
    await store.add_event_auto_index("r1", _make_event(room_id="r1"))
    sequence_holder = await asyncpg.connect(POSTGRES_DSN)
    other = await asyncpg.connect(POSTGRES_DSN)
    try:
        async with sequence_holder.transaction():
            await sequence_holder.execute(
                "SELECT 1 FROM event_sequences WHERE room_id = 'r1' FOR UPDATE"
            )
            if adding == "add_event":
                added = store.add_event(_make_event(room_id="r1", index=7))
            else:
                added = store.add_event_auto_index("r1", _make_event(room_id="r1"))
            add = asyncio.create_task(added)
            await _until_blocked_by(sequence_holder)
            async with other.transaction():
                await other.execute("SELECT 1 FROM rooms WHERE id = 'r1' FOR KEY SHARE NOWAIT")
            with pytest.raises(asyncpg.LockNotAvailableError):
                async with other.transaction():
                    await other.execute(
                        "SELECT 1 FROM rooms WHERE id = 'r1' FOR NO KEY UPDATE NOWAIT"
                    )
        await asyncio.wait_for(add, 2)
    finally:
        await sequence_holder.close()
        await other.close()


@pytest.mark.parametrize("adding", ["add_event", "add_event_auto_index"])
async def test_an_add_to_a_missing_room_stays_the_foreign_key_s_refusal(
    store, adding: str
) -> None:
    """Holding the room first finds no row and refuses nothing itself: the refusal is
    still the write's, and stores nothing."""
    import asyncpg

    if adding == "add_event":
        added = store.add_event(_make_event(room_id="gone", index=7))
    else:
        added = store.add_event_auto_index("gone", _make_event(room_id="gone"))
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await added
    async with store._pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM event_sequences") == 0


async def _until_blocked_behind(holder: Any) -> None:
    """Return once a session waits on one that waits on *holder*."""
    deadline = asyncio.get_running_loop().time() + 2
    behind = (
        "SELECT count(*) FROM pg_stat_activity a, pg_stat_activity b"
        " WHERE b.pid = ANY(pg_blocking_pids(a.pid)) AND $1 = ANY(pg_blocking_pids(b.pid))"
    )
    async with asyncio.timeout_at(deadline):
        while not await holder.fetchval(behind, holder.get_server_pid()):
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("write", ["delete_event", "write_change_target"])
async def test_a_write_of_a_message_and_its_edits_waits_on_the_room_holding_none_of_them(
    store, write: str
) -> None:
    """A room's deletion holds the room, then takes its events in the order it finds
    them, a message edited twice after its first edit. A write of a message and its
    edits waits on the room first, so that it holds nothing the deletion needs: in the
    other order each waits on what the other holds, and the server kills one (RMK-1394)."""
    import asyncpg

    await store.create_room(Room(id="r1"))
    message = await store.commit_event("r1", _make_event(room_id="r1"))
    deleter = await asyncpg.connect(POSTGRES_DSN)
    try:
        async with deleter.transaction():
            await deleter.execute("SELECT 1 FROM rooms WHERE id = 'r1' FOR UPDATE")
            if write == "delete_event":
                written = store.delete_event("r1", message.id)
            else:
                written = store.write_change_target("r1", marked_deleted(message))
            task = asyncio.create_task(written)
            await _until_blocked_by(deleter)
            await deleter.execute("DELETE FROM rooms WHERE id = 'r1'")
        if write == "delete_event":
            assert await asyncio.wait_for(task, 2) == []
        else:
            with pytest.raises(EventNotFoundError):
                await asyncio.wait_for(task, 2)
    finally:
        await deleter.close()


@pytest.mark.stress
@pytest.mark.parametrize("write", ["delete_event", "write_change_target"])
async def test_a_room_deleted_under_a_message_and_its_edits_never_deadlocks(write: str) -> None:
    """Unmodified calls: a message edited twice, its writer racing the room's deletion, on
    a table small enough for the deletion to read it in heap order, where the first edit
    comes before the message. RMK-1394 measured 29 and 51 of 3200 killed before."""
    import asyncpg

    from roomkit.store.postgres import PostgresStore

    store = PostgresStore(dsn=POSTGRES_DSN)
    await store.init(min_size=1, max_size=40)
    prefix = uuid4().hex

    async def one_room(number: int) -> list[Any]:
        room_id = f"r-{prefix}-{number}"
        await store.create_room(Room(id=room_id))
        message = await store.commit_event(room_id, _make_event(room_id=room_id))
        for _ in range(2):
            edit = _make_event(room_id=room_id).model_copy(
                update={
                    "type": EventType.EDIT,
                    "content": EditContent(
                        target_event_id=message.id, new_content=TextContent(body="edited")
                    ),
                }
            )
            await store.commit_event(room_id, edit, target=message)

        async def deleted_later() -> bool:
            await asyncio.sleep(number % 5 / 1000)
            return await store.delete_room(room_id)

        if write == "delete_event":
            written = store.delete_event(room_id, message.id)
        else:
            written = store.write_change_target(room_id, marked_deleted(message))
        return await asyncio.gather(written, deleted_later(), return_exceptions=True)

    calls: list[Any] = []
    try:
        async with store._pool.acquire() as conn:
            await conn.execute("TRUNCATE rooms, events CASCADE")
            await conn.execute("ANALYZE events")
        for batch in range(80):
            pairs = await asyncio.gather(*(one_room(batch * 40 + n) for n in range(40)))
            calls += [outcome for pair in pairs for outcome in pair]
    finally:
        await store.close()
    deadlocks = [o for o in calls if isinstance(o, asyncpg.DeadlockDetectedError)]
    assert not deadlocks, f"{len(deadlocks)} of {len(calls)} calls killed"


@pytest.mark.parametrize("adding", ["add_event", "add_event_auto_index"])
async def test_an_add_holds_a_room_whose_id_came_back_before_its_sequence(
    store, adding: str
) -> None:
    """An add waits on a room being deleted, its id created again meanwhile with a
    sequence: its wait finds no row. Had it gone on to the sequence holding nothing, it
    would then wait on the room through the event's foreign key, and a deletion holding
    the new room would wait on that sequence: the server kills one (RMK-1394). The
    trigger stops the add at its insert, the moment that deletion needs."""
    import asyncpg

    key = 0x524D4B31333934  # This test's alone: "RMK1394".
    await store.create_room(Room(id="r1"))
    await store.add_event_auto_index("r1", _make_event(room_id="r1"))
    event = _make_event(room_id="r1", index=7)
    event = event.model_copy(
        update={"source": event.source.model_copy(update={"participant_id": "stopped"})}
    )
    outside = await asyncpg.connect(POSTGRES_DSN)
    deleter = await asyncpg.connect(POSTGRES_DSN)
    try:
        await outside.execute(
            "CREATE FUNCTION rk_test_stop_insert() RETURNS trigger AS $$"
            f" BEGIN PERFORM pg_advisory_xact_lock({key}); RETURN NEW; END $$ LANGUAGE plpgsql"
        )
        await outside.execute(
            "CREATE TRIGGER rk_test_stop_insert BEFORE INSERT ON events FOR EACH ROW"
            " WHEN (NEW.source_participant_id = 'stopped')"
            " EXECUTE FUNCTION rk_test_stop_insert()"
        )
        await outside.execute("SELECT pg_advisory_lock($1)", key)
        async with deleter.transaction():
            await deleter.execute("DELETE FROM rooms WHERE id = 'r1'")
            if adding == "add_event":
                added = store.add_event(event)
            else:
                added = store.add_event_auto_index("r1", event)
            add = asyncio.create_task(added)
            await _until_blocked_by(deleter)
            await deleter.execute("INSERT INTO rooms (id) VALUES ('r1')")
            await deleter.execute(
                "INSERT INTO event_sequences (room_id, next_index) VALUES ('r1', 5)"
            )
        await _until_blocked_by(outside)
        delete = asyncio.create_task(store.delete_room("r1"))
        await _until_blocked_behind(outside)
        await outside.execute("SELECT pg_advisory_unlock($1)", key)
        stored, deleted = await asyncio.wait_for(asyncio.gather(add, delete), 3)
        assert stored.id == event.id
        assert deleted is True
    finally:
        await outside.execute("DROP TRIGGER IF EXISTS rk_test_stop_insert ON events")
        await outside.execute("DROP FUNCTION IF EXISTS rk_test_stop_insert()")
        await outside.close()
        await deleter.close()


@pytest.mark.parametrize("adding", ["add_event", "add_event_auto_index"])
async def test_a_missing_room_s_refusal_holds_the_room_that_came_back(store, adding: str) -> None:
    """An add's first statement finds no room; the room is created before the refusal's
    statement, with no sequence, and a commit holds it and reserves its index. The
    refusal holds that room before it writes anything: had it written the sequence
    first, it would wait on the room through the foreign key while the commit waits on
    that sequence, and the server kills one (RMK-1394). The trigger stops the add after
    its first statement, once per transaction; the commit's session passes it."""
    import asyncpg

    key = 0x524D4B31333935  # This test's alone: "RMK1395".
    outside = await asyncpg.connect(POSTGRES_DSN)
    committer = await asyncpg.connect(POSTGRES_DSN)
    try:
        await outside.execute(
            "CREATE FUNCTION rk_test_stop_once() RETURNS trigger AS $$ BEGIN"
            " IF current_setting('rk_test.stopped', true) IS DISTINCT FROM 'yes' THEN"
            " PERFORM set_config('rk_test.stopped', 'yes', true);"
            f" PERFORM pg_advisory_xact_lock({key}); END IF; RETURN NULL; END"
            " $$ LANGUAGE plpgsql"
        )
        await outside.execute(
            "CREATE TRIGGER rk_test_stop_once AFTER INSERT ON event_sequences"
            " FOR EACH STATEMENT EXECUTE FUNCTION rk_test_stop_once()"
        )
        await outside.execute("SELECT pg_advisory_lock($1)", key)
        if adding == "add_event":
            added = store.add_event(_make_event(room_id="r1", index=7))
        else:
            added = store.add_event_auto_index("r1", _make_event(room_id="r1"))
        add = asyncio.create_task(added)
        await _until_blocked_by(outside)
        await committer.execute("SET rk_test.stopped = 'yes'")
        await committer.execute("INSERT INTO rooms (id) VALUES ('r1')")
        async with committer.transaction():
            await committer.execute("SELECT 1 FROM rooms WHERE id = 'r1' FOR UPDATE")
            await outside.execute("SELECT pg_advisory_unlock($1)", key)
            await _until_blocked_by(committer)
            await committer.execute(
                "INSERT INTO event_sequences(room_id, next_index) VALUES('r1', 1)"
                " ON CONFLICT(room_id) DO UPDATE SET next_index = event_sequences.next_index + 1"
            )
        stored = await asyncio.wait_for(add, 3)
        assert [e.id for e in await store.list_events("r1")] == [stored.id]
    finally:
        await outside.execute("DROP TRIGGER IF EXISTS rk_test_stop_once ON event_sequences")
        await outside.execute("DROP FUNCTION IF EXISTS rk_test_stop_once()")
        await outside.close()
        await committer.close()


@pytest.mark.stress
async def test_a_room_deleted_and_created_again_under_its_adds_never_deadlocks() -> None:
    """Adds at full pace on one room id, which a loop deletes and creates again: an add is
    refused, or stored, and never killed (RMK-1394, measured 8 kills in 8 s before)."""
    import asyncpg

    from roomkit.store.postgres import PostgresStore

    store = PostgresStore(dsn=POSTGRES_DSN)
    await store.init(min_size=1, max_size=12)
    room_id = f"r-{uuid4().hex}"
    await store.create_room(Room(id=room_id))
    stop = asyncio.Event()
    killed: list[BaseException] = []

    async def adding() -> None:
        while not stop.is_set():
            try:
                await store.add_event_auto_index(room_id, _make_event(room_id=room_id))
            except asyncpg.ForeignKeyViolationError:
                pass
            except asyncpg.DeadlockDetectedError as error:
                killed.append(error)

    async def churning() -> None:
        while not stop.is_set():
            try:
                await store.delete_room(room_id)
                await store.create_room(Room(id=room_id))
            except asyncpg.DeadlockDetectedError as error:
                killed.append(error)

    tasks = [asyncio.create_task(adding()) for _ in range(8)]
    tasks.append(asyncio.create_task(churning()))
    try:
        await asyncio.sleep(8)
    finally:
        stop.set()
        await asyncio.gather(*tasks)
        await store.close()
    assert not killed, f"{len(killed)} calls killed"


async def test_a_dedupe_waiting_on_a_room_holds_nothing_its_commit_needs(store) -> None:
    """A commit with a target holds the room, then that event. A repair of duplicate
    indexes holds the room first too: renumbering its events first, it would hold what
    the commit needs while it waits on the room for its counters (RMK-1394)."""
    import asyncpg

    async with store._pool.acquire() as conn:
        await conn.execute(
            "DROP INDEX IF EXISTS idx_events_room_index; "
            "CREATE INDEX idx_events_room_index ON events(room_id, index)"
        )
    try:
        await store.create_room(Room(id="r1"))
        for number in range(3):
            await store.add_event(_make_event(room_id="r1", index=0, id=f"d{number}"))
        committer = await asyncpg.connect(POSTGRES_DSN)
        try:
            async with committer.transaction():
                await committer.execute("SELECT 1 FROM rooms WHERE id = 'r1' FOR UPDATE")
                repair = asyncio.create_task(store.dedupe_event_indices(dry_run=False))
                await _until_blocked_by(committer)
                await committer.execute("UPDATE events SET metadata = '{}'")
            report = await asyncio.wait_for(repair, 2)
        finally:
            await committer.close()
        assert report["action"] == "repaired"
    finally:
        async with store._pool.acquire() as conn:
            await conn.execute("TRUNCATE rooms, events CASCADE")
            await conn.execute(
                "DROP INDEX IF EXISTS idx_events_room_index; "
                "CREATE UNIQUE INDEX idx_events_room_index ON events(room_id, index)"
            )


async def test_a_boot_waiting_on_a_room_holds_nothing_its_commit_needs(store) -> None:
    """A commit with a target holds the room, then that event. The boot that rewrites a
    room's events (version 3), then its register (version 4), holds the room first: the
    other way, it would hold what the commit needs while it waits on the room (RMK-1394)."""
    import asyncpg

    await store.create_room(
        Room(id="r1", metadata={"author_register": {"version": 3, "names": {}}})
    )
    await store.add_binding(
        ChannelBinding(
            room_id="r1", channel_id="ch1", channel_type=ChannelType.SMS, visibility="ws"
        )
    )
    await store.add_event(_make_event(room_id="r1", visibility="all"))
    async with store._pool.acquire() as conn:
        await conn.execute("DELETE FROM schema_version WHERE version >= 3")
    committer = await asyncpg.connect(POSTGRES_DSN)
    try:
        async with committer.transaction():
            await committer.execute("SELECT 1 FROM rooms WHERE id = 'r1' FOR UPDATE")
            boot = asyncio.create_task(store.init(min_size=1, max_size=3))
            await _until_blocked_by(committer)
            await committer.execute("UPDATE events SET channel_data = '{}'")
        await asyncio.wait_for(boot, 2)
    finally:
        await committer.close()
    assert [e.visibility for e in await store.list_events("r1")] == ["ws"]


@pytest.mark.stress
async def test_the_writers_of_one_room_at_once_never_deadlock() -> None:
    """40 commits, 40 automatic adds and 40 adds at their own index, on one room, over
    40 connections: none is killed for a deadlock (RMK-1365). The own indexes stay far
    apart, so that a reserved index never meets one."""
    import asyncpg

    from roomkit.store.postgres import PostgresStore

    store = PostgresStore(dsn=POSTGRES_DSN)
    await store.init(min_size=1, max_size=40)
    room_id = f"r-{uuid4().hex}"
    await store.create_room(Room(id=room_id))
    outcomes: list[Any] = []
    try:
        for writes in range(8):
            outcomes += await asyncio.gather(
                *(store.commit_event(room_id, _make_event(room_id=room_id)) for _ in range(40)),
                *(
                    store.add_event_auto_index(room_id, _make_event(room_id=room_id))
                    for _ in range(40)
                ),
                *(
                    store.add_event(
                        _make_event(room_id=room_id, index=(writes * 40 + own + 1) * 1_000_000)
                    )
                    for own in range(40)
                ),
                return_exceptions=True,
            )
    finally:
        await store.close()
    deadlocks = [o for o in outcomes if isinstance(o, asyncpg.DeadlockDetectedError)]
    assert not deadlocks, f"{len(deadlocks)} of {len(outcomes)} calls killed"
    assert not [o for o in outcomes if isinstance(o, BaseException)]


@pytest.mark.stress
@pytest.mark.parametrize("waiting", ["commit_event", "update_room_if"])
async def test_a_stream_of_adds_starves_nothing_on_its_room(waiting: str) -> None:
    """Four sessions adding to one room without a pause: what takes the room's row
    ``FOR UPDATE`` still gets its turn (HITL 0affbb3b). An add holds the room while it
    queues on the sequence, and a lock the next add could share would never free it."""
    from roomkit.store.postgres import PostgresStore

    store = PostgresStore(dsn=POSTGRES_DSN)
    await store.init(min_size=1, max_size=8)
    room_id = f"r-{uuid4().hex}"
    room = await store.create_room(Room(id=room_id))
    stop = asyncio.Event()

    async def adding() -> None:
        while not stop.is_set():
            await store.add_event_auto_index(room_id, _make_event(room_id=room_id))

    adders = [asyncio.create_task(adding()) for _ in range(4)]
    try:
        await asyncio.sleep(0.05)
        if waiting == "commit_event":
            await asyncio.wait_for(store.commit_event(room_id, _make_event(room_id=room_id)), 2)
        else:
            await asyncio.wait_for(
                store.update_room_if(
                    room, status=room.status, paused_at=room.paused_at, last_activity_at=None
                ),
                2,
            )
    finally:
        stop.set()
        await asyncio.gather(*adders)
        await store.close()


@pytest.mark.parametrize("write", ["commit_event", "add_event", "add_event_auto_index"])
async def test_a_held_id_is_the_driver_s_refusal_whatever_the_index_order(
    store, write: str
) -> None:
    """RFC §14.3: a second write of a stored event, its key held with it, is
    the database's refusal of the id (``UniqueViolationError``), and another
    event under the key is a taken key, whichever unique index Postgres
    checks first: after ``REINDEX ... CONCURRENTLY`` the primary key is the
    newest index, checked last."""
    import asyncpg

    from roomkit.core.exceptions import IdempotencyKeyTakenError

    async with store._pool.acquire() as conn:
        await conn.execute("REINDEX INDEX CONCURRENTLY events_pkey")
    await store.create_room(Room(id="r1"))
    held = await store.commit_event("r1", _make_event(idempotency_key="k"))
    writes = {
        "commit_event": lambda event: store.commit_event("r1", event),
        "add_event": lambda event: store.add_event(event.model_copy(update={"index": 99})),
        "add_event_auto_index": lambda event: store.add_event_auto_index("r1", event),
    }

    with pytest.raises(asyncpg.UniqueViolationError):
        await writes[write](held)
    with pytest.raises(IdempotencyKeyTakenError):
        await writes[write](_make_event(idempotency_key="k"))
    assert [e.id for e in await store.list_events("r1")] == [held.id]


# Open transactions a booting worker meets: readers of the tables init()
# once added columns to, a writer of a room, and a commit reserving an index.
_OPEN_TRANSACTIONS = {
    **{
        f"reader of {table}": f"SELECT count(*) FROM {table}"
        for table in ("events", "rooms", "bindings", "participants", "identity_addresses")
    },
    "writer of a room": "UPDATE rooms SET updated_at = now() WHERE id = 'r1'",
    "commit reserving an index": (
        "UPDATE event_sequences SET next_index = next_index + 1 WHERE room_id = 'r1'"
    ),
}


@pytest.mark.parametrize("holder", list(_OPEN_TRANSACTIONS))
async def test_a_second_init_waits_on_no_open_transaction(store, holder: str) -> None:
    """``init()`` on an up-to-date schema takes no lock an open transaction
    on its tables holds against it: no ``ALTER TABLE`` (ACCESS EXCLUSIVE,
    even when ``IF NOT EXISTS`` finds the column), no ``CREATE INDEX``
    (SHARE, which waits for every writer, even when ``IF NOT EXISTS`` finds
    the index), and no write of a row a commit holds."""
    import asyncpg

    from roomkit.store.postgres import PostgresStore

    await store.create_room(Room(id="r1"))
    await store.commit_event("r1", _make_event())
    other = await asyncpg.connect(POSTGRES_DSN)
    pool = await asyncpg.create_pool(
        POSTGRES_DSN, min_size=1, max_size=1, server_settings={"lock_timeout": "500ms"}
    )
    try:
        async with other.transaction():
            await other.execute(_OPEN_TRANSACTIONS[holder])
            await PostgresStore(pool=pool).init()
    finally:
        await pool.close()
        await other.close()


# What an older deployment left in its schema: a v1 ``rooms.data`` column,
# the unique idx_participants_channel, a non-unique idx_events_room_index, the
# two-column identity key, the single-column parent index and the v0.107.0
# idempotency index.
_AN_OLDER_DEPLOYMENT = (
    "ALTER TABLE rooms ADD COLUMN data JSONB",
    "DROP INDEX idx_participants_channel",
    "CREATE UNIQUE INDEX idx_participants_channel ON participants(room_id, channel_id)",
    "DROP INDEX idx_events_room_index",
    "CREATE INDEX idx_events_room_index ON events(room_id, index)",
    "ALTER TABLE identity_addresses DROP CONSTRAINT identity_addresses_pkey,"
    " ADD PRIMARY KEY (channel_type, address)",
    "CREATE INDEX idx_events_parent ON events(parent_event_id)",
    "CREATE UNIQUE INDEX idx_events_idempotency ON events(room_id, idempotency_key)"
    " WHERE idempotency_key IS NOT NULL",
)

_GUARDED_INDEXES = [
    "idx_events_idempotency",
    "idx_events_room_idempotency",
    "idx_participants_identity",
    "idx_participants_channel",
    "idx_events_room_index",
    "idx_events_parent",
]


async def _identity_key(conn: Any, schema: str) -> int:
    return await conn.fetchval(
        "SELECT i.indexrelid::int FROM pg_index i"
        " WHERE i.indrelid = to_regclass(format('%I.%I', $1::text, 'identity_addresses'))"
        " AND i.indisprimary",
        schema,
    )


@pytest.fixture
async def second_schema(store) -> Any:
    """A second deployment's schema in the store's database, and a pool whose
    ``search_path`` is ``tenant, <the store's schema>``: the store's own
    deployment shows through it to every name the tenant does not hold."""
    import asyncpg

    tenant = f"tenant_{os.getpid()}"
    async with store._pool.acquire() as conn:
        home = await conn.fetchval("SELECT current_schema()")
        await conn.execute(f'CREATE SCHEMA "{tenant}"')
    pool = await asyncpg.create_pool(
        POSTGRES_DSN, min_size=1, max_size=1, server_settings={"search_path": f"{tenant},{home}"}
    )
    try:
        yield home, tenant, pool
    finally:
        await pool.close()
        async with store._pool.acquire() as conn:
            await conn.execute(f'DROP SCHEMA "{tenant}" CASCADE')


async def test_init_in_a_second_schema_keeps_each_schema_its_own_indexes(
    store, second_schema, caplog: pytest.LogCaptureFixture
) -> None:
    """Two deployments in one database, one per schema: ``init()`` with a
    ``search_path`` of ``tenant, <first schema>`` reads and writes the
    tenant's schema only. The first schema holds what an older deployment
    left: the tenant is not taken for a v1 schema, creates its own indexes,
    leaves the first deployment's as they are, logs no degraded index of
    another schema, and does not rebuild its identity key at each boot; its
    maintenance calls read and drop its own indexes only."""
    from roomkit.store.postgres import PostgresStore

    home, tenant, pool = second_schema
    async with store._pool.acquire() as conn:
        for statement in _AN_OLDER_DEPLOYMENT:
            await conn.execute(statement)
    try:
        other = PostgresStore(pool=pool)
        with caplog.at_level("WARNING", logger="roomkit.store.postgres"):
            await other.init()
        async with pool.acquire() as conn:
            key = await _identity_key(conn, tenant)
        await other.init()
        async with pool.acquire() as conn:
            key_again = await _identity_key(conn, tenant)
        report = await other.dedupe_event_indices()
        dropped = await other.drop_legacy_parent_index(dry_run=False)
        async with store._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT schemaname, indexname, indexdef LIKE '%UNIQUE%' AS is_unique"
                " FROM pg_indexes"
                " WHERE schemaname = ANY($1::text[]) AND indexname = ANY($2::text[])",
                [home, tenant],
                _GUARDED_INDEXES,
            )
    finally:
        async with store._pool.acquire() as conn:
            await conn.execute("ALTER TABLE rooms DROP COLUMN IF EXISTS data")
            await conn.execute("DROP INDEX IF EXISTS idx_events_parent")
        await store.init()
    assert {(row["schemaname"], row["indexname"], row["is_unique"]) for row in rows} == {
        (home, "idx_events_idempotency", True),
        (home, "idx_events_room_idempotency", False),
        (home, "idx_participants_identity", False),
        (home, "idx_participants_channel", True),
        (home, "idx_events_room_index", False),
        (home, "idx_events_parent", False),
        (tenant, "idx_events_room_idempotency", False),
        (tenant, "idx_participants_identity", False),
        (tenant, "idx_participants_channel", False),
        (tenant, "idx_events_room_index", True),
    }
    assert not [r for r in caplog.records if "idx_events_room_index" in r.getMessage()]
    assert key == key_again
    assert report["now_unique"] is True
    assert dropped == {"action": "noop"}


async def test_migrate_in_a_second_schema_drops_its_own_tables_only(store, second_schema) -> None:
    """The destructive v1 → v2 ``migrate()`` drops the tables of the schema
    its session writes to: a table the v1 tenant lacks is not taken from the
    other deployment's schema through ``search_path``."""
    from roomkit.store.postgres import PostgresStore

    home, _tenant, pool = second_schema
    other = PostgresStore(pool=pool)
    await other.init()
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE rooms ADD COLUMN data JSONB")
        await conn.execute("DROP TABLE read_markers")

    report = await other.migrate(dry_run=False, confirm=True)

    async with store._pool.acquire() as conn:
        kept = await conn.fetchval(
            "SELECT to_regclass(format('%I.%I', $1::text, 'read_markers')) IS NOT NULL", home
        )
    assert report["action"] == "migrated"
    assert kept

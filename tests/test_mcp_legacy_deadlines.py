"""Dependency network deadlines remain distinct from RoomKit operation bounds."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from roomkit.tools import MCPToolProvider
from tests.mcp_legacy_peer import LegacySSEPeer

sse = pytest.importorskip("mcp.client.sse")


@asynccontextmanager
async def stalled_tls() -> AsyncIterator[tuple[str, asyncio.Event]]:
    started = asyncio.Event()
    tasks: set[asyncio.Task[Any]] = set()

    async def hold(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            assert await reader.read(4096)  # An actual TLS ClientHello reached the peer.
            started.set()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()

    def accepted(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(hold(reader, writer))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    server = await asyncio.start_server(accepted, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        yield f"https://127.0.0.1:{port}/mcp", started
    finally:
        server.close()
        await server.wait_closed()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_legacy_default_connect_timeout_includes_the_tls_handshake() -> None:
    async with stalled_tls() as (url, started):
        provider = MCPToolProvider(url, transport="sse", discover=False, connect_timeout=30)
        before = time.monotonic()
        with pytest.raises(ExceptionGroup) as raised:
            await asyncio.wait_for(provider.__aenter__(), 8)
        elapsed = time.monotonic() - before
        assert started.is_set()
        assert 4.5 <= elapsed < 8
        assert len(raised.value.exceptions) == 1
        assert isinstance(raised.value.exceptions[0], httpx.ConnectTimeout)
        assert not provider.connected


@pytest.fixture
def short_read(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, float | None]]:
    original = httpx.Timeout
    observed: list[dict[str, float | None]] = []

    class ReadBudget(original):
        def __init__(self, timeout: Any, **kwargs: Any) -> None:
            super().__init__(timeout, **kwargs)
            observed.append(self.as_dict())
            assert self.connect == self.write == self.pool == 5
            assert self.read == 300
            self.read = 0.2

    signature = inspect.signature(sse.sse_client)
    assert signature.parameters["timeout"].default == 5
    assert signature.parameters["sse_read_timeout"].default == 300
    monkeypatch.setattr(httpx, "Timeout", ReadBudget)
    return observed


class ReadPeer(LegacySSEPeer):
    def __init__(self, stage: str) -> None:
        super().__init__({"replies": []})
        self.stage = stage

    async def _get(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.connections:
            await super()._get(reader, writer)
            return
        self.connections += 1
        self.get_opened.set()
        if self.stage == "endpoint":
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
            await writer.drain()
        try:
            await reader.read()
        finally:
            self.closed_connections += 1
            self.get_closed.set()


@pytest.mark.parametrize("stage", ["headers", "endpoint"])
async def test_legacy_read_timeout_at_opening_preserves_its_error_path_and_reentry(
    stage: str, short_read: list[dict[str, float | None]], caplog: pytest.LogCaptureFixture
) -> None:
    async with ReadPeer(stage) as peer:
        provider = MCPToolProvider(peer.url, transport="sse", discover=False, connect_timeout=0.8)
        expected = ExceptionGroup if stage == "headers" else TimeoutError
        with pytest.raises(expected) as raised:
            await provider.__aenter__()
        if stage == "headers":
            assert len(raised.value.exceptions) == 1
            assert isinstance(raised.value.exceptions[0], httpx.ReadTimeout)
        else:
            assert any(
                record.exc_info and isinstance(record.exc_info[1], httpx.ReadTimeout)
                for record in caplog.records
            )
        assert not provider.connected
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        assert peer.targets == [("GET", "/mcp")]
        async with provider:
            assert provider.connected
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        assert (peer.connections, peer.closed_connections) == (2, 2)
        assert short_read == [{"connect": 5, "read": 300, "write": 5, "pool": 5}] * 2
    assert peer.problems == []


class PostReadPeer(LegacySSEPeer):
    def __init__(self, stage: str) -> None:
        super().__init__({"replies": []})
        self.stage = stage
        self.post_held = asyncio.Event()

    async def _send(self, writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
        if self.connections != 1:
            await super()._send(writer, status, body)
            return
        self.post_held.set()
        if self.stage == "body":
            writer.write(b"HTTP/1.1 202 Accepted\r\nContent-Length: 1\r\n\r\n")
            await writer.drain()
        assert self._stream is not None
        # Keep the independent GET reader alive while the POST read stalls.
        while not self.get_closed.is_set():
            self._stream.write(b": keepalive\n\n")
            await self._stream.drain()
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("stage", ["headers", "body"])
async def test_legacy_post_read_timeout_stops_the_writer_under_the_opening_bound(
    stage: str, short_read: list[dict[str, float | None]], caplog: pytest.LogCaptureFixture
) -> None:
    async with PostReadPeer(stage) as peer:
        provider = MCPToolProvider(peer.url, transport="sse", discover=False, connect_timeout=0.8)
        with pytest.raises(TimeoutError, match="within 0.8 s"):
            await provider.__aenter__()
        assert peer.post_held.is_set()
        assert not provider.connected
        assert any(
            record.message == "Error in post_writer"
            and record.exc_info
            and isinstance(record.exc_info[1], httpx.ReadTimeout)
            for record in caplog.records
        )
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        assert [item["method"] for item in peer.messages] == ["initialize"]
        assert peer.targets == [("GET", "/mcp"), ("POST", "/messages?session=1")]
        async with provider:
            assert provider.connected
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        assert (peer.connections, peer.closed_connections) == (2, 2)
        assert short_read == [{"connect": 5, "read": 300, "write": 5, "pool": 5}] * 2
    assert peer.problems == []


class SlowPostPeer(LegacySSEPeer):
    def __init__(self) -> None:
        super().__init__({"replies": []})
        self.posts = 0
        self.initialized = asyncio.Event()

    async def _send(self, writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
        self.posts += 1
        if self.posts == 1:
            await asyncio.sleep(5.1)
        else:
            self.initialized.set()
        await super()._send(writer, status, body)


async def test_legacy_post_response_can_take_longer_than_the_five_second_write_budget() -> None:
    async with SlowPostPeer() as peer:
        provider = MCPToolProvider(peer.url, transport="sse", discover=False, connect_timeout=30)
        before = time.monotonic()
        async with provider:
            assert provider.connected
            assert time.monotonic() - before >= 5
            await asyncio.wait_for(peer.initialized.wait(), 1)
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        assert [item["method"] for item in peer.messages] == [
            "initialize",
            "notifications/initialized",
        ]
    assert peer.problems == []


class PacedHeadersPeer(LegacySSEPeer):
    async def _get(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self._stream = writer
        for fragment in [b"HTTP/1.1 200 OK\r\n", b"Content-Type: text/event-stream\r\n", b"\r\n"]:
            await asyncio.sleep(0.1)
            writer.write(fragment)
            await writer.drain()
        writer.write(f"event: endpoint\ndata: {self._endpoint()}\n\n".encode())
        await writer.drain()
        self.get_opened.set()
        try:
            await reader.read()
        finally:
            self.closed_connections += 1
            self.get_closed.set()


async def test_read_budget_restarts_after_each_fragment_of_response_headers(
    short_read: list[dict[str, float | None]],
) -> None:
    async with PacedHeadersPeer(
        {"replies": [{"method": "host/read", "params": {}, "result": {"ok": True}}]}
    ) as peer:
        async with MCPToolProvider(
            peer.url, transport="sse", discover=False, connect_timeout=1
        ) as provider:
            assert provider.connected
            assert await provider.request("host/read", {}) == {"ok": True}
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        assert short_read == [{"connect": 5, "read": 300, "write": 5, "pool": 5}]
    assert peer.problems == []


async def test_idle_connected_get_read_timeout_fails_a_pending_request_and_allows_reentry(
    short_read: list[dict[str, float | None]], caplog: pytest.LogCaptureFixture
) -> None:
    from mcp.shared.exceptions import McpError

    peer = LegacySSEPeer(
        {
            "replies": [
                {"method": "host/held", "params": {}, "hold": True},
                {"method": "host/read", "params": {}, "result": {"ok": True}},
            ]
        }
    )
    async with peer:
        provider = MCPToolProvider(peer.url, transport="sse", discover=False)
        async with provider:
            with pytest.raises(McpError, match="Connection closed"):
                await provider.request("host/held", {}, timeout=1)
            assert any(
                record.exc_info and isinstance(record.exc_info[1], httpx.ReadTimeout)
                for record in caplog.records
            )
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        async with provider:
            assert await provider.request("host/read", {}) == {"ok": True}
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        assert (peer.connections, peer.closed_connections) == (2, 2)
        assert short_read == [{"connect": 5, "read": 300, "write": 5, "pool": 5}] * 2
    assert peer.problems == []


class PacedBodyPeer(LegacySSEPeer):
    async def _get(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self._stream = writer
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
        await writer.drain()
        for _ in range(3):
            await asyncio.sleep(0.1)
            writer.write(b": keepalive\n\n")
            await writer.drain()
        writer.write(f"event: endpoint\ndata: {self._endpoint()}\n\n".encode())
        await writer.drain()
        self.get_opened.set()
        try:
            await reader.read()
        finally:
            self.closed_connections += 1
            self.get_closed.set()


async def test_read_budget_restarts_between_sse_chunks_before_the_endpoint(
    short_read: list[dict[str, float | None]],
) -> None:
    peer = PacedBodyPeer(
        {"replies": [{"method": "host/read", "params": {}, "result": {"ok": True}}]}
    )
    async with peer:
        async with MCPToolProvider(
            peer.url, transport="sse", discover=False, connect_timeout=1
        ) as provider:
            assert provider.connected
            assert await provider.request("host/read", {}) == {"ok": True}
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        assert short_read == [{"connect": 5, "read": 300, "write": 5, "pool": 5}]
    assert peer.problems == []

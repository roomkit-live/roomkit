"""A stalled large POST can reach the legacy SDK's socket write deadline."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import httpx2
import pytest
from mcp.shared.exceptions import MCPError

from roomkit.tools import MCPToolProvider
from tests.mcp_legacy_peer import LegacySSEPeer, logged

pytest.importorskip("mcp.client.sse")


class BlockedUpload(LegacySSEPeer):
    def __init__(self, early_response: bool = False) -> None:
        super().__init__({"replies": []})
        self.early_response = early_response
        self.upload_started = asyncio.Event()
        self.release = asyncio.Event()

    async def _post(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict[str, str]
    ) -> None:
        if int(headers.get("content-length", "0")) < 1024 * 1024:
            await super()._post(reader, writer, headers)
            return
        writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        if self.early_response:
            writer.write(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
        self.upload_started.set()
        # The real socket and StreamReader stop draining the body; no SDK method is replaced.
        await self.release.wait()


@pytest.mark.parametrize("early_response", [False, True])
async def test_a_stalled_upload_reaches_the_sdk_write_deadline_then_closes_its_writer(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, early_response: bool
) -> None:
    original = httpx2.Timeout
    observed: list[dict[str, float | None]] = []

    class WriteBudget(original):
        def __init__(self, timeout: Any, **kwargs: Any) -> None:
            super().__init__(timeout, **kwargs)
            observed.append(self.as_dict())
            assert self.connect == self.write == self.pool == 5
            assert self.read == 300
            self.write = 0.04

    monkeypatch.setattr(httpx2, "Timeout", WriteBudget)
    async with BlockedUpload(early_response) as peer:
        async with MCPToolProvider(peer.url, transport="sse", discover=False) as provider:
            with pytest.raises(TimeoutError):
                await provider.call_tool("upload", {"body": "x" * (8 * 1024 * 1024)}, timeout=0.3)
            assert peer.upload_started.is_set()
            assert any(
                record.message == "Error in post_writer" and logged(record, httpx2.WriteTimeout)
                for record in caplog.records
            )
            # The SDK reports the writer it lost as a closed connection.
            with pytest.raises(MCPError, match="Connection closed"):
                await provider.request("after/failed-write", {}, timeout=1)
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        assert peer.targets == [
            ("GET", "/mcp"),
            ("POST", "/messages?session=1"),
            ("POST", "/messages?session=1"),
            ("POST", "/messages?session=1"),
        ]
        assert observed == [{"connect": 5, "read": 300, "write": 5, "pool": 5}]
    assert peer.problems == []


async def test_concurrent_callers_wait_before_http_pool_acquisition() -> None:
    import inspect

    from tests.test_mcp_legacy_writer import HeldPostPeer

    limits = inspect.signature(httpx2.AsyncClient).parameters["limits"].default
    assert limits.max_connections == 100
    async with HeldPostPeer("host") as peer:
        async with MCPToolProvider(peer.url, transport="sse", discover=False) as provider:
            await asyncio.wait_for(peer.post_held.wait(), 1)
            started = 0
            all_started = asyncio.Event()

            async def call() -> None:
                nonlocal started
                started += 1
                if started == 64:
                    all_started.set()
                await provider.request("never/sent", {}, timeout=30)

            callers = [asyncio.create_task(call()) for _ in range(64)]
            await all_started.wait()
            # Fixture-owned accepted sockets, not an SDK pool or private method mock.
            assert len(peer._writers) == 2
            assert len(peer.targets) == 3  # GET, completed initialize, held initialized.
            for caller in callers:
                caller.cancel()
            cancelled = await asyncio.gather(*callers, return_exceptions=True)
            assert all(isinstance(value, asyncio.CancelledError) for value in cancelled)
            peer.release_post.set()
            assert await provider.request("host/read", {"name": "kept"}, timeout=1) == {
                "done": True
            }
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        # None of the 64 requests went out; the SDK tells the server of each
        # once its writer is free, as it counts a request its writer took as
        # sent.
        methods = [message["method"] for message in peer.messages]
        assert methods[:2] == ["initialize", "notifications/initialized"]
        assert methods[2:-1] == ["notifications/cancelled"] * 64
        assert methods[-1] == "host/read"
        assert len(peer.targets) == 4 + 64
    assert peer.problems == []

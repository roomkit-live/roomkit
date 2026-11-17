"""A stalled large POST can reach the legacy SDK's socket write deadline."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import anyio
import httpx
import pytest

from roomkit.tools import MCPToolProvider
from tests.mcp_legacy_peer import LegacySSEPeer

pytest.importorskip("mcp.client.sse")


class BlockedUpload(LegacySSEPeer):
    def __init__(self) -> None:
        super().__init__({"replies": []})
        self.upload_started = asyncio.Event()
        self.release = asyncio.Event()

    async def _post(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict[str, str]
    ) -> None:
        if int(headers.get("content-length", "0")) < 1024 * 1024:
            await super()._post(reader, writer, headers)
            return
        writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        self.upload_started.set()
        # The real socket and StreamReader stop draining the body; no SDK method is replaced.
        await self.release.wait()


async def test_a_stalled_upload_reaches_the_sdk_write_deadline_then_closes_its_writer(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original = httpx.Timeout
    observed: list[dict[str, float | None]] = []

    class WriteBudget(original):
        def __init__(self, timeout: Any, **kwargs: Any) -> None:
            super().__init__(timeout, **kwargs)
            observed.append(self.as_dict())
            assert self.connect == self.write == self.pool == 5
            assert self.read == 300
            self.write = 0.04

    monkeypatch.setattr(httpx, "Timeout", WriteBudget)
    async with BlockedUpload() as peer:
        async with MCPToolProvider(peer.url, transport="sse", discover=False) as provider:
            with pytest.raises(TimeoutError):
                await provider.call_tool("upload", {"body": "x" * (8 * 1024 * 1024)}, timeout=0.3)
            assert peer.upload_started.is_set()
            assert any(
                record.message == "Error in post_writer"
                and record.exc_info
                and isinstance(record.exc_info[1], httpx.WriteTimeout)
                for record in caplog.records
            )
            with pytest.raises(anyio.ClosedResourceError):
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

    limits = inspect.signature(httpx.AsyncClient).parameters["limits"].default
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
        assert len(peer.targets) == 4
        assert [message["method"] for message in peer.messages] == [
            "initialize",
            "notifications/initialized",
            "host/read",
        ]
    assert peer.problems == []

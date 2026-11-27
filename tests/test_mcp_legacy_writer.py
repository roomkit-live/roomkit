"""A caller owns its legacy write until the writer accepts the request."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.tools import MCPToolProvider
from tests.mcp_legacy_peer import LegacySSEPeer

pytest.importorskip("mcp.client.sse")


def message(kind: str, name: str) -> dict[str, Any]:
    if kind == "tool":
        return {"method": "tools/call", "params": {"name": name, "arguments": {}}}
    if kind == "resource":
        return {"method": "resources/read", "params": {"uri": f"file:///{name}"}}
    return {"method": "host/read", "params": {"name": name}}


def result(kind: str) -> dict[str, Any]:
    if kind == "tool":
        return {"content": [{"type": "text", "text": "done"}]}
    if kind == "resource":
        return {"contents": [{"uri": "file:///kept", "text": "done"}]}
    return {"done": True}


async def invoke(provider: MCPToolProvider, kind: str, name: str, timeout: float) -> Any:
    if kind == "tool":
        return await provider.call_tool(name, {}, timeout=timeout)
    if kind == "resource":
        return await provider.read_resource(f"file:///{name}", timeout=timeout)
    return await provider.request("host/read", {"name": name}, timeout=timeout)


class HeldPostPeer(LegacySSEPeer):
    def __init__(self, kind: str) -> None:
        super().__init__({"replies": [message(kind, "kept") | {"result": result(kind)}]})
        self.posts = 0
        self.post_held = asyncio.Event()
        self.release_post = asyncio.Event()

    async def _send(self, writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
        self.posts += 1
        if self.posts == 2:
            self.post_held.set()
            await self.release_post.wait()
        await super()._send(writer, status, body)


async def abandon(provider: MCPToolProvider, kind: str, cancellation: bool) -> None:
    if not cancellation:
        with pytest.raises(TimeoutError):
            await invoke(provider, kind, "abandoned", 0.03)
        return
    started = asyncio.Event()

    async def call() -> None:
        started.set()
        await invoke(provider, kind, "abandoned", 30)

    task = asyncio.create_task(call())
    await started.wait()  # The caller has run until its first pending I/O.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("kind", ["tool", "host", "resource"])
@pytest.mark.parametrize("cancellation", [False, True])
async def test_abandoned_request_waiting_for_the_legacy_writer_is_not_posted(
    kind: str, cancellation: bool
) -> None:
    async with HeldPostPeer(kind) as peer:
        async with MCPToolProvider(peer.url, transport="sse", discover=False) as provider:
            await asyncio.wait_for(peer.post_held.wait(), 1)
            await abandon(provider, kind, cancellation)
            assert peer.requests == [{"method": "initialize"}]
            peer.release_post.set()
            assert await invoke(provider, kind, "kept", 1) == (
                "done" if kind == "tool" else result(kind)
            )
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        # The SDK counts a request its writer took as sent: it tells the
        # server once the writer is free, though the request never went out.
        assert [item["method"] for item in peer.messages] == [
            "initialize",
            "notifications/initialized",
            "notifications/cancelled",
            message(kind, "kept")["method"],
        ]
        assert peer.requests == [{"method": "initialize"}, message(kind, "kept")]
    assert peer.problems == []


class HeldListingPeer(LegacySSEPeer):
    """A server that holds its answer to the POST of the first ``tools/list``."""

    def __init__(self) -> None:
        super().__init__({"replies": []})
        self.release = asyncio.Event()
        self.posts = 0

    def _reply(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if message.get("method") == "initialize":
            return super()._reply(message)
        return None

    async def _send(self, writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
        self.posts += 1
        if self.posts == 3:  # initialize, notifications/initialized, tools/list
            await self.release.wait()
        await super()._send(writer, status, body)


@pytest.mark.wallclock
@pytest.mark.parametrize("kind", ["tool", "host", "resource"])
async def test_a_call_cut_short_by_its_deadline_returns_at_it(kind: str) -> None:
    """The SDK, cancelled, waits up to five seconds to tell the server, behind
    a writer that is stuck; the caller does not wait for it (RFC §21.6)."""
    async with HeldPostPeer(kind) as peer:
        provider = MCPToolProvider(peer.url, transport="sse", discover=False)
        async with provider:
            await asyncio.wait_for(peer.post_held.wait(), 1)
            loop = asyncio.get_running_loop()
            began = loop.time()
            with pytest.raises(TimeoutError):
                await invoke(provider, kind, "abandoned", 0.05)
            assert loop.time() - began < 1
            peer.release_post.set()


@pytest.mark.wallclock
async def test_a_connection_cut_short_by_its_bound_returns_at_it() -> None:
    """A listing the server holds past ``connect_timeout`` ends the connection
    at its bound, the SDK's cancellation left to end on its own (RFC §21.2)."""
    async with HeldListingPeer() as peer:
        provider = MCPToolProvider(peer.url, transport="sse", connect_timeout=0.3)
        loop = asyncio.get_running_loop()
        began = loop.time()
        with pytest.raises(TimeoutError, match="did not connect within 0.3 s"):
            await provider.__aenter__()
        assert loop.time() - began < 1.5
        assert not provider.connected
        peer.release.set()


@pytest.mark.parametrize("kind", ["tool", "host", "resource"])
async def test_expired_request_already_posted_sends_a_cancellation_notification(
    kind: str,
) -> None:
    peer = LegacySSEPeer(
        {
            "replies": [
                message(kind, "sent") | {"hold": True},
                message(kind, "kept") | {"result": result(kind)},
            ]
        }
    )
    async with peer:
        async with MCPToolProvider(peer.url, transport="sse", discover=False) as provider:
            with pytest.raises(TimeoutError):
                await invoke(provider, kind, "sent", 0.03)
            assert await invoke(provider, kind, "kept", 1) == (
                "done" if kind == "tool" else result(kind)
            )
        await asyncio.wait_for(peer.get_closed.wait(), 1)
        method = message(kind, "sent")["method"]
        # The SDK tells the server a request it gave up on is cancelled.
        assert [item["method"] for item in peer.messages] == [
            "initialize",
            "notifications/initialized",
            method,
            "notifications/cancelled",
            method,
        ]
        assert peer.requests == [
            {"method": "initialize"},
            message(kind, "sent"),
            message(kind, "kept"),
        ]
    assert peer.problems == []

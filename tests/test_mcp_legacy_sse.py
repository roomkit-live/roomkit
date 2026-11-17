"""The legacy SSE choice performs real GET/POST exchanges through the MCP SDK."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.core.exceptions import UnservedToolCallError
from roomkit.tools import MCPToolProvider
from tests.mcp_legacy_peer import LegacySSEPeer

pytest.importorskip("mcp.client.sse")


def listing() -> dict[str, Any]:
    return {
        "method": "tools/list",
        "result": {
            "tools": [
                {"name": name, "description": name, "inputSchema": {"type": "object"}}
                for name in ["echo", "hidden"]
            ]
        },
    }


def echo() -> dict[str, Any]:
    return {
        "method": "tools/call",
        "params": {"name": "echo", "arguments": {}},
        "result": {"content": [{"type": "text", "text": "done"}], "isError": False},
    }


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("discover", [False, True])
@pytest.mark.parametrize("nonempty_headers", [False, True])
async def test_legacy_sse_uses_the_announced_endpoint_and_retained_headers(
    direct: bool, discover: bool, nonempty_headers: bool
) -> None:
    peer = LegacySSEPeer({"replies": ([listing()] if discover else []) + [echo()]})
    peer.discovery = discover
    async with peer:
        headers = {"X-Capture": "before"} if nonempty_headers else {}
        construct = MCPToolProvider if direct else MCPToolProvider.from_url
        provider = construct(
            peer.url,
            transport="sse",
            headers=headers,
            discover=discover,
            tool_filter=lambda name: name == "echo",
            label="capture",
        )
        assert not provider.connected and peer.targets == []
        headers["X-Capture"] = "after"
        async with provider:
            assert provider.connected
            assert provider.tool_names == (["echo"] if discover else [])
            handler = provider.as_tool_handler(gate_discovery=False)
            with pytest.raises(UnservedToolCallError):
                await handler("mcp__another__echo", {})
            assert await handler("mcp__capture__echo", {}) == "done"
        assert not provider.connected
        await asyncio.wait_for(peer.get_closed.wait(), 3)
        assert peer.targets[0] == ("GET", "/mcp")
        assert all(target == ("POST", "/messages?session=1") for target in peer.targets[1:])
        assert [item["method"] for item in peer.requests] == (
            ["initialize"] + (["tools/list"] if discover else []) + ["tools/call"]
        )
        assert all(
            headers.get("x-capture") == ("after" if nonempty_headers else None)
            for headers in peer.headers
        )
    assert peer.problems == []


async def test_reentering_legacy_sse_uses_a_fresh_endpoint_and_replaces_the_catalogue() -> None:
    peer = LegacySSEPeer({"replies": [listing(), echo(), listing(), echo()]})
    peer.discovery = True
    async with peer:
        provider = MCPToolProvider.from_url(peer.url, transport="sse")
        for session in [1, 2]:
            async with provider:
                assert provider.tool_names == ["echo", "hidden"]
                assert await provider.call_tool("echo", {}) == "done"
                with pytest.raises(RuntimeError, match="already connected"):
                    await provider.__aenter__()
            await asyncio.wait_for(peer.get_closed.wait(), 3)
            assert peer.connections == session
        assert ("POST", "/messages?session=2") in peer.targets
    assert peer.problems == []


@pytest.mark.parametrize("stage", ["endpoint", "listing"])
async def test_the_connection_bound_closes_legacy_sse_and_allows_reentry(stage: str) -> None:
    held = [{"method": "tools/list", "hold": True}] if stage == "listing" else []
    peer = LegacySSEPeer({"replies": held + [listing()]}, endpoint=stage != "endpoint")
    peer.discovery = True
    async with peer:
        provider = MCPToolProvider(peer.url, transport="sse", connect_timeout=0.05)
        with pytest.raises(TimeoutError, match="within 0.05 s"):
            await provider.__aenter__()
        await asyncio.wait_for(peer.get_closed.wait(), 3)
        assert not provider.connected
        peer.endpoint = True
        async with provider:
            assert provider.connected and provider.tool_names == ["echo", "hidden"]
        await asyncio.wait_for(peer.get_closed.wait(), 3)
        assert peer.connections == 2
    assert peer.problems == []

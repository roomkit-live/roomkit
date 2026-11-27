"""Legacy GET response boundaries observed through the public MCP provider."""

from __future__ import annotations

import asyncio

import pytest

from roomkit.tools import MCPToolProvider
from tests.mcp_legacy_peer import LegacySSEPeer

pytest.importorskip("mcp.client.sse")
from httpx2 import SSEError


class ResponsePeer(LegacySSEPeer):
    def __init__(self, content_type: str | None, *, eof: bool = False) -> None:
        super().__init__({"replies": []})
        self.content_type = content_type
        self.eof = eof

    async def _get(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self._stream = writer
        header = f"Content-Type: {self.content_type}\r\n" if self.content_type else ""
        writer.write(f"HTTP/1.1 200 OK\r\n{header}\r\n".encode())
        if not self.eof:
            writer.write(f"event: endpoint\ndata: {self._endpoint()}\n\n".encode())
        await writer.drain()
        self.get_opened.set()
        try:
            if not self.eof:
                await reader.read()
        finally:
            self.closed_connections += 1
            self.get_closed.set()


@pytest.mark.parametrize(
    ("content_type", "accepted"),
    [
        ("text/event-stream", True),
        ("text/event-stream; charset=utf-8", True),
        # Read as a media type, as the SDK reads it: the part before ";",
        # trimmed and lowercased, is text/event-stream (RMK-2203).
        ("prefix-text/event-stream-suffix", False),
        ("Text/Event-Stream", True),
        ("text/plain; text/event-stream", False),
        (None, False),
    ],
)
async def test_legacy_get_uses_the_sdk_content_type_match(
    content_type: str | None, accepted: bool
) -> None:
    async with ResponsePeer(content_type) as peer:
        provider = MCPToolProvider(peer.url, transport="sse", discover=False, connect_timeout=1)
        if accepted:
            async with provider:
                assert provider.connected
            assert peer.requests == [{"method": "initialize"}]
        else:
            with pytest.raises(ExceptionGroup) as raised:
                await provider.__aenter__()
            assert len(raised.value.exceptions) == 1
            assert isinstance(raised.value.exceptions[0], SSEError)
            assert peer.targets == [("GET", "/mcp")]
            assert peer.requests == []
        assert not provider.connected
        await asyncio.wait_for(peer.get_closed.wait(), 3)
    assert peer.problems == []


async def test_legacy_get_eof_before_endpoint_fails_without_waiting_for_open_timeout() -> None:
    async with ResponsePeer("text/event-stream", eof=True) as peer:
        provider = MCPToolProvider(peer.url, transport="sse", discover=False, connect_timeout=30)
        with pytest.raises(ExceptionGroup) as raised:
            await asyncio.wait_for(provider.__aenter__(), 1)
        assert len(raised.value.exceptions) == 1
        error = raised.value.exceptions[0]
        assert isinstance(error, RuntimeError)
        assert str(error) == "Child exited without calling task_status.started()"
        assert not provider.connected
        assert peer.targets == [("GET", "/mcp")]
        assert peer.requests == []
    assert peer.problems == []

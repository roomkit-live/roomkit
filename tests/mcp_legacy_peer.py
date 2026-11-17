"""A legacy SSE endpoint reusing the bounded public MCP scenario peer."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from tests.conformance._mcp_peer import MCPPeer


class LegacySSEPeer(MCPPeer):
    """One GET stream carries replies to POSTs sent to its advertised endpoint."""

    def __init__(self, spec: dict[str, Any], *, endpoint: bool = True) -> None:
        super().__init__(spec)
        self.endpoint = endpoint
        self.headers: list[dict[str, str]] = []
        self.targets: list[tuple[str, str]] = []
        self.get_opened = asyncio.Event()
        self.get_closed = asyncio.Event()
        self.connections = 0
        self._stream: asyncio.StreamWriter | None = None

    async def _exchange(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        first, *lines = head.decode("latin-1").split("\r\n")
        verb, target, _version = first.split(" ")
        headers = {
            name.lower(): value.strip()
            for name, value in (line.split(":", 1) for line in lines if ":" in line)
        }
        self.headers.append(headers)
        self.targets.append((verb, target))
        if verb == "GET":
            await self._get(reader, writer)
            return
        if verb != "POST":
            raise AssertionError(f"unexpected legacy SSE verb {verb}")
        assert target == f"/messages?session={self.connections}"
        length = int(headers.get("content-length", "0"))
        assert 0 < length <= 4 * 1024 * 1024
        message = json.loads(await reader.readexactly(length))
        reply = self._reply(message) if "id" in message else None
        await self._send(writer, 202, b"")
        if reply is not None:
            assert self._stream is not None
            self._stream.write(b"event: message\ndata: " + json.dumps(reply).encode() + b"\n\n")
            await self._stream.drain()

    async def _get(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self.get_closed.clear()
        self._stream = writer
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n\r\n"
        )
        if self.endpoint:
            writer.write(
                f"event: endpoint\ndata: /messages?session={self.connections}\n\n".encode()
            )
        await writer.drain()
        self.get_opened.set()
        try:
            await reader.read()
        finally:
            self.get_closed.set()

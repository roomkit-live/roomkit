"""A legacy SSE endpoint reusing the bounded public MCP scenario peer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from tests.conformance._mcp_peer import MCPPeer


class LegacySSEPeer(MCPPeer):
    """One GET stream carries replies to POSTs sent to its advertised endpoint."""

    def __init__(self, spec: dict[str, Any], *, endpoint: bool = True) -> None:
        super().__init__(spec)
        self.endpoint = endpoint
        self.headers: list[dict[str, str]] = []
        self.raw_headers: list[list[tuple[str, str]]] = []
        self.targets: list[tuple[str, str]] = []
        self.messages: list[dict[str, Any]] = []
        self.get_opened = asyncio.Event()
        self.get_closed = asyncio.Event()
        self.connections = 0
        self.closed_connections = 0
        self._stream: asyncio.StreamWriter | None = None

    async def _exchange(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        first, *lines = head.decode("latin-1").split("\r\n")
        verb, target, _version = first.split(" ")
        self.raw_headers.append(
            [
                (name, value.strip())
                for name, value in (line.split(":", 1) for line in lines if ":" in line)
            ]
        )
        headers = {
            name.lower(): value.strip()
            for name, value in (line.split(":", 1) for line in lines if ":" in line)
        }
        self.headers.append(headers)
        self.targets.append((verb, target))
        if await self._redirect(writer, verb, target):
            return
        if verb == "GET":
            await self._get(reader, writer)
            return
        if verb != "POST":
            raise AssertionError(f"unexpected legacy SSE verb {verb}")
        assert target == self._post_target()
        await self._post(reader, writer, headers)

    async def _post(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict[str, str]
    ) -> None:
        length = int(headers.get("content-length", "0"))
        assert 0 < length <= 4 * 1024 * 1024
        message = json.loads(await reader.readexactly(length))
        self.messages.append(message)
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
        endpoints = self.spec.get("endpoint_on_connections")
        if endpoints is not None and self.connections > len(endpoints):
            raise AssertionError("legacy SSE endpoint script exhausted")
        announce = endpoints[self.connections - 1] if endpoints is not None else self.endpoint
        if announce:
            writer.write(f"event: endpoint\ndata: {self._endpoint()}\n\n".encode())
        await writer.drain()
        self.get_opened.set()
        try:
            await reader.read()
        finally:
            self.closed_connections += 1
            self.get_closed.set()

    def _post_target(self) -> str:
        return f"/messages?session={self.connections}"

    def _endpoint(self) -> str:
        return self._post_target()

    async def _redirect(self, writer: asyncio.StreamWriter, verb: str, target: str) -> bool:
        return False

    async def __aexit__(self, *args: object) -> None:
        await super().__aexit__(*args)
        endpoints = self.spec.get("endpoint_on_connections")
        if endpoints is not None and len(endpoints) != self.connections:
            self.problems.append("legacy SSE peer has unconsumed endpoint announcements")


def logged(record: logging.LogRecord, kind: type[BaseException]) -> bool:
    """Whether *record* logs a *kind*, alone or in the group the SDK's post
    writer raises it in."""
    error = record.exc_info[1] if record.exc_info else None
    if isinstance(error, BaseExceptionGroup):
        return error.subgroup(kind) is not None
    return isinstance(error, kind)

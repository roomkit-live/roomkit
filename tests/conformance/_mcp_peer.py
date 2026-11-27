"""A bounded HTTP peer for public MCP host-access scenarios."""

from __future__ import annotations

import asyncio
import copy
import json
from contextlib import suppress
from typing import Any

_MODERN = "2026-07-28"
"""The MCP revision a peer with this ``era`` speaks: discovery, no handshake,
no session."""

_PROTOCOL_KEYS = (
    "io.modelcontextprotocol/protocolVersion",
    "io.modelcontextprotocol/clientInfo",
    "io.modelcontextprotocol/clientCapabilities",
)
"""The ``_meta`` entries every request of a 2026-07-28 connection carries."""


class MCPPeer:
    """Serve scripted JSON-RPC replies without replacing the client's SDK."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.requests: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self._position = 0
        self._initialized = 0
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._closing = False
        self.url = ""
        self.discovery = False
        self.tool_call = False

    async def __aenter__(self) -> MCPPeer:
        self._closing = False
        self._server = await asyncio.start_server(self._accepted, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/mcp"
        return self

    async def __aexit__(self, *_: object) -> None:
        self._closing = True
        if self._server is not None:
            self._server.close()
        for writer in self._writers:
            writer.close()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Python's server waits for accepted transports too. Close them before
        # waiting; a held or idle request need not close from the client side.
        if self._server is not None:
            await self._server.wait_closed()
        self._writers.clear()
        if self._position != len(self.spec["replies"]):
            self.problems.append("MCP peer has unconsumed scripted replies")
        if "initializations" in self.spec and self._initialized != len(
            self.spec["initializations"]
        ):
            self.problems.append("MCP peer has unconsumed initializations")

    def _accepted(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._closing:
            writer.close()
            return
        # Register the writer before its coroutine runs, so immediate shutdown
        # also closes a connection whose task has not started yet.
        self._writers.add(writer)
        task = asyncio.create_task(self._serve(reader, writer))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._exchange(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # A client can close a request when its deadline expires.
        except Exception as error:
            self.problems.append(f"MCP HTTP fixture: {type(error).__name__}: {error}")
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self._writers.discard(writer)

    async def _exchange(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        verb = head.split(b" ", 1)[0]
        if verb != b"POST":
            await self._send(writer, 200 if verb == b"DELETE" else 405, b"")
            return
        headers = dict(line.split(b":", 1) for line in head.split(b"\r\n")[1:] if b":" in line)
        length = next((int(v) for k, v in headers.items() if k.lower() == b"content-length"), 0)
        if not 0 < length <= 4 * 1024 * 1024:
            raise ValueError("MCP fixture needs a bounded JSON body")
        message = json.loads(await reader.readexactly(length))
        if self.modern:
            named = next(
                (v.strip() for k, v in headers.items() if k.lower() == b"mcp-method"), None
            )
            if named != message.get("method", "").encode():
                self.problems.append(f"MCP request {message.get('method')} named {named!r}")
        if "id" not in message:
            await self._send(writer, 202, b"", session=not self.modern)
            return
        reply = self._reply(message)
        if reply is None:
            await reader.read()  # A deliberately unanswered request ends with its client.
            return
        encoded = json.dumps(reply, ensure_ascii=False).encode()
        if self.spec.get("framing", "json") == "sse":
            body = b"event: message\ndata: " + encoded + b"\n\n"
            await self._send(writer, 200, body, sse=True, session=not self.modern)
        else:
            await self._send(writer, 200, encoded, session=not self.modern)

    @property
    def modern(self) -> bool:
        """Whether the peer is a server of MCP 2026-07-28 (``era``)."""
        return self.spec.get("era") == _MODERN

    def _reply(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message["method"]
        response = {"jsonrpc": "2.0", "id": message["id"]}
        if method in ("initialize", "server/discover"):
            return self._negotiation(method, message, response)
        if self.modern:
            message = self._without_protocol_meta(message)
        observed = {"method": method}
        if "params" in message:
            observed["params"] = message["params"]
        if (method == "tools/list" and self.discovery) or (
            method == "tools/call" and self.tool_call
        ):
            params = observed.get("params")
            meta = params.get("_meta") if isinstance(params, dict) else None
            if isinstance(meta, dict) and type(meta.get("progressToken")) is int:
                meta.pop("progressToken")
                if not meta:
                    params.pop("_meta")
        self.requests.append(observed)
        replies = self.spec["replies"]
        if self._position >= len(replies):
            self.problems.append(f"Unexpected MCP request after script exhaustion: {observed}")
            return response | {"error": {"code": -32603, "message": "fixture exhausted"}}
        scripted = replies[self._position]
        self._position += 1
        # A script without params is a request without them: an empty set
        # goes out as no params member (RK-21.2-046).
        expected = {"method": scripted["method"]}
        if "params" in scripted:
            expected["params"] = scripted["params"]
        if observed != expected:
            self.problems.append(f"MCP request {observed} differs from {expected}")
            return response | {"error": {"code": -32603, "message": "fixture request mismatch"}}
        if scripted.get("hold"):
            return None
        field = "error" if "error" in scripted else "result"
        return response | {field: scripted[field]}

    def _negotiation(
        self, method: str, message: dict[str, Any], response: dict[str, Any]
    ) -> dict[str, Any]:
        """The answer to a request that negotiates the protocol, recorded by
        its method only: a server of an earlier revision knows no discovery
        and the client falls back to the handshake; a 2026-07-28 server
        answers discovery and takes no handshake."""
        self.requests.append({"method": method})
        if method == "server/discover" and self.modern:
            return response | {"result": self._discovery()}
        if method == "initialize" and not self.modern:
            return response | {"result": self._initialization(message)}
        if method == "initialize":
            self.problems.append("MCP 2026-07-28 peer received an initialize")
        return response | {"error": {"code": -32601, "message": "Method not found"}}

    def _discovery(self) -> dict[str, Any]:
        result = self._initialization({})
        meta = {"io.modelcontextprotocol/serverInfo": result.pop("serverInfo")}
        del result["protocolVersion"]
        return result | {
            "supportedVersions": [_MODERN],
            "resultType": "complete",
            "ttlMs": 0,
            "cacheScope": "private",
            "_meta": meta,
        }

    def _without_protocol_meta(self, message: dict[str, Any]) -> dict[str, Any]:
        """*message* without the ``_meta`` entries every 2026-07-28 request
        carries, which the peer checks: its revision, the client's identity
        and capabilities. A ``_meta`` or ``params`` left empty goes too."""
        message = copy.deepcopy(message)
        params = message.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        if not isinstance(meta, dict) or meta.get(_PROTOCOL_KEYS[0]) != _MODERN:
            self.problems.append(f"MCP request {message['method']} names no 2026-07-28 revision")
            return message
        for key in _PROTOCOL_KEYS[1:]:
            if not isinstance(meta.get(key), dict):
                self.problems.append(f"MCP request {message['method']} carries no {key}")
        for key in _PROTOCOL_KEYS:
            meta.pop(key, None)
        if not meta:
            del params["_meta"]
        if not params:
            del message["params"]
        return message

    def _initialization(self, message: dict[str, Any]) -> dict[str, Any]:
        spec = self.spec
        if "initializations" in spec:
            if self._initialized >= len(spec["initializations"]):
                self.problems.append("MCP initialization script exhausted")
            else:
                spec = spec["initializations"][self._initialized]
        self._initialized += 1
        capabilities: dict[str, Any] = {"resources": {}}
        if "extensions" in spec:
            capabilities["extensions"] = copy.deepcopy(spec["extensions"])
        result = {
            "protocolVersion": "2025-11-25",
            "capabilities": capabilities,
            "serverInfo": spec.get("server_info", {"name": "host-test", "version": "1"}),
        }
        if "instructions" in spec:
            result["instructions"] = spec["instructions"]
        return result

    @staticmethod
    async def _send(
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        *,
        sse: bool = False,
        session: bool = True,
    ) -> None:
        kind = "text/event-stream" if sse else "application/json"
        # A 2026-07-28 server keeps no session.
        held = "Mcp-Session-Id: conformance-session\r\n" if session else ""
        headers = (
            f"HTTP/1.1 {status} Response\r\nContent-Type: {kind}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n{held}\r\n"
        )
        writer.write(headers.encode() + body)
        await writer.drain()

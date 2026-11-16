"""A bounded HTTP peer for public MCP host-access scenarios."""

from __future__ import annotations

import asyncio
import copy
import json
from contextlib import suppress
from typing import Any


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
        self.url = ""
        self.discovery = False

    async def __aenter__(self) -> MCPPeer:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/mcp"
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for writer in self._writers:
            writer.close()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._position != len(self.spec["replies"]):
            self.problems.append("MCP peer has unconsumed scripted replies")
        if "initializations" in self.spec and self._initialized != len(
            self.spec["initializations"]
        ):
            self.problems.append("MCP peer has unconsumed initializations")

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        self._writers.add(writer)
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
            self._tasks.discard(task)

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
        if "id" not in message:
            await self._send(writer, 202, b"")
            return
        reply = self._reply(message)
        if reply is None:
            await reader.read()  # A deliberately unanswered request ends with its client.
            return
        encoded = json.dumps(reply, ensure_ascii=False).encode()
        if self.spec.get("framing", "json") == "sse":
            await self._send(writer, 200, b"event: message\ndata: " + encoded + b"\n\n", sse=True)
        else:
            await self._send(writer, 200, encoded)

    def _reply(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message["method"]
        response = {"jsonrpc": "2.0", "id": message["id"]}
        if method == "initialize":
            self.requests.append({"method": method})
            return response | {"result": self._initialization(message)}
        observed = {"method": method}
        if "params" in message:
            observed["params"] = message["params"]
        elif method == "tools/list" and self.discovery:
            # SDK discovery may omit the empty first-page parameters. Host
            # requests retain their exact presence and value.
            observed["params"] = {}
        if method == "tools/list" and self.discovery:
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
        expected = {"method": scripted["method"], "params": scripted.get("params", {})}
        if observed != expected:
            self.problems.append(f"MCP request {observed} differs from {expected}")
        if scripted.get("hold"):
            return None
        field = "error" if "error" in scripted else "result"
        return response | {field: scripted[field]}

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
        writer: asyncio.StreamWriter, status: int, body: bytes, *, sse: bool = False
    ) -> None:
        kind = "text/event-stream" if sse else "application/json"
        headers = (
            f"HTTP/1.1 {status} Response\r\nContent-Type: {kind}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n"
            "Mcp-Session-Id: conformance-session\r\n\r\n"
        )
        writer.write(headers.encode() + body)
        await writer.drain()

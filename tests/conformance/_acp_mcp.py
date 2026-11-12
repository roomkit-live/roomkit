"""MCP session inputs, resolved by the channel through its public callback."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from acp.schema import HttpMcpServer, McpServerStdio, SseMcpServer
from pydantic import TypeAdapter

from tests.conformance._script import Gates

_SERVERS = TypeAdapter(list[HttpMcpServer | SseMcpServer | McpServerStdio])


def _members(value: Any, required: set[str], optional: set[str]) -> bool:
    return isinstance(value, dict) and required <= value.keys() <= required | optional


def _servers_valid(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for server in value:
        if not isinstance(server, dict):
            return False
        kind = server.get("type")
        if kind in ("http", "sse"):
            if not _members(server, {"type", "name", "url", "headers"}, {"_meta"}):
                return False
            pairs = server["headers"]
        elif kind is None:
            if not _members(server, {"name", "command", "args", "env"}, {"_meta"}):
                return False
            pairs = server["env"]
        else:
            return False
        if not isinstance(pairs, list) or any(
            not _members(pair, {"name", "value"}, {"_meta"}) for pair in pairs
        ):
            return False
    try:
        _SERVERS.validate_python(value, strict=True)
    except ValueError:
        return False
    return True


def mcp_problems(value: Any, path: str) -> list[str]:
    """Reject malformed scripts before they can become expected domain errors."""
    if isinstance(value, list):
        return [] if _servers_valid(value) else [f"{path}: invalid MCP descriptors"]
    if not _members(value, {"mode", "responses"}, set()):
        return [f"{path}: expected an MCP list or a mode/responses callback script"]
    if value["mode"] not in ("sync", "async") or not isinstance(value["responses"], list):
        return [f"{path}: invalid callback mode or responses"]
    problems = []
    for index, reply in enumerate(value["responses"]):
        valid = _members(reply, {"servers"}, {"gate"}) and _servers_valid(reply["servers"])
        valid |= _members(reply, {"error"}, {"gate"}) and isinstance(reply["error"], str)
        if isinstance(reply, dict) and "gate" in reply:
            valid &= value["mode"] == "async" and isinstance(reply["gate"], str)
        if not valid:
            problems.append(f"{path}.responses[{index}]: invalid MCP callback response")
    return problems


class ScriptedMCPServers:
    """A fixed SDK list or a callback, recording its actual inputs before gates."""

    def __init__(self, spec: Any, gates: Gates) -> None:
        problems = mcp_problems(spec, "acp.mcp_servers")
        if problems:
            raise ValueError("; ".join(problems))
        self._spec = spec
        self._gates = gates
        self.requests: list[str] = []
        self.problems: list[str] = []

    def source(self) -> Any:
        if isinstance(self._spec, list):
            return _SERVERS.validate_python(self._spec)
        return self._async if self._spec["mode"] == "async" else self._sync

    def _next(self, room_id: str) -> Mapping[str, Any]:
        index = len(self.requests)
        self.requests.append(room_id)
        replies = self._spec["responses"]
        if index >= len(replies):
            problem = f"acp.mcp_servers: callback script exhausted at call {index + 1}"
            self.problems.append(problem)
            raise RuntimeError(problem)
        return replies[index]

    @staticmethod
    def _resolve(reply: Mapping[str, Any]) -> Any:
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return _SERVERS.validate_python(reply["servers"])

    def _sync(self, room_id: str) -> Any:
        return self._resolve(self._next(room_id))

    async def _async(self, room_id: str) -> Any:
        reply = self._next(room_id)
        if "gate" in reply:
            await self._gates.hold(reply["gate"])
        return self._resolve(reply)

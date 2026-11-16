"""Reject malformed MCP session controls before contacting the fixture."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MCP_HOST_ARGUMENTS = {"mcp_session": {"peer", "operations", "discover", "tool_filter"}}
OPERATIONS = {
    "connect": set(),
    "close": set(),
    "connected": set(),
    "tool_names": set(),
    "tool_meta": {"mutate"},
    "server_info": {"mutate"},
    "server_extensions": {"mutate"},
    "request": {"name", "params", "timeout_ms"},
    "read_resource": {"uri", "timeout_ms"},
}


def _object(value: Any, fields: set[str], path: str, found: list[str]) -> bool:
    if not isinstance(value, Mapping):
        found.append(f"{path}: expected an object")
        return False
    found.extend(f"{path}: unknown member {key}" for key in value.keys() - fields)
    return True


def _operations(values: Any, path: str, found: list[str]) -> None:
    if not isinstance(values, list) or not values:
        found.append(f"{path}: expected a nonempty operation list")
        return
    for index, value in enumerate(values):
        at = f"{path}[{index}]"
        method = value.get("method") if isinstance(value, Mapping) else None
        if not isinstance(method, str) or method not in OPERATIONS:
            found.append(f"{at}: unknown MCP operation")
            continue
        _object(value, OPERATIONS[method] | {"method"}, at, found)
        required = {"request": "name", "read_resource": "uri"}.get(method)
        if required and not isinstance(value.get(required), str):
            found.append(f"{at}.{required}: expected a string")
        if (
            "params" in value
            and value["params"] is not None
            and not isinstance(value["params"], Mapping)
        ):
            found.append(f"{at}.params: expected an object or null")
        if "timeout_ms" in value and (
            type(value["timeout_ms"]) is not int or value["timeout_ms"] < 1
        ):
            found.append(f"{at}.timeout_ms: expected a positive integer")
        if "mutate" in value:
            _mutation(value["mutate"], f"{at}.mutate", found)


def _mutation(value: Any, path: str, found: list[str]) -> None:
    if not _object(value, {"path", "value"}, path, found):
        return
    keys = value.get("path")
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
        found.append(f"{path}.path: expected a nonempty string list")
    if "value" not in value:
        found.append(f"{path}.value: missing value")


def _replies(values: Any, path: str, found: list[str]) -> None:
    if not isinstance(values, list):
        found.append(f"{path}: expected a list")
        return
    for index, value in enumerate(values):
        at = f"{path}[{index}]"
        if not _object(value, {"method", "params", "result", "error", "hold"}, at, found):
            continue
        if not isinstance(value.get("method"), str):
            found.append(f"{at}.method: expected a string")
        if "params" in value and not isinstance(value["params"], Mapping):
            found.append(f"{at}.params: expected an object")
        if len(value.keys() & {"result", "error", "hold"}) != 1:
            found.append(f"{at}: exactly one of result/error/hold is required")
        if "hold" in value and value["hold"] is not True:
            found.append(f"{at}.hold: expected true")
        if "error" in value:
            error = value["error"]
            if _object(error, {"code", "message", "data"}, f"{at}.error", found) and (
                type(error.get("code")) is not int or not isinstance(error.get("message"), str)
            ):
                found.append(f"{at}.error: expected integer code and string message")


def _peer(peer: Any, path: str, found: list[str]) -> None:
    if not _object(
        peer,
        {"server_info", "extensions", "instructions", "replies", "framing", "initializations"},
        path,
        found,
    ):
        return
    _initialization(peer, path, found)
    if "initializations" in peer:
        values = peer["initializations"]
        if not isinstance(values, list) or not values:
            found.append(f"{path}.initializations: expected a nonempty list")
        else:
            for index, item in enumerate(values):
                at = f"{path}.initializations[{index}]"
                if _object(item, {"server_info", "extensions", "instructions"}, at, found):
                    _initialization(item, at, found)
    if peer.get("framing", "json") not in ("json", "sse"):
        found.append(f"{path}.framing: expected json or sse")
    _replies(peer.get("replies"), path + ".replies", found)


def mcp_problems(spec: Mapping[str, Any], path: str) -> list[str]:
    """Validate fixture controls; server-owned JSON payloads remain opaque."""
    found: list[str] = []
    _peer(spec.get("peer"), path + ".peer", found)
    _operations(spec.get("operations"), path + ".operations", found)
    if "discover" in spec and type(spec["discover"]) is not bool:
        found.append(f"{path}.discover: expected a boolean")
    if "tool_filter" in spec and (
        not isinstance(spec["tool_filter"], list)
        or not all(isinstance(name, str) for name in spec["tool_filter"])
    ):
        found.append(f"{path}.tool_filter: expected a string list")
    return found


def _initialization(value: Mapping[str, Any], path: str, found: list[str]) -> None:
    if "server_info" in value and not isinstance(value["server_info"], Mapping):
        found.append(f"{path}.server_info: expected an object")
    if "instructions" in value and not isinstance(value["instructions"], str):
        found.append(f"{path}.instructions: expected a string")

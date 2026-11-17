"""Validate registry-session controls independently of expected product errors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SKILL_REGISTRY_ARGUMENTS = {"skill_registry_session": {"servers", "operations"}}
OPERATIONS = {
    "add_mcp_server": {"server", "label", "allow_dynamic", "execution_tools"},
    "load_skill": {"name"},
    "get_skill": {"name"},
    "read_file": {"name", "filename"},
    "is_served": {"name"},
    "origin": {"name"},
    "copy": {"target", "names", "marks"},
    "mark_unavailable": {"name", "reason"},
    "mark_unlisted": {"name"},
    "snapshot": set(),
    "install_channel": {"channel"},
    "room_call": {"action"},
}
_REQUIRED = {
    "add_mcp_server": {"server", "label"},
    "copy": {"target"},
    "read_file": {"name", "filename"},
    "mark_unavailable": {"name", "reason"},
}


def _object(value: Any, allowed: set[str], path: str, found: list[str]) -> bool:
    if not isinstance(value, Mapping):
        found.append(f"{path}: expected an object")
        return False
    found.extend(f"{path}: unknown member {key}" for key in value.keys() - allowed)
    return True


def _operations(values: Any, path: str, found: list[str]) -> None:
    if not isinstance(values, list) or not values:
        found.append(f"{path}: expected nonempty operations")
        return
    for index, value in enumerate(values):
        at = f"{path}[{index}]"
        method = value.get("method") if isinstance(value, Mapping) else None
        if not isinstance(method, str) or method not in OPERATIONS:
            found.append(f"{at}: unknown registry operation")
            continue
        if method in {"install_channel", "room_call"}:
            key = "channel" if method == "install_channel" else "action"
            if not isinstance(value.get(key), Mapping):
                found.append(f"{at}.{key}: expected an object")
            if (
                method == "install_channel"
                and isinstance(value.get("channel"), Mapping)
                and (
                    "model" not in value.get("channel", {}) or "skills" in value.get("channel", {})
                )
            ):
                found.append(f"{at}: install requires a model and its selected registry")
        _object(value, {"method", "registry"} | OPERATIONS[method], at, found)
        required = _REQUIRED.get(method, {"name"} if "name" in OPERATIONS[method] else set())
        for key in required | ({"registry"} if "registry" in value else set()):
            if not isinstance(value.get(key), str):
                found.append(f"{at}.{key}: expected a string")
        for key in ("allow_dynamic", "marks"):
            if key in value and type(value[key]) is not bool:
                found.append(f"{at}.{key}: expected a boolean")
        for key in ("execution_tools", "names"):
            if key in value and (
                not isinstance(value[key], list)
                or not all(isinstance(item, str) for item in value[key])
            ):
                found.append(f"{at}.{key}: expected a string list")


def _server(spec: Any, path: str, found: list[str]) -> None:
    if not _object(spec, {"extensions", "replies"}, path, found):
        return
    extensions = spec.get("extensions", {})
    if not isinstance(extensions, Mapping) or not all(
        isinstance(value, Mapping) for value in extensions.values()
    ):
        found.append(f"{path}.extensions: expected extension objects")
    replies = spec.get("replies")
    if not isinstance(replies, list):
        found.append(f"{path}.replies: expected a list")
        return
    for index, reply in enumerate(replies):
        at = f"{path}.replies[{index}]"
        if not _object(reply, {"method", "params", "result", "error"}, at, found):
            continue
        if not isinstance(reply.get("method"), str) or not isinstance(
            reply.get("params"), Mapping
        ):
            found.append(f"{at}: expected string method and object params")
        if len(reply.keys() & {"result", "error"}) != 1:
            found.append(f"{at}: exactly one result or error is required")
        if "error" in reply and not isinstance(reply["error"], str):
            found.append(f"{at}.error: expected a string")


def registry_problems(spec: Mapping[str, Any], path: str) -> list[str]:
    """Malformed fixture controls cannot become expected SDK refusals."""
    found: list[str] = []
    servers = spec.get("servers")
    if not isinstance(servers, Mapping):
        found.append(f"{path}.servers: expected an object")
    else:
        for name, server in servers.items():
            _server(server, f"{path}.servers.{name}", found)
    _operations(spec.get("operations"), f"{path}.operations", found)
    return found

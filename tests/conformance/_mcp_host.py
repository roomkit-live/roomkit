"""One public MCP provider session, owned by one coroutine (RFC §21.2)."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from mcp.shared.exceptions import McpError
from pydantic import ValidationError

from roomkit.tools import MCPToolProvider
from tests.conformance._calls import UnreadableError
from tests.conformance._mcp_peer import MCPPeer
from tests.conformance._mcp_shape import mcp_problems

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness


async def _operation(provider: MCPToolProvider, operation: Mapping[str, Any]) -> Any:
    method = operation["method"]
    if method == "connect":
        await provider.__aenter__()
        return None
    if method == "close":
        await provider.__aexit__(None, None, None)
        return None
    if method == "tool_meta":
        return provider.tool_meta()
    if method == "get_tools_as_dicts":
        return provider.get_tools_as_dicts()
    if method in {"connected", "tool_names", "server_info", "server_extensions"}:
        return getattr(provider, method)
    options = {"timeout": operation["timeout_ms"] / 1000} if "timeout_ms" in operation else {}
    if method == "request":
        params = {"params": operation["params"]} if "params" in operation else {}
        return await provider.request(operation["name"], **params, **options)
    if method == "read_resource":
        result = await provider.read_resource(operation["uri"], **options)
        # Keep sent fields, including explicit nulls and extensions, while
        # allowing the public model to perform its normal parsing.
        return result.model_dump(by_alias=True, mode="json", exclude_unset=True)
    raise UnreadableError(f"Unknown MCP host operation: {method}")


async def _observed(provider: MCPToolProvider, operation: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = await _operation(provider, operation)
    except McpError as error:
        return {"error": "mcp_error", "message": str(error)}
    except TimeoutError:
        return {"error": "timeout"}
    except ValidationError:
        return {"error": "validation_error"}
    except RuntimeError as error:
        if str(error) != "MCPToolProvider is not connected. Use 'async with' to connect first.":
            raise UnreadableError(f"Unexpected MCP host error: {error}") from error
        return {"error": "not_connected"}
    except Exception as error:
        raise UnreadableError(f"Unexpected MCP host error: {error}") from error
    if "mutate" in operation:
        _mutate(value, operation["mutate"])
    return {"result": copy.deepcopy(value)}


def _mutate(value: Any, mutation: Mapping[str, Any]) -> None:
    target = value
    try:
        for key in mutation["path"][:-1]:
            target = target[key]
        target[mutation["path"][-1]] = mutation["value"]
    except (KeyError, TypeError) as error:
        raise UnreadableError(f"Invalid MCP snapshot mutation: {error}") from error


async def session(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run the named public operations; fixture mistakes never become success."""
    if problems := mcp_problems(spec, "mcp_session"):
        raise UnreadableError("; ".join(problems))
    async with MCPPeer(dict(spec["peer"])) as peer:
        selected = spec.get("tool_filter")
        provider = MCPToolProvider.from_url(
            peer.url,
            discover=spec.get("discover", True),
            tool_filter=(lambda name: name in selected) if selected is not None else None,
        )
        try:
            results = []
            for operation in spec["operations"]:
                peer.discovery = operation["method"] == "connect" and spec.get("discover", True)
                results.append(await _observed(provider, operation))
        finally:
            peer.discovery = False
            if provider.connected:
                await provider.__aexit__(None, None, None)
    if peer.problems:
        raise UnreadableError("; ".join(peer.problems))
    observed = {"results": results, "requests": peer.requests}
    # The general matcher checks partial objects. These exact observations
    # additionally prove field absence and distinguish missing from null.
    return observed | {
        f"{key}_json": json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for key, value in observed.items()
    }


MCP_HOST_CALLS: dict[str, Call] = {"mcp_session": session}

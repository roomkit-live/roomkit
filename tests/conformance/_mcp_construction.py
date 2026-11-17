"""Public construction choices and observations of the actual legacy HTTP wire."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from roomkit.tools import MCPToolProvider
from tests.conformance._mcp_peer import MCPPeer
from tests.mcp_legacy_peer import LegacySSEPeer


def peer_for(spec: Mapping[str, Any]) -> MCPPeer:
    """Select the peer's transport framing independently of the client SDK."""
    peer = LegacySSEPeer if spec.get("transport") == "sse" else MCPPeer
    return peer(dict(spec["peer"]))


def construct(spec: Mapping[str, Any], peer: MCPPeer) -> MCPToolProvider:
    selected = spec.get("tool_filter")
    headers = dict(spec["headers"]) if "headers" in spec else None
    options: dict[str, Any] = {
        "discover": spec.get("discover", True),
        "tool_filter": (lambda name: name in selected) if selected is not None else None,
        "transport": spec.get("transport", "streamable_http"),
        "headers": headers,
        "label": spec.get("label"),
    }
    if "connect_timeout_ms" in spec:
        options["connect_timeout"] = spec["connect_timeout_ms"] / 1000
    make = MCPToolProvider if spec.get("constructor") == "direct" else MCPToolProvider.from_url
    provider = make(peer.url, **options)
    if "headers_after_construction" in spec:
        assert headers is not None  # Validated as a fixture prerequisite.
        headers.clear()
        headers.update(spec["headers_after_construction"])
    return provider


async def settle_closed_stream(peer: MCPPeer, provider: MCPToolProvider) -> None:
    """A close observation waits for the peer to see the GET stream close."""
    if isinstance(peer, LegacySSEPeer) and peer.connections and not provider.connected:
        await asyncio.wait_for(peer.get_closed.wait(), 3)


def wire_observation(peer: MCPPeer, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(peer, LegacySSEPeer):
        return {}
    names = {name.lower() for name in spec.get("headers", {})}
    names.update(name.lower() for name in spec.get("headers_after_construction", {}))
    return {
        "http": {
            "connections": peer.connections,
            "closed_connections": peer.closed_connections,
            "requests": [
                {
                    "method": method,
                    "target": target,
                    "headers": [
                        [name.lower(), value] for name, value in headers if name.lower() in names
                    ],
                }
                for (method, target), headers in zip(peer.targets, peer.raw_headers, strict=True)
            ],
        }
    }

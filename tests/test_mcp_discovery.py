"""How an MCP provider negotiates the protocol with its server (RFC §21.2, RMK-664).

A provider probes ``server/discover`` and speaks MCP 2026-07-28 with a server
that answers it; a server that refuses the probe, or never answers it, is
reached through the ``initialize`` handshake. The legacy HTTP+SSE transport,
which 2026-07-28 does not define, goes straight to the handshake.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import mcp.client.session
import pytest

from roomkit.tools import MCPToolProvider
from tests.mcp_legacy_peer import LegacySSEPeer

_SCRIPTED_SERVER = textwrap.dedent(
    """\
    import json, sys

    log, plan = open(sys.argv[1], "a"), json.loads(sys.argv[2])
    for line in sys.stdin:
        log.write(line)
        log.flush()
        message = json.loads(line)
        answer = plan.get(message.get("method"))
        if "id" not in message or answer is None or answer == "silent":
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], **answer}) + "\\n")
        sys.stdout.flush()
    """
)

_TOOLS = {"result": {"tools": [{"name": "lookup", "inputSchema": {"type": "object"}}]}}

_HANDSHAKE = {
    "result": {
        "protocolVersion": "2025-11-25",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "legacy", "version": "1"},
        "instructions": "from the handshake",
    }
}

_DISCOVERY = {
    "result": {
        "supportedVersions": ["2026-07-28"],
        "capabilities": {"tools": {}, "extensions": {"io.example/ext": {"level": 1}}},
        "instructions": "from discovery",
        "resultType": "complete",
        "ttlMs": 0,
        "cacheScope": "public",
        "_meta": {
            "io.modelcontextprotocol/serverInfo": {"name": "modern", "version": "2", "vendor": "x"}
        },
    }
}

_REFUSED = {"error": {"code": -32601, "message": "Method not found"}}


def _provider(tmp_path: Path, plan: dict[str, Any], **options: Any) -> MCPToolProvider:
    script = tmp_path / "server.py"
    script.write_text(_SCRIPTED_SERVER)
    log = tmp_path / "received.jsonl"
    options.setdefault("connect_timeout", 5)
    return MCPToolProvider.from_command(
        sys.executable, [str(script), str(log), json.dumps(plan)], **options
    )


def _received(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "received.jsonl").read_text().splitlines()]


async def test_a_server_answering_discovery_is_reached_without_a_handshake(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path, {"server/discover": _DISCOVERY, "tools/list": _TOOLS})
    async with provider:
        assert provider.tool_names == ["lookup"]
        assert provider.server_info == {"name": "modern", "version": "2", "vendor": "x"}
        assert provider.instructions == "from discovery"
        assert provider.server_extensions == {"io.example/ext": {"level": 1}}
    received = _received(tmp_path)
    assert [message["method"] for message in received] == ["server/discover", "tools/list"]
    # Every request of a 2026-07-28 connection names its protocol revision.
    meta = received[1]["params"]["_meta"]
    assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"


@pytest.mark.parametrize("probe", [_REFUSED, "silent"], ids=["refused", "unanswered"])
async def test_a_server_that_does_not_discover_is_reached_by_the_handshake(
    probe: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe refused, or never answered within the SDK's own probe bound,
    falls back to ``initialize``; the whole connect stays within
    ``connect_timeout`` (RFC §21.2)."""
    monkeypatch.setattr(mcp.client.session, "DISCOVER_TIMEOUT_SECONDS", 0.05)
    plan = {"server/discover": probe, "initialize": _HANDSHAKE, "tools/list": _TOOLS}
    provider = _provider(tmp_path, plan)
    async with provider:
        assert provider.tool_names == ["lookup"]
        assert provider.server_info == {"name": "legacy", "version": "1"}
        assert provider.instructions == "from the handshake"
    methods = [message["method"] for message in _received(tmp_path)]
    assert methods[:3] == ["server/discover", "initialize", "notifications/initialized"]
    assert methods[-1] == "tools/list"


async def test_a_probe_past_the_connect_bound_fails_the_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp.client.session, "DISCOVER_TIMEOUT_SECONDS", 30)
    provider = _provider(tmp_path, {"server/discover": "silent"}, connect_timeout=0.2)
    with pytest.raises(TimeoutError, match="did not connect within 0.2 s"):
        await provider.__aenter__()
    assert not provider.connected


async def test_the_legacy_sse_transport_goes_straight_to_the_handshake() -> None:
    async with LegacySSEPeer({"replies": []}) as peer:
        provider = MCPToolProvider(peer.url, transport="sse", discover=False)
        async with provider:
            pass
    assert [request["method"] for request in peer.requests] == ["initialize"]
    assert peer.problems == []

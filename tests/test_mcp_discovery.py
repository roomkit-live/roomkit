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
    """The probe counts within ``connect_timeout``: one that outlasts it fails
    the connection, though the server would take the handshake after it."""
    monkeypatch.setattr(mcp.client.session, "DISCOVER_TIMEOUT_SECONDS", 30)
    plan = {"server/discover": "silent", "initialize": _HANDSHAKE, "tools/list": _TOOLS}
    provider = _provider(tmp_path, plan, connect_timeout=0.2)
    with pytest.raises(TimeoutError, match="did not connect within 0.2 s"):
        await provider.__aenter__()
    assert not provider.connected


_HEADED_TOOLS = {
    "result": {
        "tools": [
            {
                "name": "routed",
                "inputSchema": {
                    "type": "object",
                    "properties": {"region": {"type": "string", "x-mcp-header": "Region"}},
                },
            },
            {
                # x-mcp-header names a scalar argument only: this one is invalid.
                "name": "misrouted",
                "inputSchema": {
                    "type": "object",
                    "properties": {"blob": {"type": "object", "x-mcp-header": "Blob"}},
                },
            },
            {"name": "plain", "inputSchema": {"type": "object"}},
        ]
    }
}


@pytest.mark.parametrize(
    ("probe", "kept"),
    [(_DISCOVERY, ["routed", "plain"]), (_REFUSED, ["routed", "misrouted", "plain"])],
    ids=["2026-07-28", "earlier"],
)
async def test_a_tool_whose_header_annotation_is_invalid_is_left_out_on_2026_07_28(
    probe: Any, kept: list[str], tmp_path: Path
) -> None:
    """MCP 2026-07-28 has a client leave out a tool whose ``x-mcp-header``
    annotations are invalid; an earlier revision knows no such annotation."""
    plan = {"server/discover": probe, "initialize": _HANDSHAKE, "tools/list": _HEADED_TOOLS}
    async with _provider(tmp_path, plan) as provider:
        assert provider.tool_names == kept


async def test_the_legacy_protocol_goes_straight_to_the_handshake(tmp_path: Path) -> None:
    """``protocol="legacy"`` sends no probe, for a server that would fail on
    it, even to a server that would answer it."""
    plan = {"server/discover": _DISCOVERY, "initialize": _HANDSHAKE, "tools/list": _TOOLS}
    async with _provider(tmp_path, plan, protocol="legacy") as provider:
        assert provider.server_info == {"name": "legacy", "version": "1"}
    methods = [message["method"] for message in _received(tmp_path)]
    assert methods == ["initialize", "notifications/initialized", "tools/list"]


def test_a_protocol_other_than_auto_or_legacy_is_refused() -> None:
    with pytest.raises(ValueError, match="protocol must be 'auto' or 'legacy'"):
        MCPToolProvider.from_command(sys.executable, protocol="2026-07-28")

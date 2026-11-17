"""A broken MCP peer script cannot pass as an expected server refusal."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tests.conformance._harness import Harness
from tests.conformance._mcp_peer import MCPPeer
from tests.conformance._mcp_shape import mcp_problems


async def test_peer_exit_closes_an_accepted_connection_before_waiting_for_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer({"replies": []})
    accepted = asyncio.Event()
    exchange = peer._exchange

    async def observe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted.set()
        await exchange(reader, writer)

    monkeypatch.setattr(peer, "_exchange", observe)
    await peer.__aenter__()
    assert peer._server is not None
    port = peer._server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        async with asyncio.timeout(1):
            await accepted.wait()
            await peer.__aexit__()
            assert await reader.read() == b""
    finally:
        writer.close()
        await writer.wait_closed()
        await peer.__aexit__()
    assert peer.problems == []


@pytest.mark.parametrize("after_shutdown", [False, True])
async def test_peer_closes_connections_accepted_at_shutdown_without_starting_their_work(
    monkeypatch: pytest.MonkeyPatch, after_shutdown: bool
) -> None:
    peer = MCPPeer({"replies": []})
    started = asyncio.Event()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        started.set()

    monkeypatch.setattr(peer, "_serve", serve)
    writer = Mock(spec=asyncio.StreamWriter)
    if after_shutdown:
        await peer.__aexit__()
    # No scheduling checkpoint between acceptance and shutdown.
    peer._accepted(asyncio.StreamReader(), writer)
    await peer.__aexit__()
    writer.close.assert_called_once()
    assert not started.is_set()


@pytest.mark.parametrize(
    "update",
    [
        {"peer": None},
        {"peer": {"replies": [], "reply": {}}},
        {"peer": {"replies": None}},
        {"peer": {"replies": [{"method": "read", "result": {}, "error": {}}]}},
        {"peer": {"replies": [{"method": "read", "hold": False}]}},
        {"peer": {"replies": [{"method": "read", "error": {"code": True, "message": "x"}}]}},
        {"peer": {"replies": [], "framing": "later"}},
        {"peer": {"replies": [], "initializations": []}},
        {"peer": {"replies": [], "initializations": [{"server_info": False}]}},
        {"peer": {"replies": [], "initializations": [{"instruction": "unknown"}]}},
        {"operations": []},
        {"operations": [{"method": "unknown"}]},
        {"operations": [{"method": "connect", "timeout_ms": 10}]},
        {"operations": [{"method": "request"}]},
        {"operations": [{"method": "request", "name": "read", "params": []}]},
        {"operations": [{"method": "read_resource", "uri": "resource://x", "timeout_ms": 0}]},
        {"operations": [{"method": "tool_meta", "mutate": {"path": [], "value": None}}]},
        {"discover": None},
        {"tool_filter": "all"},
        {"constructor": "later"},
        {"constructor": []},
        {"transport": "legacy"},
        {"label": 1},
        {"connect_timeout_ms": 0},
        {"connect_timeout_ms": True},
        {"headers": None},
        {"headers": {"X-Test": 1}},
        {"headers_after_construction": {}},
        {"headers": {}, "headers_after_construction": []},
        {"peer": {"replies": [], "endpoint_on_connections": []}, "transport": "sse"},
        {"peer": {"replies": [], "endpoint_on_connections": [1]}, "transport": "sse"},
        {"peer": {"replies": [], "endpoint_on_connections": [True]}},
        {"operations": [{"method": "call_tool"}]},
        {"operations": [{"method": "call_tool", "name": "echo", "arguments": []}]},
    ],
)
def test_malformed_host_controls_are_refused(update: dict[str, Any]) -> None:
    spec = {"peer": {"replies": []}, "operations": [{"method": "connect"}], **update}
    assert mcp_problems(spec, "mcp_session")


@pytest.mark.parametrize(
    "replies", [[], [{"method": "another", "result": {}}], [{"method": "another", "hold": True}]]
)
async def test_an_expected_server_error_cannot_hide_a_wrong_peer_script(
    replies: list[dict[str, Any]],
) -> None:
    scenario = {
        "steps": [
            {
                "call": "mcp_session",
                "discover": False,
                "peer": {"replies": replies},
                "operations": [
                    {"method": "connect"},
                    {"method": "request", "name": "read"},
                    {"method": "close"},
                ],
                "expect": {"result": {"results": [None, {"error": "mcp_error"}, None]}},
            }
        ]
    }
    problems = await Harness(scenario).run()
    assert any("the runner cannot read the call" in problem for problem in problems)


def test_a_wrong_held_request_is_refused_instead_of_hanging() -> None:
    peer = MCPPeer({"replies": [{"method": "vendor/read", "hold": True}]})
    reply = peer._reply({"jsonrpc": "2.0", "id": 2, "method": "vendor/next", "params": {}})
    assert reply == {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {"code": -32603, "message": "fixture request mismatch"},
    }
    assert len(peer.problems) == 1
    assert "vendor/next" in peer.problems[0]


@pytest.mark.parametrize("params", ["omitted", None, [], False])
@pytest.mark.parametrize("method", ["vendor/read", "tools/list"])
def test_host_request_parameters_are_not_normalized_into_an_empty_object(
    params: Any, method: str
) -> None:
    peer = MCPPeer({"replies": [{"method": method, "params": {}, "result": {}}]})
    message = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params != "omitted":
        message["params"] = params
    peer._reply(message)
    assert peer.problems


def test_only_discovery_ignores_the_sdks_generated_progress_token() -> None:
    reply = {"method": "tools/list", "params": {}, "result": {"tools": []}}
    for discovery in (False, True):
        peer = MCPPeer({"replies": [reply]})
        peer.discovery = discovery
        peer._reply(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {"_meta": {"progressToken": 0}},
            }
        )
        assert bool(peer.problems) is not discovery


@pytest.mark.parametrize("mutation", ["extra", "missing-null"])
async def test_host_result_oracles_reject_extra_fields_and_missing_nulls(mutation: str) -> None:
    name = "raw-results-json" if mutation == "extra" else "resource-results-json"
    path = Path(__file__).parent / "scenarios" / "mcp-host" / f"{name}.json"
    scenario = json.loads(path.read_text())
    for reply in scenario["steps"][0]["peer"]["replies"]:
        if mutation == "extra":
            reply["result"]["unexpected"] = True
        else:
            reply["result"].pop("optional", None)
            reply["result"]["contents"][0].pop("optional", None)
    assert await Harness(scenario).run()


@pytest.mark.parametrize(
    "operation",
    [
        {"method": "server_info", "mutate": {"path": ["absent", "child"], "value": True}},
        {"method": "server_info", "mutate": {"path": ["name", "child"], "value": True}},
    ],
)
async def test_invalid_observations_are_independent_fixture_failures(
    operation: dict[str, Any],
) -> None:
    scenario = {
        "steps": [
            {
                "call": "mcp_session",
                "discover": False,
                "peer": {"replies": []},
                "operations": [{"method": "connect"}, operation],
                "expect": {"error": {"name": "<unreadable>"}},
            }
        ]
    }
    problems = await Harness(scenario).run()
    assert any("the runner cannot read the call" in problem for problem in problems)


async def test_an_unexpected_runtime_error_during_connect_is_a_fixture_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from roomkit.tools import MCPToolProvider

    async def fail(provider: MCPToolProvider) -> MCPToolProvider:
        raise RuntimeError("unexpected connection refusal")

    monkeypatch.setattr(MCPToolProvider, "__aenter__", fail)
    scenario = {
        "steps": [
            {
                "call": "mcp_session",
                "discover": False,
                "peer": {"replies": []},
                "operations": [{"method": "connect"}],
                "expect": {"result": {"results": [{"error": "already_connected"}]}},
            }
        ]
    }
    problems = await Harness(scenario).run()
    assert any("the runner cannot read the call" in problem for problem in problems)
    assert any("unexpected connection refusal" in problem for problem in problems)

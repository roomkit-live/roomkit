"""A broken MCP peer script cannot pass as an expected server refusal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conformance._harness import Harness
from tests.conformance._mcp_peer import MCPPeer
from tests.conformance._mcp_shape import mcp_problems


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
    ],
)
def test_malformed_host_controls_are_refused(update: dict[str, Any]) -> None:
    spec = {"peer": {"replies": []}, "operations": [{"method": "connect"}], **update}
    assert mcp_problems(spec, "mcp_session")


@pytest.mark.parametrize("replies", [[], [{"method": "another", "result": {}}]])
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
        {"method": "connect"},
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

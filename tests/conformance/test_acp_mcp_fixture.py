"""An invalid or exhausted MCP fixture cannot pass as an expected channel error."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._acp_mcp import ScriptedMCPServers, mcp_problems
from tests.conformance._harness import Harness
from tests.conformance._script import Gates


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {},
        {"mode": "later", "responses": []},
        {"mode": "async", "responses": [{"servers": [], "error": "no"}]},
        {"mode": "sync", "responses": [{"servers": [], "gate": "hold"}]},
        {"mode": "async", "responses": [{"servers": [], "gates": "hold"}]},
        [{"name": "tools", "command": "tool", "args": [], "env": {}, "unknown": True}],
        [{"type": "stdio", "name": "tools", "command": "tool", "args": [], "env": []}],
        [{"type": "http", "name": "tools", "url": "https://example.test", "headers": [1]}],
    ],
)
def test_malformed_mcp_sources_are_rejected(spec: Any) -> None:
    assert mcp_problems(spec, "acp.mcp_servers")
    with pytest.raises(ValueError):
        ScriptedMCPServers(spec, Gates())


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_expected_callback_error_does_not_hide_exhaustion(mode: str) -> None:
    scenario = {
        "setup": {
            "channels": [
                {"id": "ws", "type": "websocket", "category": "transport"},
                {
                    "id": "agent",
                    "type": "ai",
                    "acp": {
                        "cwd": "/srv/project",
                        "agent": {},
                        "mcp_servers": {"mode": mode, "responses": []},
                    },
                },
            ],
            "rooms": [{"id": "room", "bindings": [{"channel_id": "ws"}, {"channel_id": "agent"}]}],
        },
        "steps": [
            {
                "call": "process_inbound",
                "room_id": "room",
                "message": {
                    "channel_id": "ws",
                    "sender_id": "alice",
                    "content": {"type": "text", "body": "Hello"},
                },
                "expect": {"result": {"error": {"$contains": "callback script exhausted"}}},
            }
        ],
        "expect": {"acp_mcp_requests": {"agent": ["room"]}},
    }
    problems = await Harness(scenario).run()
    assert any("acp.mcp_servers: callback script exhausted" in problem for problem in problems)

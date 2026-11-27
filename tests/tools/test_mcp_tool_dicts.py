"""A dictionary snapshot owns its data and keeps the AITool model fields."""

from __future__ import annotations

from tests.conformance._mcp_peer import MCPPeer

from roomkit.tools import MCPToolProvider


async def test_tool_dictionaries_own_their_nested_values() -> None:
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}
    tool = {"name": "echo", "description": "Echo.", "inputSchema": parameters}
    peer = MCPPeer({"replies": [{"method": "tools/list", "result": {"tools": [tool]}}]})
    expected = [
        {
            "name": "echo",
            "description": "Echo.",
            "parameters": parameters,
            "tags": [],
            "defer_loading": False,
            "changes_state": False,
        }
    ]
    async with peer:
        peer.discovery = True
        async with MCPToolProvider.from_url(peer.url) as provider:
            peer.discovery = False
            snapshot = provider.get_tools_as_dicts()
            assert snapshot == expected
            snapshot[0]["parameters"]["properties"]["text"]["type"] = "integer"
            snapshot[0]["tags"].append("changed")
            snapshot[0].pop("name")
            assert provider.get_tools_as_dicts() == expected
            assert provider.get_tools()[0].name == "echo"
    assert not peer.problems
    assert [request["method"] for request in peer.requests] == [
        "server/discover",
        "initialize",
        "tools/list",
    ]

"""An MCP server over stdio for the conformance runner's ``mcp_servers``.

Started as ``python _mcp_server.py <server spec as JSON>``: it lists the
spec's ``tools`` as written (README, "Scripted agents") and answers every
call of one with its ``returns``, a result that says ``isError`` when the
tool's ``is_error`` is true.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


def _server(spec: dict[str, Any]) -> Server:
    tools = spec["tools"]

    async def _list(
        _ctx: Any, _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        # As the spec writes them, a name listed twice included.
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=tool["name"],
                    description=tool.get("description"),
                    inputSchema=tool.get("input_schema", {"type": "object"}),
                )
                for tool in tools
            ]
        )

    async def _call(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        # The low-level server applies no input schema: any arguments reach here.
        tool = next(tool for tool in tools if tool["name"] == params.name)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=tool["returns"])],
            isError=tool.get("is_error", False),
        )

    return Server(spec.get("label") or "scripted", on_list_tools=_list, on_call_tool=_call)


async def _serve(spec: dict[str, Any]) -> None:
    server = _server(spec)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(_serve, json.loads(sys.argv[1]))

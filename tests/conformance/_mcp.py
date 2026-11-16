"""A scripted agent's MCP servers (README, "Scripted agents", ``mcp_servers``).

Each server runs as a subprocess of the runner (``_mcp_server.py``) and is
reached through roomkit's own ``MCPToolProvider`` over stdio, connected
before the run, its tools listed then, and closed after it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from roomkit.providers.ai.base import AITool
from roomkit.tools import MCPToolProvider
from roomkit.tools.compose import ToolHandler

_SERVER = str(Path(__file__).with_name("_mcp_server.py"))

Connected = Sequence[tuple[Mapping[str, Any], MCPToolProvider]]
"""A channel's servers, each its spec and its connected provider, in order."""


class MCPServers:
    """Every setup channel's MCP providers, connected on one stack."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.by_channel: dict[str, list[tuple[Mapping[str, Any], MCPToolProvider]]] = {}

    async def connect(self, channels: Sequence[Mapping[str, Any]]) -> None:
        """Start and connect the servers of each of *channels*, in order."""
        for channel in channels:
            for spec in channel.get("mcp_servers", []):
                provider = MCPToolProvider.from_command(
                    sys.executable, [_SERVER, json.dumps(spec)], label=spec.get("label")
                )
                await self._stack.enter_async_context(provider)
                self.by_channel.setdefault(channel["id"], []).append((spec, provider))

    async def close(self) -> None:
        """Disconnect every provider and stop its server."""
        await self._stack.aclose()


def declared_tools(servers: Connected) -> list[AITool]:
    """The definitions the agent declares for *servers*, as each ``declare``
    says: the provider's own, under their aliases, or none."""
    tools: list[AITool] = []
    for spec, provider in servers:
        declare = spec.get("declare", "bare")
        for tool in provider.get_tools():
            if declare == "bare":
                tools.append(tool)
            elif declare == "aliases":
                alias = f"mcp__{spec['label']}__{tool.name}"
                tools.append(tool.model_copy(update={"name": alias}))
    return tools


def tool_handlers(servers: Connected) -> list[ToolHandler]:
    """Each server's tool handler, in the chain's order."""
    return [
        provider.as_tool_handler(gate_discovery=spec.get("gate_discovery", True))
        for spec, provider in servers
    ]

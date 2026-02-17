"""MCP tool provider with compose_tool_handlers.

Demonstrates how to combine MCP-discovered tools with custom local tools
using compose_tool_handlers, and wire them into an AIChannel.

Since this example doesn't require a real MCP server, it uses a mock
MCPToolProvider setup to show the composition pattern.

Run with:
    uv run python examples/mcp_tool_provider.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from roomkit import (
    AIChannel,
    AITool,
    ChannelCategory,
    InboundMessage,
    MockAIProvider,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
    compose_tool_handlers,
)

logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")


# -- Custom local tool handler ------------------------------------------------

CLOCK_TOOL = AITool(
    name="get_time",
    description="Get the current time",
    parameters={
        "type": "object",
        "properties": {"timezone": {"type": "string", "default": "UTC"}},
    },
)


async def local_tool_handler(name: str, arguments: dict[str, Any]) -> str:
    """Handle locally-defined tools."""
    if name == "get_time":
        tz = arguments.get("timezone", "UTC")
        return json.dumps({"time": "2026-02-16T12:00:00", "timezone": tz})
    return json.dumps({"error": f"Unknown tool: {name}"})


# -- Simulated MCP tools (in a real app these come from MCPToolProvider) ------

MCP_SEARCH_TOOL = AITool(
    name="web_search",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


async def mcp_tool_handler(name: str, arguments: dict[str, Any]) -> str:
    """Simulates an MCP server's tool handler."""
    if name == "web_search":
        query = arguments.get("query", "")
        return json.dumps({"results": [f"Result for: {query}"]})
    return json.dumps({"error": f"Unknown tool: {name}"})


# -- Main ---------------------------------------------------------------------


async def main() -> None:
    kit = RoomKit()

    # Compose handlers: local tools first, then MCP tools
    combined = compose_tool_handlers(local_tool_handler, mcp_tool_handler)

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=MockAIProvider(
            responses=[
                "The current time is 12:00 UTC.",
                "Here are the search results for 'RoomKit'.",
            ]
        ),
        system_prompt="You are a helpful assistant with access to local and MCP tools.",
        tool_handler=combined,
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "ws-user")
    await kit.attach_channel(
        "demo-room",
        "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
    )

    # -- Demo: tools are dispatched through composed handler --
    print("=== Composed Tool Handler Demo ===\n")

    # Local tool
    result = await combined("get_time", {"timezone": "America/Montreal"})
    print(f"  get_time → {result}")

    # MCP tool
    result = await combined("web_search", {"query": "RoomKit framework"})
    print(f"  web_search → {result}")

    # Unknown tool (falls through both)
    result = await combined("nonexistent", {})
    print(f"  nonexistent → {result}")

    # -- Demo: end-to-end with AIChannel --
    print("\n=== AIChannel Integration ===\n")

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="What time is it?"),
        )
    )

    for ev in inbox:
        if ev.source.channel_id == "ai-assistant":
            print(f"  AI: {ev.content.body}")  # type: ignore[union-attr]

    await kit.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

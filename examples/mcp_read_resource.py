"""RoomKit -- read an MCP server's resource without listing its tools.

A host that only reads a resource (an MCP App's HTML, a document) has no use
for the server's tool catalogue, which on a gateway can weigh hundreds of
kilobytes. ``discover=False`` connects without sending ``tools/list``: the
read costs the connection and one request, however many tools the server has.
A tool call on such a connection never lists the tools either (RFC §21.2).

    MCPToolProvider(discover=False) ── stdio ── examples/mcp_servers/notes_server.py
        ├── read_resource("notes://about")
        └── call_tool("add_note", ...)

Requirements:
    pip install roomkit[mcp]

Run:
    uv run python examples/mcp_read_resource.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit.tools import MCPToolProvider

logger = setup_logging("mcp_read_resource")

NOTES_SERVER = Path(__file__).resolve().parent / "mcp_servers" / "notes_server.py"


async def main() -> None:
    async with MCPToolProvider.from_command(
        sys.executable, [str(NOTES_SERVER)], discover=False
    ) as mcp:
        logger.info("Tools listed: %d", len(mcp.get_tools()))  # 0: nothing was listed

        about = await mcp.read_resource("notes://about")
        for content in about.contents:
            logger.info("notes://about says: %s", content.text)

        logger.info("add_note: %s", await mcp.call_tool("add_note", {"text": "Buy coffee."}))


if __name__ == "__main__":
    asyncio.run(main())

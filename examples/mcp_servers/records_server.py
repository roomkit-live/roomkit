"""A tiny MCP server over stdio, for examples/mcp_server_labels.py.

Serves ``search_records`` and ``delete_records`` for the store named by its
first argument, and says in each answer which store ran the call: started
twice, as ``crm`` and ``archive``, it shows which server an MCP alias reached.

    python examples/mcp_servers/records_server.py crm   # speaks MCP on stdin/stdout
"""

from __future__ import annotations

import sys

from mcp.server import MCPServer

STORE = sys.argv[1] if len(sys.argv) > 1 else "records"
server = MCPServer(STORE, log_level="WARNING")  # logs to our stderr


@server.tool()
def search_records(query: str = "") -> str:
    """Search the store's records."""
    return f"{STORE}: no record matches {query!r}"


@server.tool()
def delete_records(query: str = "") -> str:
    """Delete the store's records that match the query."""
    return f"{STORE}: deleted the records matching {query!r}"


if __name__ == "__main__":
    server.run()

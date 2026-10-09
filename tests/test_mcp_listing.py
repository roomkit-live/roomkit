"""What MCPToolProvider lists, and when (RFC §21.2), against a real HTTP server.

The server counts every ``tools/list`` a client sends (its own cache refresh is
not one) and pages its listing two tools at a time, so the tests read what
crosses the wire rather than what the provider believes it did.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from roomkit.tools.mcp import MCPToolProvider, _list_tools

pytest.importorskip("mcp.server.fastmcp")

_SERVER = textwrap.dedent(
    """\
    import sys
    from pydantic import BaseModel
    from starlette.responses import PlainTextResponse
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, ListToolsRequest, ListToolsResult, TextContent

    server = FastMCP("roomkit-listing-test", port=int(sys.argv[1]), log_level="WARNING")
    listings = 0
    PAGE = 2
    HIDDEN = {"unlisted"}


    class Count(BaseModel):
        n: int


    @server.tool()
    def add(a: int, b: int) -> int:
        \"\"\"Add two integers.\"\"\"
        return a + b


    @server.tool()
    def good_count() -> Count:
        \"\"\"A count that matches its output schema.\"\"\"
        return Count(n=1)


    @server.tool()
    def bad_count() -> Count:
        \"\"\"A count that breaks its output schema (see call below).\"\"\"
        return Count(n=0)


    @server.tool()
    def bare_count() -> Count:
        \"\"\"A count sent without structured content (see call below).\"\"\"
        return Count(n=0)


    @server.tool()
    def failed_count() -> Count:
        \"\"\"A count that fails: an error result carries no structured content.\"\"\"
        raise ValueError("no count")


    @server.tool()
    def unlisted() -> str:
        \"\"\"Served, but never named by tools/list.\"\"\"
        return "served"


    async def list_page(req: ListToolsRequest) -> ListToolsResult:
        global listings
        tools = [t for t in await server.list_tools() if t.name not in HIDDEN]
        tools.sort(key=lambda t: t.name)
        if req is None:  # the server refreshing its own cache, not a client
            return ListToolsResult(tools=tools)
        listings += 1
        start = int(req.params.cursor) if req.params and req.params.cursor else 0
        end = start + PAGE
        return ListToolsResult(
            tools=tools[start:end], nextCursor=str(end) if end < len(tools) else None
        )


    # FastMCP checks a tool's result against its own model before it leaves:
    # these two answers go out as raw results, as a server that does not check.
    RAW = {
        "bad_count": CallToolResult(
            content=[TextContent(type="text", text="many")], structuredContent={"n": "many"}
        ),
        "bare_count": CallToolResult(content=[TextContent(type="text", text="1")]),
    }


    async def call(name, arguments):
        if name in RAW:
            return RAW[name]
        return await server.call_tool(name, arguments)


    server._mcp_server.list_tools()(list_page)
    server._mcp_server.call_tool(validate_input=False)(call)


    @server.custom_route("/listings", methods=["GET"])
    async def listings_route(request):
        return PlainTextResponse(str(listings))


    server.run(transport="streamable-http")
    """
)

LISTED = ["add", "bad_count", "bare_count", "failed_count", "good_count"]


@pytest.fixture(scope="module")
def server_port(tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    script = tmp_path_factory.mktemp("mcp") / "listing_server.py"
    script.write_text(_SERVER)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = script.with_suffix(".log")
    with log.open("wb") as out:
        process = subprocess.Popen(
            [sys.executable, str(script), str(port)], stdout=out, stderr=out
        )
    try:
        _wait_listening(port, log)
        yield port
    finally:
        process.terminate()
        process.wait(timeout=10)


def _wait_listening(port: int, log: Path) -> None:
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.1)
    pytest.fail(f"MCP server never listened:\n{log.read_text()}")


async def _listings(port: int) -> int:
    """How many ``tools/list`` requests clients have sent the server so far."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://127.0.0.1:{port}/listings")
    return int(response.text)


def _provider(port: int, **kwargs: Any) -> MCPToolProvider:
    return MCPToolProvider.from_url(f"http://127.0.0.1:{port}/mcp", **kwargs)


async def test_discovery_reads_every_page(server_port: int) -> None:
    before = await _listings(server_port)
    async with _provider(server_port) as mcp:
        assert mcp.tool_names == LISTED
    assert await _listings(server_port) - before == 3  # five tools, two a page


class _LoopingSession:
    """A server whose listing hands out the same cursor forever."""

    def __init__(self) -> None:
        self.pages = 0

    async def list_tools(self, *, params: Any = None) -> Any:
        self.pages += 1
        return SimpleNamespace(tools=[SimpleNamespace(name=f"t{self.pages}")], nextCursor="again")


async def test_a_repeated_cursor_ends_the_listing(caplog: pytest.LogCaptureFixture) -> None:
    session = _LoopingSession()
    tools = await _list_tools(session)
    assert [tool.name for tool in tools] == ["t1", "t2"]
    assert "repeated a cursor" in caplog.text

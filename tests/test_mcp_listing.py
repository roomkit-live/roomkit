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
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from roomkit.core.exceptions import UnservedToolCallError
from roomkit.tools._mcp_result import check_structured_content
from roomkit.tools.mcp import _MAX_LIST_PAGES, MCPToolProvider, _list_tools

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


async def test_calls_never_list_again_listed_or_not(server_port: int) -> None:
    async with _provider(server_port) as mcp:
        handler = mcp.as_tool_handler(gate_discovery=False)
        before = await _listings(server_port)
        for _ in range(3):
            assert await mcp.call_tool("add", {"a": 1, "b": 2}) == "3"
            assert await mcp.call_tool("unlisted", {}) == "served"
            assert (await mcp.call_tool_result("good_count", {})).structuredContent == {"n": 1}
            assert await handler("unlisted", {}) == "served"
        assert await _listings(server_port) == before


async def test_a_listed_tool_that_breaks_its_output_schema_fails(server_port: int) -> None:
    async with _provider(server_port) as mcp:
        with pytest.raises(RuntimeError, match="Invalid structured content returned by tool"):
            await mcp.call_tool_result("bad_count", {})
        with pytest.raises(RuntimeError, match="did not return structured content"):
            await mcp.call_tool_result("bare_count", {})


async def test_the_handler_fails_on_a_broken_result_too(server_port: int) -> None:
    async with _provider(server_port) as mcp:
        with pytest.raises(RuntimeError, match="Invalid structured content"):
            await mcp.as_tool_handler()("bad_count", {})


async def test_an_error_result_is_not_validated(server_port: int) -> None:
    async with _provider(server_port) as mcp:
        result = await mcp.call_tool_result("failed_count", {})
    assert result.isError


async def test_a_tool_the_filter_hides_is_still_validated(server_port: int) -> None:
    """The schemas come from the whole listing: a host calls a tool the model
    never saw (an MCP App's app-only tool) and it is checked as before."""
    async with _provider(server_port, tool_filter=lambda name: name == "add") as mcp:
        assert mcp.tool_names == ["add"]
        with pytest.raises(RuntimeError, match="Invalid structured content"):
            await mcp.call_tool_result("bad_count", {})


async def test_without_discovery_the_connection_lists_nothing(server_port: int) -> None:
    before = await _listings(server_port)
    async with _provider(server_port, discover=False) as mcp:
        assert mcp.connected
        assert mcp.get_tools() == []
        assert mcp.tool_meta() == {}
        assert await mcp.call_tool("add", {"a": 2, "b": 2}) == "4"
        # Nothing was listed, so nothing is validated: the broken result passes.
        assert (await mcp.call_tool_result("bad_count", {})).structuredContent == {"n": "many"}
    assert await _listings(server_port) == before


async def test_without_discovery_the_gated_handler_serves_no_name(server_port: int) -> None:
    async with _provider(server_port, discover=False) as mcp:
        with pytest.raises(UnservedToolCallError):
            await mcp.as_tool_handler()("add", {"a": 1, "b": 1})


_STDIO_SERVER = textwrap.dedent(
    """\
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("roomkit-stdio-listing-test", log_level="WARNING")

    @server.tool()
    def echo(text: str) -> str:
        \"\"\"Say it back.\"\"\"
        return text

    server.run()
    """
)


async def test_from_command_connects_without_listing(tmp_path: Path) -> None:
    script = tmp_path / "stdio_server.py"
    script.write_text(_STDIO_SERVER)
    async with MCPToolProvider.from_command(sys.executable, [str(script)], discover=False) as mcp:
        assert mcp.get_tools() == []
        assert await mcp.call_tool("echo", {"text": "hi"}) == "hi"
    async with MCPToolProvider.from_command(sys.executable, [str(script)]) as mcp:
        assert mcp.tool_names == ["echo"]


def test_a_remote_ref_is_refused_without_being_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[Any] = []

    def offline(*args: Any, **kwargs: Any) -> Any:
        fetched.append(args)
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    schema = {"$ref": "https://schemas.invalid/count.json"}
    with pytest.raises(RuntimeError, match="Invalid schema for tool count"):
        check_structured_content("count", schema, SimpleNamespace(structuredContent={"n": 1}))
    assert fetched == []


class _PagingSession:
    """A server whose listing pages as *cursors* says, one tool a page while
    *tools_until* allows, then empty pages."""

    def __init__(self, cursors: Callable[[int], str], tools_until: int) -> None:
        self.pages = 0
        self._cursors = cursors
        self._tools_until = tools_until

    async def list_tools(self, *, params: Any = None) -> Any:
        self.pages += 1
        tools = [SimpleNamespace(name=f"t{self.pages}")] if self.pages <= self._tools_until else []
        return SimpleNamespace(tools=tools, nextCursor=self._cursors(self.pages))


@pytest.mark.parametrize(
    ("cursors", "tools_until", "pages", "kept"),
    [
        pytest.param(lambda page: "again", 10**9, 2, 2, id="a repeated cursor"),
        pytest.param(lambda page: str(page * 2), 1, 2, 1, id="an empty page with a cursor"),
        pytest.param(lambda page: str(page), 10**9, _MAX_LIST_PAGES, _MAX_LIST_PAGES, id="no end"),
    ],
)
async def test_a_listing_that_pages_without_end_stops(
    caplog: pytest.LogCaptureFixture,
    cursors: Callable[[int], str],
    tools_until: int,
    pages: int,
    kept: int,
) -> None:
    session = _PagingSession(cursors, tools_until)
    tools = await _list_tools(session, "http://looping.invalid/mcp")
    assert session.pages == pages
    assert len(tools) == kept
    assert "http://looping.invalid/mcp pages tools/list without end" in caplog.text

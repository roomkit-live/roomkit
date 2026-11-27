"""What a host reads from an MCP server beside the model's tools (RMK-408).

An MCP App renders a tool's result in a frame: the host reads the tool's
``_meta.ui`` from the listing, the app's HTML as a resource, and relays the
frame's own calls, an app-only tool hidden from the model among them, with the
server's raw result. Against a real MCPServer over stdio.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from roomkit.tools.mcp import MCPToolProvider

pytest.importorskip("mcp.server.mcpserver")
MCPError = pytest.importorskip("mcp.shared.exceptions").MCPError

_SERVER = textwrap.dedent(
    """\
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer("roomkit-apps")

    @server.tool(meta={"ui": {"resourceUri": "ui://board/view.html", "csp": "self"}})
    def show_board(name: str) -> str:
        \"\"\"Show a board.\"\"\"
        return f"board {name}"

    @server.tool()
    def plain() -> str:
        \"\"\"No UI.\"\"\"
        return "plain"

    @server.tool()
    def app_only_save(state: str) -> str:
        \"\"\"Called by the app's frame, never by the model.\"\"\"
        return f"saved {state}"

    @server.tool()
    def refuse() -> str:
        \"\"\"Always refuses.\"\"\"
        raise ToolError("not allowed")

    @server.resource("ui://board/view.html", mime_type="text/html")
    def view() -> str:
        return "<html>board</html>"

    server.run()
    """
)


@pytest.fixture
def server_script(tmp_path: Path) -> str:
    path = tmp_path / "server.py"
    path.write_text(_SERVER)
    return str(path)


def _provider(script: str) -> MCPToolProvider:
    """The model is offered every tool but the app-only one."""
    return MCPToolProvider(
        transport="stdio",
        command=sys.executable,
        args=[script],
        tool_filter=lambda name: not name.startswith("app_only_"),
    )


async def test_the_tool_meta_comes_from_the_listing_made_at_connection(server_script: str) -> None:
    async with _provider(server_script) as mcp:
        meta = mcp.tool_meta()

    assert meta["show_board"]["ui"] == {"resourceUri": "ui://board/view.html", "csp": "self"}
    assert "app_only_save" not in meta  # not discovered: filtered


async def test_a_resource_is_read_as_the_server_returns_it(server_script: str) -> None:
    async with _provider(server_script) as mcp:
        result = await mcp.read_resource("ui://board/view.html")
        with pytest.raises(MCPError):
            await mcp.read_resource("ui://board/missing.html")

    assert result["contents"][0]["text"] == "<html>board</html>"


async def test_the_raw_result_reaches_a_tool_the_filter_hid_from_the_model(
    server_script: str,
) -> None:
    async with _provider(server_script) as mcp:
        assert "app_only_save" not in mcp.tool_names
        saved = await mcp.call_tool_result("app_only_save", {"state": "s1"})
        refused = await mcp.call_tool_result("refuse", {})

    assert saved.is_error is False
    assert saved.content[0].text == "saved s1"
    assert refused.is_error is True  # a refusal is the result's, not an exception


async def test_the_meta_handed_out_is_a_copy(server_script: str) -> None:
    async with _provider(server_script) as mcp:
        mcp.tool_meta()["show_board"]["ui"]["csp"] = "*"

        assert mcp.tool_meta()["show_board"]["ui"]["csp"] == "self"


async def test_connected_says_whether_the_connection_is_live(server_script: str) -> None:
    provider = _provider(server_script)
    assert provider.connected is False
    async with provider as mcp:
        assert mcp.connected is True
    assert provider.connected is False


async def test_a_catalogue_that_fails_releases_the_connection(server_script: str) -> None:
    """What was opened is closed, the stdio server included (RFC §21.2), and
    the provider can connect again."""

    def broken_filter(name: str) -> bool:
        raise ZeroDivisionError("filter failed")

    provider = MCPToolProvider(
        transport="stdio", command=sys.executable, args=[server_script], tool_filter=broken_filter
    )
    with pytest.raises(ZeroDivisionError):
        async with provider:
            pass

    assert provider.connected is False
    assert provider._stack is None


async def test_reading_before_connecting_is_refused() -> None:
    mcp = MCPToolProvider("http://localhost:1/mcp")

    with pytest.raises(RuntimeError, match="not connected"):
        mcp.tool_meta()
    with pytest.raises(RuntimeError, match="not connected"):
        await mcp.read_resource("ui://x")

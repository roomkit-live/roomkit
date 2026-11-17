"""The direct MCP constructor keeps the same public behavior as its factories."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from roomkit.core.exceptions import UnservedToolCallError
from roomkit.tools import MCPToolProvider


@pytest.mark.parametrize(
    ("options", "refusal"),
    [
        ({"label": "bad__label", "transport": "invalid"}, "cannot name its server"),
        ({"transport": "invalid"}, "Unsupported transport"),
        ({"transport": "stdio", "url": "http://example.test"}, "needs a command"),
        ({"transport": "stdio", "command": ""}, "needs a command"),
        (
            {"transport": "stdio", "command": "server", "url": "http://example.test"},
            "HTTP transports",
        ),
        (
            {"transport": "stdio", "command": "server", "headers": {"X-Test": "1"}},
            "HTTP transports",
        ),
        ({"transport": "sse", "command": "server"}, "needs a url"),
        ({"url": "", "env": {}}, "needs a url"),
        ({"url": "http://example.test", "command": "server"}, "for stdio"),
        ({"url": "http://example.test", "args": ["arg"]}, "for stdio"),
        ({"url": "http://example.test", "env": {}}, "for stdio"),
        ({"url": "http://example.test", "cwd": ""}, "for stdio"),
        ({"url": "invalid", "headers": {"Accept": "invalid"}}, "endpoint"),
        (
            {"url": "http://example.test", "headers": {"Accept": "invalid"}},
            "reserved for the transport",
        ),
    ],
)
def test_direct_construction_refuses_crossed_options_in_source_order(
    options: dict[str, Any], refusal: str
) -> None:
    with pytest.raises(ValueError, match=refusal):
        MCPToolProvider(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"url": "http://example.test"},
        {"url": "http://example.test", "command": "", "args": [], "env": None, "cwd": None},
        {"url": "http://example.test", "transport": "sse", "headers": {}},
        {"transport": "stdio", "command": "roomkit-does-not-exist"},
        {"transport": "stdio", "command": "server", "url": "", "headers": {}},
        {"transport": "stdio", "command": "server", "env": {}, "cwd": ""},
    ],
)
def test_direct_construction_is_lazy_and_keeps_accepted_empty_options(
    options: dict[str, Any],
) -> None:
    provider = MCPToolProvider(**options)
    assert not provider.connected
    with pytest.raises(RuntimeError, match="not connected"):
        provider.get_tools()


_STDIO_SERVER = textwrap.dedent(
    """\
    import json
    import os
    import sys
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("constructor-test")

    @server.tool()
    def probe() -> str:
        return json.dumps({
            "args": sys.argv[1:],
            "env": os.environ.get("ROOMKIT_CONSTRUCTOR_TEST"),
            "cwd": os.getcwd(),
        })

    @server.tool()
    def ignored() -> str:
        return "not listed"

    server.run()
    """
)


@pytest.mark.parametrize("discover", [False, True])
async def test_direct_stdio_copies_arguments_and_reads_the_retained_environment_on_entry(
    tmp_path: Path, discover: bool
) -> None:
    pytest.importorskip("mcp.server.fastmcp")
    script = tmp_path / "server.py"
    script.write_text(_STDIO_SERVER)
    args = [str(script), "original"]
    env = {"ROOMKIT_CONSTRUCTOR_TEST": "before"}
    provider = MCPToolProvider(
        transport="stdio",
        command=sys.executable,
        args=args,
        env=env,
        cwd=str(tmp_path),
        label="capture",
        tool_filter=lambda name: name == "probe",
        discover=discover,
        connect_timeout=5,
    )
    args.clear()
    env["ROOMKIT_CONSTRUCTOR_TEST"] = "after"
    assert not provider.connected
    async with provider:
        assert provider.connected
        assert provider.tool_names == (["probe"] if discover else [])
        expected = {"args": ["original"], "env": "after", "cwd": str(tmp_path.resolve())}
        assert json.loads(await provider.call_tool("probe", {})) == expected
        handler = provider.as_tool_handler(gate_discovery=False)
        assert json.loads(await handler("mcp__capture__probe", {})) == expected
        with pytest.raises(UnservedToolCallError):
            await handler("mcp__another__probe", {})
    assert not provider.connected

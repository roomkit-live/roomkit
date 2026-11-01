"""MCPToolProvider against real MCP servers: stdio, streamable HTTP and SSE (FastMCP)."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from roomkit.core.exceptions import ToolFailedError, UnservedToolCallError
from roomkit.tools.mcp import MCPToolProvider

pytest.importorskip("mcp.server.fastmcp")
McpError = pytest.importorskip("mcp.shared.exceptions").McpError

_SERVER = textwrap.dedent(
    """\
    import os
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("roomkit-test")

    @server.tool()
    def add(a: int, b: int) -> int:
        \"\"\"Add two integers.\"\"\"
        return a + b

    @server.tool()
    def refuse() -> str:
        \"\"\"Always refuses.\"\"\"
        raise ValueError("not allowed today")

    @server.tool()
    def read_env(name: str) -> str:
        \"\"\"An environment variable of the server process.\"\"\"
        return os.environ.get(name, "<unset>")

    @server.tool()
    def pid() -> int:
        \"\"\"The server's process id.\"\"\"
        return os.getpid()

    @server.tool()
    def cwd() -> str:
        \"\"\"The server's working directory.\"\"\"
        return os.getcwd()

    server.run()
    """
)


@pytest.fixture
def server_script(tmp_path: Path) -> str:
    path = tmp_path / "server.py"
    path.write_text(_SERVER)
    return str(path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_discovers_and_calls_the_tools_of_a_stdio_server(server_script: str) -> None:
    async with MCPToolProvider.from_command(sys.executable, [server_script]) as mcp:
        assert sorted(mcp.tool_names) == ["add", "cwd", "pid", "read_env", "refuse"]
        add = next(t for t in mcp.get_tools() if t.name == "add")
        assert add.parameters["required"] == ["a", "b"]

        assert await mcp.call_tool("add", {"a": 2, "b": 3}) == "5"

        handler = mcp.as_tool_handler()
        assert await handler("add", {"a": 1, "b": 1}) == "2"
        with pytest.raises(ToolFailedError, match="not allowed today"):
            await handler("refuse", {})
        with pytest.raises(UnservedToolCallError):
            await handler("nope", {})


async def test_tool_filter_applies_to_stdio(server_script: str) -> None:
    provider = MCPToolProvider.from_command(
        sys.executable, [server_script], tool_filter=lambda name: name == "add"
    )
    async with provider as mcp:
        assert mcp.tool_names == ["add"]


async def test_env_reaches_the_server_and_the_rest_of_ours_does_not(
    monkeypatch: pytest.MonkeyPatch, server_script: str
) -> None:
    monkeypatch.setenv("ROOMKIT_TEST_PARENT_ONLY", "leak")
    provider = MCPToolProvider.from_command(
        sys.executable, [server_script], env={"ROOMKIT_TEST_TOKEN": "s3cret"}
    )
    async with provider as mcp:
        assert await mcp.call_tool("read_env", {"name": "ROOMKIT_TEST_TOKEN"}) == "s3cret"
        # The MCP SDK hands a server a minimal environment, plus env=.
        assert await mcp.call_tool("read_env", {"name": "ROOMKIT_TEST_PARENT_ONLY"}) == "<unset>"


async def test_exiting_stops_the_server_process(server_script: str) -> None:
    async with MCPToolProvider.from_command(sys.executable, [server_script]) as mcp:
        server_pid = int(await mcp.call_tool("pid", {}))
        assert _alive(server_pid)

    for _ in range(500):
        if not _alive(server_pid):
            break
        await asyncio.sleep(0.01)
    assert not _alive(server_pid)


async def test_a_server_that_dies_at_startup_leaves_the_provider_closed() -> None:
    provider = MCPToolProvider.from_command(sys.executable, ["-c", "import sys; sys.exit(3)"])

    with pytest.raises(McpError, match="Connection closed"):
        await asyncio.wait_for(provider.__aenter__(), timeout=10)

    assert provider._stack is None
    with pytest.raises(RuntimeError, match="not connected"):
        provider.get_tools()


async def test_a_command_that_does_not_exist_says_so() -> None:
    provider = MCPToolProvider.from_command("roomkit-no-such-mcp-server")

    with pytest.raises(FileNotFoundError):
        await provider.__aenter__()
    assert provider._stack is None


async def test_a_server_that_never_answers_is_stopped_at_the_connect_bound() -> None:
    """RMK-1021 item 38: entering waits for the handshake and the listing
    within ``connect_timeout``, then stops the server and raises; a server
    that never answered held its host for ever."""
    # Reads until its input closes and writes nothing: no handshake answer.
    provider = MCPToolProvider.from_command(
        sys.executable, ["-I", "-c", "import sys; sys.stdin.read()"], connect_timeout=0.05
    )

    with pytest.raises(TimeoutError, match="within 0.05 s"):
        await provider.__aenter__()
    assert provider._stack is None


async def test_cwd_is_the_server_working_directory(server_script: str, tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    provider = MCPToolProvider.from_command(sys.executable, [server_script], cwd=str(workdir))
    async with provider as mcp:
        assert Path(await mcp.call_tool("cwd", {})).resolve() == workdir.resolve()


async def test_entering_a_connected_provider_again_is_refused(server_script: str) -> None:
    provider = MCPToolProvider.from_command(sys.executable, [server_script])
    async with provider:
        with pytest.raises(RuntimeError, match="already connected"):
            await provider.__aenter__()


async def test_reconnecting_does_not_duplicate_the_tools(server_script: str) -> None:
    provider = MCPToolProvider.from_command(sys.executable, [server_script])
    async with provider as mcp:
        first = mcp.tool_names
    async with provider as mcp:
        assert mcp.tool_names == first


class TestConstruction:
    def test_stdio_needs_a_command(self) -> None:
        with pytest.raises(ValueError, match="needs a command"):
            MCPToolProvider(transport="stdio")

    def test_http_needs_a_url(self) -> None:
        with pytest.raises(ValueError, match="needs a url"):
            MCPToolProvider(transport="sse")

    @pytest.mark.parametrize("option", [{"env": {"A": "1"}}, {"cwd": "/tmp"}])
    def test_stdio_options_are_refused_on_http(self, option: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="for stdio"):
            MCPToolProvider("http://localhost/mcp", **option)

    def test_http_options_are_refused_on_stdio(self) -> None:
        with pytest.raises(ValueError, match="HTTP transports"):
            MCPToolProvider(transport="stdio", command="srv", headers={"A": "1"})

    def test_unknown_transport_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="Unsupported transport"):
            MCPToolProvider("http://localhost/mcp", transport="grpc")

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://mcp.example/mcp",
            "mcp.example/mcp",
            "http:///mcp",
            "https://user:secret@mcp.example/mcp",
            "https://user@mcp.example/mcp",
            "https://mcp.example/mcp#tools",
            "https://mcp.example:99999/mcp",
        ],
    )
    def test_an_endpoint_that_cannot_be_reached_as_given_is_refused(self, url: str) -> None:
        """RMK-1068: refused at construction, as roomkit-rs's from_url: no
        credentials in a URL every log and proxy sees, no fragment no request
        carries, an HTTP(S) URL naming a host."""
        with pytest.raises(ValueError, match="endpoint"):
            MCPToolProvider.from_url(url)

    @pytest.mark.parametrize(
        ("headers", "refusal"),
        [
            ({"Accept": "text/plain"}, "reserved for the transport"),
            ({"Content-Type": "text/plain"}, "reserved for the transport"),
            ({"Host": "elsewhere"}, "reserved for the transport"),
            ({"Mcp-Session-Id": "forged"}, "reserved for the transport"),
            ({"mcp-protocol-version": "2024-11-05"}, "reserved for the transport"),
            ({"Proxy-Authorization": "Basic x"}, "reserved for the transport"),
            ({"X Bad": "1"}, "invalid MCP header name"),
            ({"X-Line": "a\r\nInjected: 1"}, "invalid value"),
            ({"X-Big": "a" * 8193}, "exceeds 8192 bytes"),
            ({f"X-{n}": "1" for n in range(33)}, "more than 32"),
        ],
        ids=[
            "accept",
            "content-type",
            "host",
            "mcp-session-id",
            "mcp-protocol-version",
            "proxy-authorization",
            "invalid name",
            "header injection",
            "too long",
            "too many",
        ],
    )
    def test_a_header_that_cannot_be_sent_as_given_is_refused(
        self, headers: dict[str, str], refusal: str
    ) -> None:
        """RMK-1068: refused at construction, as roomkit-rs's from_url."""
        with pytest.raises(ValueError, match=refusal):
            MCPToolProvider.from_url("https://mcp.example/mcp", headers=headers)

    def test_the_headers_a_server_takes_are_kept(self) -> None:
        headers = {"Authorization": "Bearer t", **{f"X-{n}": "v" * 8192 for n in range(31)}}
        provider = MCPToolProvider.from_url("https://mcp.example:8443/mcp?k=v", headers=headers)
        assert provider._headers == headers


async def test_an_http_server_that_never_answers_is_left_at_the_connect_bound() -> None:
    """RMK-1108: entering waits for the handshake and the listing within
    ``connect_timeout``, as a started server's, then closes the connection
    and raises; a server that took the connection and never answered held
    its host for ever."""
    held: list[asyncio.StreamWriter] = []

    async def accept(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        held.append(writer)  # reads nothing, answers nothing

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    provider = MCPToolProvider.from_url(f"http://127.0.0.1:{port}/mcp", connect_timeout=0.05)
    try:
        with pytest.raises(TimeoutError, match="within 0.05 s"):
            await provider.__aenter__()
        assert (provider._stack, len(held)) == (None, 1)
    finally:
        for writer in held:
            writer.close()
        server.close()
        await server.wait_closed()


_HTTP_SERVER = textwrap.dedent(
    """\
    import sys
    from mcp.server.fastmcp import FastMCP

    from mcp.server.fastmcp import Context

    server = FastMCP("roomkit-http-test", port=int(sys.argv[1]))

    @server.tool()
    def add(a: int, b: int) -> int:
        \"\"\"Add two integers.\"\"\"
        return a + b

    @server.tool()
    def header(name: str, ctx: Context) -> str:
        \"\"\"A header of the HTTP request that called this tool.\"\"\"
        return ctx.request_context.request.headers.get(name, "<missing>")

    server.run(transport=sys.argv[2])
    """
)


@pytest.mark.parametrize(("transport", "path"), [("streamable-http", "/mcp"), ("sse", "/sse")])
async def test_http_transports_against_a_real_server(
    tmp_path: Path, transport: str, path: str
) -> None:
    script = tmp_path / "http_server.py"
    script.write_text(_HTTP_SERVER)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = tmp_path / "server.log"
    with log.open("wb") as out:
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(script), str(port), transport, stdout=out, stderr=out
        )
    try:
        for _ in range(500):
            try:
                _reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                break
            except OSError:
                await asyncio.sleep(0.02)
        else:
            pytest.fail(f"MCP {transport} server never listened:\n{log.read_text()}")
        provider = MCPToolProvider.from_url(
            f"http://127.0.0.1:{port}{path}",
            transport="streamable_http" if transport == "streamable-http" else "sse",
            headers={"X-Test": "1"},
        )
        async with provider as mcp:
            assert sorted(mcp.tool_names) == ["add", "header"]
            assert await mcp.call_tool("add", {"a": 20, "b": 22}) == "42"
            assert await mcp.call_tool("header", {"name": "x-test"}) == "1"
    finally:
        process.terminate()
        await process.wait()

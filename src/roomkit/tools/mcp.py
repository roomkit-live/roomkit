"""MCPToolProvider — bridge MCP servers into RoomKit's AITool/ToolHandler system."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any

from roomkit.core.exceptions import ToolFailedError, UnservedToolCallError
from roomkit.providers.ai.base import AITool, some_vendor_accepts_tool_name
from roomkit.tools._mcp_result import (
    check_structured_content,
    error_text,
    handler_result,
    text_body,
)
from roomkit.tools.compose import ToolHandler, ToolResult
from roomkit.tools.policy import served_tool_name

logger = logging.getLogger("roomkit.tools.mcp")


def _definition(tool: Any) -> AITool | None:
    """The definition of a listed MCP tool, or ``None`` for one whose name no
    provider accepts: skipped, so the server's other tools stay usable
    (RFC §6.7)."""
    # FastMCP serializes a tool's tags into `_meta["fastmcp"]["tags"]`;
    # surface them so Tool Search can match this tool cross-lingually.
    if not some_vendor_accepts_tool_name(tool.name):
        logger.warning("MCP tool %r skipped: no provider accepts its name", tool.name)
        return None
    meta = getattr(tool, "meta", None)
    tags = meta.get("fastmcp", {}).get("tags", []) if isinstance(meta, dict) else []
    return AITool(
        name=tool.name,
        description=tool.description or "",
        parameters=tool.inputSchema if tool.inputSchema else {},
        tags=tags or [],
    )


async def _list_tools(session: Any) -> list[Any]:
    """Every tool the server lists, following its cursor across pages (RFC §21.2).

    A cursor the server hands out twice ends the reading where it stands: a
    server that loops must not hold the connection open forever.
    """
    from mcp.types import PaginatedRequestParams

    tools: list[Any] = []
    seen: set[str] = set()
    params: Any = None
    while True:
        page = await session.list_tools(params=params)
        tools.extend(page.tools)
        cursor = getattr(page, "nextCursor", None)
        if not isinstance(cursor, str) or not cursor:
            return tools
        if cursor in seen:
            logger.warning("MCP tools/list repeated a cursor; keeping %d tools", len(tools))
            return tools
        seen.add(cursor)
        params = PaginatedRequestParams(cursor=cursor)


_DEFAULT_CALL_TIMEOUT = 30.0
"""Seconds a call to the server waits: one default shared by
:meth:`MCPToolProvider.call_tool`, :meth:`~MCPToolProvider.call_tool_result`,
:meth:`~MCPToolProvider.read_resource` and the tool handler, so they cannot
drift apart."""


# Upper bound for publishing a structured result on the tool-call context
# (serialized size). Tool-call events ride the room event pipeline — DB rows,
# WebSocket broadcasts, audit — so a pathological multi-megabyte payload must
# not tag along; every realistic widget payload is far below this.
_STRUCTURED_CONTENT_MAX_BYTES = 512 * 1024


def _publish_structured_content(result: Any) -> None:
    """Expose ``CallToolResult.structuredContent`` to the tool-call context.

    The ToolHandler contract renders results for the model (text, or content
    parts), which large-result eviction may later bound. UI surfaces
    (MCP Apps widgets) need the structured payload verbatim, so it travels
    out-of-band on the ToolCallContext when one is active.
    """
    structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, dict):
        return
    from roomkit.tools.context import _current_tool_call

    ctx = _current_tool_call.get()
    if ctx is None:
        return
    try:
        if len(json.dumps(structured)) > _STRUCTURED_CONTENT_MAX_BYTES:
            logger.warning(
                "structuredContent dropped: exceeds %d bytes", _STRUCTURED_CONTENT_MAX_BYTES
            )
            return
    except (TypeError, ValueError):
        return
    ctx.structured_content = structured


class MCPToolProvider:
    """Discover and invoke tools from an MCP server.

    Three transports: ``streamable_http`` (default) and ``sse`` for a server
    reached by URL, and ``stdio`` for a server RoomKit starts as a subprocess
    and talks to over its stdin/stdout — the way most MCP servers are run.

    Usage::

        async with MCPToolProvider.from_url("http://localhost:8000/mcp") as mcp:
            tools = mcp.get_tools()          # list[AITool]
            handler = mcp.as_tool_handler()   # ToolHandler for AIChannel

        async with MCPToolProvider.from_command("uvx", ["mcp-server-time"]) as mcp:
            ...

    The tools are listed once, on entry; no call lists them again (RFC §21.2).
    ``discover=False`` connects without listing, for a host that only reads
    the server's resources.

    Enter and exit it in the same task (``async with``): the MCP SDK's
    transports hold anyio cancel scopes that must close where they opened.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        transport: str = "streamable_http",
        tool_filter: Callable[[str], bool] | None = None,
        headers: dict[str, str] | None = None,
        command: str | None = None,
        args: Sequence[str] = (),
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        discover: bool = True,
    ) -> None:
        if transport not in ("streamable_http", "sse", "stdio"):
            raise ValueError(f"Unsupported transport: {transport!r}")
        if transport == "stdio":
            if not command:
                raise ValueError("the stdio transport needs a command to start the server")
            if url or headers:
                raise ValueError("url and headers are for the HTTP transports, not stdio")
        else:
            if not url:
                raise ValueError(f"the {transport} transport needs a url")
            if command or args or env is not None or cwd is not None:
                raise ValueError(f"command, args, env and cwd are for stdio, not {transport}")
        self._url = url
        self._transport = transport
        self._tool_filter = tool_filter
        self._headers = headers or {}
        self._command = command
        self._args = list(args)
        self._env = env
        self._cwd = cwd
        self._discover = discover
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._tools: list[AITool] = []
        self._tool_set: set[str] = set()
        self._tool_meta: dict[str, dict[str, Any]] = {}
        self._output_schemas: dict[str, dict[str, Any]] = {}
        self._connected = False

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        transport: str = "streamable_http",
        tool_filter: Callable[[str], bool] | None = None,
        headers: dict[str, str] | None = None,
        discover: bool = True,
    ) -> MCPToolProvider:
        """Create an MCPToolProvider for the given URL.

        The provider is not connected until used as an async context manager.

        Args:
            url: MCP server URL.
            transport: ``"streamable_http"`` (default) or ``"sse"``.
            tool_filter: Optional predicate to include only matching tool names.
            headers: Optional HTTP headers sent with every request.
            discover: List the server's tools on entry (the default). ``False``
                connects without listing, for a host that only reads
                resources: :meth:`get_tools` and :meth:`tool_meta` are then
                empty, and no call is checked against an output schema.

        Returns:
            An MCPToolProvider instance (not yet connected).
        """
        return cls(
            url, transport=transport, tool_filter=tool_filter, headers=headers, discover=discover
        )

    @classmethod
    def from_command(
        cls,
        command: str,
        args: Sequence[str] = (),
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        tool_filter: Callable[[str], bool] | None = None,
        discover: bool = True,
    ) -> MCPToolProvider:
        """Create an MCPToolProvider for a server started as a subprocess (stdio).

        Entering the provider starts ``command`` with ``args``; exiting stops it.

        Args:
            command: Executable of the MCP server (``"uvx"``, ``"npx"``, a path).
            args: Its arguments, passed as a list: no shell is involved.
            env: Extra environment variables for the server. The MCP SDK starts
                it with a minimal environment (``HOME``, ``PATH``, ``USER``…),
                not the whole of this process's: pass what the server needs,
                an API key say, here.
            cwd: Working directory of the server.
            tool_filter: Optional predicate to include only matching tool names.
            discover: List the server's tools on entry (the default); see
                :meth:`from_url`.

        Returns:
            An MCPToolProvider instance (not yet connected).
        """
        return cls(
            transport="stdio",
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            tool_filter=tool_filter,
            discover=discover,
        )

    @property
    def _target(self) -> str:
        """The server, as logs name it: the command alone, as its args may hold secrets."""
        if self._transport == "stdio":
            return self._command or ""
        return self._url or ""

    async def __aenter__(self) -> MCPToolProvider:
        try:
            from mcp import ClientSession
        except ImportError:
            raise ImportError(
                "MCPToolProvider requires the 'mcp' package. "
                "Install it with: pip install roomkit[mcp]"
            ) from None
        if self._stack is not None:
            raise RuntimeError("MCPToolProvider is already connected; exit it before re-entering")

        # Everything entered goes on one stack, so a failure half-way (a server
        # that exits, an initialize that errors) still closes what was opened:
        # for stdio, that is the server process.
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await self._open_transport(stack)
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            # Inside the try: a listing the catalogue cannot read releases what
            # was opened, the stdio server included (RFC §21.2).
            if self._discover:
                await self._discover_tools(session)
        except BaseException:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session
        self._connected = True
        logger.info(
            "Connected to MCP server %s (%s) — %s",
            self._target,
            self._transport,
            f"discovered {len(self._tools)} tools" if self._discover else "tools not listed",
        )
        return self

    async def _discover_tools(self, session: Any) -> None:
        """Read the server's whole tool listing into this provider.

        The catalogue keeps what the filter admits; the output schemas cover
        every listed tool, as a host calls tools the model never saw (an MCP
        App's app-only tool) and their results are checked all the same.
        """
        listed = await _list_tools(session)
        self._catalogue(listed)
        self._output_schemas = {
            tool.name: schema
            for tool in listed
            if isinstance(schema := getattr(tool, "outputSchema", None), dict)
        }

    def _catalogue(self, listed: Sequence[Any]) -> None:
        """Keep the listed tools the filter admits and a provider accepts:
        their definitions, their names and the ``_meta`` they were listed with."""
        tools: list[AITool] = []
        meta_by_name: dict[str, dict[str, Any]] = {}
        for tool in listed:
            if self._tool_filter and not self._tool_filter(tool.name):
                continue
            ai_tool = _definition(tool)
            if ai_tool is None:
                continue
            tools.append(ai_tool)
            if isinstance(meta := getattr(tool, "meta", None), dict):
                meta_by_name[tool.name] = meta
        self._tools = tools
        self._tool_set = {tool.name for tool in tools}
        self._tool_meta = meta_by_name

    async def _open_transport(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        """Open this provider's MCP transport on *stack*; return its two streams."""
        if self._transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self._command or "",
                args=self._args,
                env=self._env,
                cwd=self._cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write
        if self._transport == "sse":
            from mcp.client.sse import sse_client

            client = sse_client(self._url or "", headers=self._headers)
            read, write = await stack.enter_async_context(client)
            return read, write
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        # The SDK leaves a client it is handed open: the stack closes it.
        http = await stack.enter_async_context(create_mcp_http_client(headers=self._headers))
        client = streamable_http_client(self._url or "", http_client=http)
        read, write, _session_id = await stack.enter_async_context(client)
        return read, write

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._connected = False
        self._session = None
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def connected(self) -> bool:
        """Whether the provider holds a live connection: entered, not yet exited."""
        return self._connected

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "MCPToolProvider is not connected. Use 'async with' to connect first."
            )

    def get_tools(self) -> list[AITool]:
        """Return discovered tools as RoomKit AITool instances."""
        self._ensure_connected()
        return list(self._tools)

    def get_tools_as_dicts(self) -> list[dict[str, Any]]:
        """Return discovered tools as plain dicts (for binding metadata)."""
        self._ensure_connected()
        return [t.model_dump() for t in self._tools]

    @property
    def tool_names(self) -> list[str]:
        """Return the names of all discovered tools."""
        self._ensure_connected()
        return [t.name for t in self._tools]

    def tool_meta(self) -> dict[str, dict[str, Any]]:
        """The ``_meta`` each discovered tool was listed with, by tool name.

        Read from the listing made at connection, so no second ``tools/list``:
        an MCP App's ``ui`` (``resourceUri``, ``csp``) among the rest. A tool
        listed without one, or not discovered (``tool_filter``, a name no
        provider accepts, ``discover=False``), is absent.
        """
        self._ensure_connected()
        return copy.deepcopy(self._tool_meta)

    async def read_resource(self, uri: str, *, timeout: float = _DEFAULT_CALL_TIMEOUT) -> Any:
        """The server's ``ReadResourceResult`` for *uri* (an MCP App's HTML, say).

        Raises:
            McpError: The server refused the read (an unknown resource).
            TimeoutError: No answer within *timeout* seconds.
        """
        self._ensure_connected()
        return await asyncio.wait_for(self._session.read_resource(uri), timeout=timeout)

    async def call_tool_result(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = _DEFAULT_CALL_TIMEOUT,
    ) -> Any:
        """Call the tool and return the server's ``CallToolResult`` as it is.

        A failure is the result's ``isError``, not an exception: :meth:`call_tool`
        renders it into its error envelope, the tool handler raises it
        (:class:`~roomkit.core.exceptions.ToolFailedError`). For a
        host relaying the raw result (an MCP App's frame calling its server).
        A successful result's ``structuredContent`` is also published to the
        tool call in progress, when there is one (the model's own calls).

        Like :meth:`call_tool`, it calls any tool the server has: ``tool_filter``
        shapes what discovery offers a model, not what the host may call (an
        app-only tool the frame calls, say). The host authorizes the call.

        A successful result of a tool listed with an output schema is checked
        against it, and one that breaks it raises ``RuntimeError``, as the MCP
        SDK does. A tool the listing did not name is called as it is: no call
        lists the server's tools (RFC §21.2).
        """
        self._ensure_connected()
        from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult

        # Sent as a request: ``ClientSession.call_tool`` lists every tool to
        # check a name its listing lacks, on every call of that name.
        request = CallToolRequest(params=CallToolRequestParams(name=name, arguments=arguments))
        sent = self._session.send_request(request, CallToolResult)
        result = await asyncio.wait_for(sent, timeout=timeout)
        if not result.isError:
            if (schema := self._output_schemas.get(name)) is not None:
                check_structured_content(name, schema, result)
            _publish_structured_content(result)
        return result

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = _DEFAULT_CALL_TIMEOUT,
    ) -> str:
        """Call a tool on the MCP server and return the result as a string.

        Args:
            name: Tool name.
            arguments: Tool arguments dict.
            timeout: Maximum seconds to wait for a response.

        Returns:
            Result string. Single TextContent → plain text; multi-part → JSON array;
            error results → ``{"error": "..."}``.

        The error envelope is this method's contract and does not change. A tool
        loop reads :meth:`as_tool_handler` instead, which raises
        :class:`~roomkit.core.exceptions.ToolFailedError` so the outcome does
        not have to be recognised in the body.
        """
        result = await self.call_tool_result(name, arguments, timeout=timeout)
        if result.isError:
            return json.dumps({"error": error_text(result)})
        return text_body(result)

    def as_tool_handler(self, *, gate_discovery: bool = True) -> ToolHandler:
        """Return a ToolHandler suitable for ``AIChannel(tool_handler=...)``.

        A tool that is not from this MCP server raises
        :class:`~roomkit.core.exceptions.UnservedToolCallError`, which hands
        the call to the next handler of a ``compose_tool_handlers`` chain and
        reads as served by nothing on a channel (RFC §21.4).

        ``gate_discovery=False`` forwards every name to the server instead. A
        gateway that routes by name prefix and authenticates the caller per
        call serves tools this connection never listed — a server whose
        ``tools/list`` answers only behind the caller's own credential, say —
        and a host with its own allow-list in front has already decided what
        the model may call. Such a handler never raises
        ``UnservedToolCallError``, so it sits last in a
        ``compose_tool_handlers`` chain: nothing after it would be reached.
        A provider connected with ``discover=False`` discovered nothing, so
        its gated handler refuses every name.

        A tool whose result says ``isError`` raises
        :class:`~roomkit.core.exceptions.ToolFailedError` either way: the tool
        ran and failed, so the tool loop marks the call failed (not refused)
        and hands the server's message to the model unchanged.

        A result that carries an image (PNG, JPEG, GIF or WebP, with a payload
        that decodes) comes back as content parts
        (:class:`~roomkit.providers.ai.base.AITextPart` and
        :class:`~roomkit.providers.ai.base.AIImagePart`), so the model sees the
        image; any other result is the string :meth:`call_tool` returns. Binary
        content the model cannot take (another image format, audio, a blob
        resource) becomes a one-line note rather than its base64.
        """
        self._ensure_connected()

        async def _handler(name: str, arguments: dict[str, Any]) -> ToolResult:
            # An MCP alias (``mcp__<server>__<tool>``, from a prompt's naming)
            # runs the tool it names; the gate judged both names.
            lookup = served_tool_name(name)
            if gate_discovery and lookup not in self._tool_set:
                raise UnservedToolCallError(f"tool {name!r} is not served here")
            result = await self.call_tool_result(lookup, arguments, timeout=_DEFAULT_CALL_TIMEOUT)
            if result.isError:
                # The tool ran and failed (MCP ``isError``); say so instead of
                # returning a body the loop would have to recognise, and keep
                # the server's words: they are what the model is meant to read.
                raise ToolFailedError(error_text(result))
            return handler_result(result)

        return _handler

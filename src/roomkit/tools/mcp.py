"""MCPToolProvider — bridge MCP servers into RoomKit's AITool/ToolHandler system."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import warnings
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from roomkit._text import bounded_text, is_label
from roomkit.core.exceptions import ToolFailedError, UnservedToolCallError
from roomkit.providers.ai.base import AITool, some_vendor_accepts_tool_name
from roomkit.tools._mcp_capture import WireCapture
from roomkit.tools._mcp_result import (
    check_structured_content,
    error_text,
    handler_result,
    text_body,
)
from roomkit.tools._mcp_wire import (
    RAW_RESULT,
    compatible_handshake,
    host_params,
    host_result,
    resource_result,
    resource_uri,
    server_extensions,
    server_info,
)
from roomkit.tools.compose import ToolHandler, ToolResult
from roomkit.tools.context import turn_admits_alias
from roomkit.tools.policy import alias_server, served_tool_name

logger = logging.getLogger("roomkit.tools.mcp")


_NAME_LIMIT = 128
"""The longest tool name kept, in characters: the longest a vendor accepts."""

_DESCRIPTION_LIMIT = 4096
"""The longest tool description kept, in characters: a longer one is cut at a
word (RFC §21.2)."""

_SCHEMA_LIMIT = 32 * 1024
"""The largest parameters' schema kept, serialized, in bytes: a tool with a
larger one is left out (RFC §21.2)."""

_TOOL_LIMIT = 512
"""The most tools a server's listing keeps (RFC §21.2)."""


def _definition(tool: Any, target: str) -> AITool | None:
    """The definition of a listed MCP tool, as its server declared it and
    bounded (RFC §21.2), or ``None`` for one no model can be handed: a name no
    provider accepts or longer than the longest a vendor takes (RFC §6.7), or
    a parameters' schema over :data:`_SCHEMA_LIMIT` bytes. Skipped with a
    warning, so the server's other tools stay usable. A description over
    :data:`_DESCRIPTION_LIMIT` characters is cut at a word."""
    name = tool.name
    if len(name) > _NAME_LIMIT or not some_vendor_accepts_tool_name(name):
        logger.warning(
            "MCP tool %r skipped: no provider accepts its name (server %s)",
            name[:_NAME_LIMIT],
            target,
        )
        return None
    parameters = tool.input_schema if tool.input_schema else {}
    schema = json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
    if len(schema.encode()) > _SCHEMA_LIMIT:
        logger.warning(
            "MCP tool %r of %s skipped: its parameters' schema is over %d bytes",
            name,
            target,
            _SCHEMA_LIMIT,
        )
        return None
    description = tool.description or ""
    if len(description) > _DESCRIPTION_LIMIT:
        logger.warning(
            "MCP tool %r of %s: its description is cut to %d characters",
            name,
            target,
            _DESCRIPTION_LIMIT,
        )
    return AITool(
        name=name,
        description=bounded_text(description, _DESCRIPTION_LIMIT),
        parameters=parameters,
        tags=_published_tags(getattr(tool, "meta", None)),
    )


def _published_tags(meta: Any) -> list[str]:
    """The tags FastMCP wrote in a tool's ``_meta["fastmcp"]["tags"]``, so Tool
    Search can match the tool cross-lingually. Optional metadata of a server: what
    is no list of texts is left out, and costs it no tool."""
    fastmcp = meta.get("fastmcp") if isinstance(meta, dict) else None
    tags = fastmcp.get("tags") if isinstance(fastmcp, dict) else None
    return [tag for tag in tags if isinstance(tag, str)] if isinstance(tags, list) else []


_MAX_LIST_PAGES = 1000
"""Pages of ``tools/list`` read at most: a bound on a server whose cursor never
ends, far above any real catalogue."""


async def _list_tools(
    read_page: Callable[[Any], Awaitable[dict[str, Any]]], target: str
) -> list[Any]:
    """Every tool the server lists, following its cursor across pages (RFC §21.2).

    *read_page* reads one page, given its parameters, as the server sent it;
    the SDK's model of a listing checks it, no stricter than RoomKit's
    reading. The SDK's session would check a page against the negotiated
    revision's schema, where a tool whose ``inputSchema`` names no ``type``
    fails the whole listing.

    The reading stops where it stands, with a warning naming *target*, when the
    server pages on without bringing anything new (an empty page that still
    carries a cursor, a cursor handed out twice) or past
    :data:`_MAX_LIST_PAGES`: a server that pages forever must not hold the
    connection open forever.
    """
    from mcp.types import ListToolsResult, PaginatedRequestParams

    tools: list[Any] = []
    seen: set[str] = set()
    params: Any = None
    for _ in range(_MAX_LIST_PAGES):
        page = ListToolsResult.model_validate(await read_page(params), by_name=False)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not isinstance(cursor, str) or not cursor:
            return tools
        if not page.tools or cursor in seen:
            break
        seen.add(cursor)
        params = PaginatedRequestParams(cursor=cursor)
    logger.warning(
        "MCP server %s pages tools/list without end; keeping the %d tools read",
        target,
        len(tools),
    )
    return tools


_DEFAULT_CALL_TIMEOUT = 30.0
"""Seconds a call to the server waits: one default shared by
:meth:`MCPToolProvider.call_tool`, :meth:`~MCPToolProvider.call_tool_result`,
:meth:`~MCPToolProvider.read_resource` and the tool handler, so they cannot
drift apart."""

# The seconds connecting to a server may take by default: the handshake and
# the listing, and for a server RoomKit starts its start, a first start that
# fetches its package included (roomkit-rs's default too).
_CONNECT_TIMEOUT = 60.0

# Headers the HTTP transports write themselves: HTTP's framing and MCP's
# negotiation (its session, its protocol version), which a caller's value
# would contradict (roomkit-rs's McpHttp refuses the same).
_TRANSPORT_HEADERS = frozenset(
    {
        "accept",
        "content-type",
        "content-length",
        "host",
        "connection",
        "transfer-encoding",
        "last-event-id",
        "upgrade",
        "te",
        "trailer",
        "proxy-authorization",
    }
)
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_HEADER_VALUE = re.compile(r"[\t\x20-\x7e]*")
_MAX_HEADERS = 32
_MAX_HEADER_VALUE_BYTES = 8192


# Upper bound for publishing a structured result on the tool-call context
# (serialized size). Tool-call events ride the room event pipeline — DB rows,
# WebSocket broadcasts, audit — so a pathological multi-megabyte payload must
# not tag along; every realistic widget payload is far below this.
_STRUCTURED_CONTENT_MAX_BYTES = 512 * 1024


def _acknowledged(_result: dict[str, Any]) -> dict[str, Any]:
    """What the SDK reads in place of a host request's result."""
    return {}


def _retrieved(task: asyncio.Task[Any]) -> None:
    """Read an abandoned call's outcome, so its error is not reported unread."""
    if not task.cancelled():
        task.exception()


def _empty_listing(_result: dict[str, Any]) -> dict[str, Any]:
    """What the SDK reads in place of a ``tools/list`` page."""
    return {"tools": []}


def _empty_read(_result: dict[str, Any]) -> dict[str, Any]:
    """What the SDK reads in place of a ``resources/read`` result."""
    return {"contents": []}


_MESSAGE_LIMIT = 4 * 1024 * 1024
"""The largest event a Streamable HTTP response's SSE stream may carry, in
bytes: a larger one fails the request it answers (``MCPError``), the
connection going on. roomkit-rs's bound on any MCP message. The SDK's
default would be 1 MiB."""


def _publish_structured_content(result: Any) -> None:
    """Expose ``CallToolResult.structuredContent`` to the tool-call context.

    The ToolHandler contract renders results for the model (text, or content
    parts), which large-result eviction may later bound. UI surfaces
    (MCP Apps widgets) need the structured payload verbatim, so it travels
    out-of-band on the ToolCallContext when one is active.
    """
    structured = result.structured_content
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


def _check_label(label: str | None) -> None:
    """Refuse a *label* that cannot name its server in an alias
    ``mcp__<label>__<tool>``: one that is not an identifier (the rule of the
    host's other labels, ``is_label``), or holds ``__`` or ends with ``_``,
    which would make the alias read another server (``crm_`` gives
    ``mcp__crm___delete``, server ``crm``)."""
    if label is None:
        return
    if not is_label(label) or "__" in label or label.endswith("_"):
        raise ValueError(
            f"MCP server label {label!r} cannot name its server in an alias: use "
            "letters, digits and '_ . -', 64 at most, no '__' and no final '_'"
        )


def _check_endpoint(url: str) -> None:
    """Refuse an endpoint *url* the HTTP transports cannot reach as given:
    one that is not HTTP(S) or names no host, or carries credentials (sent
    on to every log and proxy the URL reaches: pass them as a header) or a
    fragment, which no request carries."""
    try:
        parts = urlsplit(url)
        _port = parts.port
    except ValueError:
        raise ValueError("invalid MCP endpoint URL") from None
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or "#" in url
    ):
        raise ValueError("use an HTTP(S) MCP endpoint without URL credentials or a fragment")


def _check_headers(headers: Mapping[str, str]) -> None:
    """Refuse *headers* the HTTP transports cannot send as given: an invalid
    name or value, a value over 8192 bytes, more than 32 headers, or one the
    transport writes itself (:data:`_TRANSPORT_HEADERS`, ``mcp-*``)."""
    names: set[str] = set()
    for name, value in headers.items():
        if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name):
            raise ValueError(f"invalid MCP header name {name!r}")
        lowered = name.lower()
        if lowered.startswith("mcp-") or lowered in _TRANSPORT_HEADERS:
            raise ValueError(f"MCP header {name!r} is reserved for the transport")
        if not isinstance(value, str):
            raise ValueError(f"invalid value for MCP header {name!r}")
        if len(value.encode()) > _MAX_HEADER_VALUE_BYTES:
            raise ValueError(f"MCP header {name!r} exceeds {_MAX_HEADER_VALUE_BYTES} bytes")
        if not _HEADER_VALUE.fullmatch(value):
            raise ValueError(f"invalid value for MCP header {name!r}")
        names.add(lowered)
    if len(names) > _MAX_HEADERS:
        raise ValueError(f"more than {_MAX_HEADERS} MCP headers")


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

    ``label`` is the host's name for the server, the ``<server>`` of an MCP
    alias ``mcp__<server>__<tool>`` (RFC §21.2): the handler of a labelled
    provider serves only the aliases that name it, so an alias runs on the
    server it names. Without a label the handler serves an alias whatever
    server it names; give one wherever a host composes several servers.

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
        label: str | None = None,
        connect_timeout: float | None = None,
    ) -> None:
        _check_label(label)
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
            _check_endpoint(url)
            _check_headers(headers or {})
        if transport == "sse":
            warnings.warn(
                "MCP's HTTP+SSE transport is deprecated by the MCP 2026-07-28 revision: "
                "use streamable_http",
                DeprecationWarning,
                # Attributed to the host's call, through from_url as well.
                skip_file_prefixes=(os.path.dirname(__file__),),
            )
        self._url = url
        self._transport = transport
        self._tool_filter = tool_filter
        self._headers = headers or {}
        self._command = command
        self._args = list(args)
        self._env = env
        self._cwd = cwd
        # Every connection is bounded (RFC §21.2): a server that accepts it
        # and never answers would hold its host for ever.
        self._connect_timeout = _CONNECT_TIMEOUT if connect_timeout is None else connect_timeout
        self._discover = discover
        self._label = label
        self._session: Any = None
        self._capture: WireCapture | None = None
        self._abandoned: set[asyncio.Task[Any]] = set()
        self._stack: AsyncExitStack | None = None
        self._tools: list[AITool] = []
        self._tool_set: set[str] = set()
        self._tool_meta: dict[str, dict[str, Any]] = {}
        self._output_schemas: dict[str, dict[str, Any]] = {}
        self._server_extensions: dict[str, dict[str, Any]] = {}
        self._instructions: str | None = None
        self._server_info: dict[str, Any] = {}
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
        label: str | None = None,
        connect_timeout: float = _CONNECT_TIMEOUT,
    ) -> MCPToolProvider:
        """Create an MCPToolProvider for the given URL.

        The provider is not connected until used as an async context manager.

        Args:
            url: MCP server URL: HTTP(S), naming a host, with no credentials
                (pass them as a header) and no fragment.
            transport: ``"streamable_http"`` (default) or ``"sse"``.
            tool_filter: Optional predicate to include only matching tool names.
            headers: Optional HTTP headers sent with every request: 32 at
                most, each value 8192 bytes at most, none the transport writes
                itself (``Accept``, ``Content-Type``, ``Host``, ``mcp-*``...).
            discover: List the server's tools on entry (the default). ``False``
                connects without listing, for a host that only reads
                resources: :meth:`get_tools` and :meth:`tool_meta` are then
                empty, and no call is checked against an output schema.
            label: The host's name for the server, the ``<server>`` of an
                MCP alias ``mcp__<server>__<tool>``: letters, digits and
                ``_ . -``, 64 at most, no ``__`` and no final ``_`` (either
                would make an alias read another server). The tool handler then serves
                only the aliases naming this server. ``None`` (the default)
                serves an alias whatever server it names.
            connect_timeout: The seconds entering may take: the handshake and
                the listing of the tools. Past it the connection is closed,
                its session deleted, and entering raises :class:`TimeoutError`.

        Returns:
            An MCPToolProvider instance (not yet connected).

        Raises:
            ValueError: *label* cannot name the server in an alias, or *url*
                or *headers* cannot be sent as given.
        """
        return cls(
            url,
            transport=transport,
            tool_filter=tool_filter,
            headers=headers,
            discover=discover,
            label=label,
            connect_timeout=connect_timeout,
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
        label: str | None = None,
        connect_timeout: float = _CONNECT_TIMEOUT,
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
            label: The host's name for the server; see :meth:`from_url`.
            connect_timeout: The seconds entering may take: the server's
                start, the handshake and the listing of its tools, a first
                start that fetches its package included. Past it the server
                is stopped and entering raises :class:`TimeoutError`.

        Returns:
            An MCPToolProvider instance (not yet connected).

        Raises:
            ValueError: *label* cannot name the server in an alias.
        """
        return cls(
            transport="stdio",
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            tool_filter=tool_filter,
            discover=discover,
            label=label,
            connect_timeout=connect_timeout,
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
        bound = asyncio.timeout(self._connect_timeout)
        try:
            async with bound:
                capture = WireCapture()
                read_stream, write_stream = capture.streams(*await self._open_transport(stack))
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                # The SDK runs its own handshake on a copy it can parse; the
                # provider keeps the server's result as it sent it.
                with capture.ticket(compatible_handshake) as handshake:
                    await session.initialize()
                init = handshake.taken()
                # Inside the try: a listing the catalogue cannot read releases
                # what was opened, the stdio server included (RFC §21.2).
                if self._discover:
                    await self._discover_tools(session, capture)
        except BaseException as exc:
            await stack.aclose()
            if isinstance(exc, TimeoutError) and bound.expired():
                raise TimeoutError(
                    f"MCP server {self._target} did not connect within {self._connect_timeout} s"
                ) from exc
            raise

        self._stack = stack
        self._session = session
        self._capture = capture
        self._keep_handshake(init)
        self._connected = True
        logger.info(
            "Connected to MCP server %s (%s) — %s",
            self._target,
            self._transport,
            f"discovered {len(self._tools)} tools" if self._discover else "tools not listed",
        )
        return self

    def _keep_handshake(self, init: dict[str, Any]) -> None:
        """Keep what the server's ``initialize`` result says of it, as it sent
        it (RFC §21.2)."""
        self._server_extensions = server_extensions(init)
        instructions = init.get("instructions")
        self._instructions = instructions if isinstance(instructions, str) else None
        self._server_info = server_info(init)

    async def _discover_tools(self, session: Any, capture: WireCapture) -> None:
        """Read the server's whole tool listing into this provider, each page
        off the wire (:func:`_list_tools`)."""

        async def read(params: Any) -> dict[str, Any]:
            with capture.ticket(_empty_listing) as ticket:
                await session.list_tools(params=params)
            return ticket.taken()

        async def read_page(params: Any) -> dict[str, Any]:
            # Bounded by the connection's own deadline, which it never outlasts.
            return await self._within(read(params), None)

        self._keep_listing(await _list_tools(read_page, self._target))

    def _keep_listing(self, listed: Sequence[Any]) -> None:
        """Keep a listing: the catalogue keeps what the filter admits; the
        output schemas cover every listed tool, as a host calls tools the
        model never saw (an MCP App's app-only tool) and their results are
        checked all the same."""
        self._catalogue(listed)
        self._output_schemas = {}
        for tool in listed:
            # A name listed twice keeps its first definition, schema included.
            if isinstance(schema := tool.output_schema, dict):
                self._output_schemas.setdefault(tool.name, schema)

    def _catalogue(self, listed: Sequence[Any]) -> None:
        """Keep the listed tools the filter admits and a provider accepts,
        :data:`_TOOL_LIMIT` at most, a name listed twice keeping its first
        definition: their definitions, their names and the ``_meta`` they were
        listed with."""
        tools: list[AITool] = []
        kept: set[str] = set()
        meta_by_name: dict[str, dict[str, Any]] = {}
        for tool in listed:
            if self._tool_filter and not self._tool_filter(tool.name):
                continue
            if len(tools) == _TOOL_LIMIT:
                logger.warning(
                    "MCP server %s lists more tools than the %d kept", self._target, _TOOL_LIMIT
                )
                break
            if tool.name in kept:
                # One tool per name (RFC §21.1): the first definition stands.
                logger.warning(
                    "MCP tool %r of %s skipped: its name is listed already",
                    tool.name[:_NAME_LIMIT],
                    self._target,
                )
                continue
            ai_tool = _definition(tool, self._target)
            if ai_tool is None:
                continue
            tools.append(ai_tool)
            kept.add(ai_tool.name)
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
        client = streamable_http_client(
            self._url or "", http_client=http, max_sse_event_size=_MESSAGE_LIMIT
        )
        read, write = await stack.enter_async_context(client)
        return read, write

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._connected = False
        self._session = None
        self._capture = None
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

    async def _within[T](self, work: Coroutine[Any, Any, T], timeout: float | None) -> T:
        """*work*'s result, the caller waiting *timeout* seconds at most
        (``None``: no bound of its own, the caller's own deadline is the
        bound), and never past it (RFC §21.2, §21.6).

        A call cut short is cancelled and left to end on its own: the SDK
        tells the server with ``notifications/cancelled`` and waits up to five
        seconds for a writer that may be stuck, which the caller does not
        wait for. The provider holds the call until it ends.
        """
        if timeout is not None and timeout <= 0:
            # No time at all: the call is never started, so nothing is sent.
            work.close()
            raise TimeoutError
        task = asyncio.ensure_future(work)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        except BaseException:
            self._abandon(task)
            raise
        if task not in done:
            self._abandon(task)
            raise TimeoutError
        return task.result()

    def _abandon(self, task: asyncio.Task[Any]) -> None:
        task.cancel()
        self._abandoned.add(task)
        task.add_done_callback(self._abandoned.discard)
        task.add_done_callback(_retrieved)

    def _wire(self) -> WireCapture:
        """The capture of the live connection's streams."""
        self._ensure_connected()
        if self._capture is None:
            raise RuntimeError("MCPToolProvider has no transport: connect it with 'async with'")
        return self._capture

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

    async def read_resource(
        self, uri: str, *, timeout: float = _DEFAULT_CALL_TIMEOUT
    ) -> dict[str, Any]:
        """The server's ``resources/read`` result for *uri* (an MCP App's HTML,
        say), as it sent it: its ``contents`` in order, each with its ``uri``
        and its ``text`` or ``blob``, extension fields and explicit nulls
        included (RFC §21.2). *uri*, and each content's, is normalized
        (``https://EXAMPLE.com/a b`` reads ``https://example.com/a%20b``).

        Raises:
            pydantic.ValidationError: *uri* is not an absolute URI, or the
                result is not a resource read's.
            MCPError: The server refused the read (an unknown resource).
            TimeoutError: No answer within *timeout* seconds.
        """
        capture = self._wire()
        sent = resource_uri(uri)

        async def read() -> dict[str, Any]:
            # The SDK reads an empty read in place of the server's result,
            # which RoomKit checks itself (roomkit-rs gives rmcp the same).
            with capture.ticket(_empty_read) as ticket:
                await self._session.read_resource(sent)
            return resource_result(ticket.taken())

        return await self._within(read(), timeout)

    @property
    def server_extensions(self) -> dict[str, dict[str, Any]]:
        """The extensions the server declared when it connected, by identifier,
        with their settings (``io.modelcontextprotocol/skills`` among them);
        empty for a server that declared none (RFC §21.2)."""
        self._ensure_connected()
        return copy.deepcopy(self._server_extensions)

    @property
    def instructions(self) -> str | None:
        """What the server published, when it connected, on how to use its
        tools (``InitializeResult.instructions``); ``None`` when it published
        nothing (RFC §21.2).

        The server's text, as it sent it: hand it to a channel's
        ``server_instructions``, which sets it apart and bounds it, rather than
        into a prompt directly.
        """
        self._ensure_connected()
        return self._instructions

    @property
    def server_info(self) -> dict[str, Any]:
        """What the server said of itself when it connected (``serverInfo``:
        ``name``, ``version``...), as it sent it (RFC §21.2).

        The server's own words, not a name for it: a host names a server by a
        label of its own (``server_instructions``, ``add_mcp_server``).
        """
        self._ensure_connected()
        return copy.deepcopy(self._server_info)

    async def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float = _DEFAULT_CALL_TIMEOUT,
    ) -> dict[str, Any]:
        """Send *method*, a request the SDK does not model (``skills/list``,
        say), with *params* as written, and return its result as the server
        sent it, but for its members sent as ``null`` (RFC §21.2).

        No parameters, or an empty set, go out as no ``params`` member; an
        empty or ``None`` ``_meta`` as none (RK-21.2-046).

        Raises:
            pydantic.ValidationError: The ``_meta`` of *params*, or of the
                result, is not an object.
            MCPError: The server answered with an error.
            TimeoutError: No answer within *timeout* seconds.
        """
        capture = self._wire()
        from mcp.types import Request

        # The SDK's own request type: its session reads the class for the
        # headers a method needs, and a method it does not model needs none.
        request = Request[dict[str, Any], str](method=method, params=host_params(params))

        async def send() -> dict[str, Any]:
            # Read off the wire: the SDK's ``Result`` keeps only the fields it
            # models, and checks a method it knows against its schema. It
            # reads an empty result instead, which a method it knows refuses
            # once the server's own result is already read.
            with capture.ticket(_acknowledged, (params or {}).get("_meta")) as ticket:
                try:
                    await self._session.send_request(request, RAW_RESULT)
                except ValidationError:
                    if ticket.result is None:
                        raise
            return host_result(ticket.taken())

        return await self._within(send(), timeout)

    async def call_tool_result(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = _DEFAULT_CALL_TIMEOUT,
    ) -> Any:
        """Call the tool and return its result, the SDK's ``CallToolResult``:
        what the SDK models of the server's result, without extension fields.

        A failure is the result's ``isError``, not an exception: :meth:`call_tool`
        renders it into its error envelope, the tool handler raises it
        (:class:`~roomkit.core.exceptions.ToolFailedError`). For a
        host relaying the result (an MCP App's frame calling its server).
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
        result = await self._within(sent, timeout)
        if not result.is_error:
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
            error results → ``{"error": "..."}``. A text resource reads as its
            text; an image, audio, a blob resource or a resource link reads as
            a one-line note, ``[image content (image/png) not shown to the
            model]``, never its base64.

        The error envelope is this method's contract and does not change. A tool
        loop reads :meth:`as_tool_handler` instead, which raises
        :class:`~roomkit.core.exceptions.ToolFailedError` so the outcome does
        not have to be recognised in the body.
        """
        result = await self.call_tool_result(name, arguments, timeout=timeout)
        if result.is_error:
            return json.dumps({"error": error_text(result)})
        return text_body(result)

    def _admitted_here(self, tool: str) -> bool:
        """Whether the turn's policy lets *tool*, called by its bare name, run
        on this labelled server: false when the policy names the server to keep
        it out through its alias ``mcp__<label>__<tool>``, so the next handler
        of the chain takes the call (RFC §21.1, §21.2)."""
        return turn_admits_alias(f"mcp__{self._label}__{tool}")

    def as_tool_handler(self, *, gate_discovery: bool = True) -> ToolHandler:
        """Return a ToolHandler suitable for ``AIChannel(tool_handler=...)``.

        A tool that is not from this MCP server raises
        :class:`~roomkit.core.exceptions.UnservedToolCallError`, which hands
        the call to the next handler of a ``compose_tool_handlers`` chain and
        reads as served by nothing on a channel (RFC §21.4).

        An MCP alias ``mcp__<server>__<tool>`` runs ``<tool>``. A provider
        given a ``label`` serves only the aliases naming that server: one
        naming another raises ``UnservedToolCallError`` either way, so the
        next handler of the chain takes it. A provider without a label serves
        an alias whatever server it names, so a policy that names a server
        (``deny=["mcp__b__*"]``) cannot hold a call to it. A labelled
        provider serves a bare name (``delete_records``) unless the turn's
        tool policy names its server to keep it out, through a deny pattern
        covering its alias (``mcp__<label>__delete_records``) or allow
        patterns naming aliases none of which covers it: it then raises
        ``UnservedToolCallError``, so a bare name runs on the first server of
        the chain the policy admits, and on none when it keeps each out.

        ``gate_discovery=False`` forwards every name to the server instead. A
        gateway that routes by name prefix and authenticates the caller per
        call serves tools this connection never listed — a server whose
        ``tools/list`` answers only behind the caller's own credential, say —
        and a host with its own allow-list in front has already decided what
        the model may call. Such a handler raises ``UnservedToolCallError``
        only for an alias naming another server than its label, so it sits
        last in a ``compose_tool_handlers`` chain: nothing after it would be
        reached for any other name.
        A provider connected with ``discover=False`` discovered nothing, so
        its gated handler serves no name: every call raises
        ``UnservedToolCallError``.

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
        resource) becomes a one-line note rather than its base64. Such a
        result keeps its texts within 100,000 characters in all, the text past
        the bound cut with a note when the note fits, and at most four of its
        images within 5 MiB of data URL in all, each image past either a note
        that counts as a text.
        """
        self._ensure_connected()

        async def _handler(name: str, arguments: dict[str, Any]) -> ToolResult:
            # An MCP alias (``mcp__<server>__<tool>``, from a prompt's naming)
            # runs the tool it names; the gate judged both names. A labelled
            # provider is that one server only.
            server = alias_server(name)
            if server is not None and self._label is not None and server != self._label:
                raise UnservedToolCallError(f"tool {name!r} names another server")
            lookup = served_tool_name(name)
            if gate_discovery and lookup not in self._tool_set:
                raise UnservedToolCallError(f"tool {name!r} is not served here")
            if server is None and self._label is not None and not self._admitted_here(lookup):
                raise UnservedToolCallError(f"tool {name!r} is refused on this server")
            result = await self.call_tool_result(lookup, arguments, timeout=_DEFAULT_CALL_TIMEOUT)
            if result.is_error:
                # The tool ran and failed (MCP ``isError``); say so instead of
                # returning a body the loop would have to recognise, and keep
                # the server's words: they are what the model is meant to read.
                raise ToolFailedError(error_text(result))
            return handler_result(result, lookup)

        return _handler

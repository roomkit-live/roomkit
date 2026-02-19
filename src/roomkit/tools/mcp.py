"""MCPToolProvider — bridge MCP servers into RoomKit's AITool/ToolHandler system."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any

from roomkit.providers.ai.base import AITool

logger = logging.getLogger("roomkit.tools.mcp")

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


class MCPToolProvider:
    """Discover and invoke tools from an MCP server.

    Supports both ``streamable_http`` (default) and ``sse`` transports.

    Usage::

        async with MCPToolProvider.from_url("http://localhost:8000/mcp") as mcp:
            tools = mcp.get_tools()          # list[AITool]
            handler = mcp.as_tool_handler()   # ToolHandler for AIChannel
    """

    def __init__(
        self,
        url: str,
        *,
        transport: str = "streamable_http",
        tool_filter: Callable[[str], bool] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._url = url
        self._transport = transport
        self._tool_filter = tool_filter
        self._headers = headers or {}
        self._session: Any = None
        self._context: Any = None  # async context manager from client
        self._tools: list[AITool] = []
        self._tool_set: set[str] = set()
        self._connected = False

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        transport: str = "streamable_http",
        tool_filter: Callable[[str], bool] | None = None,
        headers: dict[str, str] | None = None,
    ) -> MCPToolProvider:
        """Create an MCPToolProvider for the given URL.

        The provider is not connected until used as an async context manager.

        Args:
            url: MCP server URL.
            transport: ``"streamable_http"`` (default) or ``"sse"``.
            tool_filter: Optional predicate to include only matching tool names.
            headers: Optional HTTP headers sent with every request.

        Returns:
            An MCPToolProvider instance (not yet connected).
        """
        return cls(url, transport=transport, tool_filter=tool_filter, headers=headers)

    async def __aenter__(self) -> MCPToolProvider:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            raise ImportError(
                "MCPToolProvider requires the 'mcp' package. "
                "Install it with: pip install roomkit[mcp]"
            ) from None

        if self._transport == "sse":
            from mcp.client.sse import sse_client

            client_cm = sse_client(self._url, headers=self._headers)
        elif self._transport == "streamable_http":
            client_cm = streamablehttp_client(self._url, headers=self._headers)
        else:
            raise ValueError(f"Unsupported transport: {self._transport!r}")

        # Enter the transport context manager to get read/write streams
        self._context = client_cm
        streams = await self._context.__aenter__()

        # streamable_http returns (read, write, session_id); sse returns (read, write)
        if len(streams) == 3:
            read_stream, write_stream, _ = streams
        else:
            read_stream, write_stream = streams

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

        # Discover tools
        result = await self._session.list_tools()
        for tool in result.tools:
            if self._tool_filter and not self._tool_filter(tool.name):
                continue
            ai_tool = AITool(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.inputSchema if tool.inputSchema else {},
            )
            self._tools.append(ai_tool)
            self._tool_set.add(tool.name)

        self._connected = True
        logger.info(
            "Connected to MCP server at %s — discovered %d tools",
            self._url,
            len(self._tools),
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._connected = False
        if self._session is not None:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
            self._session = None
        if self._context is not None:
            await self._context.__aexit__(exc_type, exc_val, exc_tb)
            self._context = None

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

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> str:
        """Call a tool on the MCP server and return the result as a string.

        Args:
            name: Tool name.
            arguments: Tool arguments dict.
            timeout: Maximum seconds to wait for a response.

        Returns:
            Result string. Single TextContent → plain text; multi-part → JSON array;
            error results → ``{"error": "..."}``.
        """
        import asyncio

        self._ensure_connected()
        result = await asyncio.wait_for(self._session.call_tool(name, arguments), timeout=timeout)

        if result.isError:
            parts = [getattr(c, "text", str(c)) for c in result.content]
            return json.dumps({"error": " ".join(parts)})

        # Extract text from content parts
        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
            else:
                texts.append(str(content))

        if len(texts) == 1:
            return str(texts[0])
        return json.dumps(texts)

    def as_tool_handler(self) -> ToolHandler:
        """Return a ToolHandler suitable for ``AIChannel(tool_handler=...)``.

        Unknown tools (not from this MCP server) return
        ``{"error": "Unknown tool: <name>"}``, which allows composition
        via ``compose_tool_handlers``.
        """
        self._ensure_connected()

        async def _handler(name: str, arguments: dict[str, Any]) -> str:
            if name not in self._tool_set:
                return json.dumps({"error": f"Unknown tool: {name}"})
            return await self.call_tool(name, arguments)

        return _handler

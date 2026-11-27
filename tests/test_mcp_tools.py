"""Tests for MCPToolProvider (tools/mcp.py)."""

from __future__ import annotations

import pytest


class TestMCPToolProvider:
    def test_constructor(self) -> None:
        from roomkit.tools.mcp import MCPToolProvider

        provider = MCPToolProvider("http://localhost:8000/mcp")
        assert provider._url == "http://localhost:8000/mcp"
        assert provider._transport == "streamable_http"
        assert provider._connected is False

    def test_from_url(self) -> None:
        from roomkit.tools.mcp import MCPToolProvider

        provider = MCPToolProvider.from_url(
            "http://localhost:8000/mcp",
            transport="sse",
            tool_filter=lambda n: n.startswith("test"),
        )
        assert provider._url == "http://localhost:8000/mcp"
        assert provider._transport == "sse"

    def test_get_tools_not_connected_raises(self) -> None:
        from roomkit.tools.mcp import MCPToolProvider

        provider = MCPToolProvider("http://localhost:8000/mcp")
        with pytest.raises(RuntimeError, match="not connected"):
            provider.get_tools()

"""Tests for MCPToolProvider (tools/mcp.py)."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AITool


def _build_mock_mcp() -> MagicMock:
    """Build a mock mcp module with required submodules."""
    mock_tool = SimpleNamespace(
        name="test_tool",
        description="A test tool",
        inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
    )

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[mock_tool]))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_mcp = MagicMock()
    mock_mcp.ClientSession = MagicMock(return_value=mock_session)

    # streamable_http client
    mock_streams = (AsyncMock(), AsyncMock(), "session-id")
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_streams)
    mock_client_cm.__aexit__ = AsyncMock(return_value=None)

    mock_streamable = MagicMock()
    mock_streamable.streamablehttp_client = MagicMock(return_value=mock_client_cm)

    mock_sse = MagicMock()

    return mock_mcp, mock_streamable, mock_sse


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

    async def test_context_manager_discovers_tools(self) -> None:
        mock_mcp, mock_streamable, mock_sse = _build_mock_mcp()
        with patch.dict(
            sys.modules,
            {
                "mcp": mock_mcp,
                "mcp.client": MagicMock(),
                "mcp.client.streamable_http": mock_streamable,
                "mcp.client.sse": mock_sse,
            },
        ):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.tools.mcp")
            importlib.reload(mod)

            provider = mod.MCPToolProvider("http://localhost:8000/mcp")
            async with provider as p:
                tools = p.get_tools()
                assert len(tools) == 1
                assert isinstance(tools[0], AITool)
                assert tools[0].name == "test_tool"

    async def test_tool_filter(self) -> None:
        mock_mcp, mock_streamable, mock_sse = _build_mock_mcp()
        with patch.dict(
            sys.modules,
            {
                "mcp": mock_mcp,
                "mcp.client": MagicMock(),
                "mcp.client.streamable_http": mock_streamable,
                "mcp.client.sse": mock_sse,
            },
        ):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.tools.mcp")
            importlib.reload(mod)

            # Filter excludes all tools
            provider = mod.MCPToolProvider(
                "http://localhost:8000/mcp",
                tool_filter=lambda n: n.startswith("xxx"),
            )
            async with provider as p:
                tools = p.get_tools()
                assert len(tools) == 0

"""Tests for MCPToolProvider using mock MCP session."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.providers.ai.base import AITool
from roomkit.tools.mcp import MCPToolProvider

# ---------------------------------------------------------------------------
# Mock MCP types
# ---------------------------------------------------------------------------


class MockTool:
    """Mimics mcp.types.Tool."""

    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class MockListToolsResult:
    """Mimics the result from session.list_tools()."""

    def __init__(self, tools: list[MockTool]) -> None:
        self.tools = tools


class MockTextContent:
    """Mimics mcp.types.TextContent."""

    def __init__(self, text: str) -> None:
        self.text = text


class MockCallToolResult:
    """Mimics the result from session.call_tool()."""

    def __init__(
        self,
        content: list[MockTextContent],
        is_error: bool = False,
    ) -> None:
        self.content = content
        self.isError = is_error


def _make_provider_connected(
    tools: list[MockTool],
    call_tool_side_effect: Any = None,
) -> MCPToolProvider:
    """Create a provider and wire up a mock session directly."""
    provider = MCPToolProvider("http://fake:8000/mcp")

    session = AsyncMock()
    session.list_tools = AsyncMock(return_value=MockListToolsResult(tools))
    if call_tool_side_effect:
        session.call_tool = AsyncMock(side_effect=call_tool_side_effect)
    else:
        session.call_tool = AsyncMock(return_value=MockCallToolResult([MockTextContent("ok")]))

    provider._session = session
    provider._connected = True

    # Populate tools as __aenter__ would
    for tool in tools:
        ai_tool = AITool(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema if tool.inputSchema else {},
        )
        provider._tools.append(ai_tool)
        provider._tool_set.add(tool.name)

    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SEARCH_TOOL = MockTool(
    "search",
    "Search the web",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

CALC_TOOL = MockTool(
    "calculate",
    "Do math",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
        },
        "required": ["expression"],
    },
)


async def test_get_tools_mapping() -> None:
    provider = _make_provider_connected([SEARCH_TOOL, CALC_TOOL])
    tools = provider.get_tools()
    assert len(tools) == 2
    assert all(isinstance(t, AITool) for t in tools)
    assert tools[0].name == "search"
    assert tools[0].description == "Search the web"
    assert tools[0].parameters["required"] == ["query"]
    assert tools[1].name == "calculate"


async def test_get_tools_as_dicts() -> None:
    provider = _make_provider_connected([SEARCH_TOOL])
    dicts = provider.get_tools_as_dicts()
    assert len(dicts) == 1
    assert dicts[0]["name"] == "search"
    assert isinstance(dicts[0]["parameters"], dict)


async def test_tool_names() -> None:
    provider = _make_provider_connected([SEARCH_TOOL, CALC_TOOL])
    assert provider.tool_names == ["search", "calculate"]


async def test_tool_filter() -> None:
    provider = MCPToolProvider(
        "http://fake:8000/mcp",
        tool_filter=lambda name: name.startswith("search"),
    )
    provider._session = AsyncMock()
    provider._connected = True
    # Manually apply filter like __aenter__ does
    for tool in [SEARCH_TOOL, CALC_TOOL]:
        if provider._tool_filter and not provider._tool_filter(tool.name):
            continue
        provider._tools.append(
            AITool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            )
        )
        provider._tool_set.add(tool.name)

    assert provider.tool_names == ["search"]


async def test_call_tool_single_text() -> None:
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: MockCallToolResult(
            [MockTextContent("result text")]
        ),
    )
    result = await provider.call_tool("search", {"query": "hello"})
    assert result == "result text"


async def test_call_tool_error() -> None:
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: MockCallToolResult(
            [MockTextContent("something failed")], is_error=True
        ),
    )
    result = await provider.call_tool("search", {"query": "hello"})
    parsed = json.loads(result)
    assert parsed == {"error": "something failed"}


async def test_call_tool_multi_part() -> None:
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: MockCallToolResult(
            [MockTextContent("part1"), MockTextContent("part2")]
        ),
    )
    result = await provider.call_tool("search", {"query": "hello"})
    parsed = json.loads(result)
    assert parsed == ["part1", "part2"]


async def test_as_tool_handler() -> None:
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: MockCallToolResult([MockTextContent("found it")]),
    )
    handler = provider.as_tool_handler()
    result = await handler("search", {"query": "test"})
    assert result == "found it"


async def test_as_tool_handler_unknown_tool() -> None:
    provider = _make_provider_connected([SEARCH_TOOL])
    handler = provider.as_tool_handler()
    result = await handler("nonexistent", {})
    parsed = json.loads(result)
    assert parsed == {"error": "Unknown tool: nonexistent"}


async def test_not_connected_guard() -> None:
    provider = MCPToolProvider("http://fake:8000/mcp")
    with pytest.raises(RuntimeError, match="not connected"):
        provider.get_tools()

    with pytest.raises(RuntimeError, match="not connected"):
        await provider.call_tool("x", {})

    with pytest.raises(RuntimeError, match="not connected"):
        provider.as_tool_handler()

    with pytest.raises(RuntimeError, match="not connected"):
        provider.tool_names  # noqa: B018

    with pytest.raises(RuntimeError, match="not connected"):
        provider.get_tools_as_dicts()

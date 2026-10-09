"""Tests for MCPToolProvider using mock MCP session."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.core.exceptions import ToolFailedError, UnservedToolCallError
from roomkit.providers.ai.base import AIImagePart, AITextPart, AITool
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


class MockImageContent:
    """Mimics mcp.types.ImageContent: base64 data, no ``text``."""

    type = "image"

    def __init__(self, data: str, mime_type: str) -> None:
        self.data = data
        self.mimeType = mime_type

    def __str__(self) -> str:
        return f"type='image' data='{self.data}' mimeType='{self.mimeType}'"


class MockAudioContent:
    """Mimics mcp.types.AudioContent."""

    type = "audio"

    def __init__(self, data: str, mime_type: str) -> None:
        self.data = data
        self.mimeType = mime_type

    def __str__(self) -> str:
        return f"type='audio' data='{self.data}' mimeType='{self.mimeType}'"


class MockBlobResource:
    """Mimics mcp.types.EmbeddedResource around BlobResourceContents."""

    type = "resource"

    def __init__(self, blob: str, mime_type: str) -> None:
        self.resource = type("Blob", (), {"blob": blob, "mimeType": mime_type})()

    def __str__(self) -> str:
        return f"type='resource' resource=blob='{self.resource.blob}'"


class MockCallToolResult:
    """Mimics the ``CallToolResult`` a ``tools/call`` request returns."""

    def __init__(
        self,
        content: list[Any],
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
        # The provider sends ``tools/call`` itself (RMK-662): the side effect
        # still reads the tool's name and arguments.
        session.send_request = AsyncMock(
            side_effect=lambda request, _result_type: call_tool_side_effect(
                request.params.name, request.params.arguments
            )
        )
    else:
        session.send_request = AsyncMock(return_value=MockCallToolResult([MockTextContent("ok")]))

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


async def test_tool_handler_raises_on_a_failed_call() -> None:
    """The handler a tool loop reads states the failure (MCP ``isError``: the
    tool ran and failed, RMK-459); ``call_tool`` renders it.

    The loop cannot recognise a failure in a body, so the outcome has to reach
    it some other way. ``call_tool`` keeps the envelope its own callers have
    always been given, and both read the same server verdict.
    """
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: MockCallToolResult(
            [MockTextContent("Missing X-Tenant-ID header")], is_error=True
        ),
    )
    handler = provider.as_tool_handler()

    with pytest.raises(ToolFailedError) as raised:
        await handler("search", {"query": "hello"})
    # The server's words reach the model unchanged — no envelope around them.
    assert raised.value.message == "Missing X-Tenant-ID header"


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


async def test_tool_handler_returns_an_image_as_an_image_part() -> None:
    """The model sees the image, not its base64 spelled out as text."""
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: MockCallToolResult(
            [MockTextContent("the page"), MockImageContent("iVBORw0KGgo", "image/jpeg")]
        ),
    )
    handler = provider.as_tool_handler()

    result = await handler("search", {"query": "hello"})

    assert result == [
        AITextPart(text="the page"),
        AIImagePart(url="data:image/jpeg;base64,iVBORw0KGgo", mime_type="image/jpeg"),
    ]


async def test_call_tool_keeps_its_string_contract_for_an_image() -> None:
    """``call_tool`` returns a string whatever the server sent."""
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: MockCallToolResult(
            [MockImageContent("iVBORw0KGgo", "image/png")]
        ),
    )

    result = await provider.call_tool("search", {"query": "hello"})

    assert result == "type='image' data='iVBORw0KGgo' mimeType='image/png'"


@pytest.mark.parametrize(
    ("content", "note"),
    [
        (MockImageContent("not base64!", "image/png"), "[image content (image/png)"),
        (MockImageContent("PHN2Zz4=", "image/svg+xml"), "[image content (image/svg+xml)"),
        (MockAudioContent("UklGRg==", "audio/wav"), "[audio content (audio/wav)"),
        (MockBlobResource("JVBERi0=", "application/pdf"), "[resource content (application/pdf)"),
    ],
    ids=["corrupt-image", "svg", "audio", "blob"],
)
async def test_tool_handler_notes_what_the_model_cannot_take(content: Any, note: str) -> None:
    """A bad image would fail the whole request at the vendor, and audio or a
    blob spelled out is base64 noise: each becomes a one-line note."""
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: MockCallToolResult(
            [MockTextContent("the page"), content]
        ),
    )
    handler = provider.as_tool_handler()

    result = await handler("search", {"query": "hello"})

    assert isinstance(result, str)
    texts = json.loads(result)
    assert texts[0] == "the page"
    assert texts[1].startswith(note)
    assert texts[1].endswith("not shown to the model]")


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
    with pytest.raises(UnservedToolCallError):
        await handler("nonexistent", {})


async def test_as_tool_handler_ungated_forwards_an_undiscovered_name() -> None:
    """``gate_discovery=False`` hands every name to the server.

    A gateway that routes by prefix serves tools this connection never
    listed; the host that mounts it has its own allow-list in front. The
    prefix normalisation still applies.
    """
    calls: list[tuple[str, dict[str, Any]]] = []

    def record(name: str, args: dict[str, Any]) -> MockCallToolResult:
        calls.append((name, args))
        return MockCallToolResult([MockTextContent("found it")])

    provider = _make_provider_connected([SEARCH_TOOL], call_tool_side_effect=record)
    handler = provider.as_tool_handler(gate_discovery=False)

    assert await handler("nonexistent", {"q": 1}) == "found it"
    assert await handler("mcp__server__search", {"query": "x"}) == "found it"
    assert calls == [("nonexistent", {"q": 1}), ("search", {"query": "x"})]


async def test_as_tool_handler_ungated_still_raises_on_a_failed_call() -> None:
    """Skipping the gate skips only the gate: the server's failure is still raised."""
    provider = _make_provider_connected(
        [SEARCH_TOOL],
        call_tool_side_effect=lambda name, args: MockCallToolResult(
            [MockTextContent("Missing X-Tenant-ID header")], is_error=True
        ),
    )
    handler = provider.as_tool_handler(gate_discovery=False)

    with pytest.raises(ToolFailedError) as raised:
        await handler("nonexistent", {})
    assert raised.value.message == "Missing X-Tenant-ID header"


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

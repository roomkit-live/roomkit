"""MCP structuredContent must survive past result flattening and eviction.

The ToolHandler contract flattens tool results to the LLM-facing string, and
large-result eviction may later replace that string with a placeholder. MCP
Apps widgets render from ``CallToolResult.structuredContent`` verbatim, so the
provider publishes it out-of-band on the active ``ToolCallContext`` and the
tool-call events carry it untouched. These tests pin that side channel.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.models.event import ToolCallContent
from roomkit.models.streaming import ToolCallEndMarker
from roomkit.providers.ai.base import AIToolResultPart
from roomkit.tools.human_input import ToolCallContext, _current_tool_call
from roomkit.tools.mcp import MCPToolProvider


class _TextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _CallToolResult:
    def __init__(
        self,
        content: list[_TextContent],
        is_error: bool = False,
        structured_content: Any = None,
    ) -> None:
        self.content = content
        self.isError = is_error
        self.structuredContent = structured_content


def _provider(result: _CallToolResult) -> MCPToolProvider:
    provider = MCPToolProvider("http://fake:8000/mcp")
    session = AsyncMock()
    session.call_tool = AsyncMock(return_value=result)
    provider._session = session
    provider._connected = True
    provider._tool_set.add("get-menu")
    return provider


async def _call_with_ctx(provider: MCPToolProvider) -> tuple[str, ToolCallContext]:
    ctx = ToolCallContext(room_id="r1", tool_call_id="call_1", channel_id="ai")
    token = _current_tool_call.set(ctx)
    try:
        text = await provider.call_tool("get-menu", {})
    finally:
        _current_tool_call.reset(token)
    return text, ctx


@pytest.mark.asyncio
async def test_structured_content_published_to_tool_call_context() -> None:
    menu = {"menu": {"items": [{"id": "1", "name": "Espresso"}]}}
    provider = _provider(_CallToolResult([_TextContent('{"menu": ...}')], structured_content=menu))

    text, ctx = await _call_with_ctx(provider)

    assert text == '{"menu": ...}'
    assert ctx.structured_content == menu


@pytest.mark.asyncio
async def test_no_context_and_no_structured_content_are_harmless() -> None:
    provider = _provider(_CallToolResult([_TextContent("ok")], structured_content=None))
    text, ctx = await _call_with_ctx(provider)
    assert text == "ok"
    assert ctx.structured_content is None

    # Without an active context the call still works (plain handler usage).
    provider2 = _provider(_CallToolResult([_TextContent("ok")], structured_content={"a": 1}))
    assert await provider2.call_tool("get-menu", {}) == "ok"


@pytest.mark.asyncio
async def test_error_results_never_publish_structured_content() -> None:
    provider = _provider(
        _CallToolResult([_TextContent("boom")], is_error=True, structured_content={"a": 1})
    )
    text, ctx = await _call_with_ctx(provider)
    assert "boom" in text
    assert ctx.structured_content is None


@pytest.mark.asyncio
async def test_oversized_structured_content_is_dropped() -> None:
    huge = {"blob": "x" * (600 * 1024)}
    provider = _provider(_CallToolResult([_TextContent("ok")], structured_content=huge))
    _, ctx = await _call_with_ctx(provider)
    assert ctx.structured_content is None


@pytest.mark.asyncio
async def test_non_dict_structured_content_is_ignored() -> None:
    provider = _provider(_CallToolResult([_TextContent("ok")], structured_content=[1, 2, 3]))
    _, ctx = await _call_with_ctx(provider)
    assert ctx.structured_content is None


def test_event_models_carry_structured_content() -> None:
    # The carriers between the tool run and the persisted room event: each
    # must accept the field so eviction of `result` cannot lose the payload.
    part = AIToolResultPart(
        tool_call_id="call_1",
        name="get-menu",
        result="Result too large (…). Use read_stored_result…",
        structured_content={"menu": {"items": []}},
    )
    marker = ToolCallEndMarker(
        tool_name="get-menu",
        tool_id="call_1",
        result=part.result,
        structured_content=part.structured_content,
    )
    content = ToolCallContent(
        tool_name=marker.tool_name,
        tool_id=marker.tool_id,
        result=marker.result,
        status="completed",
        structured_content=marker.structured_content,
    )
    assert content.structured_content == {"menu": {"items": []}}
    # Round-trips through model serialization (the persistence boundary).
    assert ToolCallContent.model_validate(content.model_dump()).structured_content == {
        "menu": {"items": []}
    }

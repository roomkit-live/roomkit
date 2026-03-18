"""Tests for the Tool protocol and extract_tools utility."""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import AITool
from roomkit.tools.base import Tool
from roomkit.tools.compose import extract_tools


class FakeTool:
    """A minimal Tool implementation for testing."""

    def __init__(self, name: str, result: str) -> None:
        self._name = name
        self._result = result

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "description": f"Fake {self._name}",
            "parameters": {"type": "object", "properties": {}},
        }

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        if name != self._name:
            return f'{{"error": "Unknown tool: {name}"}}'
        return self._result


class TestToolProtocol:
    def test_fake_tool_satisfies_protocol(self) -> None:
        tool = FakeTool("test", "ok")
        assert isinstance(tool, Tool)

    def test_dict_does_not_satisfy_protocol(self) -> None:
        assert not isinstance({"name": "test"}, Tool)

    def test_aitool_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(AITool(name="test", description="d"), Tool)


class TestExtractTools:
    def test_extract_tool_objects(self) -> None:
        t1 = FakeTool("alpha", "a")
        t2 = FakeTool("beta", "b")
        defs, handler = extract_tools([t1, t2])

        assert len(defs) == 2
        assert defs[0].name == "alpha"
        assert defs[1].name == "beta"
        assert handler is not None

    async def test_composed_handler_dispatches(self) -> None:
        t1 = FakeTool("alpha", "result-a")
        t2 = FakeTool("beta", "result-b")
        _, handler = extract_tools([t1, t2])

        assert handler is not None
        assert await handler("alpha", {}) == "result-a"
        assert await handler("beta", {}) == "result-b"

    def test_extract_aitool_objects(self) -> None:
        ai = AITool(name="foo", description="a foo tool")
        defs, handler = extract_tools([ai])

        assert len(defs) == 1
        assert defs[0].name == "foo"
        assert handler is None

    def test_extract_dicts(self) -> None:
        d = {"name": "bar", "description": "a bar", "parameters": {}}
        defs, handler = extract_tools([d])

        assert len(defs) == 1
        assert defs[0].name == "bar"
        assert handler is None

    def test_mixed_tools_and_aitool(self) -> None:
        t = FakeTool("webcam", "frame")
        ai = AITool(name="search", description="web search")
        defs, handler = extract_tools([t, ai])

        assert len(defs) == 2
        assert defs[0].name == "webcam"
        assert defs[1].name == "search"
        assert handler is not None

    async def test_single_tool_handler_not_composed(self) -> None:
        t = FakeTool("only", "result")
        _, handler = extract_tools([t])

        assert handler is not None
        assert await handler("only", {}) == "result"

    def test_empty_list(self) -> None:
        defs, handler = extract_tools([])
        assert defs == []
        assert handler is None

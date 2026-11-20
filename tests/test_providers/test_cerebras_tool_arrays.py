"""Cerebras's wire-format arrays are repaired before ordinary tool validation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import httpx2
import pytest

from roomkit.providers.ai.base import (
    AIContext,
    AITool,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.ai.tool_declaration import declared_parameters
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.tools.validation import validate_tool_arguments
from tests.test_providers.test_cerebras import (
    _context,
    _provider,
    _response,
    _stream_response,
)

_BLOCKS = [{"type": "checklist", "items": [{"label": "Prepare"}, {"label": "Cook"}]}]
_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
        "content": {"type": "string"},
        "flexible": {"type": ["array", "string"]},
    },
}


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("omit_object_types", [False, True])
@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        (_BLOCKS, _BLOCKS),
        (json.dumps(_BLOCKS), _BLOCKS),
        ([{"type": "checklist", "items": json.dumps(_BLOCKS[0]["items"])}], _BLOCKS),
        ("broken [", "broken ["),
        ('{"items": []}', '{"items": []}'),
        ("null", "null"),
        ("42", "42"),
        (json.dumps(json.dumps(_BLOCKS)), json.dumps(json.dumps(_BLOCKS))),
    ],
)
async def test_only_schema_declared_arrays_are_decoded(
    streaming: bool, omit_object_types: bool, blocks: Any, expected: Any
) -> None:
    args = {"blocks": blocks, "content": "[]", "flexible": "[]", "unknown": "[]"}
    tool = AITool(name="display", description="Show a checklist", parameters=deepcopy(_SCHEMA))
    if omit_object_types:
        del tool.parameters["type"]
        del tool.parameters["properties"]["blocks"]["items"]["type"]
    context = _context(tools=[tool])
    original = context.model_dump()

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        # Declared as given, a root without a type typed an object (RMK-398).
        assert body["tools"][0]["function"]["parameters"] == declared_parameters(tool.parameters)
        call = {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "display",
                "arguments": json.dumps(args),
            },
        }
        if streaming:
            return _stream_response({"tool_calls": [{"index": 0, **call}]}, finish="tool_calls")
        return _response(tool_calls=[call])

    async with _provider(handle, model="qwen-3.8-27b") as provider:
        if streaming:
            events = [event async for event in provider.generate_structured_stream(context)]
            call = next(event for event in events if isinstance(event, StreamToolCall))
        else:
            call = (await provider.generate(context)).tool_calls[0]
    assert call.arguments == {**args, "blocks": expected}
    assert context.model_dump() == original
    error = validate_tool_arguments(_SCHEMA, call.arguments)
    assert (error is None) == isinstance(expected, list)


@pytest.mark.parametrize("streaming", [False, True])
async def test_undeclared_tools_are_not_repaired(streaming: bool) -> None:
    args = {"blocks": json.dumps(_BLOCKS)}

    def handle(request: httpx2.Request) -> httpx2.Response:
        call = {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "undeclared",
                "arguments": json.dumps(args),
            },
        }
        if streaming:
            return _stream_response({"tool_calls": [{"index": 0, **call}]}, finish="tool_calls")
        return _response(tool_calls=[call])

    async with _provider(handle) as provider:
        context = _context(tools=[AITool(name="display", description="", parameters=_SCHEMA)])
        if streaming:
            calls = [
                e
                async for e in provider.generate_structured_stream(context)
                if isinstance(e, StreamToolCall)
            ]
        else:
            calls = (await provider.generate(context)).tool_calls
    assert calls[0].arguments == args


async def test_closing_the_adapter_joins_the_transport_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    async def transport(self: OpenAIAIProvider, context: AIContext) -> AsyncIterator[StreamEvent]:
        nonlocal closed
        try:
            yield StreamTextDelta(text="partial")
        finally:
            closed = True

    monkeypatch.setattr(OpenAIAIProvider, "generate_structured_stream", transport)
    async with _provider(lambda request: _response()) as provider:
        stream = provider.generate_structured_stream(_context())
        await anext(stream)
        await stream.aclose()
        assert closed

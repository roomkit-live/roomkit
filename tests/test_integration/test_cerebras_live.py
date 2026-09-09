"""Opt-in Cerebras smoke tests with synthetic prompts and bounded output.

Run with CEREBRAS_API_KEY and ROOMKIT_RUN_CEREBRAS_LIVE=1 in the environment:
    uv run --extra cerebras pytest tests/test_integration/test_cerebras_live.py -q

These tests make paid API requests. No credentials or response text are logged.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from roomkit import CerebrasAIProvider, CerebrasConfig
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ROOMKIT_RUN_CEREBRAS_LIVE") != "1" or not os.environ.get("CEREBRAS_API_KEY"),
    reason="Set ROOMKIT_RUN_CEREBRAS_LIVE=1 and CEREBRAS_API_KEY to run live tests",
)


@pytest.fixture(params=["gpt-oss-120b", "qwen-3.8-27b"])
async def provider(request: pytest.FixtureRequest) -> AsyncIterator[CerebrasAIProvider]:
    pytest.importorskip("openai")
    instance = CerebrasAIProvider(
        CerebrasConfig(
            api_key=os.environ["CEREBRAS_API_KEY"],
            model=request.param,
            reasoning_effort="low",
            max_tokens=1024,
            timeout=30.0,
        )
    )
    try:
        yield instance
    finally:
        await instance.close()


async def test_generate(
    provider: CerebrasAIProvider, record_property: Callable[[str, Any], None]
) -> None:
    result = await provider.generate(
        AIContext(messages=[AIMessage(role="user", content="Reply with exactly the word pong.")])
    )
    assert "pong" in result.content.lower()
    assert result.finish_reason == "stop"
    assert result.usage["output_tokens"] > 0
    record_property("usage", result.usage)


async def test_stream(
    provider: CerebrasAIProvider, record_property: Callable[[str, Any], None]
) -> None:
    started = time.monotonic()
    first_text_ms: float | None = None
    text = ""
    done: StreamDone | None = None
    context = AIContext(
        messages=[AIMessage(role="user", content="Reply with exactly the word pong.")]
    )
    async for event in provider.generate_structured_stream(context):
        if isinstance(event, StreamTextDelta):
            if first_text_ms is None:
                first_text_ms = (time.monotonic() - started) * 1000
            text += event.text
        elif isinstance(event, StreamDone):
            done = event
    assert "pong" in text.lower()
    assert done is not None and done.finish_reason == "stop"
    assert done.usage["output_tokens"] > 0
    record_property("first_text_ms", first_text_ms)
    record_property("usage", done.usage)


async def test_streamed_tool_round(
    provider: CerebrasAIProvider, record_property: Callable[[str, Any], None]
) -> None:
    context = AIContext(
        system_prompt=(
            "Use the add tool to answer addition questions. "
            "After its result, answer with the number only."
        ),
        messages=[AIMessage(role="user", content="Use the add tool to add 17 and 25.")],
        tools=[
            AITool(
                name="add",
                description="Add two integers",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            )
        ],
    )
    events = [event async for event in provider.generate_structured_stream(context)]
    calls = [event for event in events if isinstance(event, StreamToolCall)]
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "add"
    assert call.arguments == {"a": 17, "b": 25}
    thinking = "".join(
        event.thinking for event in events if isinstance(event, StreamThinkingDelta)
    )
    context.messages.extend(
        [
            AIMessage(
                role="assistant",
                content=[
                    AIThinkingPart(thinking=thinking),
                    AIToolCallPart(id=call.id, name=call.name, arguments=call.arguments),
                ],
            ),
            AIMessage(
                role="tool",
                content=[
                    AIToolResultPart(tool_call_id=call.id, name=call.name, result="42"),
                ],
            ),
        ]
    )
    context.tools = []
    result = await provider.generate(context)
    assert "42" in result.content
    assert result.finish_reason == "stop"
    record_property("usage", result.usage)

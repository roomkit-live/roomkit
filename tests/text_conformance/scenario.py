"""What every scenario uses: a tool, a context and one generation, run as the
loop runs it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIThinkingPart,
    AITool,
    AIToolCall,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
    thinking_parts_of,
)
from roomkit.providers.ai.thinking_blocks import ThinkingBlocks
from tests.text_conformance.driver import Driver
from tests.text_conformance.script import Script

LOOKUP = AITool(
    name="lookup",
    description="Look it up.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)
PNG = "data:image/png;base64,iVBORw0KGgo="


@dataclass
class Answer:
    """What a generation handed the loop, streamed or not."""

    calls: list[StreamToolCall | AIToolCall]
    deltas: list[StreamToolCallDelta] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    reasoning: list[AIThinkingPart] = field(default_factory=list)
    finish_reason: str | None = None
    text: str = ""
    """The answer's text, reasoning taken out."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """What the response says beside its content (its model)."""


def tool_context(*tools: AITool, messages: list[AIMessage] | None = None) -> AIContext:
    return AIContext(
        messages=messages or [AIMessage(role="user", content="go")], tools=list(tools)
    )


async def generation(driver: Driver, script: Script, mode: str, context: AIContext) -> Answer:
    """Run *script* through the driver's provider, streamed or through
    ``generate()``, and collect what the loop reads of it."""
    provider = driver.provider(script)
    if mode == "generate":
        response = await provider.generate(context)
        return Answer(
            calls=list(response.tool_calls),
            usage=dict(response.usage),
            reasoning=thinking_parts_of(response),
            finish_reason=response.finish_reason,
            text=response.content,
            metadata=dict(response.metadata),
        )
    events = [event async for event in provider.generate_structured_stream(context)]
    done = next((e for e in events if isinstance(e, StreamDone)), None)
    # The reasoning as the loop assembles it from the stream.
    reasoning = ThinkingBlocks()
    for event in events:
        if isinstance(event, StreamThinkingDelta):
            reasoning.add(event)
    return Answer(
        calls=[e for e in events if isinstance(e, StreamToolCall)],
        deltas=[e for e in events if isinstance(e, StreamToolCallDelta)],
        usage=dict(done.usage) if done is not None else {},
        reasoning=reasoning.parts(),
        finish_reason=done.finish_reason if done is not None else None,
        text="".join(e.text for e in events if isinstance(e, StreamTextDelta)),
        metadata=dict(done.metadata) if done is not None else {},
    )

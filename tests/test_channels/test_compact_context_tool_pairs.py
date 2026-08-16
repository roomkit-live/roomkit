"""Emergency compaction never splits an assistant/tool-result pair.

``_compact_context`` cuts the message list in half to recover from a context
overflow. A cut landing on a ``tool`` message strands the results from the
assistant turn that called for them — an orphan every strict provider rejects
with a 400, turning a recoverable overflow into a dead turn.
"""

from __future__ import annotations

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.ai.mock import MockAIProvider


def _orphaned_result_ids(messages: list[AIMessage]) -> list[str]:
    """Tool-result ids whose calling assistant turn is absent or later."""
    seen_calls: set[str] = set()
    orphans: list[str] = []
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if isinstance(part, AIToolCallPart):
                seen_calls.add(part.id)
            elif isinstance(part, AIToolResultPart) and part.tool_call_id not in seen_calls:
                orphans.append(part.tool_call_id)
    return orphans


def _channel() -> AIChannel:
    return AIChannel("ai1", provider=MockAIProvider())


async def test_a_cut_landing_on_a_tool_message_keeps_the_pair_whole() -> None:
    # Eight messages: the naive halfway cut lands exactly on the tool-result
    # message (index 4), stranding it from its assistant call at index 3.
    messages = [
        AIMessage(role="user", content="q1"),
        AIMessage(role="assistant", content="a1"),
        AIMessage(role="user", content="q2"),
        AIMessage(
            role="assistant",
            content=[AIToolCallPart(id="t1", name="echo", arguments={})],
        ),
        AIMessage(
            role="tool",
            content=[AIToolResultPart(tool_call_id="t1", name="echo", result="ok")],
        ),
        AIMessage(role="user", content="q3"),
        AIMessage(role="assistant", content="a3"),
        AIMessage(role="user", content="q4"),
    ]

    compacted = await _channel()._compact_context(AIContext(messages=messages))

    assert _orphaned_result_ids(compacted.messages) == []
    # The summary replaced the old half; the conversation goes on from there.
    assert compacted.messages[0].role == "user"
    assert compacted.messages[-1].content == "q4"


async def test_a_cut_between_plain_messages_is_untouched() -> None:
    messages = [
        AIMessage(role="user" if i % 2 == 0 else "assistant", content=f"m{i}") for i in range(8)
    ]

    compacted = await _channel()._compact_context(AIContext(messages=messages))

    # Halfway cut: one summary message + the recent half.
    assert len(compacted.messages) == 5
    assert compacted.messages[-1].content == "m7"


async def test_too_small_to_compact_still_raises() -> None:
    messages = [AIMessage(role="user", content="q")] * 4

    with pytest.raises(ProviderError, match="cannot compact"):
        await _channel()._compact_context(AIContext(messages=messages))

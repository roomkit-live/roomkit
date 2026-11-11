"""An unidentifiable deferred input protects the hook's whole context (RFC §6.4)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels._discussion_turn import DISCUSSION_TURN, issue_mark, retire_mark
from roomkit.channels._tool_eviction import is_eviction_placeholder
from roomkit.channels.ai import AIChannel
from roomkit.core.hooks import SyncPipelineResult
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai.base import AIMessage, AIResponse, AIToolCallPart, AIToolResultPart
from tests.conftest import make_event
from tests.test_channels.test_ai_compaction_between_rounds import (
    _calls,
    _CompactingMemory,
    _Recorded,
    _result,
)
from tests.tool_loop_modes import respond


@pytest.mark.parametrize("rewrite", ["in_place", "replace", "copy_after_mutation", "remove"])
async def test_ambiguous_input_keeps_old_tools_and_reduces_only_new_results(
    streaming: bool, rewrite: str
) -> None:
    """Old tool results stay whole, even after a memory inserts another message."""
    history = [
        make_event(room_id="r1", channel_id="sms1", body="Earlier.", index=0),
        make_event(room_id="r1", channel_id="sms1", body="Later.", index=2),
    ]
    summary = AIMessage(role="assistant", content="Memory added this.")
    memory = _CompactingMemory([[summary], []], messages=[], events=history)
    provider = _Recorded(
        ai_responses=[_calls("c1"), _calls("c2"), AIResponse(content="done")], streaming=streaming
    )
    long = "A whole result.\n" * 250
    left: list[AIMessage] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return long

    async def hook(event: AIGenerationEvent) -> SyncPipelineResult:
        messages = event.ai_context.messages
        if rewrite == "remove":
            messages.pop(1)
        elif rewrite == "replace":
            messages[1] = AIMessage(role="user", content="Redacted question.")
        else:
            messages[1].content = "Redacted question."
        messages[:0] = [
            AIMessage(
                role="assistant",
                content=[AIToolCallPart(id="old", name="lookup", arguments={})],
            ),
            AIMessage(
                role="tool",
                content=[AIToolResultPart(tool_call_id="old", name="lookup", result=long)],
            ),
        ]
        if rewrite == "copy_after_mutation":
            messages[:] = [m.model_copy(deep=True) for m in messages]
        left.extend(m.model_copy(deep=True) for m in messages)
        return SyncPipelineResult(allowed=True)

    channel = AIChannel("ai1", provider=provider, tool_handler=handler, memory=memory)
    channel._before_generation_hook = hook
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [{"name": "lookup", "description": "Look it up."}]},
    )
    mark = issue_mark([])
    try:
        run = await respond(
            channel,
            make_event(
                room_id="r1",
                channel_id="sms1",
                body="Private question.",
                index=1,
                metadata={DISCUSSION_TURN: mark},
            ),
            binding,
            RoomContext(room=Room(id="r1"), recent_events=history),
        )
        assert run.text == "done"
        assert [call[1] for call in memory.compactions] == [[], []]
        assert len(provider.sent) == 3
        for request in provider.sent[1:]:
            assert request[: len(left) + 1] == [summary, *left]
            assert _result(request[2]) == long
        assert _result(provider.sent[1][-1]) == long
        assert is_eviction_placeholder(_result(provider.sent[2][-3]))
        assert _result(provider.sent[2][-1]) == long
    finally:
        retire_mark(mark)
        await channel.close()

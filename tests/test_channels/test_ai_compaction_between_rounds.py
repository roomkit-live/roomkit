"""A turn's history compacted between two of its tool rounds (RFC §6.4, §20.1).

After each tool round it ran, the AI channel hands its memory the history
before the turn's input and what the round's generation measured
(``MemoryProvider.compact``). A memory that returns ``None`` changes nothing.
A history it returns stands for the old one in the next requests: the
turn's input kept whole, a user text the history ends on joined to it, the
long results of the turn's earlier rounds stored for ``read_stored_result``,
the last round's whole. No emergency compaction is needed for it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from roomkit.channels._discussion_turn import DISCUSSION_TURN, issue_mark, retire_mark
from roomkit.channels._tool_eviction import is_eviction_placeholder
from roomkit.channels.ai import AIChannel
from roomkit.core.hooks import SyncPipelineResult
from roomkit.memory.mock import MockMemoryProvider
from roomkit.memory.summarizing import SummarizingMemory
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AIToolCall,
    AIToolResultPart,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_HISTORY = [
    AIMessage(role="user", content="earlier 0"),
    AIMessage(role="assistant", content="answer 0"),
    AIMessage(role="user", content="earlier 1"),
    AIMessage(role="assistant", content="answer 1"),
]


class _CompactingMemory(MockMemoryProvider):
    """A memory whose n-th compaction answers its n-th reaction: ``None``,
    a history, or an exception it raises; each compaction it is asked is
    kept."""

    def __init__(
        self,
        reactions: list[Any],
        messages: list[AIMessage],
        *,
        events: list[RoomEvent] | None = None,
    ) -> None:
        super().__init__(messages=messages, events=events)
        self._reactions = list(reactions)
        self.compactions: list[tuple[str, list[AIMessage], dict[str, int], str | None]] = []

    async def compact(
        self,
        room_id: str,
        history: list[AIMessage],
        *,
        usage: dict[str, int],
        channel_id: str | None = None,
    ) -> list[AIMessage] | None:
        self.compactions.append((room_id, history, usage, channel_id))
        reaction = self._reactions.pop(0) if self._reactions else None
        if isinstance(reaction, Exception):
            raise reaction
        return reaction


class _Recorded(MockAIProvider):
    """The mock, each request's messages kept as it sent them: the loop
    builds a round's request on the list the previous one sent."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sent: list[list[AIMessage]] = []

    async def generate(self, context: AIContext) -> AIResponse:
        self.sent.append(list(context.messages))
        return await super().generate(context)


def _calls(call_id: str, input_tokens: int = 7000) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name="lookup", arguments={"n": call_id})],
        usage={"input_tokens": input_tokens, "output_tokens": 10},
    )


async def _turn(
    memory: MockMemoryProvider,
    streaming: bool,
    results: dict[str, str] | None = None,
    rounds: int = 3,
) -> tuple[_Recorded, LoopRun]:
    """A turn on "go" whose model calls ``lookup`` *rounds* times, then
    answers "done"; each call answers its *results* entry, else "found"."""
    answers = [_calls(f"c{n}") for n in range(1, rounds + 1)]
    provider = _Recorded(ai_responses=[*answers, AIResponse(content="done")], streaming=streaming)

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return (results or {}).get(arguments["n"], "found")

    channel = AIChannel("ai1", provider=provider, tool_handler=handler, memory=memory)
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [{"name": "lookup", "description": "Look it up."}]},
    )
    run = await respond(
        channel,
        make_event(body="go", channel_id="sms1"),
        binding,
        RoomContext(room=Room(id="r1")),
    )
    return provider, run


def _contents(messages: list[AIMessage]) -> list[Any]:
    return [m.content if isinstance(m.content, str) else m.role for m in messages]


def _result(message: AIMessage) -> Any:
    part = message.content[0]
    assert isinstance(part, AIToolResultPart)
    return part.result


async def test_a_memory_that_compacts_nothing_sends_every_round_whole(streaming: bool) -> None:
    """A turn whose rounds each measure 85% of the model's window (the mock's
    8,192 tokens), with a memory that compacts nothing, reads its memory once,
    at its start, and sends every round whole."""
    memory = MockMemoryProvider(messages=_HISTORY)
    provider, run = await _turn(memory, streaming)

    assert run.text == "done"
    assert provider.context_window == 8192
    assert len(memory.retrieve_calls) == 1
    assert _contents(provider.sent[3]) == [
        *(m.content for m in _HISTORY),
        "go",
        *["assistant", "tool"] * 3,
    ]


async def test_a_memory_compacts_the_history_between_two_tool_rounds(streaming: bool) -> None:
    """The memory is handed the history before the turn's input and the
    round's usage after each round; what it returns is what the next
    requests send, a user text it ends on joined to the input, which the
    next compaction finds again."""
    first = [AIMessage(role="user", content="Summary."), _HISTORY[3]]
    second = [AIMessage(role="user", content="Summary 2.")]
    memory = _CompactingMemory([first, second, None], messages=_HISTORY)
    provider, run = await _turn(memory, streaming)

    assert run.text == "done"
    assert len(provider.sent) == 4
    room, history, usage, channel = memory.compactions[0]
    assert (room, channel) == ("r1", "ai1")
    assert history == _HISTORY
    assert usage == {"input_tokens": 7000, "output_tokens": 10}
    assert _contents(provider.sent[1]) == [
        "Summary.",
        "answer 1",
        "go",
        "assistant",
        "tool",
    ]
    assert memory.compactions[1][1] == first
    assert _contents(provider.sent[2]) == [
        "Summary 2.\n\ngo",
        *["assistant", "tool"] * 2,
    ]
    # The input, with the summary joined ahead of it, is still the turn's.
    assert memory.compactions[2][1] == []
    assert _contents(provider.sent[3]) == [
        "Summary 2.\n\ngo",
        *["assistant", "tool"] * 3,
    ]


async def test_a_compaction_stores_the_earlier_rounds_long_results(streaming: bool) -> None:
    """A history returned after the second round stores the first round's long
    result for re-reading, a preview in its place, and sends the second's,
    which the model has not read yet, whole."""
    long_results = {"c1": "x" * 5000, "c2": "y" * 5000}
    memory = _CompactingMemory([None, list(_HISTORY)], messages=_HISTORY)
    provider, run = await _turn(memory, streaming, long_results, rounds=2)

    assert run.text == "done"
    before = provider.sent[2]
    assert _contents(before[:5]) == [*(m.content for m in _HISTORY), "go"]
    assert is_eviction_placeholder(_result(before[6]))
    assert _result(before[8]) == "y" * 5000
    # Before the compaction, the first round's result went whole.
    assert _result(provider.sent[1][6]) == "x" * 5000


async def test_a_memory_that_fails_to_compact_leaves_the_turn_as_it_was(
    streaming: bool, caplog: pytest.LogCaptureFixture
) -> None:
    memory = _CompactingMemory([RuntimeError("summarizer down")], messages=_HISTORY)
    with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
        provider, run = await _turn(memory, streaming)

    assert run.text == "done"
    assert len(memory.compactions) == 3
    assert _contents(provider.sent[1]) == [
        *(m.content for m in _HISTORY),
        "go",
        "assistant",
        "tool",
    ]
    assert "Memory compaction between two tool rounds failed" in caplog.text


async def test_a_summarizing_memory_compacts_a_turn_past_its_threshold_between_rounds(
    streaming: bool,
) -> None:
    """RMK-1533's proof: a turn whose first round measures 85% of the window,
    past a summarizing memory's threshold of 80%, has its older history
    summarized before the second call, with no emergency compaction."""
    history = [
        AIMessage(role="user" if n % 2 == 0 else "assistant", content=f"turn {n} " + "w" * 400)
        for n in range(8)
    ]
    summarizer = MockAIProvider(responses=["What was said before."])
    memory = SummarizingMemory(
        MockMemoryProvider(messages=history),
        summarizer,
        8192,
        tier2_ratio=0.80,
        summary_max_tokens=100,
        min_events=2,
    )
    provider, run = await _turn(memory, streaming, rounds=1)

    assert run.text == "done"
    assert len(provider.sent) == 2
    assert len(summarizer.calls) == 1
    compacted = provider.sent[1]
    # The 7,010 tokens measured pass the threshold of 6,553 by 457: with the
    # summary's 100, the history gives back 557 of its 848, so the latest two
    # turns, 212 tokens, are kept whole, the summary joined to the first.
    assert "What was said before." in compacted[0].content
    assert compacted[0].content.endswith(f"\n\n{history[6].content}")
    assert compacted[1].content == history[7].content
    assert _contents(compacted[2:]) == ["go", "assistant", "tool"]


@pytest.mark.parametrize("copy_mode", ["none", "copy", "mutate_discarded"])
async def test_editing_a_follower_keeps_the_answered_duplicate(
    streaming: bool, copy_mode: str
) -> None:
    """A hook editing a later message cannot move the compaction boundary
    to an older message with the same words as the answered input."""
    history = [
        make_event(room_id="r1", channel_id="sms1", body="Same question.", index=0),
        make_event(room_id="r1", channel_id="sms1", body="Later detail.", index=2),
    ]
    memory = _CompactingMemory([None], messages=[], events=history)
    provider = _Recorded(
        ai_responses=[_calls("c1"), AIResponse(content="done")], streaming=streaming
    )

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    async def hook(event: AIGenerationEvent) -> SyncPipelineResult:
        if copy_mode != "none":
            original = event.ai_context.messages
            event.ai_context = event.ai_context.model_copy(
                update={
                    "messages": [
                        message.model_copy(deep=True) for message in event.ai_context.messages
                    ]
                }
            )
            if copy_mode == "mutate_discarded":
                # The context returned is unchanged, whatever happened to the abandoned one.
                original[1].content = "Discarded rewrite."
        last = event.ai_context.messages[-1]
        event.ai_context.messages[-1] = last.model_copy(
            update={"content": f"{last.content} [checked]"}
        )
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
                body="Same question.",
                index=1,
                metadata={DISCUSSION_TURN: mark},
            ),
            binding,
            RoomContext(room=Room(id="r1"), recent_events=history),
        )
        assert run.text == "done"
        assert len(memory.compactions) == 1
        assert _contents(memory.compactions[0][1]) == ["Same question."]
        assert len(provider.sent) == 2
        assert str(provider.sent[1][2].content).endswith("[checked]")
    finally:
        retire_mark(mark)
        await channel.close()

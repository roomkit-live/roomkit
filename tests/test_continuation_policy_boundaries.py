"""Public continuation boundaries shared by both text provider paths (RFC §6.4)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from roomkit import AIChannel, ChannelBinding, ChannelCategory, ChannelType, Room, RoomContext
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, ServedCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

LOOKUP = AITool(name="lookup", description="Read a record", parameters={})


async def _serve(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _run(provider: MockAIProvider, policy: Callable[[str], str | None]) -> LoopRun:
    channel = AIChannel(
        "agent", provider=provider, tools=[LOOKUP], tool_handler=_serve, continuation=policy
    )
    binding = ChannelBinding(
        room_id="room",
        channel_id="agent",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    try:
        return await respond(
            channel,
            make_event(body="Go", channel_id="member"),
            binding,
            RoomContext(room=Room(id="room")),
        )
    finally:
        await channel.close()


@pytest.mark.parametrize("nudge", [None, "", "  proceed\n"])
async def test_only_none_ends_continuation_and_text_is_preserved(
    streaming: bool, nudge: str | None
) -> None:
    read: list[str] = []

    def policy(text: str) -> str | None:
        read.append(text)
        return nudge if len(read) == 1 else None

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(content="  I will check.\n", finish_reason="stop"),
            AIResponse(content="Done.", finish_reason="stop"),
        ],
    )
    run = await _run(provider, policy)

    assert run.reason == "completed"
    assert len(provider.calls) == (1 if nudge is None else 2)
    assert read == (["  I will check.\n"] if nudge is None else ["  I will check.\n", "Done."])
    if nudge is not None:
        assert [(m.role, m.content) for m in provider.calls[1].messages[-2:]] == [
            ("assistant", "  I will check.\n"),
            ("user", nudge),
        ]


@pytest.mark.parametrize("text", ["", " \n\t"])
async def test_blank_text_never_reaches_the_policy(streaming: bool, text: str) -> None:
    read: list[str] = []

    def policy(value: str) -> str:
        read.append(value)
        return "Go on"

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(content=text, finish_reason="stop"),
        ],
    )
    run = await _run(provider, policy)

    assert run.reason == "completed"
    assert read == []
    assert len(provider.calls) == 1


@pytest.mark.parametrize("served", [False, True], ids=["local", "provider"])
async def test_a_round_with_calls_is_not_submitted_to_the_policy(
    streaming: bool, served: bool
) -> None:
    read: list[str] = []

    def policy(text: str) -> None:
        read.append(text)

    call = AIToolCall(
        id="c1",
        name="lookup",
        arguments={},
        served=ServedCall(result="found") if served else None,
    )
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(content="I will check.", finish_reason="stop", tool_calls=[call]),
            AIResponse(content="Done.", finish_reason="stop"),
        ],
    )
    run = await _run(provider, policy)

    assert run.reason == "completed"
    assert len(provider.calls) == (1 if served else 2)
    assert read == ([] if served else ["Done."])
    assert len(run.calls) == 1
    assert run.calls[0].result == "found"


@pytest.mark.parametrize("after_call", [False, True])
async def test_policy_exceptions_keep_the_original_error(
    streaming: bool, after_call: bool
) -> None:
    error = RuntimeError("continuation callback failed")
    read: list[str] = []

    def policy(text: str) -> None:
        read.append(text)
        raise error

    rounds = [AIResponse(content="I will check.", finish_reason="stop")]
    if after_call:
        rounds.insert(
            0,
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(id="c1", name="lookup", arguments={}),
                ],
            ),
        )
    provider = MockAIProvider(streaming=streaming, ai_responses=rounds)
    with pytest.raises(RuntimeError) as caught:
        await _run(provider, policy)

    assert caught.value is error
    assert read == ["I will check."]
    assert len(provider.calls) == (2 if after_call else 1)

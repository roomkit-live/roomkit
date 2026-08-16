"""A streaming tool loop says why it stopped, instead of just stopping.

The loop knows which of its rules fired. It used to log that and ``return``,
so the stream simply ended and a consumer had to re-derive the reason by
counting tool calls and reading a clock. That guess is what reported a loop
the platform had cut mid-work as "the model returned an empty response".

``LoopEndMarker`` is emitted on every exit, ``completed`` included, so the end
of the stream is never itself the signal.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.streaming import LoopEndMarker
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider


def _tool(i: int = 0) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"t{i}", name="echo", arguments={"value": str(i)})],
    )


async def _run(ch: AIChannel, context: AIContext) -> list[object]:
    return [delta async for delta in ch._run_streaming_tool_loop(context)]


def _ctx() -> AIContext:
    return AIContext(messages=[AIMessage(role="user", content="go")])


def _markers(deltas: list[object]) -> list[LoopEndMarker]:
    return [d for d in deltas if isinstance(d, LoopEndMarker)]


def _channel(responses: list[AIResponse], **kwargs: object) -> AIChannel:
    from unittest.mock import AsyncMock

    return AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=responses, streaming=True),
        tool_handler=AsyncMock(return_value="ok"),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_a_plain_answer_is_marked_completed() -> None:
    ch = _channel([AIResponse(content="hello")])

    marks = _markers(await _run(ch, _ctx()))

    assert [m.reason for m in marks] == ["completed"]


async def test_an_answer_after_tools_is_still_completed() -> None:
    ch = _channel([_tool(), AIResponse(content="done")])

    marks = _markers(await _run(ch, _ctx()))

    assert [m.reason for m in marks] == ["completed"]
    assert marks[0].rounds == 1, "one tool round ran before the answer"


async def test_a_truncated_round_is_named_truncated() -> None:
    # The reasoning model that spent its whole output budget thinking: empty
    # content, finish_reason=length. Distinguishing this from silence is the
    # difference between "raise max_tokens" and "change model".
    ch = _channel([_tool(), AIResponse(content="", finish_reason="length")])

    marks = _markers(await _run(ch, _ctx()))

    assert [m.reason for m in marks] == ["truncated"]


async def test_a_silent_round_is_named_empty_response() -> None:
    ch = _channel(
        [_tool(), AIResponse(content="", finish_reason="stop"), AIResponse(content="")],
        max_empty_retries=1,
    )

    marks = _markers(await _run(ch, _ctx()))

    assert [m.reason for m in marks] == ["empty_response"]


async def test_the_round_cap_is_named() -> None:
    ch = _channel([_tool(0), _tool(1), _tool(2)], max_tool_rounds=1)

    marks = _markers(await _run(ch, _ctx()))

    assert [m.reason for m in marks] == ["max_rounds"]


async def test_the_deadline_is_named() -> None:
    # The loop tests its deadline at a round boundary, so the round that
    # crossed it still ran — and the tool call it asked for is dropped unrun.
    # That is the case the platform used to report as a model failure.
    ch = _channel([_tool(0), _tool(1), AIResponse(content="never")])

    async def slow(*args: object, **kwargs: object) -> str:
        await asyncio.sleep(0.05)
        return "ok"

    ch._tool_handler = slow  # type: ignore[assignment]
    ch._tool_loop_timeout_seconds = 0.01

    marks = _markers(await _run(ch, _ctx()))

    assert [m.reason for m in marks] == ["timeout"]


async def test_cancellation_is_named() -> None:
    ch = _channel([_tool(), AIResponse(content="never")])
    deltas: list[object] = []
    async for delta in ch._run_streaming_tool_loop(_ctx()):
        deltas.append(delta)
        for loop_ctx in list(ch._active_loops.values()):
            loop_ctx.cancel_event.set()

    assert [m.reason for m in _markers(deltas)] == ["cancelled"]


@pytest.mark.parametrize(
    "responses",
    [
        [AIResponse(content="hello")],
        [_tool(), AIResponse(content="done")],
        [_tool(), AIResponse(content="", finish_reason="length")],
    ],
)
async def test_exactly_one_marker_and_it_comes_last(responses: list[AIResponse]) -> None:
    """The invariant a consumer relies on: one terminal marker, at the end."""
    ch = _channel(responses)

    deltas = await _run(ch, _ctx())

    assert len(_markers(deltas)) == 1
    assert isinstance(deltas[-1], LoopEndMarker)

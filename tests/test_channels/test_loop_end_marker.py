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
from roomkit.core.mixins._streaming_segments import _ended_record, _GenerationsRead
from roomkit.models.streaming import (
    LoopEndMarker,
    SegmentBreakMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
    UsageMarker,
)
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse, AIToolCall, ServedCall
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


@pytest.mark.parametrize(
    "finish_reason",
    ["length", "max_tokens", "MAX_TOKENS", "model_context_window_exceeded", "model_length"],
)
async def test_a_truncated_round_is_named_truncated(finish_reason: str) -> None:
    # The reasoning model that spent its whole output budget thinking, or a
    # context window that filled up: empty content, out of room. Telling this
    # from silence is the difference between "raise the cap" and "change model",
    # and the round is not tried again, since the same room runs out again
    # (RFC §6.4).
    ch = _channel(
        [_tool(), AIResponse(content="", finish_reason=finish_reason), AIResponse(content="x")],
        max_empty_retries=1,
    )

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

    ch.tool_handler = slow
    ch._tool_loop_timeout_seconds = 0.01

    marks = _markers(await _run(ch, _ctx()))

    assert [m.reason for m in marks] == ["timeout"]


async def test_the_anti_loop_ripcord_is_named_not_disguised_as_an_answer() -> None:
    """The exit that used to read ``completed``, and the reason this value
    exists.

    The ripcord works by demanding prose, so the model DOES produce text — and
    text was the whole test for "the model answered".
    A run cut after hammering one call was therefore indistinguishable from a
    finished one, and callers delivered the cut turn's summary as the result.

    ``_REPEAT_CALL_LIMIT`` short-circuits the 3rd identical call and
    ``_REPEAT_FORCE_STOP_AT`` pulls the ripcord 3 blocked repeats later, so
    the same call re-issued six times reaches it.
    """
    ch = _channel([*[_tool(0) for _ in range(6)], AIResponse(content="here is what I found")])

    deltas = await _run(ch, _ctx())
    marks = _markers(deltas)

    assert [m.reason for m in marks] == ["force_stopped"]
    # It is emphatically NOT empty: the text is what made this exit invisible.
    assert "here is what I found" in "".join(d for d in deltas if isinstance(d, str))


async def test_a_loop_that_never_repeats_is_untouched_by_the_ripcord() -> None:
    """The positive control: the same amount of work with DIFFERENT arguments
    is legitimate and must stay ``completed``. It is also the guard's blind
    spot — a model permuting its arguments burns the whole round budget
    without ever tripping this."""
    ch = _channel([*[_tool(i) for i in range(6)], AIResponse(content="done")])

    marks = _markers(await _run(ch, _ctx()))

    assert [m.reason for m in marks] == ["completed"]


async def test_a_cancel_queued_before_the_first_round_is_still_named() -> None:
    """The pre-loop drain exit used to ``return`` bare — no marker, ever."""
    ch = _channel([AIResponse(content="never")])
    ch._drain_steering_queue = lambda context, loop_ctx: (context, True)  # type: ignore[method-assign]

    deltas = await _run(ch, _ctx())

    assert [m.reason for m in _markers(deltas)] == ["cancelled"]
    assert isinstance(deltas[-1], LoopEndMarker)


async def test_provider_owned_tool_calls_still_end_with_a_marker() -> None:
    """The provider ran its calls itself (their results ride them): the loop
    only reports them, and that exit used to be the one carve-out from
    "every exit yields a marker"."""
    provider_run = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="b1", name="Bash", served=ServedCall(result="a.txt"))],
    )
    ch = AIChannel("ai1", provider=MockAIProvider(ai_responses=[provider_run], streaming=True))

    deltas = await _run(ch, _ctx())

    assert [m.reason for m in _markers(deltas)] == ["completed"]
    assert isinstance(deltas[-1], LoopEndMarker)


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


@pytest.mark.parametrize(
    ("marker", "record"),
    [
        # A channel that says nothing of its generations writes the two keys.
        (
            LoopEndMarker(reason="completed", usage={"input_tokens": 1}),
            {"ai_usage": {"input_tokens": 1}, "loop_end_reason": "completed"},
        ),
        # Its last generation failed: none, written, never left out.
        (
            LoopEndMarker(reason="error", usage_complete=False, model="m", provider="p"),
            {
                "ai_usage": {},
                "loop_end_reason": "error",
                "ai_last_usage": None,
                "ai_usage_complete": False,
                "ai_model": "m",
                "ai_provider": "p",
            },
        ),
    ],
)
def test_the_record_holds_what_its_marker_says(
    marker: LoopEndMarker, record: dict[str, object]
) -> None:
    assert _ended_record(marker) == record


_TOLD = UsageMarker(usage={"input_tokens": 5, "output_tokens": 2})
_START = ToolCallStartMarker(tool_name="t", tool_id="c")
_END = ToolCallEndMarker(tool_name="t", tool_id="c")


@pytest.mark.parametrize(
    ("items", "complete"),
    [
        pytest.param([_TOLD], True, id="told-and-nothing-after"),
        pytest.param([_TOLD, _START], True, id="its-call-runs"),
        pytest.param([_TOLD, _START, _START, _END], True, id="one-of-two-calls-runs"),
        pytest.param([_TOLD, _START, _END], False, id="its-calls-over-the-next-starts"),
        pytest.param([_TOLD, SegmentBreakMarker()], False, id="tried-again"),
        pytest.param([_START, _END, _TOLD], True, id="a-provider-s-call-within-it"),
        pytest.param([_TOLD, _START, _END, _START], False, id="a-provider-s-call-in-the-next-one"),
        pytest.param([_TOLD, _START, _END, _START, _END, _TOLD], True, id="told-again"),
    ],
)
def test_a_cut_reads_incomplete_with_a_generation_in_flight(
    items: list[object], complete: bool
) -> None:
    read = _GenerationsRead()
    for item in items:
        read.read(item)

    assert read.cut_record()["ai_usage_complete"] is complete


def test_a_stream_that_told_nothing_records_only_its_end() -> None:
    read = _GenerationsRead()
    read.read("text")

    assert read.cut_record() == {"loop_end_reason": "cancelled"}

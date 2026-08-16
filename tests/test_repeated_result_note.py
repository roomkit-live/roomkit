"""A tool that keeps giving the same answer says so.

``_repeated_call_guard`` keys on the ARGUMENTS, which leaves a blind spot the
size of the failure it was written for: a model that permutes its arguments
never trips it, and is never told that nothing is changing. Measured on a stuck
turn — 54 tool calls, **44 distinct argument sets**, but only **25 distinct
results**, one of them (an empty search) returned 23 times. The guard fired on
10 of the 54 calls, all at the very end; the first already-seen RESULT had come
back 40 calls earlier.

The note is deliberately not a block. Identical results are not by themselves a
fault — six deletions each answering ``{"success": true}`` are six correct
operations with one result — so short-circuiting on result identity would
destroy real work to save latency. Blocking stays with the argument guard,
which cannot mistake legitimate work for a loop.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

NOTE = "identical result"


def _search(i: int) -> AIResponse:
    """One search call, with arguments that differ every time — the shape the
    argument guard is blind to."""
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"t{i}", name="boards", arguments={"search": f"term-{i}"})],
    )


def _channel(responses: list[AIResponse], handler) -> AIChannel:
    return AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=responses, streaming=True),
        tool_handler=handler,
    )


@contextmanager
def _in_a_turn(ch: AIChannel):
    """Bind a tool-loop context, the way the real loop does.

    Outside one, ``_get_loop_ctx`` hands back a FRESH context per call — so a
    direct call to the guard would count to one forever and every assertion
    below would pass for the wrong reason."""
    from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext

    token = _current_loop_ctx.set(_ToolLoopContext())
    try:
        yield ch
    finally:
        _current_loop_ctx.reset(token)


async def _run(ch: AIChannel, ctx: AIContext) -> list[object]:
    return [d async for d in ch._run_streaming_tool_loop(ctx)]


def _tool_results(ctx: AIContext) -> list[str]:
    """Every tool result the loop put in front of the model, in order."""
    return [
        part.result
        for m in ctx.messages
        if m.role == "tool" and isinstance(m.content, list)
        for part in m.content
    ]


async def test_the_note_reaches_the_model_where_the_argument_guard_is_silent() -> None:
    """The whole point, end to end through the real loop.

    Every call carries DIFFERENT arguments, so ``_repeated_call_guard`` never
    fires — this is the exact shape that burned 40 further calls in production.
    The answer is the same every time, and by the third one the result the model
    reads says so.
    """

    async def always_empty(name: str, args: dict) -> str:
        return json.dumps({"success": True, "cards": [], "total": 0})

    ch = _channel([*[_search(i) for i in range(4)], AIResponse(content="done")], always_empty)
    ctx = AIContext(messages=[AIMessage(role="user", content="go")])

    await _run(ch, ctx)
    results = _tool_results(ctx)

    assert len(results) == 4
    # The argument guard never spoke: four distinct argument sets.
    assert all("EXACT arguments" not in r for r in results)
    # First two ordinary, third and fourth named.
    assert NOTE not in results[0]
    assert NOTE not in results[1]
    assert NOTE in results[2]
    assert NOTE in results[3]
    # The payload the model needs is still there, ahead of the note.
    assert results[2].startswith('{"success": true, "cards": [], "total": 0}')


async def test_the_first_two_identical_answers_pass_unremarked() -> None:
    """A retry, a poll, a second row deleted: two identical answers are
    ordinary. Only the third is a pattern."""
    with _in_a_turn(_channel([AIResponse(content="x")], lambda n, a: None)) as ch:
        note_1 = ch._repeated_result_note("boards", "same")
        note_2 = ch._repeated_result_note("boards", "same")
        note_3 = ch._repeated_result_note("boards", "same")

    assert NOTE not in note_1
    assert NOTE not in note_2
    assert NOTE in note_3
    # The payload is preserved — the note is appended, never a replacement.
    assert note_3.startswith("same")


async def test_a_different_result_never_accumulates() -> None:
    """The positive control: work that keeps producing new answers is never
    annotated, however many calls it makes."""
    with _in_a_turn(_channel([AIResponse(content="x")], lambda n, a: None)) as ch:
        for i in range(10):
            assert NOTE not in ch._repeated_result_note("boards", f"row-{i}")


async def test_two_tools_returning_the_same_string_are_counted_apart() -> None:
    """``{"success": true}`` from two different tools is two facts, not one
    repetition — the key carries the tool name."""
    with _in_a_turn(_channel([AIResponse(content="x")], lambda n, a: None)) as ch:
        ok = json.dumps({"success": True})
        for _ in range(2):
            ch._repeated_result_note("todos", ok)
            ch._repeated_result_note("boards", ok)
        assert NOTE not in ch._repeated_result_note("notes", ok)


async def test_the_advisory_error_is_not_itself_annotated() -> None:
    """The argument guard's own error repeats verbatim by construction.
    Annotating it would stack a second advisory on a first that already says
    what is wrong."""
    with _in_a_turn(_channel([AIResponse(content="x")], lambda n, a: None)) as ch:
        advisory = json.dumps(
            {"error": "You already called 'boards' with these EXACT arguments 2 time(s)"}
        )
        for _ in range(5):
            assert NOTE not in ch._repeated_result_note("boards", advisory)


@pytest.mark.parametrize("n", [3, 8, 23])
async def test_the_note_counts_what_it_saw(n: int) -> None:
    """The count is the model's evidence — it is the difference between "this
    happened" and "this has been happening"."""
    with _in_a_turn(_channel([AIResponse(content="x")], lambda n_, a: None)) as ch:
        out = ""
        for _ in range(n):
            out = ch._repeated_result_note("boards", "same")
    assert f"{n} times this turn" in out

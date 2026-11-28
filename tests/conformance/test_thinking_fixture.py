"""A scripted thinker observes real calls and rejects malformed controls early."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.providers.ai import AIContext, AIMessage
from roomkit.speaking import Thought
from tests.conformance._agent import thinking_context
from tests.conformance._script import Gates
from tests.conformance._shape import unknown_members
from tests.conformance._thinking import ScriptedThinker, ThinkingFailedError


@pytest.mark.parametrize(
    "spec",
    [
        None,
        [],
        {},
        {"thoughts": []},
        {"thoughts": [None]},
        {"thoughts": [{"thought": None}]},
        {"thoughts": [{"thought": {"text": 1}}]},
        {"thoughts": [{"thought": {"urgent": 1}}]},
        {"thoughts": [{"thought": {"want_to_say": "words"}}]},
        {"thoughts": [{"thought": {"want_to_say": [None]}}]},
        {"thoughts": [{"thought": {"misspelled": 1}}]},
        {"thoughts": [{"thought": {}, "fail": "failed"}]},
        {"thoughts": [{"fail": None}]},
        {"thoughts": [{"thought": {}, "gate": None}]},
        {"thoughts": [{"thought": {}, "misspelled": 1}]},
        {"thoughts": [{"thought": {}}], "misspelled": 1},
        {"llm": None},
        {"llm": {}},
        {"llm": {"model": {"rounds": []}, "timeout_ms": "1"}},
        {"llm": {"model": {"rounds": []}, "max_tokens": 1.5}},
        {"llm": {"model": {"rounds": []}, "misspelled": 1}},
        {"llm": {"model": {"rounds": []}}, "thoughts": [{"thought": {}}]},
    ],
)
def test_invalid_thinker_is_refused_before_product_execution(spec: Any) -> None:
    scenario = {"setup": {"channels": [{"id": "ai", "type": "ai", "thinker": spec}]}}
    assert any("thinker" in problem for problem in unknown_members(scenario))
    with pytest.raises(ValueError, match="thinker"):
        ScriptedThinker(spec, Gates(), [], thinking_context)


async def test_thinker_records_before_its_gate_and_repeats_the_last_failure() -> None:
    calls: list[Any] = []
    gates = Gates()
    thinker = ScriptedThinker(
        {
            "thoughts": [
                {"thought": {"text": "Known.", "want_to_say": ["A fact."]}, "gate": "entered"},
                {"fail": "failed"},
            ]
        },
        gates,
        calls,
        thinking_context,
    )
    context = AIContext(messages=[AIMessage(role="user", content="Read this.")])
    task = asyncio.create_task(thinker.think(Thought("Before."), context))
    try:
        await gates.wait_for("entered")
        assert not task.done()
        assert calls[0]["previous"] == {"text": "Before.", "want_to_say": [], "urgent": False}
        assert calls[0]["context"]["messages"] == [
            {"role": "user", "text": "Read this.", "parts": None, "round": None}
        ]
        gates.open("entered")
        async with asyncio.timeout(2):
            assert await task == Thought("Known.", ("A fact.",))
        for _ in range(2):
            with pytest.raises(ThinkingFailedError, match="failed"):
                await thinker.think(Thought(), context)
        assert len(calls) == 3
    finally:
        gates.open("entered")
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

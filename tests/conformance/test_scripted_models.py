"""The scripted provider selects a real generation path and reserves gated rounds."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.providers.ai import AIContext
from roomkit.providers.ai.base import StreamTextDelta
from tests.conformance._agent import ModelFailedError, ScriptedModel
from tests.conformance._script import Gates
from tests.conformance._shape import unknown_members


async def _text(model: ScriptedModel) -> str:
    return "".join(
        [
            event.text
            async for event in model.generate_structured_stream(AIContext())
            if isinstance(event, StreamTextDelta)
        ]
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_a_held_generation_reserves_its_round(streaming: bool, cancel: bool) -> None:
    gates = Gates()
    model = ScriptedModel(
        [{"gate": "first", "text": "first"}, {"text": "second"}, {"text": "third"}],
        gates,
        streaming=streaming,
    )
    held = asyncio.create_task(_text(model))
    try:
        await gates.wait_for("first")
        assert await _text(model) == "second"
        if cancel:
            held.cancel()
            with pytest.raises(asyncio.CancelledError):
                await held
        else:
            gates.open("first")
            assert await held == "first"
        assert await _text(model) == "third"
        method = "generate_structured_stream" if streaming else "generate"
        assert [request["method"] for request in model.requests] == [method] * 3
    finally:
        held.cancel()
        await asyncio.gather(held, return_exceptions=True)


async def test_a_whole_generation_failure_is_not_a_successful_empty_response() -> None:
    model = ScriptedModel([{"fails": "broken"}, {"text": "next"}], Gates(), streaming=False)
    with pytest.raises(ModelFailedError, match="broken"):
        await _text(model)
    assert await _text(model) == "next"
    assert [request["method"] for request in model.requests] == ["generate", "generate"]


def _scenario(model: dict[str, Any], location: str) -> dict[str, Any]:
    channel: dict[str, Any] = {"id": "agent", "type": "ai", "model": {"rounds": []}}
    if location == "summarizer":
        channel["memory"] = {"summarizer": model}
    else:
        channel[location] = model
    return {"setup": {"channels": [channel]}}


@pytest.mark.parametrize(
    "field", ["deltas", "thinking", "stream", "done", "failure_after_calls", "failure_after_done"]
)
@pytest.mark.parametrize("location", ["model", "fallback_model", "summarizer"])
def test_stream_only_round_fields_are_refused_without_streaming(field: str, location: str) -> None:
    model = {"streaming": False, "rounds": [{field: []}]}
    scenario = _scenario(model, location)
    assert any(field in problem for problem in unknown_members(scenario))


@pytest.mark.parametrize("flag", [None, 0, "false", []])
@pytest.mark.parametrize("location", ["model", "fallback_model", "summarizer"])
def test_the_streaming_mode_is_not_coerced(flag: Any, location: str) -> None:
    model = {"streaming": flag, "rounds": []}
    scenario = _scenario(model, location)
    assert any("streaming" in problem for problem in unknown_members(scenario))


@pytest.mark.parametrize(
    "failure", [{"fails": "broken"}, {"refuses": {"status": 500, "message": "broken"}}]
)
@pytest.mark.parametrize("location", ["model", "fallback_model", "summarizer"])
def test_a_whole_response_cannot_also_fail_without_streaming(
    failure: dict[str, Any], location: str
) -> None:
    model = {"streaming": False, "rounds": [{"whole_response": {}, **failure}]}
    scenario = _scenario(model, location)
    assert any("whole_response" in problem for problem in unknown_members(scenario))

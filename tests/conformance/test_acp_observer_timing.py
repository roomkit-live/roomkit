"""ACP fixtures can distinguish work during a prompt from work after it."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from tests.conformance._acp import ScriptedACPAgent
from tests.conformance._script import Gates, perform
from tests.conformance._shape import unknown_members


@pytest.mark.parametrize("error", [None, {"code": -32000, "message": "refused"}])
async def test_prompt_reply_waits_at_its_final_gate(error: dict[str, Any] | None) -> None:
    gates = Gates()
    agent = ScriptedACPAgent({}, gates)
    prompt: dict[str, Any] = {"before_reply_gate": "reply"}
    if error is not None:
        prompt["error"] = error
    # No wire messages are emitted by this prompt; only its answer is held.
    task = asyncio.create_task(
        agent._prompt_reply({"params": {"sessionId": "s"}}, prompt, None)  # type: ignore[arg-type]
    )
    try:
        await gates.wait_for("reply")
        assert not task.done()
        gates.open("reply")
        assert await task == (
            ({"stopReason": "end_turn"}, None) if error is None else (None, error)
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_after_do_gate_is_reached_only_after_actions_return() -> None:
    gates = Gates()
    harness = Mock(gates=gates, run_action=AsyncMock())

    async def action(*_args: Any) -> None:
        await gates.hold("action")

    harness.run_action.side_effect = action
    task = asyncio.create_task(
        perform(harness, {"do": [{"call": "get_room"}], "after_do_gate": "done"}, {})
    )
    try:
        await gates.wait_for("action")
        assert not gates._pair("done")[1].is_set()
        gates.open("action")
        await gates.wait_for("done")
        harness.run_action.assert_awaited_once()
        assert not task.done()
        gates.open("done")
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("value", [None, True, 1, [], {}])
@pytest.mark.parametrize("control", ["before_reply_gate", "after_do_gate"])
def test_timing_controls_reject_non_string_gate_names(control: str, value: Any) -> None:
    setup: dict[str, Any]
    if control == "before_reply_gate":
        setup = {
            "channels": [
                {
                    "id": "agent",
                    "type": "ai",
                    "acp": {
                        "cwd": "/srv/project",
                        "agent": {"prompts": [{control: value}]},
                    },
                }
            ]
        }
    else:
        setup = {
            "hooks": [
                {
                    "name": "tools",
                    "trigger": "on_tool_call",
                    "reactions": [
                        {control: value},
                    ],
                }
            ]
        }
    assert any(control in problem for problem in unknown_members({"setup": setup}))

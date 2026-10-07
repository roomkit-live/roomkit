"""Every door that answers a room event tells its caller how the answering
agent's turn ended under ``turns``, a cut read there with no error: a room
turn, a Supervisor's task-formulation pass and a synchronous Loop alike
(RMK-529, RFC §6.4, §19.7.4)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.orchestration.strategies.loop import Loop
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="Look up", parameters={"type": "object"})


class _Loops(MockAIProvider):
    """A tool round on every generation: only a limit ends the turn."""

    def __init__(self) -> None:
        super().__init__(streaming=True)

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        return AIResponse(
            content="Still checking.",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
            usage={"input_tokens": 10, "output_tokens": 2},
        )


async def _slow_found(name: str, arguments: dict[str, Any]) -> str:
    await asyncio.sleep(0.05)
    return "found"


def _agent(end: str) -> Agent:
    limits: dict[str, dict[str, Any]] = {
        "completed": {
            "provider": MockAIProvider(responses=["The answer."], streaming=True),
            "max_tool_rounds": 1,
        },
        "max_rounds": {"provider": _Loops(), "max_tool_rounds": 1},
        "budget_exceeded": {
            "provider": _Loops(),
            "max_tool_rounds": 10,
            "turn_budget_tokens": 15,
        },
        "timeout": {
            "provider": _Loops(),
            "max_tool_rounds": 10,
            "tool_loop_timeout_seconds": 0.01,
        },
    }
    return Agent(
        "agent", tools=[LOOKUP], tool_handler=_slow_found, tool_search=False, **limits[end]
    )


async def _answer(door: str, agent: Agent) -> InboundResult:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(agent)
    if door == "room":
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    elif door == "pass1":
        worker = Agent("worker", provider=MockAIProvider(responses=["Worker answer."]))
        kit.register_channel(worker)
        supervisor = Supervisor(
            agent, [worker], strategy="parallel", auto_delegate=True, refine_task=True
        )
        await kit.create_room(room_id="r", orchestration=supervisor)
    else:
        reviewer = Agent("reviewer", provider=MockAIProvider(responses=["APPROVED"]))
        loop = Loop(agent=agent, reviewer=reviewer, max_iterations=2)
        await kit.create_room(room_id="r", orchestration=loop)
    await kit.attach_channel("r", "sms")
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go."))
    result = await asyncio.wait_for(kit.process_inbound(message), timeout=10.0)
    await kit.close()
    return result


@pytest.mark.parametrize("end", ["completed", "max_rounds", "budget_exceeded", "timeout"])
@pytest.mark.parametrize("door", ["room", "pass1", "loop"])
async def test_the_caller_reads_how_the_agents_turn_ended(door: str, end: str) -> None:
    result = await _answer(door, _agent(end))

    turns = dict(result.response_metadata).get("turns", {})
    assert turns["agent"]["loop_end_reason"] == end
    assert "ai_usage" in turns["agent"]
    assert result.error is None

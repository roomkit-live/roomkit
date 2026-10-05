"""A turn's end is reported and logged once, on every door (RMK-513, RFC §15.2).

An expected end (completed, its round cap, a stop) fires no ON_ERROR; a
failure fires one, typed as the provider raised it, where it happened, and is
logged once. The doors: a room turn, a regenerated reply, a supervisor's
task-formulation pass, a delegated turn, a synchronous Loop's producer, and
the turn a background result handed back starts.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.models.steering import Cancel
from roomkit.orchestration.strategies.loop import Loop
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel

_LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})


def _round(n: int) -> AIResponse:
    return AIResponse(
        content="Checking.",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"c{n}", name="lookup", arguments={})],
        usage={"input_tokens": 10, "output_tokens": 2},
    )


def _answers(n: int) -> AIResponse:
    return _round(n) if n % 2 else AIResponse(content="the answer APPROVED")


def _fails(n: int) -> AIResponse:
    if n % 2:
        return _round(n)
    raise ProviderError("upstream 400", provider="mock", status_code=400)


_ENDS: dict[str, tuple[Callable[[int], AIResponse], dict[str, Any]]] = {
    "completed": (_answers, {}),
    "cap": (_round, {"max_tool_rounds": 1}),
    "cancelled": (_round, {"max_tool_rounds": 10}),
    "failure": (_fails, {}),
}


class _Scripted(MockAIProvider):
    def __init__(self, script: Callable[[int], AIResponse]) -> None:
        super().__init__(streaming=True)
        self._script = script

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        return self._script(len(self.calls))


def _agent(channel_id: str, end: str) -> Agent:
    """An agent whose turn ends *end*: a tool round, then its ending."""
    script, options = _ENDS[end]
    agents: list[Agent] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        if end == "cancelled":
            agents[0].steer(Cancel())
        return "found"

    agent = Agent(
        channel_id,
        provider=_Scripted(script),
        tools=[_LOOKUP],
        tool_handler=lookup,
        tool_search=False,
        **options,
    )
    agents.append(agent)
    return agent


class _Errors:
    """What ON_ERROR heard: each failure's room, type and category."""

    def __init__(self, kit: RoomKit) -> None:
        self.heard: list[tuple[str, Any, Any, Any]] = []

        @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
        async def on_error(event: Any, context: Any) -> None:
            metadata = event.metadata or {}
            self.heard.append(
                (
                    event.room_id,
                    metadata.get("error_type"),
                    metadata.get("error_category"),
                    event.correlation_id,
                )
            )


async def _kit() -> tuple[RoomKit, _Errors]:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    return kit, _Errors(kit)


async def _ask(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    )


async def _room_turn(end: str) -> tuple[RoomKit, _Errors]:
    kit, errors = await _kit()
    kit.register_channel(_agent("a", end))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "a", category=ChannelCategory.INTELLIGENCE)
    await _ask(kit)
    return kit, errors


async def _regenerate(end: str) -> tuple[RoomKit, _Errors]:
    kit, errors = await _room_turn(end)
    errors.heard.clear()
    await kit.regenerate_response("r")
    return kit, errors


async def _pass1(end: str) -> tuple[RoomKit, _Errors]:
    kit, errors = await _kit()
    supervisor = _agent("sup", end)
    worker = Agent("w", provider=MockAIProvider(responses=["worker answer"]))
    kit.register_channel(supervisor)
    kit.register_channel(worker)
    strategy = Supervisor(supervisor, [worker], strategy="parallel", auto_delegate=True)
    await kit.create_room(room_id="r", orchestration=strategy)
    await kit.attach_channel("r", "sms")
    await _ask(kit)
    return kit, errors


async def _delegated(end: str) -> tuple[RoomKit, _Errors]:
    kit, errors = await _kit()
    kit.register_channel(_agent("w", end))
    await kit.create_room(room_id="r")
    await kit.delegate("r", "w", "go", wait=True)
    return kit, errors


async def _loop_sync(end: str) -> tuple[RoomKit, _Errors]:
    kit, errors = await _kit()
    producer = _agent("prod", end)
    kit.register_channel(producer)
    reviewer = Agent("rev", provider=MockAIProvider(responses=["APPROVED"]))
    loop = Loop(agent=producer, reviewer=reviewer, max_iterations=1)
    await kit.create_room(room_id="r", orchestration=loop)
    await kit.attach_channel("r", "sms")
    await _ask(kit)
    return kit, errors


async def _hand_back(end: str) -> tuple[RoomKit, _Errors]:
    kit, errors = await _kit()
    boss = _agent("boss", end)
    kit.register_channel(boss)
    kit.register_channel(Agent("w", provider=MockAIProvider(responses=["the work"])))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "boss", category=ChannelCategory.INTELLIGENCE)
    handle = await kit.delegate("r", "w", "go", notify="boss")
    await handle.wait(timeout=5)
    provider = boss._provider
    await until(lambda: bool(provider.calls), timeout=5)
    await asyncio.sleep(0.2)
    return kit, errors


_DOORS: dict[str, Callable[[str], Awaitable[tuple[RoomKit, _Errors]]]] = {
    "room": _room_turn,
    "regenerate": _regenerate,
    "pass-1": _pass1,
    "delegated": _delegated,
    "loop-sync": _loop_sync,
    "hand-back": _hand_back,
}


def _lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        f"{record.name}: {record.getMessage()}"
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name.startswith("roomkit")
    ]


@pytest.mark.parametrize("end", list(_ENDS))
@pytest.mark.parametrize("door", list(_DOORS))
async def test_a_turns_end_is_reported_and_logged_once(
    door: str, end: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="roomkit")
    kit, errors = await _DOORS[door](end)
    await asyncio.sleep(0.1)
    await kit.close()

    if end != "failure":
        # An expected end: no failure to report.
        assert errors.heard == []
        return
    [(_, error_type, category, correlation)] = errors.heard
    assert (error_type, category) == ("ProviderError", "streaming")
    assert correlation
    assert len(_lines(caplog)) <= 1, _lines(caplog)

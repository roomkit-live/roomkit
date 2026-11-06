"""A supervisor serving two rooms keeps their delegations apart (RFC §23.4; RMK-275).

A worker busy with one room's task is free for another room's, and one room's
running pipeline does not hold up another room's call.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.agent import Agent
from roomkit.models.room import Room
from roomkit.orchestration.strategies.supervisor import Supervisor, _inject_strategy
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import current_tool_chain_depth
from tests.orchestration.test_strategy_supervisor import _make_mock_kit
from tests.tool_room import owner_tool_call_in, owner_tools_in, tool_call_in


def _agent(channel_id: str) -> Agent:
    return Agent(channel_id, provider=MockAIProvider(responses=["ok"]))


async def _installed_in_two_rooms(supervisor: Supervisor) -> MagicMock:
    kit = _make_mock_kit(Room(id="tenant-A"))
    task = MagicMock()
    task.id = "task-1"
    kit.delegate = AsyncMock(return_value=task)  # never completes
    await supervisor.install(kit, "tenant-A")
    await supervisor.install(kit, "tenant-B")
    return kit


async def _call(kit: Any, agent: Agent, room_id: str, tool: str) -> Any:
    with owner_tool_call_in(kit, agent.channel_id, room_id):
        return await agent._channel_tool_handler(tool, {"task": "look into it"})


async def test_a_worker_busy_in_one_room_is_free_in_another() -> None:
    boss, researcher = _agent("boss"), _agent("researcher")
    kit = await _installed_in_two_rooms(Supervisor(boss, [researcher], wait_for_result=False))

    first = json.loads(await _call(kit, boss, "tenant-A", "delegate_to_researcher"))
    second = json.loads(await _call(kit, boss, "tenant-B", "delegate_to_researcher"))
    again = json.loads(await _call(kit, boss, "tenant-A", "delegate_to_researcher"))

    assert first["status"] == "delegated"
    assert second["status"] == "delegated"
    assert again["status"] == "already_running"


async def test_one_rooms_pipeline_does_not_hold_up_another_rooms_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()

    async def run_parallel(kit: Any, room_id: str, *args: Any, **kwargs: Any) -> list[Any]:
        if room_id == "tenant-A":
            await release.wait()
        return []

    monkeypatch.setattr(_inject_strategy, "_run_parallel", run_parallel)
    boss = _agent("boss")
    kit = await _installed_in_two_rooms(
        Supervisor(boss, [_agent("researcher")], strategy="parallel")
    )

    room_a = asyncio.ensure_future(_call(kit, boss, "tenant-A", "delegate_workers"))
    await asyncio.sleep(0)
    await asyncio.wait_for(_call(kit, boss, "tenant-B", "delegate_workers"), 1)

    assert not room_a.done()
    release.set()
    await asyncio.wait_for(room_a, 1)


async def test_async_results_continue_the_chain_of_the_turn_that_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RFC §19.7.3, §23.3: the results come back at the dispatching turn's depth."""
    dispatched: list[tuple[str, int]] = []

    async def run_and_deliver(**kwargs: Any) -> None:
        dispatched.append((kwargs["room_id"], current_tool_chain_depth()))
        kwargs["on_done"]()

    monkeypatch.setattr(_inject_strategy, "_async_run_and_deliver", run_and_deliver)
    boss = _agent("boss")
    kit = await _installed_in_two_rooms(
        Supervisor(boss, [_agent("researcher")], strategy="parallel", async_delivery=True)
    )

    with tool_call_in("tenant-A", chain_depth=2), owner_tools_in(kit, "boss", "tenant-A"):
        await boss._channel_tool_handler("delegate_workers", {"task": "look into it"})
    await asyncio.sleep(0)

    assert dispatched == [("tenant-A", 2)]

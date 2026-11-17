"""Task-plan limits and payload values at the Python contract boundaries."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError, asdict
from datetime import UTC, datetime

import pytest

from roomkit import Agent, PlanUpdatedEvent
from roomkit.channels._task_planner import TaskPlanner
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai import AITool, MockAIProvider
from tests.planning_support import answer, message, plan, room
from tests.test_planning_hooks import PlanPublication


@pytest.mark.parametrize(("count", "length"), [(100, 500), (101, 1), (1, 501)])
async def test_task_and_unicode_title_limits_reach_the_actual_tool_result(
    count: int, length: int, streaming: bool
) -> None:
    tasks = [{"title": "é" * length, "status": "pending"} for _ in range(count)]
    kit, _, model = await room([plan(tasks), answer()], streaming)
    try:
        await kit.process_inbound(message())
        result = next(
            part for msg in model.calls[1].messages if msg.role == "tool" for part in msg.content
        )
        body = json.loads(result.result)
        if count == 100:
            assert body == {
                "status": "ok",
                "task_count": 100,
                "pending": 100,
                "in_progress": 0,
                "completed": 0,
                "blocked": 0,
            }
        elif count == 101:
            assert body == {"error": "Invalid plan: at most 100 tasks are allowed"}
        else:
            assert body == {"error": "Invalid plan: task 0 title must be at most 500 characters"}
        assert result.is_error is False
        assert result.outcome == "served"
        ends = [
            e.content
            for e in await kit.store.list_events("r")
            if e.type == EventType.TOOL_CALL_END
        ]
        assert len(ends) == 1 and isinstance(ends[0], ToolCallContent)
        assert ends[0].result == result.result
        assert ends[0].outcome == "served"
    finally:
        await kit.close()


async def test_reading_a_plan_refreshes_its_lru_position() -> None:
    planner = TaskPlanner()
    for index in range(100):
        await planner.handle_plan_tasks(
            {"tasks": [{"title": str(index), "status": "pending"}]}, room_id=f"r{index}"
        )
    assert planner.plan_for("r0") == [{"title": "0", "status": "pending"}]
    await planner.handle_plan_tasks({"tasks": []}, room_id="new")
    assert planner.plan_for("r1") is None
    assert planner.plan_for("r0") == [{"title": "0", "status": "pending"}]
    assert planner.plan_for("new") == []
    assert planner.current_plan is None


def test_plan_event_defaults_are_independent_and_its_public_shape_is_preserved() -> None:
    before = datetime.now(UTC)
    first, second = PlanUpdatedEvent("r", "agent"), PlanUpdatedEvent("r2", "other")
    after = datetime.now(UTC)
    assert first.tasks == second.tasks == []
    first.tasks.append({"title": "one", "status": "pending", "extra": 1})
    assert second.tasks == []
    assert before <= first.timestamp <= after
    assert asdict(first) == {
        "room_id": "r",
        "channel_id": "agent",
        "tasks": first.tasks,
        "timestamp": first.timestamp,
    }
    assert not hasattr(first, "channel_type")
    with pytest.raises(FrozenInstanceError):
        first.room_id = "changed"  # type: ignore[misc]


async def test_the_private_planner_waits_for_its_publisher_before_its_callback() -> None:
    planner, realtime = TaskPlanner(), PlanPublication()
    seen: list[list[dict[str, str]]] = []
    tasks = [{"title": "one", "status": "pending"}]

    async def callback(room_id: str, written: list[dict[str, str]]) -> None:
        assert room_id == "r" and planner.plan_for(room_id) == tasks
        seen.append(written)

    call = asyncio.create_task(
        planner.handle_plan_tasks(
            {"tasks": tasks},
            room_id="r",
            channel_id="agent",
            realtime=realtime,
            on_plan_updated=callback,
        )
    )
    try:
        await asyncio.wait_for(realtime.entered.wait(), 3)
        assert planner.plan_for("r") == tasks and seen == []
        realtime.release.set()
        assert json.loads(await asyncio.wait_for(call, 3))["status"] == "ok"
        assert seen == [tasks]
    finally:
        realtime.release.set()
        await asyncio.gather(call, return_exceptions=True)


async def test_a_private_callback_without_a_backend_can_fail_after_the_plan_is_written() -> None:
    planner = TaskPlanner()
    seen: list[tuple[str, list[dict[str, str]]]] = []

    async def callback(room_id: str, tasks: list[dict[str, str]]) -> None:
        seen.append((room_id, tasks))
        raise RuntimeError("callback failed")

    result = await planner.handle_plan_tasks({}, room_id="r", on_plan_updated=callback)
    assert json.loads(result) == {
        "status": "ok",
        "task_count": 0,
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "blocked": 0,
    }
    assert seen == [("r", [])] and planner.plan_for("r") == []


@pytest.mark.parametrize("agent", [False, True])
def test_enabling_planning_refuses_a_host_tool_of_its_reserved_name(agent: bool) -> None:
    constructor = Agent if agent else AIChannel
    with pytest.raises(ValueError, match="plan_tasks"):
        constructor(
            "agent",
            provider=MockAIProvider(),
            enable_planning=True,
            tools=[AITool(name="plan_tasks", description="host", parameters={})],
        )

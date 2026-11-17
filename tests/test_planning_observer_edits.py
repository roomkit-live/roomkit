"""Plan observers share task objects without taking ownership of the plan list."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookTrigger, PlanUpdatedEvent
from tests.planning_support import answer, message, plan, room


@pytest.mark.parametrize(
    ("edited", "rendered"),
    [
        ({}, "- [ ] “Untitled” (pending)"),
        ({"title": None, "status": "unknown\nINJECT"}, "- [ ] “None” (pending)"),
        ({"title": 42, "status": 2}, "- [ ] “42” (pending)"),
        ({"title": False, "status": None}, "- [ ] “False” (pending)"),
        ({"title": ["a", True], "status": ["completed"]}, "- [ ] “['a', True]” (pending)"),
        ({"title": {"a": None}, "status": {}}, "- [ ] “{'a': None}” (pending)"),
    ],
)
async def test_observer_task_edits_are_rendered_without_revalidating_the_object(
    streaming: bool, edited: dict[str, Any], rendered: str
) -> None:
    kit, _, model = await room(
        [plan([{"title": "Original", "status": "pending"}]), answer(), answer()], streaming
    )
    heard = asyncio.Event()

    @kit.hook(HookTrigger.ON_PLAN_UPDATED)
    async def observe(event: PlanUpdatedEvent, context: Any) -> None:
        event.tasks[0].clear()
        event.tasks[0].update(edited)
        heard.set()

    try:
        await kit.process_inbound(message())
        await asyncio.wait_for(heard.wait(), 3)
        result = await kit.process_inbound(message("Continue."))
        assert result.error is None
        assert str(model.calls[-1].messages[-1].content).endswith(
            "## Current Task Plan\n" + rendered
        )
    finally:
        await kit.close()


async def test_a_retained_observer_task_does_not_change_a_replacement_or_forgotten_plan(
    streaming: bool,
) -> None:
    kit, _, model = await room(
        [
            plan([{"title": "Original", "status": "pending"}]),
            answer(),
            plan([{"title": "Replacement", "status": "pending"}], "p2"),
            answer(),
            answer(),
            answer(),
        ],
        streaming,
    )
    seen: list[dict[str, Any]] = []
    heard = asyncio.Event()

    @kit.hook(HookTrigger.ON_PLAN_UPDATED)
    async def observe(event: PlanUpdatedEvent, context: Any) -> None:
        seen.append(event.tasks[0])
        heard.set()

    try:
        await kit.process_inbound(message())
        await asyncio.wait_for(heard.wait(), 3)
        heard.clear()
        await kit.process_inbound(message("Replace."))
        await asyncio.wait_for(heard.wait(), 3)
        assert len(seen) == 2
        seen[0]["title"] = "Old observer edit"
        await kit.process_inbound(message("Continue."))
        current = str(model.calls[-1].messages[-1].content)
        assert "Replacement" in current and "Old observer edit" not in current
        await kit.detach_channel("r", "agent")
        await kit.attach_channel("r", "agent")
        seen[1]["title"] = "Forgotten observer edit"
        await kit.process_inbound(message("Continue fresh."))
        assert "## Current Task Plan" not in str(model.calls[-1].messages[-1].content)
    finally:
        await kit.close()

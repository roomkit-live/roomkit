"""A Loop reviewer's delegation ends on a terminal status entry, its task's
failure included (RMK-478, RFC §19.7.4).

A review that approves is completed, one that asks for a revision is info,
and one whose task failed is failed: never read as a rejection.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.loop import _execute_loop
from roomkit.providers.ai.mock import MockAIProvider
from tests.loop_review import review_answers


class _Raising(MockAIProvider):
    async def generate(self, context: Any) -> Any:
        raise RuntimeError("reviewer down")


@pytest.mark.parametrize(
    ("reviewer", "level"),
    [
        (review_answers("APPROVED"), StatusLevel.COMPLETED),
        (MockAIProvider(responses=["Needs work."]), StatusLevel.INFO),
        (_Raising(), StatusLevel.FAILED),
    ],
    ids=["approves", "asks-for-revision", "task-failed"],
)
async def test_a_review_ends_on_its_terminal_entry(reviewer: Any, level: StatusLevel) -> None:
    kit = RoomKit()
    writer = Agent("writer", provider=MockAIProvider(responses=["The draft."]))
    editor = Agent("editor", provider=reviewer)
    for agent in (writer, editor):
        kit.register_channel(agent)
    await kit.create_room(room_id="r")

    await _execute_loop(
        kit=kit,
        room_id="r",
        producer=writer,
        reviewers=[editor],
        strategy=None,
        task_desc="Write it.",
        max_iterations=1,
    )
    await asyncio.sleep(0.01)  # the bus records its posts in the background
    entries = await kit.status_bus.recent(5, agent_id="editor")
    await kit.close()

    assert [e.status for e in entries][-1] == level

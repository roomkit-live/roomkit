"""An agent cancels a task of its room with ``cancel_task``, and is not told twice
(RMK-549; RFC §23.3, §23.4).

A request the person changed or dropped ("the weather in Québec… no, Montréal")
must be able to stop its work: the task ends ``cancelled`` as any task cancelled
from outside, and the agent that cancelled it is not handed the cancellation
back, which would have it say so twice. A host's ``kit.cancel_task`` still tells
the agent, which did not ask for it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.models.enums import ChannelCategory, EventType, TaskStatus
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks import CANCEL_TASK_TOOL, CancelTaskTool
from roomkit.tools.context import ToolCallContext, tool_turn_context
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel


class _Endless(MockAIProvider):
    """A worker that never finishes on its own."""

    def __init__(self) -> None:
        super().__init__(responses=["never"])
        self.started = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _Room:
    """Room ``r``: a transport, the speaking agent, a worker; what is handed back
    and which tasks ended."""

    def __init__(self, worker: MockAIProvider) -> None:
        self.kit = RoomKit()
        self.worker = worker
        self.handed_back: list[dict[str, Any]] = []
        self.ended: list[dict[str, Any]] = []

    async def open(self) -> _Room:
        kit = self.kit
        kit.register_channel(SimpleChannel("sms"))
        kit.register_channel(Agent("speaker", provider=MockAIProvider(responses=["Voilà."])))
        kit.register_channel(Agent("worker", provider=self.worker))
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "sms")
        await kit.attach_channel("r", "speaker", category=ChannelCategory.INTELLIGENCE)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
        async def hand_back(event: Any, ctx: Any) -> HookResult:
            self.handed_back.append(dict(event.metadata))
            return HookResult.allow()

        @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
        async def completed(event: Any, ctx: Any) -> None:
            self.ended.append(dict(event.metadata))

        return self

    async def running_task(self, room_id: str = "r") -> Any:
        task = await self.kit.delegate(
            room_id, "worker", "Compte jusqu'à mille.", notify="speaker"
        )
        await asyncio.wait_for(self.worker.started.wait(), 5)
        return task


def _call(channel_id: str = "speaker") -> ToolCallContext:
    return ToolCallContext(room_id="r", tool_call_id="call-1", channel_id=channel_id)


async def _cancel(tool: CancelTaskTool, task_id: str, *, channel_id: str = "speaker") -> Any:
    with tool_turn_context(room_id="r", call=_call(channel_id)):
        return json.loads(await tool.handler(CANCEL_TASK_TOOL, {"task_id": task_id}))


async def test_the_agent_cancels_a_running_task_and_is_not_told_twice() -> None:
    room = await _Room(_Endless()).open()
    task = await room.running_task()

    answer = await _cancel(CancelTaskTool(room.kit), task.id)
    result = await task.wait(timeout=5)
    await until(lambda: bool(room.ended), timeout=5)
    await asyncio.sleep(0.1)  # a hand-back would be under way by now
    entries = [e for e in await room.kit.status_bus.recent(50) if e.action == "task"]
    await room.kit.close()

    assert answer == {
        "task_id": task.id,
        "status": "cancelled",
        "message": "Cancelled: its result will not come back.",
    }
    assert result.status == TaskStatus.CANCELLED
    assert room.ended[0]["task_status"] == TaskStatus.CANCELLED
    assert entries[-1].metadata["task_status"] == "cancelled"
    assert room.handed_back == []  # the tool's answer told it


async def test_a_task_the_host_cancels_is_still_handed_back_to_the_agent() -> None:
    room = await _Room(_Endless()).open()
    task = await room.running_task()

    assert await room.kit.cancel_task(task.id) is True
    await until(lambda: bool(room.handed_back), timeout=5)
    await room.kit.close()

    assert room.handed_back[0]["task_id"] == task.id
    assert room.handed_back[0]["task_status"] == "cancelled"


async def test_a_cancel_by_another_agent_still_tells_the_notified_one() -> None:
    room = await _Room(_Endless()).open()
    task = await room.running_task()

    answer = await _cancel(CancelTaskTool(room.kit), task.id, channel_id="assistant-bis")
    await until(lambda: bool(room.handed_back), timeout=5)
    await room.kit.close()

    assert answer["status"] == "cancelled"
    assert room.handed_back[0]["task_status"] == "cancelled"


async def test_another_rooms_task_is_out_of_reach() -> None:
    room = await _Room(_Endless()).open()
    await room.kit.create_room(room_id="other")
    await room.kit.attach_channel("other", "speaker", category=ChannelCategory.INTELLIGENCE)
    theirs = await room.running_task("other")

    answer = await _cancel(CancelTaskTool(room.kit), theirs.id)
    still_running = theirs.result is None
    await room.kit.close()

    assert answer == {
        "task_id": theirs.id,
        "status": "unknown",
        "message": f"No task {theirs.id} here.",
    }
    assert still_running


async def test_a_task_that_already_ended_is_left_as_it_stands() -> None:
    room = await _Room(MockAIProvider(responses=["Mille."])).open()
    task = await room.kit.delegate("r", "worker", "Compte jusqu'à mille.", notify="speaker")
    await task.wait(timeout=5)
    await until(lambda: bool(room.ended), timeout=5)

    answer = await _cancel(CancelTaskTool(room.kit), task.id)
    await room.kit.close()

    assert answer == {
        "task_id": task.id,
        "status": "completed",
        "message": "The task had already ended.",
    }
    assert task.result is not None and task.result.status == TaskStatus.COMPLETED


async def test_a_call_without_a_task_or_outside_a_conversation_is_refused() -> None:
    kit = RoomKit()
    tool = CancelTaskTool(kit)
    with tool_turn_context(room_id="r"):
        missing = json.loads(await tool.handler(CANCEL_TASK_TOOL, {}))
    outside = json.loads(await tool.handler(CANCEL_TASK_TOOL, {"task_id": "t"}))
    await kit.close()

    assert missing == {"error": "task_id is required"}
    assert "error" in outside


async def test_cancel_task_declines_another_tools_call() -> None:
    kit = RoomKit()
    with pytest.raises(UnservedToolCallError):
        await CancelTaskTool(kit).handler("task_status", {})
    await kit.close()


def test_the_definition_asks_for_the_task_id() -> None:
    definition = CancelTaskTool(RoomKit()).definition
    assert definition["name"] == CANCEL_TASK_TOOL
    assert definition["parameters"]["required"] == ["task_id"]

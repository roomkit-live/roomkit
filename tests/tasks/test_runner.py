"""Tests for InMemoryTaskRunner."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelCategory, ChannelType, TaskStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.tasks.memory import InMemoryTaskRunner
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

# -- Helpers ------------------------------------------------------------------


class _MockBroadcastResult:
    def __init__(self, response_text: str | None = None):
        self.outputs: dict[str, MagicMock] = {}
        if response_text:
            output = MagicMock()
            output.responded = True
            resp_event = RoomEvent(
                room_id="child-1",
                source=EventSource(
                    channel_id="agent-a",
                    channel_type=ChannelType.AI,
                ),
                content=TextContent(body=response_text),
            )
            output.response_events = [resp_event]
            self.outputs["agent-a"] = output
        else:
            output = MagicMock()
            output.responded = False
            output.response_events = []
            self.outputs["agent-a"] = output


def _make_mock_kit(
    child_room: Room | None = None,
    broadcast_result: _MockBroadcastResult | None = None,
) -> MagicMock:
    room = child_room or Room(
        id="child-1",
        metadata={
            "parent_room_id": "parent-1",
            "task_agent_id": "agent-a",
            "task_status": "pending",
        },
    )
    bindings = [
        ChannelBinding(
            channel_id="agent-a",
            room_id="child-1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
    ]

    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.list_bindings = AsyncMock(return_value=bindings)
    kit.store.add_event = AsyncMock()
    kit.store.update_room = AsyncMock()

    router = MagicMock()
    result = broadcast_result or _MockBroadcastResult("task done")
    router.broadcast = AsyncMock(return_value=result)
    kit._get_router = MagicMock(return_value=router)

    return kit


def _make_task(**kwargs: object) -> DelegatedTask:
    defaults = {
        "id": "t1",
        "child_room_id": "child-1",
        "parent_room_id": "parent-1",
        "agent_id": "agent-a",
        "task": "review PR",
    }
    defaults.update(kwargs)
    return DelegatedTask(**defaults)  # type: ignore[arg-type]


# -- Tests --------------------------------------------------------------------


class TestInMemoryTaskRunner:
    async def test_submit_and_wait_completed(self):
        kit = _make_mock_kit()
        runner = InMemoryTaskRunner()
        task = _make_task()

        await runner.submit(kit, task)
        result = await task.wait(timeout=5.0)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == "task done"
        assert result.task_id == "t1"
        assert result.child_room_id == "child-1"
        assert result.parent_room_id == "parent-1"
        assert result.duration_ms > 0

    async def test_on_complete_callback_called(self):
        kit = _make_mock_kit()
        runner = InMemoryTaskRunner()
        task = _make_task()

        callback_results: list[DelegatedTaskResult] = []

        async def on_complete(r: DelegatedTaskResult) -> None:
            callback_results.append(r)

        await runner.submit(kit, task, on_complete=on_complete)
        await task.wait(timeout=5.0)

        assert len(callback_results) == 1
        assert callback_results[0].status == TaskStatus.COMPLETED

    async def test_agent_returns_nothing_fails(self):
        kit = _make_mock_kit(broadcast_result=_MockBroadcastResult(None))
        runner = InMemoryTaskRunner()
        task = _make_task()

        await runner.submit(kit, task)
        result = await task.wait(timeout=5.0)

        assert result.status == TaskStatus.FAILED

    async def test_cancel_running_task(self):
        kit = _make_mock_kit()

        # Make broadcast hang so we can cancel
        async def slow_broadcast(*args: object, **kwargs: object) -> _MockBroadcastResult:
            await asyncio.sleep(10)
            return _MockBroadcastResult("should not reach")

        kit._get_router().broadcast = slow_broadcast

        runner = InMemoryTaskRunner()
        task = _make_task()

        await runner.submit(kit, task)
        await asyncio.sleep(0.05)  # let it start

        cancelled = await runner.cancel("t1")
        assert cancelled is True
        assert task.status == TaskStatus.CANCELLED

    async def test_cancel_unknown_task(self):
        runner = InMemoryTaskRunner()
        assert await runner.cancel("nonexistent") is False

    async def test_close_cancels_all_inflight(self):
        kit = _make_mock_kit()

        async def slow_broadcast(*args: object, **kwargs: object) -> _MockBroadcastResult:
            await asyncio.sleep(10)
            return _MockBroadcastResult("should not reach")

        kit._get_router().broadcast = slow_broadcast

        runner = InMemoryTaskRunner()
        t1 = _make_task(id="t1")
        t2 = _make_task(id="t2")

        await runner.submit(kit, t1)
        await runner.submit(kit, t2)
        await asyncio.sleep(0.05)

        await runner.close()

        assert t1.status == TaskStatus.CANCELLED
        assert t2.status == TaskStatus.CANCELLED

    async def test_child_room_metadata_updated(self):
        kit = _make_mock_kit()
        runner = InMemoryTaskRunner()
        task = _make_task()

        await runner.submit(kit, task)
        await task.wait(timeout=5.0)

        # Should have called update_room at least twice (in_progress + completed)
        assert kit.store.update_room.call_count >= 2

"""Tests for DelegatedTask and DelegatedTaskResult models."""

from __future__ import annotations

import asyncio

import pytest

from roomkit.models.enums import TaskStatus
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult


class TestDelegatedTaskResult:
    def test_defaults(self):
        result = DelegatedTaskResult(
            task_id="t1",
            child_room_id="child-1",
            parent_room_id="parent-1",
            agent_id="agent-a",
        )
        assert result.status == TaskStatus.COMPLETED
        assert result.output is None
        assert result.error is None
        assert result.duration_ms == 0.0
        assert result.metadata == {}

    def test_serialization_round_trip(self):
        result = DelegatedTaskResult(
            task_id="t1",
            child_room_id="child-1",
            parent_room_id="parent-1",
            agent_id="agent-a",
            status=TaskStatus.FAILED,
            error="timeout",
            duration_ms=1234.5,
            metadata={"key": "value"},
        )
        data = result.model_dump()
        restored = DelegatedTaskResult.model_validate(data)
        assert restored == result

    def test_json_round_trip(self):
        result = DelegatedTaskResult(
            task_id="t1",
            child_room_id="child-1",
            parent_room_id="parent-1",
            agent_id="agent-a",
            output="review complete",
        )
        json_str = result.model_dump_json()
        restored = DelegatedTaskResult.model_validate_json(json_str)
        assert restored.output == "review complete"


class TestDelegatedTask:
    def _make_task(self, **kwargs: object) -> DelegatedTask:
        defaults = {
            "id": "t1",
            "child_room_id": "child-1",
            "parent_room_id": "parent-1",
            "agent_id": "agent-a",
            "task": "review PR",
        }
        defaults.update(kwargs)
        return DelegatedTask(**defaults)  # type: ignore[arg-type]

    def test_initial_state(self):
        t = self._make_task()
        assert t.status == TaskStatus.PENDING
        assert t.result is None
        assert t.id == "t1"
        assert t.task == "review PR"

    async def test_wait_returns_after_set_result(self):
        t = self._make_task()
        result = DelegatedTaskResult(
            task_id="t1",
            child_room_id="child-1",
            parent_room_id="parent-1",
            agent_id="agent-a",
            output="done",
        )

        async def _set_later():
            await asyncio.sleep(0.01)
            t._set_result(result)

        asyncio.create_task(_set_later())
        got = await t.wait(timeout=2.0)
        assert got.output == "done"
        assert t.status == TaskStatus.COMPLETED

    async def test_wait_timeout(self):
        t = self._make_task()
        with pytest.raises(asyncio.TimeoutError):
            await t.wait(timeout=0.01)

    def test_cancel(self):
        t = self._make_task()
        t.cancel()
        assert t.status == TaskStatus.CANCELLED
        assert t.result is not None
        assert t.result.status == TaskStatus.CANCELLED

    def test_cancel_idempotent(self):
        t = self._make_task()
        t.cancel()
        t.cancel()  # second cancel is a no-op
        assert t.status == TaskStatus.CANCELLED

    async def test_wait_after_cancel(self):
        t = self._make_task()
        t.cancel()
        result = await t.wait(timeout=1.0)
        assert result.status == TaskStatus.CANCELLED

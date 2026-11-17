"""Tests for the public background setup and delegated task models."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from roomkit.models.enums import TaskStatus
from roomkit.tasks import BackgroundSetup
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult


class TestBackgroundSetup:
    def test_default_worker_and_hand_back(self) -> None:
        setup = BackgroundSetup()
        assert setup.worker is None
        assert setup.hand_back == "relay"

    @pytest.mark.parametrize("worker", ["", " leading", "a,b"])
    def test_worker_text_is_preserved(self, worker: str) -> None:
        setup = BackgroundSetup(worker=worker, hand_back="post")
        assert setup.worker == worker
        assert setup.hand_back == "post"

    def test_unknown_hand_back_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError) as raised:
            BackgroundSetup(hand_back="bad")
        assert str(raised.value) == "hand_back must be one of ('relay', 'post'), not 'bad'"

    def test_setup_is_frozen(self) -> None:
        setup = BackgroundSetup()
        with pytest.raises(FrozenInstanceError):
            setup.worker = "other"  # type: ignore[misc]


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

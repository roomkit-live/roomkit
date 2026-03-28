"""Tests for roomkit.core.task_utils."""

from __future__ import annotations

import asyncio
import contextlib

from roomkit.core.task_utils import log_task_exception


class TestLogTaskException:
    async def test_logs_exception(self, caplog):
        async def _fail():
            raise ValueError("boom")

        task = asyncio.create_task(_fail())
        with contextlib.suppress(ValueError):
            await task
        with caplog.at_level("ERROR", logger="roomkit.tasks"):
            log_task_exception(task)
        assert "boom" in caplog.text

    async def test_no_log_on_success(self, caplog):
        async def _ok():
            return 42

        task = asyncio.create_task(_ok())
        await task
        with caplog.at_level("ERROR", logger="roomkit.tasks"):
            log_task_exception(task)
        assert caplog.text == ""

    async def test_no_log_on_cancel(self, caplog):
        async def _hang():
            await asyncio.sleep(999)

        task = asyncio.create_task(_hang())
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with caplog.at_level("ERROR", logger="roomkit.tasks"):
            log_task_exception(task)
        assert caplog.text == ""

"""Delegating a report in a child task still counts as reporting that call."""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit.tools import PolicyExternalToolHandler
from tests.test_refused_reports import _acp


class ChildReport(PolicyExternalToolHandler):
    async def on_tool_result(
        self, tool_name: str, tool_input: dict[str, Any], result: str, **kwargs: Any
    ) -> None:
        await asyncio.create_task(super().on_tool_result(tool_name, tool_input, result, **kwargs))
        raise RuntimeError("audit sink down")


async def test_a_child_task_report_then_failure_does_not_report_the_call_twice() -> None:
    heard = await _acp(ChildReport(), status="completed", raw_output="read")
    assert len(heard.sync) == 1
    assert len(heard.observed) == 1
    assert heard.observed[0].result == "read"


async def test_concurrent_handler_reports_with_the_same_id_have_separate_witnesses() -> None:
    first_reported, second_reporting = asyncio.Event(), asyncio.Event()

    class First(PolicyExternalToolHandler):
        async def on_tool_result(
            self, name: str, arguments: Any, result: str, **kwargs: Any
        ) -> None:
            await super().on_tool_result(name, arguments, result, **kwargs)
            first_reported.set()
            await second_reporting.wait()
            raise RuntimeError("audit sink down")

    class Second(PolicyExternalToolHandler):
        async def on_tool_result(
            self, name: str, arguments: Any, result: str, **kwargs: Any
        ) -> None:
            second_reporting.set()
            raise RuntimeError("audit sink down")

    first = asyncio.create_task(_acp(First(), status="completed", raw_output="original"))
    await asyncio.wait_for(first_reported.wait(), 3)
    second = asyncio.create_task(_acp(Second(), status="completed", raw_output="original"))
    heard = await asyncio.wait_for(asyncio.gather(first, second), 3)
    for report in heard:
        assert len(report.sync) == 1
        assert len(report.observed) == 1
        assert report.observed[0].tool_call_id == "tool-1"
        assert report.observed[0].result == "original"

"""Owner tools carry independent records and merge their changes (RFC §10.6.5)."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import pytest

from roomkit.channels._tool_registry import orchestration_tool
from roomkit.core.exceptions import ToolRefusedError, UnservedToolCallError
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.tools.context import current_response_metadata, current_tool_call
from tests.test_turns_owner_tools import PROBE, _agent, _call, _kit, _reports, _say


@pytest.mark.parametrize("streaming", [False, True])
async def test_parallel_owner_tools_merge_only_their_changes(streaming: bool) -> None:
    both = asyncio.Barrier(2)
    seen: dict[str, Any] = {}

    async def serve(arguments: dict[str, Any]) -> str:
        record = current_response_metadata()
        call = current_tool_call()
        assert record is not None and call is not None
        if arguments.get("seed"):
            record.update(names={"a": 1}, stale=True)
            return "seeded"
        name = arguments["name"]
        seen[name] = (record, deepcopy(record["names"]))
        await asyncio.wait_for(both.wait(), 2)
        record["names"][name] = 1
        if name == "b":
            del record["names"]["a"]
            del record["stale"]
        else:
            record["new"] = "kept"
        call.structured_content = {"name": name}
        await asyncio.wait_for(both.wait(), 2)
        assert set(record["names"]) == ({"b"} if name == "b" else {"a", "c"})
        return name

    parallel = AIResponse(
        content="",
        tool_calls=[
            AIToolCall(id="b", name="probe", arguments={"name": "b"}),
            AIToolCall(id="c", name="probe", arguments={"name": "c"}),
        ],
    )
    agent = _agent("agent", _call("probe", {"seed": True}), parallel, "done", streaming=streaming)
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    reports = _reports(kit)
    try:
        result = await asyncio.wait_for(_say(kit, "go"), 5)
        assert seen["b"][0] is not seen["c"][0]
        assert seen["b"][1] == seen["c"][1] == {"a": 1}
        assert result.response_metadata["names"] == {"b": 1, "c": 1}
        assert result.response_metadata["new"] == "kept"
        assert "stale" not in result.response_metadata
        copies = {report.tool_call_id: report.structured_content for report in reports}
        assert copies == {"c1": None, "b": {"name": "b"}, "c": {"name": "c"}}
        assert len(agent.provider.calls) == 3
    finally:
        await kit.close()


@pytest.mark.parametrize("outcome", ["served", "refused", "failed", "unserved"])
async def test_owner_record_changes_precede_every_outcome(outcome: str) -> None:
    async def serve(arguments: dict[str, Any]) -> str:
        record = current_response_metadata()
        call = current_tool_call()
        assert record is not None and call is not None
        record["learned"] = outcome
        call.structured_content = {"private": outcome}
        if outcome == "refused":
            raise ToolRefusedError("refused")
        if outcome == "failed":
            raise ValueError("failed")
        if outcome == "unserved":
            raise UnservedToolCallError("unserved")
        return "served"

    agent = _agent("agent", _call("probe", {}), "done")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    reports = _reports(kit)
    try:
        result = await _say(kit, "go")
        assert result.response_metadata["learned"] == outcome
        [report] = reports
        assert report.is_error == (outcome != "served")
        assert report.structured_content == ({"private": outcome} if outcome == "served" else None)
        assert len(agent.provider.calls) == 2
    finally:
        await kit.close()

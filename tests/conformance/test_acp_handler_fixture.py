"""Handler fixture mistakes must fail even when ACP catches callback errors."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._acp_handler import ScriptedExternalHandler, handler_problems


@pytest.mark.parametrize(
    "spec",
    [
        None,
        [],
        {"decision": {}},
        {"decisions": None},
        {"decisions": [{}]},
        {"decisions": [{"approved": "false"}]},
        {"decisions": [{"approved": True, "result": 1}]},
        {"decisions": [{"raise": "broken", "approved": True}]},
        {"policy": []},
        {"policy": {"deny": "Read"}},
        {"reports": "ignore"},
    ],
)
def test_handler_fixture_rejects_malformed_controls(spec: Any) -> None:
    assert handler_problems(spec, "handler")


async def test_handler_fixture_records_exhaustion_separately_from_a_decision_failure() -> None:
    handler = ScriptedExternalHandler({"decisions": [{"raise": "approval service down"}]})
    with pytest.raises(RuntimeError, match="approval service down"):
        await handler.process_tool_call("Read", {})
    assert handler.problems == []
    with pytest.raises(RuntimeError, match="fixture exhausted"):
        await handler.process_tool_call("Read", {})
    assert handler.problems == ["external_tool_handler.decisions exhausted"]


async def test_handler_fixture_snapshots_actual_arguments() -> None:
    handler = ScriptedExternalHandler({})
    arguments = {"nested": {"value": "original"}}
    assert (await handler.process_tool_call("Read", arguments, room_id="r")).approved
    arguments["nested"]["value"] = "changed"
    assert handler.calls == [
        {
            "method": "process_tool_call",
            "channel_id": "",
            "name": "Read",
            "arguments": {"nested": {"value": "original"}},
            "room_id": "r",
        }
    ]

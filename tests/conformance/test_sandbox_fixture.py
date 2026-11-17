"""Sandbox controls cannot hide host entry or silently lose misspelled input."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._sandbox import ScriptedSandbox
from tests.conformance._shape import unknown_members


async def test_the_fixture_records_before_failure_and_does_not_filter_tools() -> None:
    record: dict[str, Any] = {"calls": [], "closes": 0}
    fixture = ScriptedSandbox(
        {
            "tools": [{"name": "not_a_sandbox_tool"}],
            "raises": {"kind": "ConnectionError", "message": "unavailable"},
        },
        record,
    )
    assert fixture.tool_definitions() == [{"name": "not_a_sandbox_tool"}]
    with pytest.raises(Exception, match="unavailable") as caught:
        await fixture.execute("unfiltered", {"path": "../outside"})
    assert type(caught.value).__name__ == "ConnectionError"
    assert record["calls"] == [{"command": "unfiltered", "arguments": {"path": "../outside"}}]
    await fixture.close()
    assert record["closes"] == 1


async def test_the_fixture_preserves_none_and_empty_arguments_and_default_result() -> None:
    record: dict[str, Any] = {"calls": [], "closes": 0}
    fixture = ScriptedSandbox({}, record)
    assert fixture.tool_definitions() == []
    for args in (None, {}):
        result = await fixture.execute("read", args)
        assert result.model_dump() == {
            "exit_code": 0,
            "output": "",
            "error": "",
            "metadata": {},
            "success": True,
        }
    assert record["calls"] == [
        {"command": "read", "arguments": None},
        {"command": "read", "arguments": {}},
    ]


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {"toools": []},
        {"tools": None},
        {"tools": [None]},
        {"tools": [{}]},
        {"tools": [{"name": 1}]},
        {"tools": [{"name": "sandbox_x", "parameters": None}]},
        {"result": None},
        {"result": {}},
        {"result": {"exit_code": True}},
        {"result": {"exit_code": 0, "success": False}},
        {"result": {"exit_code": 0, "metadata": None}},
        {"raises": None},
        {"raises": {"kind": "Error"}},
        {"raises": {"kind": 1, "message": "x"}},
    ],
)
def test_sandbox_controls_refuse_invalid_or_unknown_members(spec: Any) -> None:
    scenario = {"setup": {"channels": [{"id": "agent", "type": "ai", "sandbox": spec}]}}
    assert unknown_members(scenario)


def test_a_direct_fixture_refuses_null() -> None:
    with pytest.raises(ValueError, match="sandbox: not an object"):
        ScriptedSandbox(None, {"calls": [], "closes": 0})  # type: ignore[arg-type]

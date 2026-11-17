"""Declared callback scripts never silently ignore misspelled contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.conformance._shape import unknown_members


@pytest.mark.parametrize("configuration", [False, True], ids=["channel", "turn-config"])
@pytest.mark.parametrize("nested", [False, True], ids=["script", "call"])
def test_unknown_host_call_members_are_refused(configuration: bool, nested: bool) -> None:
    script = {"calls": [{"verb": "app.read"}]}
    if nested:
        script["calls"][0]["argument"] = {}
    else:
        script["concurrant"] = True
    tool = {"name": "lookup", "host_calls": script}
    channel = {"id": "agent", "type": "ai"}
    if configuration:
        channel["config_provider"] = {"r1": {"tool_handler": [tool]}}
    else:
        channel["tools"] = [tool]

    assert unknown_members({"setup": {"channels": [channel]}})


@pytest.mark.parametrize("nested", [False, True], ids=["handler", "error"])
def test_unknown_declared_handler_members_are_refused(nested: bool) -> None:
    handler = {"raises": {"kind": "PermissionError", "message": "private"}}
    invalid = deepcopy(handler)
    if nested:
        invalid["raises"]["messege"] = "private"
    else:
        invalid["writes_records"] = {}
    scenario = {"steps": [{"call": "declare_host_verb", "name": "app.read", "handler": invalid}]}

    assert unknown_members(scenario)
    scenario["steps"][0]["handler"] = handler
    assert unknown_members(scenario) == []

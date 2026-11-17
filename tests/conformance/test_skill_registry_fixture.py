"""Broken server scripts cannot pass as sanitized registry errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conformance._harness import Harness
from tests.conformance._skill_registry_shape import registry_problems


@pytest.mark.parametrize(
    "update",
    [
        {"servers": []},
        *[
            {"servers": {"remote": {"replies": [], "approvals": value}}}
            for value in [None, True, [1], [{}], [{"error": 1}], [{"error": "x", "extra": 1}]]
        ],
        {"servers": {"remote": {"replies": [], "reply": {}}}},
        {"servers": {"remote": {"replies": [], "extensions": {"skills": None}}}},
        {"servers": {"remote": {"replies": [{"method": "skills/list", "result": {}}]}}},
        {
            "servers": {
                "remote": {
                    "replies": [
                        {"method": "skills/list", "params": {}, "result": {}, "error": "x"}
                    ]
                }
            }
        },
        {"operations": []},
        {"operations": [{"method": "unknown"}]},
        {"operations": [{"method": "add_mcp_server", "server": "remote"}]},
        {"operations": [{"method": "copy", "target": "copy", "marks": 1}]},
        {"operations": [{"method": "copy", "target": "copy", "names": "all"}]},
        {"operations": [{"method": "snapshot", "hidden": True}]},
        {"operations": [{"method": "install_channel", "channel": None}]},
        {"operations": [{"method": "install_channel", "channel": {"id": "agent"}}]},
        {"operations": [{"method": "install_channel", "channel": {"model": {}, "skills": []}}]},
        {"operations": [{"method": "room_call", "action": None}]},
    ],
)
def test_malformed_registry_controls_are_refused(update: dict[str, Any]) -> None:
    spec = {"servers": {}, "operations": [{"method": "snapshot"}], **update}
    assert registry_problems(spec, "skill_registry_session")


@pytest.mark.parametrize(
    "replies",
    [
        [],
        [{"method": "another", "params": {}, "result": {}}],
        [{"method": "skills/list", "params": {"cursor": "wrong"}, "result": {}}],
        [
            {"method": "skills/list", "params": {}, "result": {"skills": []}},
            {"method": "skills/list", "params": {}, "result": {"skills": []}},
        ],
    ],
)
async def test_wrong_server_scripts_fail_outside_expected_product_errors(
    replies: list[dict[str, Any]],
) -> None:
    scenario = {
        "steps": [
            {
                "call": "skill_registry_session",
                "servers": {"remote": {"replies": replies}},
                "operations": [{"method": "add_mcp_server", "server": "remote", "label": "docs"}],
            }
        ]
    }
    problems = await Harness(scenario).run()
    assert any("the runner cannot read the call" in problem for problem in problems)


@pytest.mark.parametrize(
    "operations",
    [
        [{"method": "snapshot", "registry": "missing"}],
        [{"method": "add_mcp_server", "server": "missing", "label": "docs"}],
        [{"method": "copy", "target": "main"}],
    ],
)
async def test_unknown_fixture_objects_are_not_sdk_refusals(
    operations: list[dict[str, Any]],
) -> None:
    scenario = {
        "steps": [{"call": "skill_registry_session", "servers": {}, "operations": operations}]
    }
    problems = await Harness(scenario).run()
    assert any("the runner cannot read the call" in problem for problem in problems)


async def test_a_refusal_cannot_pass_as_an_expected_null_result() -> None:
    path = Path(__file__).parent / "scenarios/served-skills/only-full-copies-fetch-unlisted.json"
    scenario = json.loads(path.read_text())
    scenario["steps"][0]["expect"]["result"]["results"][3] = {"status": "ok", "result": None}
    problems = await Harness(scenario).run()
    assert any("status" in problem for problem in problems)


async def test_a_sanitized_load_error_cannot_hide_a_wrong_resource_request() -> None:
    path = Path(__file__).parent / "scenarios/served-skills/load-refuses-size.json"
    scenario = json.loads(path.read_text())
    scenario["steps"][0]["servers"]["remote"]["replies"][1]["method"] = "wrong/read"
    problems = await Harness(scenario).run()
    assert any("the runner cannot read the call" in problem for problem in problems)


async def test_approval_exhaustion_cannot_hide_behind_an_execution_refusal() -> None:
    path = Path(__file__).parent / "scenarios/served-skills/execution-absent-approval.json"
    scenario = json.loads(path.read_text())
    scenario["steps"][0]["servers"]["remote"]["approvals"] = []
    problems = await Harness(scenario).run()
    assert any("the runner cannot read the call" in problem for problem in problems)

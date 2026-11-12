"""Malformed ACP fixtures fail rather than silently changing their behavior."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.conformance._acp import ScriptedACPAgent, ScriptedACPTransport
from tests.conformance._script import Gates
from tests.conformance._shape import unknown_members


@pytest.mark.parametrize(
    "transport",
    [
        {"refuse_prompts": []},
        {"refuse_prompt": [{"reason": "unusable", "authorized": True}]},
        {"refuse_prompt": [{"reason": "unusable", "after": True}]},
    ],
)
def test_recovery_fixture_rejects_unknown_members(transport: dict[str, Any]) -> None:
    scenario = {
        "setup": {
            "channels": [
                {
                    "id": "agent",
                    "type": "ai",
                    "acp": {"cwd": "/srv/project", "agent": {}, "transport": transport},
                }
            ]
        }
    }
    assert unknown_members(scenario)


async def test_scripted_transport_closes_its_sdk_tasks() -> None:
    """Closing a fixture must stop its SDK sender as well as its agent pipe."""
    before = asyncio.all_tasks()
    transport = ScriptedACPTransport(ScriptedACPAgent({}, Gates()), {})
    connection = await transport.open(object(), queue=None)
    await connection.initialize(protocol_version=1)
    await transport.close()
    assert not (asyncio.all_tasks() - before)


@pytest.mark.parametrize("directories", [None, "/srv/shared", {}, [1], ["/srv/shared", False]])
def test_directory_fixture_rejects_malformed_lists(directories: Any) -> None:
    scenario = {
        "setup": {
            "channels": [
                {
                    "id": "agent",
                    "type": "ai",
                    "acp": {
                        "cwd": "/srv/project",
                        "agent": {},
                        "additional_directories": directories,
                    },
                }
            ]
        }
    }
    assert any("additional_directories" in problem for problem in unknown_members(scenario))

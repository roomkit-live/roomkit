"""Broken context scripts cannot masquerade as the channel's fail-open path."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._acp_context import ScriptedContextContributor, context_problems
from tests.conformance._harness import Harness
from tests.conformance._script import Gates
from tests.conformance._shape import unknown_members


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {},
        {"responses": None},
        {"responses": [None]},
        {"responses": [{"blocks": "not a list"}]},
        {"responses": [{"blocks": [12]}]},
        {"responses": [{"blocks": [], "error": "both"}]},
        {"responses": [{"error": 12}]},
        {"responses": [{"blocks": [], "gate": None}]},
        {"responses": [{"blocks": [], "unknown": True}]},
    ],
)
def test_malformed_context_is_refused_before_running(spec: Any) -> None:
    assert context_problems(spec, "acp.context_contributor")
    with pytest.raises(ValueError):
        ScriptedContextContributor(spec, Gates())


def _scenario(**options: Any) -> dict[str, Any]:
    return {
        "setup": {
            "channels": [
                {"id": "ws", "type": "websocket", "category": "transport"},
                {"id": "agent", "type": "ai", "acp": {"cwd": "/srv", "agent": {}, **options}},
            ],
            "rooms": [{"id": "room", "bindings": [{"channel_id": "ws"}, {"channel_id": "agent"}]}],
        },
        "steps": [
            {
                "call": "process_inbound",
                "room_id": "room",
                "message": {
                    "channel_id": "ws",
                    "sender_id": "alice",
                    "content": {"type": "text", "body": "Hello"},
                },
            }
        ],
    }


@pytest.mark.parametrize("value", [True, -1, "20", 0.5, None])
def test_malformed_history_window_is_a_fixture_failure(value: Any) -> None:
    assert any("room_history" in error for error in unknown_members(_scenario(room_history=value)))


async def test_callback_exhaustion_is_not_a_successful_fail_open_turn() -> None:
    problems = await Harness(_scenario(context_contributor={"responses": []})).run()
    assert any("context_contributor: callback script exhausted" in problem for problem in problems)

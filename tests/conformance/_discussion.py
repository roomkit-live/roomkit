"""A discussion built and read (README, "Calls"; RFC §19.7.5).

``discussion`` builds the implementation's Discussion from its keywords, each
agent a channel id or an agent given live, and reads it, installed nowhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import Agent, RoomKit
from roomkit.channels.declared import id_of
from roomkit.orchestration import Discussion

if TYPE_CHECKING:
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness

_DEFAULTED = ("handles", "people", "everyone", "max_turns", "max_depth", "rules")
"""The keywords a call may leave out, each None when it does."""


def _agent(spec: str | Mapping[str, Any]) -> str | Agent:
    if isinstance(spec, str):
        return spec
    return Agent(spec["id"], name=spec.get("name"))


async def _discussion(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    built = Discussion(
        [_agent(agent) for agent in spec["agents"]],
        addressed_only=spec.get("addressed_only", False),
        silent_token=spec.get("silent_token", "(silent)"),
        **{keyword: spec.get(keyword) for keyword in _DEFAULTED},
    )
    return {"agents": [id_of(agent) for agent in built.agents()], "handles": built.handles()}


DISCUSSION_CALLS: dict[str, Call] = {"discussion": _discussion}

DISCUSSION_ARGUMENTS: dict[str, set[str]] = {
    "discussion": {
        "agents",
        "handles",
        "people",
        "addressed_only",
        "everyone",
        "max_turns",
        "max_depth",
        "silent_token",
        "rules",
    },
}

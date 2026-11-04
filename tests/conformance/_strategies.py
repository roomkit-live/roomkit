"""Scripted strategies installed on a room (README, "Setup"; RFC §19.7).

A scripted strategy is a host's own: the agents it names by their channel
ids, what its install does (a gate, kit calls made as a reaction's, then a
refusal of the room) and what its own uninstall step does. A strategy with a
``loop`` is the implementation's Loop instead, its agents named by their ids
(RFC §19.7.4), and one with a ``supervisor`` the implementation's Supervisor
(RFC §19.7.3). One object stands for each id, whatever the room it is
installed in.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.orchestration import Loop, Pipeline, Supervisor, Swarm
from roomkit.orchestration.base import Orchestration
from tests.conformance._discussion import installed_discussion
from tests.conformance._script import perform

if TYPE_CHECKING:
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness

STRATEGY = {
    "id",
    "agents",
    "install",
    "uninstall",
    "loop",
    "supervisor",
    "swarm",
    "pipeline",
    "discussion",
}
SWARM = {"agents", "entry"}
PIPELINE = {"agents", "supervisor_id"}
STRATEGY_STEP = {"gate", "do", "refuse"}
STRATEGY_UNINSTALL = {"gate", "do"}
LOOP = {"agent", "reviewer", "reviewers", "max_iterations", "strategy"}
SUPERVISOR = {
    "supervisor",
    "workers",
    "strategy",
    "share_channels",
    "task_timeout_ms",
    "max_revisions",
    "auto_delegate",
    "refine_task",
    "refine_instruction",
    "delegation_message",
    "wait_for_result",
}


class ScriptedStrategy(Orchestration):
    """A scripted strategy: it attaches its agents by id, then does what
    its install says, as a reaction does, in the room it is installed in."""

    def __init__(self, spec: Mapping[str, Any], harness: Harness) -> None:
        self.strategy_id: str = spec["id"]
        self._agents: list[str] = list(spec.get("agents", []))
        self._install: Mapping[str, Any] = spec.get("install", {})
        self._uninstall: Mapping[str, Any] = spec.get("uninstall", {})
        self._harness = harness

    def agents(self) -> list[str]:
        return list(self._agents)

    async def install(self, kit: RoomKit, room_id: str) -> None:
        await _step(self._harness, self._install, room_id)
        if (reason := self._install.get("refuse")) is not None:
            raise ValueError(reason)

    async def uninstall(self, kit: RoomKit, room_id: str) -> None:
        await _step(self._harness, self._uninstall, room_id)


async def _step(harness: Harness, step: Mapping[str, Any], room_id: str) -> None:
    """Hold at the step's gate, then make its actions, ``$room`` the room."""
    reaction = {key: step[key] for key in ("gate", "do") if key in step}
    await perform(harness, reaction, {"room": room_id})


def _loop(spec: Mapping[str, Any]) -> Loop:
    """The implementation's Loop a strategy's ``loop`` builds, its agents
    named by their channel ids."""
    named = ("reviewer", "reviewers", "max_iterations", "strategy")
    options: dict[str, Any] = {key: spec[key] for key in named if key in spec}
    return Loop(spec["agent"], **options)


def _supervisor(spec: Mapping[str, Any]) -> Supervisor:
    """The implementation's Supervisor a strategy's ``supervisor`` builds, its
    agents named by their channel ids, its task timeout given in
    milliseconds."""
    named = (
        "strategy",
        "share_channels",
        "max_revisions",
        "auto_delegate",
        "refine_task",
        "refine_instruction",
        "delegation_message",
        "wait_for_result",
    )
    options: dict[str, Any] = {key: spec[key] for key in named if key in spec}
    if "task_timeout_ms" in spec:
        options["task_timeout"] = spec["task_timeout_ms"] / 1000
    return Supervisor(spec["supervisor"], spec["workers"], **options)


def _strategy(spec: Mapping[str, Any], harness: Harness) -> Orchestration:
    """The strategy *spec* describes: a built-in one, or a scripted one."""
    if "loop" in spec:
        return _loop(spec["loop"])
    if "supervisor" in spec:
        return _supervisor(spec["supervisor"])
    if "swarm" in spec:
        return Swarm(spec["swarm"]["agents"], entry=spec["swarm"].get("entry"))
    if "pipeline" in spec:
        pipeline = spec["pipeline"]
        return Pipeline(pipeline["agents"], supervisor_id=pipeline.get("supervisor_id"))
    if "discussion" in spec:
        return installed_discussion(spec["id"], spec["discussion"], harness)
    return ScriptedStrategy(spec, harness)


def strategies(setup: Mapping[str, Any], harness: Harness) -> dict[str, Orchestration]:
    """The scripted strategies of a scenario's setup, by id."""
    return {spec["id"]: _strategy(spec, harness) for spec in setup.get("strategies", [])}


def kit_orchestration(kit: Mapping[str, Any], harness: Harness) -> dict[str, Any]:
    """The kit's default strategy a scenario's ``setup.kit`` names."""
    if "orchestration" not in kit:
        return {}
    return {"orchestration": harness.strategy(kit["orchestration"])}


def room_orchestration(spec: Mapping[str, Any], harness: Harness) -> dict[str, Any]:
    """The strategy a ``create_room`` call names: an id, or null for none."""
    if "orchestration" not in spec:
        return {}
    named = spec["orchestration"]
    return {"orchestration": None if named is None else harness.strategy(named)}


def _org(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {"organization_id": spec["organization_id"]} if "organization_id" in spec else {}


async def _install(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    strategy = harness.strategy(spec["strategy"])
    return await kit.install_strategy(spec["room_id"], strategy, **_org(spec))


async def _uninstall(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.uninstall_strategy(spec["room_id"], **_org(spec))


async def _claim(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    strategy = harness.strategy(spec["strategy"])
    return await kit.claim_room_strategy(spec["room_id"], strategy, **_org(spec))


async def _room_strategy(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    strategy = kit.room_strategy(spec["room_id"], **_org(spec))
    if strategy is None:
        return None
    for strategy_id, named in harness.strategies.items():
        if named is strategy:
            return strategy_id
    raise TypeError(f"room {spec['room_id']} holds a strategy no scenario named")


STRATEGY_CALLS: dict[str, Call] = {
    "install_strategy": _install,
    "uninstall_strategy": _uninstall,
    "claim_room_strategy": _claim,
    "room_strategy": _room_strategy,
}

STRATEGY_ARGUMENTS: dict[str, set[str]] = {
    "install_strategy": {"room_id", "strategy"},
    "uninstall_strategy": {"room_id"},
    "claim_room_strategy": {"room_id", "strategy"},
    "room_strategy": {"room_id"},
}

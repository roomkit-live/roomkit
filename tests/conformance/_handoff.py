"""The handoff (README, "Calls"; RFC §19.6).

``handoff_handler`` builds the implementation's HandoffHandler,
``install_router`` the one a router's ``install`` returns, each under a name
the later calls use; ``setup_handoff`` serves the tool on an agent;
``handoff.handle`` and ``handoff.set_language`` are the handler's own calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit, setup_handoff
from roomkit.orchestration import HandoffHandler, build_handoff_tool
from tests.conformance._routing import router_of

if TYPE_CHECKING:
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness


async def _handoff_handler(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    allowed = spec.get("allowed_transitions")
    harness.handoff_handlers[spec["handler"]] = HandoffHandler(
        kit,
        router_of(spec["router"]),
        agent_aliases=spec.get("agent_aliases"),
        phase_map=spec.get("phase_map"),
        allowed_transitions=(
            None if allowed is None else {phase: set(to) for phase, to in allowed.items()}
        ),
        known_agents=set(spec["known_agents"]) if "known_agents" in spec else None,
    )


async def _install_router(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    priority = {"hook_priority": spec["hook_priority"]} if "hook_priority" in spec else {}
    harness.handoff_handlers[spec["handler"]] = router_of(spec["router"]).install(
        kit,
        spec["agents"],
        agent_aliases=spec.get("agent_aliases"),
        phase_map=spec.get("phase_map"),
        **priority,
    )


async def _setup_handoff(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    targets = spec.get("targets")
    setup_handoff(
        spec["agent_id"],
        harness.handoff_handlers[spec["handler"]],
        tool=None if targets is None else build_handoff_tool([tuple(t) for t in targets]),
        room_id=spec.get("room_id"),
    )


async def _handle(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    handler = harness.handoff_handlers[spec["handler"]]
    return await handler.handle(spec["room_id"], spec["calling_agent_id"], spec["arguments"])


async def _set_language(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    handler = harness.handoff_handlers[spec["handler"]]
    await handler.set_language(spec["room_id"], spec["language"])


HANDOFF_CALLS: dict[str, Call] = {
    "handoff_handler": _handoff_handler,
    "install_router": _install_router,
    "setup_handoff": _setup_handoff,
    "handoff.handle": _handle,
    "handoff.set_language": _set_language,
}

HANDOFF_ARGUMENTS: dict[str, set[str]] = {
    "handoff_handler": {
        "handler",
        "router",
        "agent_aliases",
        "phase_map",
        "allowed_transitions",
        "known_agents",
    },
    "install_router": {
        "handler",
        "router",
        "agents",
        "agent_aliases",
        "phase_map",
        "hook_priority",
    },
    "setup_handoff": {"agent_id", "handler", "targets", "room_id"},
    "handoff.handle": {"handler", "room_id", "calling_agent_id", "arguments"},
    "handoff.set_language": {"handler", "room_id", "language"},
}

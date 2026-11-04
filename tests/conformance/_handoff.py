"""The handoff (README, "Calls"; RFC §19.6).

``handoff_handler`` builds the implementation's HandoffHandler,
``install_router`` the one a router's ``install`` returns and
``install_pipeline`` the one a ConversationPipeline's ``install`` returns, each
under a name the later calls use; ``conversation_pipeline`` builds a pipeline
and reads it; ``setup_handoff`` serves the tool on an agent; ``handoff.handle``,
``handoff.set_language`` and ``handoff.send_greeting`` are the handler's own
calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit, setup_handoff
from roomkit.orchestration import (
    ConversationPipeline,
    HandoffHandler,
    PipelineStage,
    build_handoff_tool,
)
from tests.conformance._routing import router_of

if TYPE_CHECKING:
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness


async def _handoff_handler(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    allowed = spec.get("allowed_transitions")
    handler = HandoffHandler(
        kit,
        router_of(spec["router"]),
        agent_aliases=spec.get("agent_aliases"),
        phase_map=spec.get("phase_map"),
        allowed_transitions=(
            None if allowed is None else {phase: set(to) for phase, to in allowed.items()}
        ),
        known_agents=set(spec["known_agents"]) if "known_agents" in spec else None,
        event_channel_id=spec.get("event_channel_id"),
    )
    if "greeting_map" in spec:
        handler.greeting_map = spec["greeting_map"]
    harness.handoff_handlers[spec["handler"]] = handler


async def _install_router(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    priority = {"hook_priority": spec["hook_priority"]} if "hook_priority" in spec else {}
    harness.handoff_handlers[spec["handler"]] = router_of(spec["router"]).install(
        kit,
        spec["agents"],
        agent_aliases=spec.get("agent_aliases"),
        phase_map=spec.get("phase_map"),
        **priority,
    )


def _pipeline_of(spec: Mapping[str, Any]) -> ConversationPipeline:
    """The ConversationPipeline a call's ``stages`` describe."""
    return ConversationPipeline(
        [PipelineStage(**stage) for stage in spec["stages"]],
        default_phase=spec.get("default_phase"),
        supervisor_id=spec.get("supervisor_id"),
    )


async def _conversation_pipeline(
    kit: RoomKit, harness: Harness, spec: Mapping[str, Any]
) -> dict[str, Any]:
    pipeline = _pipeline_of(spec)
    allowed = pipeline.get_allowed_transitions()
    return {
        "stages": [
            stage.model_dump() | {"can_return_to": sorted(stage.can_return_to)}
            for stage in pipeline.stages
        ],
        "phase_map": pipeline.get_phase_map(),
        "allowed_transitions": {phase: sorted(to) for phase, to in allowed.items()},
    }


async def _install_pipeline(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    priority = {"hook_priority": spec["hook_priority"]} if "hook_priority" in spec else {}
    _router, handler = _pipeline_of(spec).install(
        kit, spec["agents"], agent_aliases=spec.get("agent_aliases"), **priority
    )
    harness.handoff_handlers[spec["handler"]] = handler


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


async def _send_greeting(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    handler = harness.handoff_handlers[spec["handler"]]
    await handler.send_greeting(spec["room_id"], channel_id=spec.get("channel_id"))


HANDOFF_CALLS: dict[str, Call] = {
    "handoff_handler": _handoff_handler,
    "install_router": _install_router,
    "conversation_pipeline": _conversation_pipeline,
    "install_pipeline": _install_pipeline,
    "setup_handoff": _setup_handoff,
    "handoff.handle": _handle,
    "handoff.set_language": _set_language,
    "handoff.send_greeting": _send_greeting,
}

HANDOFF_ARGUMENTS: dict[str, set[str]] = {
    "handoff_handler": {
        "handler",
        "router",
        "agent_aliases",
        "phase_map",
        "allowed_transitions",
        "known_agents",
        "greeting_map",
        "event_channel_id",
    },
    "install_router": {
        "handler",
        "router",
        "agents",
        "agent_aliases",
        "phase_map",
        "hook_priority",
    },
    "conversation_pipeline": {"stages", "default_phase", "supervisor_id"},
    "install_pipeline": {
        "handler",
        "stages",
        "default_phase",
        "supervisor_id",
        "agents",
        "agent_aliases",
        "hook_priority",
    },
    "setup_handoff": {"agent_id", "handler", "targets", "room_id"},
    "handoff.handle": {"handler", "room_id", "calling_agent_id", "arguments"},
    "handoff.set_language": {"handler", "room_id", "language"},
    "handoff.send_greeting": {"handler", "room_id", "channel_id"},
}

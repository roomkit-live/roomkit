"""Phases and routing (README, "Setup" and "Calls"; RFC §19.2, §19.4).

A hook with a ``router`` is a conversation router's hook: the
ConversationRouter built from it, its hook the hook's handler, recording no
call. ``save_conversation_state`` makes phase transitions from the state of
the room the store holds, then saves the state the last one leaves.
``default_router.route`` asks the default router over the kit's store.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import DefaultInboundRoomRouter, Identity, RoomKit
from roomkit.models.enums import ChannelType
from roomkit.orchestration.router import ConversationRouter, RoutingConditions, RoutingRule
from roomkit.orchestration.state import (
    ConversationState,
    get_conversation_state,
    save_conversation_state,
)

if TYPE_CHECKING:
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness
    from tests.conformance._script import HookHandler

ROUTER = {"rules", "default_agent_id", "supervisor_id"}
ROUTING_RULE = {"agent_id", "priority", "conditions"}
CONDITIONS = {"phases", "channel_types", "intents", "source_channel_ids"}
TRANSITION = {"to_phase", "to_agent", "reason", "metadata"}


def router_of(spec: Mapping[str, Any]) -> ConversationRouter:
    """The ConversationRouter a ``router`` member describes."""
    return ConversationRouter(
        rules=[_rule(rule) for rule in spec.get("rules", [])],
        default_agent_id=spec.get("default_agent_id"),
        supervisor_id=spec.get("supervisor_id"),
    )


def router_hook(spec: Mapping[str, Any]) -> HookHandler:
    """The hook of the ConversationRouter a hook's ``router`` describes."""
    return router_of(spec).as_hook()


def _rule(spec: Mapping[str, Any]) -> RoutingRule:
    conditions = spec["conditions"]
    sets = {
        key: set(conditions[key])
        for key in ("phases", "intents", "source_channel_ids")
        if key in conditions
    }
    if "channel_types" in conditions:
        sets["channel_types"] = {ChannelType(value) for value in conditions["channel_types"]}
    return RoutingRule(
        agent_id=spec["agent_id"],
        conditions=RoutingConditions(**sets),
        priority=spec.get("priority", 0),
    )


async def _save_conversation_state(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    room = await kit.store.get_room(spec["room_id"])
    state = ConversationState() if room is None else get_conversation_state(room)
    for transition in spec["transitions"]:
        state = state.transition(
            transition["to_phase"],
            to_agent=transition.get("to_agent"),
            reason=transition.get("reason", ""),
            metadata=transition.get("metadata"),
        )
    await save_conversation_state(kit.store, spec["room_id"], state)


async def _default_route(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> str | None:
    identity_id = spec.get("identity_id")
    return await DefaultInboundRoomRouter(kit.store).route(
        spec["channel_id"],
        ChannelType(spec["channel_type"]),
        participant_id=spec.get("participant_id"),
        organization_id=spec.get("organization_id"),
        identity=None if identity_id is None else Identity(id=identity_id),
    )


ROUTING_CALLS: dict[str, Call] = {
    "save_conversation_state": _save_conversation_state,
    "default_router.route": _default_route,
}

ROUTING_ARGUMENTS: dict[str, set[str]] = {
    "save_conversation_state": {"room_id", "transitions"},
    "default_router.route": {
        "channel_id",
        "channel_type",
        "participant_id",
        "identity_id",
        "organization_id",
    },
}

"""The routing stamps a conversation router writes (RFC §19.4).

``_routed_to`` names the agent an unaddressed event goes to, and
``_always_process`` the agents asked beside it, the supervisor: the
broadcast reads both to decide which agents an event asks. Only a hook
writes them, the router among hooks, so the kit drops them from an event's
metadata before its BEFORE_BROADCAST hooks run: a sender who wrote them
into its message would otherwise choose the agent that answers it, or be
heard by one it did not address.
"""

from __future__ import annotations

import logging

from roomkit.models.event import RoomEvent

logger = logging.getLogger("roomkit.routing_stamps")

ROUTED_TO = "_routed_to"
ALWAYS_PROCESS = "_always_process"
ROUTING_STAMPS = (ROUTED_TO, ALWAYS_PROCESS)

ORCHESTRATION_INTERNAL = "_orchestration_internal"
"""Marks an event as the orchestration's own record, a handoff's, which asks no
agent (RFC §19.6). Only the runtime and code the host runs write it: a sender
who wrote it into its message would silence the room's agents, so the kit drops
it from what arrives through its door (RFC §10.1)."""


def without_routing_stamps(event: RoomEvent) -> RoomEvent:
    """*event* without the routing stamps its metadata arrived with, each
    dropped with a warning; *event* itself when it carries none."""
    metadata = event.metadata or {}
    stamped = [key for key in ROUTING_STAMPS if key in metadata]
    if not stamped:
        return event
    logger.warning(
        "Event %s of room %s arrived with the routing stamps %s, which only a hook "
        "writes: dropped",
        event.id,
        event.room_id,
        stamped,
    )
    kept = {key: value for key, value in metadata.items() if key not in ROUTING_STAMPS}
    return event.model_copy(update={"metadata": kept})

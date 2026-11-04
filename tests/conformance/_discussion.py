"""A discussion in the runner (README, "Calls", "Setup"; RFC §19.7.5).

``discussion`` builds the implementation's Discussion from its keywords, each
agent a channel id or an agent given live, and reads it, installed nowhere. A
setup's strategy with a ``discussion`` is one installed through the kit, its
``done`` read from the ``discussion.done`` call; ``speak_queue`` reads its queue,
and an ``ON_SPEAK_QUEUE`` hook records each change it is handed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import Agent, RoomKit
from roomkit.channels.declared import id_of
from roomkit.orchestration import Discussion, SpeakQueue, SpeakQueueEvent

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


def queue_record(queue: SpeakQueue) -> dict[str, Any]:
    """A speak queue as a scenario reads it."""
    return {
        "speaking": queue.speaking,
        "queue": list(queue.queue),
        "listening": sorted(queue.listening),
        "asked": [list(pair) for pair in queue.asked],
        "waiting": queue.waiting,
        "over": queue.over,
    }


def speak_queue_record(payload: SpeakQueueEvent) -> dict[str, Any]:
    """What an ``ON_SPEAK_QUEUE`` hook is handed, as a scenario reads it."""
    return {
        "room_id": payload.room_id,
        "change": payload.change.value,
        "channel_ids": list(payload.channel_ids),
        "event_id": payload.event_id,
        "queue": queue_record(payload.queue),
    }


async def _speak_queue(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    queue = kit.speak_queue(spec["room_id"], organization_id=spec.get("organization_id"))
    return None if queue is None else queue_record(queue)


async def _discussion_done(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    harness.discussion_done[spec["strategy"]] = spec["holds"]


def installed_discussion(
    strategy_id: str, spec: Mapping[str, Any], harness: Harness
) -> Discussion:
    """The Discussion a setup's strategy describes: its agents by id, its
    ``done`` what the last ``discussion.done`` call named, false until one."""
    options: dict[str, Any] = {keyword: spec.get(keyword) for keyword in _DEFAULTED}
    if spec.get("done"):
        options["done"] = lambda _room_id: harness.discussion_done.get(strategy_id, False)
    return Discussion(
        list(spec["agents"]),
        addressed_only=spec.get("addressed_only", False),
        silent_token=spec.get("silent_token", "(silent)"),
        **options,
    )


def discussions_busy(kit: RoomKit) -> bool:
    """Whether a discussion of *kit* has a turn to give or an announcement to
    make: its driver, which polls, counts only once it found nothing to do."""
    for discussion in kit._discussions.values():
        if discussion._announcing:
            return True
        driver = discussion.driver
        if driver is not None and not driver.settled:
            return True
    return False


DISCUSSION_CALLS: dict[str, Call] = {
    "discussion": _discussion,
    "speak_queue": _speak_queue,
    "discussion.done": _discussion_done,
}

DISCUSSION_MEMBERS = {
    "agents",
    "handles",
    "people",
    "addressed_only",
    "everyone",
    "max_turns",
    "max_depth",
    "silent_token",
    "rules",
    "done",
}
"""The members a setup's strategy with a ``discussion`` may give."""

DISCUSSION_ARGUMENTS: dict[str, set[str]] = {
    "speak_queue": {"room_id", "organization_id"},
    "discussion.done": {"strategy", "holds"},
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

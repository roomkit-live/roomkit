"""A discussion in the runner (README, "Calls", "Setup"; RFC §19.7.5).

``discussion`` builds the implementation's Discussion from its keywords, each
agent a channel id or an agent given live, and reads it, installed nowhere. A
setup's strategy with a ``discussion`` is one installed through the kit, its
``done`` read from the ``discussion.done`` call; ``speak_queue`` reads its queue,
``listen_only`` and ``talk_again`` change it, and an ``ON_SPEAK_QUEUE`` hook
records each change it is handed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import Agent, RoomKit
from roomkit.channels.declared import id_of
from roomkit.orchestration import (
    Discussion,
    DispatchDecision,
    DispatchDecisionEvent,
    DispatchTurn,
    MockDispatchPolicy,
    SpeakQueue,
    SpeakQueueEvent,
)

if TYPE_CHECKING:
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness

_DEFAULTED = ("handles", "people", "everyone", "max_turns", "max_depth", "rules")
"""The keywords a call may leave out, each None when it does."""


def _agent(spec: str | Mapping[str, Any]) -> str | Agent:
    if isinstance(spec, str):
        return spec
    return Agent(spec["id"], name=spec.get("name"))


class ScriptedDispatch(MockDispatchPolicy):
    """A setup's scripted dispatch policy (README, "Setup"): the mock, held
    at its gate, each turn it is handed recorded as *strategy_id*'s."""

    def __init__(self, spec: Mapping[str, Any], harness: Harness, strategy_id: str | None) -> None:
        super().__init__(
            [_decision(decision) for decision in spec.get("decisions", [])],
            delay=spec.get("delay_ms", 0) / 1000,
            error=RuntimeError(spec["error"]) if "error" in spec else None,
        )
        self._harness, self._strategy_id, self._gate = harness, strategy_id, spec.get("gate")

    async def decide(self, turn: DispatchTurn) -> DispatchDecision:
        if self._strategy_id is not None:
            self._harness.record_dispatch_turn(self._strategy_id, dispatch_turn_record(turn))
        if self._gate is not None:
            await self._harness.gates.hold(self._gate)
        return await super().decide(turn)


def _decision(spec: Any) -> DispatchDecision | list[str]:
    if isinstance(spec, list):
        return spec
    return DispatchDecision(
        tuple(spec.get("agents", ())), spec.get("reason", ""), dict(spec.get("judgments", {}))
    )


def _dispatch_options(
    spec: Mapping[str, Any], harness: Harness, strategy_id: str | None
) -> dict[str, Any]:
    """The ``dispatch`` and ``dispatch_timeout`` a discussion *spec* gives."""
    options: dict[str, Any] = {}
    if "dispatch" in spec:
        options["dispatch"] = ScriptedDispatch(spec["dispatch"], harness, strategy_id)
    if "dispatch_timeout_ms" in spec:
        options["dispatch_timeout"] = spec["dispatch_timeout_ms"] / 1000
    return options


def dispatch_turn_record(turn: DispatchTurn) -> dict[str, Any]:
    """What a scripted dispatch policy was handed, as a scenario reads it."""
    return {
        "room_id": turn.room_id,
        "event": turn.event.model_dump(mode="json"),
        "recent": [event.model_dump(mode="json") for event in turn.recent],
        "speakers": [[event_id, label] for event_id, label in turn.speakers.items()],
        "candidates": [
            {
                "channel_id": c.channel_id,
                "name": c.name,
                "role": c.role,
                "description": c.description,
                "handle": c.handle,
            }
            for c in turn.candidates
        ],
        "asked": list(turn.asked),
    }


def dispatch_decision_record(payload: DispatchDecisionEvent) -> dict[str, Any]:
    """What an ``ON_DISPATCH_DECISION`` hook is handed, as a scenario reads it."""
    decision = payload.decision
    return {
        "room_id": payload.room_id,
        "event": payload.event.model_dump(mode="json"),
        "candidates": list(payload.candidates),
        "decision": {
            "agents": list(decision.agents),
            "reason": decision.reason,
            "judgments": dict(decision.judgments),
        },
        "duration_ms": payload.duration_ms,
    }


async def _discussion(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    built = Discussion(
        [_agent(agent) for agent in spec["agents"]],
        addressed_only=spec.get("addressed_only", False),
        silent_token=spec.get("silent_token", "(silent)"),
        **{keyword: spec.get(keyword) for keyword in _DEFAULTED},
        **_dispatch_options(spec, harness, None),
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


async def _listen_only(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    await kit.listen_only(
        spec["room_id"], spec["channel_ids"], organization_id=spec.get("organization_id")
    )


async def _talk_again(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    await kit.talk_again(
        spec["room_id"], spec["channel_ids"], organization_id=spec.get("organization_id")
    )


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
    options.update(_dispatch_options(spec, harness, strategy_id))
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
    "listen_only": _listen_only,
    "talk_again": _talk_again,
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
    "dispatch",
    "dispatch_timeout_ms",
}
"""The members a setup's strategy with a ``discussion`` may give."""

DISPATCH_MEMBERS = {"decisions", "delay_ms", "error", "gate"}
"""The members a scripted dispatch policy may give."""

DISCUSSION_ARGUMENTS: dict[str, set[str]] = {
    "speak_queue": {"room_id", "organization_id"},
    "listen_only": {"room_id", "channel_ids", "organization_id"},
    "talk_again": {"room_id", "channel_ids", "organization_id"},
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
        "dispatch",
        "dispatch_timeout_ms",
    },
}

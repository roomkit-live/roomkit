"""The scripted parts of identity resolution: the kit's resolver and its
identity hooks (RFC §11.1 to §11.3).

roomkit-specs ``conformance/README.md`` ("Identity resolution") defines what
each does; this module builds them on the Python core.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.identity.base import IdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventDirection, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.models.hook import InjectedEvent
from roomkit.models.identity import IdentityHookResult, IdentityResult
from tests.conformance._calls import read
from tests.conformance._script import (
    SYSTEM_SOURCE,
    Reactions,
    ScriptedFailureError,
    complete_event,
    event_subject,
    perform,
)

if TYPE_CHECKING:
    from tests.conformance._harness import Harness


def _message_subject(message: InboundMessage, room_id: str) -> dict[str, Any]:
    """What a resolver answer's ``when`` reads of the message it is asked about."""
    return {
        "room_id": room_id,
        "sender_id": message.sender_id,
        "channel_id": message.channel_id,
        "text": getattr(message.content, "body", None),
    }


class ScriptedResolver(IdentityResolver):
    """Records each call, then answers with the first answer whose ``when`` holds."""

    def __init__(self, spec: Mapping[str, Any], harness: Harness) -> None:
        self._answers = Reactions(spec.get("answers"))
        self._harness = harness

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        room_id = context.room.id
        self._harness.record_resolver(room_id, message)
        answer = self._answers.pick(_message_subject(message, room_id)) or {}
        if (gate := answer.get("gate")) is not None:
            await self._harness.gates.hold(gate)
        if (failure := answer.get("fail")) is not None:
            raise ScriptedFailureError(failure)
        return IdentityResult.model_validate(answer.get("result", {"status": "unknown"}))


def kit_options(kit: Mapping[str, Any], harness: Harness) -> dict[str, Any]:
    """The kit's identity keywords a scenario's ``setup.kit`` names."""
    options: dict[str, Any] = {}
    if "identity_resolver" in kit:
        options["identity_resolver"] = ScriptedResolver(kit["identity_resolver"], harness)
    if "identity_channel_types" in kit:
        types = kit["identity_channel_types"]
        options["identity_channel_types"] = read(lambda: {ChannelType(t) for t in types})
    if "identity_timeout_ms" in kit:
        options["identity_timeout"] = kit["identity_timeout_ms"] / 1000
    return options


IdentityHandler = Callable[
    [RoomEvent, RoomContext, IdentityResult], Awaitable[IdentityHookResult | None]
]


def register_identity_hook(kit: RoomKit, harness: Harness, hook: Mapping[str, Any]) -> None:
    """Registers a scripted identity hook (§11.2) on the kit."""
    kwargs: dict[str, Any] = {}
    if "channel_types" in hook:
        kwargs["channel_types"] = read(lambda: {ChannelType(t) for t in hook["channel_types"]})
    if "channel_ids" in hook:
        kwargs["channel_ids"] = set(hook["channel_ids"])
    if "directions" in hook:
        kwargs["directions"] = read(lambda: {EventDirection(d) for d in hook["directions"]})
    trigger = read(lambda: HookTrigger(hook["trigger"]))
    kit.identity_hook(trigger, **kwargs)(_handler(hook, harness))


def _handler(hook: Mapping[str, Any], harness: Harness) -> IdentityHandler:
    reactions = Reactions(hook.get("reactions"))
    name = hook["name"]

    async def handler(
        event: RoomEvent, context: RoomContext, id_result: IdentityResult
    ) -> IdentityHookResult | None:
        room_id = context.room.id
        harness.record_identity_hook(name, hook["trigger"], room_id, event, id_result)
        reaction = reactions.pick(event_subject(event, room_id))
        if reaction is None:
            return None
        await perform(harness, {"gate": reaction.get("gate"), "fail": reaction.get("fail")}, {})
        result = reaction.get("result")
        return None if result is None else _hook_result(result, room_id)

    return handler


def _hook_result(spec: Mapping[str, Any], room_id: str) -> IdentityHookResult:
    """An IdentityHookResult from a scenario's, its injected event completed."""
    data = dict(spec)
    if (inject := data.get("inject")) is not None:
        data["inject"] = InjectedEvent(
            event=complete_event(inject["event"], room_id, SYSTEM_SOURCE),
            target_channel_ids=inject.get("target_channel_ids"),
        )
    return IdentityHookResult.model_validate(data)

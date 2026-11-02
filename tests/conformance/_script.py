"""The scripted parts of a scenario: gates, reactions, channels and hooks.

roomkit-specs ``conformance/README.md`` ("Scripted channels", "Scripted
hooks") defines what each does; this module builds them on the Python core.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.channels.base import Channel
from roomkit.core.exceptions import ProviderDeliveryError
from roomkit.core.inbound_router import InboundRoomRouter
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.enums import ChannelCategory, ChannelDirection, ChannelType, EventDirection
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.pending_input import PendingInputEvent
from roomkit.models.task import Observation, Task
from roomkit.models.tool_call import AIGenerationEvent, ToolCallEvent, ToolRoundEvent
from tests.conformance._agent import patched_tool_call
from tests.conformance._ai_hooks import patched_generation, patched_tool_round

if TYPE_CHECKING:
    from tests.conformance._harness import Harness

# How long a step waits for a call to reach a gate before the scenario fails.
GATE_BOUND_S = 5.0

# The payload and the context of the scripted hook whose actions run, which
# ``sender_verification`` reads (identity verification, §11.7.3).
HOOKED: ContextVar[tuple[Any, RoomContext] | None] = ContextVar("hooked", default=None)


class ScriptedFailureError(Exception):
    """What a reaction's ``fail`` raises."""


class Gates:
    """Named gates: a call held at one waits until a step opens it."""

    def __init__(self) -> None:
        self._opened: dict[str, asyncio.Event] = {}
        self._reached: dict[str, asyncio.Event] = {}

    def _pair(self, name: str) -> tuple[asyncio.Event, asyncio.Event]:
        return (
            self._opened.setdefault(name, asyncio.Event()),
            self._reached.setdefault(name, asyncio.Event()),
        )

    async def hold(self, name: str) -> None:
        opened, reached = self._pair(name)
        reached.set()
        await opened.wait()

    async def wait_for(self, name: str) -> None:
        _, reached = self._pair(name)
        await asyncio.wait_for(reached.wait(), GATE_BOUND_S)

    def open(self, name: str) -> None:
        opened, _ = self._pair(name)
        opened.set()


class Reactions:
    """A list of reactions, each used once unless it repeats."""

    def __init__(self, reactions: list[dict[str, Any]] | None) -> None:
        self._items = list(reactions or [])
        self._used: set[int] = set()

    def pick(self, subject: Mapping[str, Any]) -> dict[str, Any] | None:
        for position, reaction in enumerate(self._items):
            if position in self._used or not _holds(reaction.get("when"), subject):
                continue
            if not reaction.get("repeat"):
                self._used.add(position)
            return reaction
        return None


def _holds(condition: Mapping[str, Any] | None, subject: Mapping[str, Any]) -> bool:
    return all(subject.get(key) == value for key, value in (condition or {}).items())


def event_subject(event: Any, room_id: str) -> dict[str, Any]:
    """What a reaction's ``when`` reads of the call's event."""
    if not isinstance(event, RoomEvent):
        return {"room_id": room_id}
    return {
        "room_id": room_id,
        "text": getattr(event.content, "body", None),
        "type": str(event.type),
        "source_channel_id": event.source.channel_id,
        "chain_depth": event.chain_depth,
    }


def message_subject(message: InboundMessage, room_id: str) -> dict[str, Any]:
    return {
        "room_id": room_id,
        "text": getattr(message.content, "body", None),
        "type": str(message.event_type),
        "source_channel_id": message.channel_id,
        "chain_depth": message.chain_depth,
    }


async def perform(harness: Harness, reaction: Mapping[str, Any], scope: Mapping[str, str]) -> None:
    """Hold at the reaction's gate, run its actions, leave those for after
    the turn to work it began, then raise its failure."""
    if (gate := reaction.get("gate")) is not None:
        await harness.gates.hold(gate)
    for action in reaction.get("do", []):
        await harness.run_action(action, scope)
    if (later := reaction.get("after_turn")) is not None:
        harness.leave_after_turn(later, scope)
    if (message := reaction.get("fail")) is not None:
        raise ScriptedFailureError(message)
    if (refusal := reaction.get("refuse")) is not None:
        raise ProviderDeliveryError(
            ProviderResult(
                success=False,
                provider_message_id=refusal.get("provider_message_id"),
                error=refusal.get("error"),
                metadata={key: refusal[key] for key in ("code", "retryable") if key in refusal},
            )
        )


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def complete_event(
    template: Mapping[str, Any], room_id: str, source: Mapping[str, Any]
) -> RoomEvent:
    """A RoomEvent from a scenario's template, its missing required members filled."""
    data = {"id": uuid4().hex, "room_id": room_id, "type": "message", "created_at": _now()}
    data.update(template)
    data["source"] = {**source, **template.get("source", {})}
    return RoomEvent.model_validate(data)


def complete_participant(template: Mapping[str, Any], room_id: str) -> Participant:
    return Participant.model_validate({"room_id": room_id, "joined_at": _now(), **template})


def complete_task(template: Mapping[str, Any], room_id: str) -> Task:
    return Task.model_validate(
        {"id": uuid4().hex, "room_id": room_id, "created_at": _now(), **template}
    )


def complete_observation(
    template: Mapping[str, Any], room_id: str, channel_id: str
) -> Observation:
    return Observation.model_validate(
        {
            "id": uuid4().hex,
            "room_id": room_id,
            "channel_id": channel_id,
            "created_at": _now(),
            **template,
        }
    )


def parsed_event(event: RoomEvent, patch: Mapping[str, Any]) -> RoomEvent:
    """*event* with a parse reaction's ``parsed`` patch: its members replace
    the event's, those of its ``source`` the event's source's."""
    data = event.model_dump(mode="json")
    patched = {**data, **patch, "source": {**data["source"], **patch.get("source", {})}}
    return RoomEvent.model_validate(patched)


SYSTEM_SOURCE = {"channel_id": "system", "channel_type": "system", "direction": "inbound"}


class ScriptedChannel(Channel):
    """A channel whose parse, turn and delivery follow the scenario's reactions."""

    def __init__(self, spec: Mapping[str, Any], harness: Harness) -> None:
        super().__init__(spec["id"])
        self.channel_type = ChannelType(spec["type"])
        default = "intelligence" if spec["type"] == "ai" else "transport"
        self.category = ChannelCategory(spec.get("category", default))
        self.direction = ChannelDirection(spec.get("direction", "bidirectional"))
        self._capabilities = ChannelCapabilities.model_validate(spec.get("capabilities", {}))
        self._recipient_key = spec.get("recipient_key")
        self._harness = harness
        self._parse = Reactions(spec.get("parse"))
        self._on_event = Reactions(spec.get("on_event"))
        self._deliver = Reactions(spec.get("deliver"))
        self._streams = bool(spec.get("streams", False))
        self._streams_in = spec.get("streams_in")
        self._deliver_stream = Reactions(spec.get("deliver_stream"))
        self._addresses = spec.get("addresses")
        self._recent_events_window = int(spec.get("recent_events_window", 0))
        if "sender_is_participant" in spec:
            # Declared on the class by a real channel; per scenario here.
            self.sender_is_participant = bool(spec["sender_is_participant"])

    def capabilities(self) -> ChannelCapabilities:
        return self._capabilities

    @property
    def recipient_key(self) -> str | None:
        return self._recipient_key

    @property
    def recent_events_window(self) -> int:
        return self._recent_events_window

    # What the scenario's ``addresses`` says the channel's own address
    # methods answer; the defaults without it.

    def normalize_address(self, address: str) -> str:
        if self._addresses is None:
            return super().normalize_address(address)
        return self._addresses.get("normalizes", {}).get(address, address)

    def conversation_address(self, message: InboundMessage) -> str:
        key = (self._addresses or {}).get("conversation_key")
        chat = message.metadata.get(key) if key is not None else None
        if isinstance(chat, str) and chat.strip():
            return chat
        return super().conversation_address(message)

    def reply_metadata(self, message: InboundMessage) -> dict[str, str]:
        if self._addresses is None or self._recipient_key is None:
            return super().reply_metadata(message)
        return {self._recipient_key: self.conversation_address(message)}

    def recipient_metadata(self, address: str) -> dict[str, str]:
        if self._addresses is None or self._recipient_key is None:
            return super().recipient_metadata(address)
        return {self._recipient_key: address}

    def recipient_address(self, binding: ChannelBinding) -> str | None:
        if self._addresses is None or self._recipient_key is None:
            return super().recipient_address(binding)
        held = binding.metadata.get(self._recipient_key)
        return self.normalize_address(held) if isinstance(held, str) and held.strip() else None

    def supports_streaming_delivery_for(self, room_id: str) -> bool:
        if self._streams_in is None:
            return super().supports_streaming_delivery_for(room_id)
        return self._streams and room_id in self._streams_in

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        room_id = context.room.id
        reaction = self._parse.pick(message_subject(message, room_id)) or {}
        await perform(self._harness, reaction, {"room": room_id})
        event = RoomEvent(
            room_id=room_id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                direction=EventDirection.INBOUND,
                participant_id=message.sender_id or None,
                external_id=message.external_id,
                raw_payload=message.raw_payload,
                provider_message_id=message.provider_message_id,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
            metadata=message.metadata,
        )
        if (patch := reaction.get("parsed")) is None:
            return event
        return parsed_event(event, await self._harness.resolve(patch, {"room": room_id}))

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self._harness.record("deliveries", event.room_id, self.channel_id, event)
        self._harness.record_context(event.room_id, self.channel_id, "deliver", context)
        return await self._react(self._deliver, event, respond=False)

    @property
    def supports_streaming_delivery(self) -> bool:
        return self._streams

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """Reads the stream it is handed, recording each delta and row, until
        its end, or until the delta its reaction's ``stop_after`` names."""
        room_id = binding.room_id
        reaction = self._deliver_stream.pick({"room_id": room_id}) or {}
        await perform(self._harness, {**reaction, "fail": None}, {"room": room_id})
        stop_after = reaction.get("stop_after")
        try:
            async for item in text_stream:
                if isinstance(item, str):
                    self._harness.record_streamed(room_id, self.channel_id, {"text": item})
                    if item == stop_after:
                        break
                elif isinstance(item, RoomEvent):
                    row = {"row": item.model_dump(mode="json")}
                    self._harness.record_streamed(room_id, self.channel_id, row)
        except Exception:
            self._harness.record_streamed(room_id, self.channel_id, {"failed": True})
            raise
        if (message := reaction.get("fail")) is not None:
            raise ScriptedFailureError(message)
        return self._side_effects(reaction.get("output", {}), room_id)

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if self.category is ChannelCategory.INTELLIGENCE:
            self._harness.record("turns", event.room_id, self.channel_id, event)
            self._harness.record_context(event.room_id, self.channel_id, "on_event", context)
        return await self._react(self._on_event, event, respond=True)

    async def _react(
        self, reactions: Reactions, event: RoomEvent, *, respond: bool
    ) -> ChannelOutput:
        reaction = reactions.pick(event_subject(event, event.room_id))
        if reaction is None:
            return ChannelOutput.empty()
        await perform(self._harness, reaction, {"room": event.room_id, "event": event.id})
        output = reaction.get("output", {})
        source = {
            "channel_id": self.channel_id,
            "channel_type": str(self.channel_type),
            "direction": "outbound",
        }
        responses = [
            complete_event(
                {"chain_depth": event.chain_depth + 1, **template}, event.room_id, source
            )
            for template in reaction.get("respond", [])
        ]
        return self._side_effects(output, event.room_id).model_copy(
            update={
                "responded": respond and "respond" in reaction,
                "response_events": responses if respond else [],
            }
        )

    def _side_effects(self, output: Mapping[str, Any], room_id: str) -> ChannelOutput:
        """The output a reaction's ``output`` gives: its tasks, observations
        and room metadata updates, completed for *room_id*."""
        return ChannelOutput(
            tasks=[complete_task(t, room_id) for t in output.get("tasks", [])],
            observations=[
                complete_observation(o, room_id, self.channel_id)
                for o in output.get("observations", [])
            ],
            metadata_updates=output.get("metadata_updates", {}),
        )


class ScriptedRouter(InboundRoomRouter):
    """A router answering every routed message with one room, once held at
    its gate when it has one."""

    def __init__(self, room_id: str, gate: str | None = None, gates: Gates | None = None) -> None:
        self._room_id = room_id
        self._gate = gate
        self._gates = gates

    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
        *,
        organization_id: str | None = None,
        identity: Identity | None = None,
    ) -> str | None:
        if self._gate is not None and self._gates is not None:
            await self._gates.hold(self._gate)
        return self._room_id


HookHandler = Callable[[Any, RoomContext], Awaitable[HookResult | None]]


def hook_handler(spec: Mapping[str, Any], harness: Harness) -> HookHandler:
    """The handler of a scripted hook: it records its call, then reacts."""
    reactions = Reactions(spec.get("reactions"))
    name = spec["name"]

    async def handler(payload: Any, context: RoomContext) -> HookResult | None:
        room_id = context.room.id
        harness.record_hook(name, spec["trigger"], room_id, payload)
        reaction = reactions.pick(event_subject(payload, room_id))
        if reaction is None:
            return HookResult.allow()
        scope = {"room": room_id}
        if isinstance(payload, RoomEvent):
            scope["event"] = payload.id
        if isinstance(payload, PendingInputEvent):
            scope["request"] = payload.pending_id
        hooked = HOOKED.set((payload, context))
        try:
            await perform(harness, reaction, scope)
        finally:
            HOOKED.reset(hooked)
        return hook_result(reaction.get("result"), payload, room_id)

    return handler


def hook_result(spec: Mapping[str, Any] | None, payload: Any, room_id: str) -> HookResult:
    """A HookResult from a scenario's, a modify's event read as a patch on the payload."""
    if spec is None:
        return HookResult.allow()
    data = dict(spec)
    if data.get("action") == "modify" and "event" in data and isinstance(payload, RoomEvent):
        patched = {**payload.model_dump(mode="json"), **data["event"]}
        data["event"] = RoomEvent.model_validate(patched)
    if data.get("action") == "modify" and "event" in data and isinstance(payload, ToolCallEvent):
        data["event"] = patched_tool_call(payload, data["event"])
    if data.get("action") == "modify" and "event" in data:
        if isinstance(payload, AIGenerationEvent):
            data["event"] = patched_generation(payload, data["event"])
        elif isinstance(payload, ToolRoundEvent):
            data["event"] = patched_tool_round(payload, data["event"])
    data["injected_events"] = [
        InjectedEvent(
            event=complete_event(item["event"], room_id, SYSTEM_SOURCE),
            target_channel_ids=item.get("target_channel_ids"),
        )
        for item in data.get("injected_events", [])
    ]
    data["tasks"] = [complete_task(t, room_id) for t in data.get("tasks", [])]
    data["observations"] = [
        complete_observation(o, room_id, "system") for o in data.get("observations", [])
    ]
    return HookResult.model_validate(data)

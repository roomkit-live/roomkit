"""Runs one conformance scenario on the Python core and compares what it did.

roomkit-specs ``conformance/README.md`` is the contract: the setup is made
before anything is recorded, the steps run in order, the kit is left to go
idle, and what the run produced is normalized and matched against the
scenario's expectations.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from roomkit import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.store_filter import EventFilter
from roomkit.models.tool_call import ToolCallEvent
from roomkit.store.base import ConversationStore
from tests.conformance._agent import ScriptedModel, tool_call_record
from tests.conformance._calls import (
    CALLS,
    UnmappedError,
    UnreadableError,
    add_participant,
    attach_channel,
    create_room,
    dump,
    error_of,
    register_channel,
    register_hook,
    result_event_id,
)
from tests.conformance._compare import Check, compare
from tests.conformance._normalize import Observed, is_lifecycle_record
from tests.conformance._script import Gates, ScriptedRouter
from tests.conformance._store import STORE_CALLS

# The framework events of the §26.1 table, and Processing timeout's (§8.2).
LEVEL_0_EVENTS = (
    "room_created",
    "room_paused",
    "room_closed",
    "room_archived",
    "room_refused_event",
    "channel_registered",
    "channel_unregistered",
    "room_channel_attached",
    "room_channel_detached",
    "event_processed",
    "event_blocked",
    "delivery_succeeded",
    "delivery_failed",
    "broadcast_partial_failure",
    "chain_depth_exceeded",
    "hook_error",
    "hook_timeout",
    "session_started",
    "process_timeout",
)

# A call that has not returned by then fails the scenario instead of hanging
# it; so does a kit that does not go idle.
CALL_BOUND_S = 10.0
_EVENT_ALIAS = re.compile(r"^(.+)#(\d+)$")
_CREATED_ROOM = re.compile(r"^\$r(\d+)$")


class Harness:
    """One run of one scenario."""

    def __init__(
        self, scenario: Mapping[str, Any], store: ConversationStore | None = None
    ) -> None:
        self.scenario = scenario
        self.gates = Gates()
        self.observed = Observed(literal_rooms=_literal_rooms(scenario))
        self._recording = False
        self._labels: dict[str, str | None] = {}
        self._background: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._checks: list[Check] = []
        self._step_outcomes: list[dict[str, Any]] = []
        self._problems: list[str] = []
        self.models: dict[str, ScriptedModel] = {}
        setup = scenario.get("setup", {})
        kit_setup = setup.get("kit", {})
        kwargs: dict[str, Any] = {}
        if "max_chain_depth" in kit_setup:
            kwargs["max_chain_depth"] = kit_setup["max_chain_depth"]
        if "process_timeout_ms" in kit_setup:
            kwargs["process_timeout"] = kit_setup["process_timeout_ms"] / 1000
        if store is not None:
            kwargs["store"] = store
        if "router" in kit_setup:
            kwargs["inbound_router"] = ScriptedRouter(kit_setup["router"]["room_id"])
        self.kit = RoomKit(**kwargs)

    # -- recording --------------------------------------------------------

    def record(self, kind: str, room_id: str, channel_id: str, event: RoomEvent) -> None:
        if not self._recording:
            return
        per_room = getattr(self.observed, kind).setdefault(room_id, {})
        per_room.setdefault(channel_id, []).append(event.model_dump(mode="json"))

    def record_context(
        self, room_id: str, channel_id: str, call: str, context: RoomContext
    ) -> None:
        """The context a scripted channel's ``deliver`` or ``on_event`` was
        handed: the ids of its events, lifecycle records left out."""
        if not self._recording:
            return
        events = [event.model_dump(mode="json") for event in context.recent_events]
        recent = [event["id"] for event in events if not is_lifecycle_record(event)]
        per_room = self.observed.contexts.setdefault(room_id, {})
        per_room.setdefault(channel_id, []).append({"call": call, "recent_events": recent})

    def record_streamed(self, room_id: str, channel_id: str, item: dict[str, Any]) -> None:
        """One item a streaming channel's ``deliver_stream`` read, as the
        scenario reads it: ``{"text": ...}``, ``{"row": event}``, or
        ``{"failed": true}`` where the stream raised."""
        if not self._recording:
            return
        per_room = self.observed.streamed.setdefault(room_id, {})
        per_room.setdefault(channel_id, []).append(item)

    def record_hook(self, name: str, trigger: str, room_id: str, payload: Any) -> None:
        if not self._recording:
            return
        event = payload.model_dump(mode="json") if isinstance(payload, RoomEvent) else None
        call: dict[str, Any] = {"trigger": trigger, "room_id": room_id, "event": event}
        if isinstance(payload, ToolCallEvent):
            call["tool_call"] = tool_call_record(payload)
        self.observed.hooks.setdefault(name, []).append(call)
        self.observed.hook_order.append(name)

    async def _on_framework_event(self, event: FrameworkEvent) -> None:
        if event.type == "room_created" and event.room_id is not None:
            self.observed.created_rooms.append(event.room_id)
        if not self._recording:
            return
        # The members the RFC's Data column names, flat; `data` kept whole
        # too, for the rows that name `data.x` (§8.2).
        flat: dict[str, Any] = {"event": event.type, **event.data, "data": event.data}
        for key in ("room_id", "channel_id", "event_id"):
            value = getattr(event, key)
            if value is not None:
                flat[key] = value
        self.observed.framework_events.append(flat)

    # -- the run ----------------------------------------------------------

    async def run(self) -> list[str]:
        """Runs the scenario; returns every mismatch with its expectations."""
        try:
            await self._setup()
            self._recording = True
            for position, step in enumerate(self.scenario["steps"]):
                await self._step(step, f"steps[{position}]")
            await self.idle()
            await self._observe()
        finally:
            await self.kit.close()
        return self._problems + compare(
            self.scenario.get("expect", {}), self.observed, self._checks
        )

    async def _setup(self) -> None:
        setup = self.scenario.get("setup", {})
        for event_type in LEVEL_0_EVENTS:
            self.kit.on(event_type)(self._on_framework_event)
        for channel in setup.get("channels", []):
            register_channel(self.kit, self, channel)
        for room in setup.get("rooms", []):
            await create_room(self.kit, room)
            for binding in room.get("bindings", []):
                await attach_channel(self.kit, room["id"], binding)
                if binding.get("muted"):
                    await self.kit.mute(room["id"], binding["channel_id"])
            for participant in room.get("participants", []):
                await add_participant(self.kit, room["id"], participant)
            if room.get("status") in ("closed", "archived"):
                await self.kit.close_room(room["id"])
            if room.get("status") == "archived":
                await self.kit.archive_room(room["id"])
        for hook in setup.get("hooks", []):
            register_hook(self.kit, self, hook)
        await self.idle()

    async def _step(self, step: Mapping[str, Any], path: str) -> None:
        if "call" in step:
            if step.get("background"):
                self._background[step["as"]] = asyncio.create_task(self.invoke(step, {}))
                return
            outcome = await self.invoke(step, {})
            self._step_outcomes.append(outcome)
            self._checks.append((path, step.get("expect"), outcome))
        elif "await" in step:
            outcome = await self._background.pop(step["await"])
            self._step_outcomes.append(outcome)
            self._checks.append((path, step.get("expect"), outcome))
        elif "concurrent" in step:
            calls = step["concurrent"]
            outcomes = await asyncio.gather(*(self.invoke(call, {}) for call in calls))
            for position, (call, outcome) in enumerate(zip(calls, outcomes, strict=True)):
                self._step_outcomes.append(outcome)
                self._checks.append(
                    (f"{path}.concurrent[{position}]", call.get("expect"), outcome)
                )
        elif "wait_for" in step:
            try:
                await self.gates.wait_for(step["wait_for"])
            except TimeoutError:
                self._problems.append(f"{path}: no call reached the gate {step['wait_for']}")
        elif "open" in step:
            self.gates.open(step["open"])
        elif "idle" in step:
            await self.idle()

    async def run_action(self, action: Mapping[str, Any], scope: Mapping[str, str]) -> None:
        """A call a hook or a channel makes while it runs; its outcome is kept."""
        outcome = await self.invoke(action, scope)
        label = action.get("as")
        if label is not None:
            self.observed.actions.setdefault(label, []).append(outcome)
        # An action's outcome is checked by its own expect; with none, it
        # must not fail, unless the scenario's expect.actions checks its label.
        asserted = self.scenario.get("expect", {}).get("actions", {})
        if "expect" in action or label not in asserted:
            self._checks.append(
                (f"action {label or action['call']}", action.get("expect"), outcome)
            )

    async def invoke(self, spec: Mapping[str, Any], scope: Mapping[str, str]) -> dict[str, Any]:
        """Makes one call; returns its outcome, its error named, never raising."""
        resolved = await self.resolve(dict(spec), scope)
        try:
            result = await asyncio.wait_for(
                (CALLS | STORE_CALLS)[spec["call"]](self.kit, self, resolved), CALL_BOUND_S
            )
        except TimeoutError:
            self._problems.append(f"{spec['call']} did not return within {CALL_BOUND_S} s")
            return {"result": None, "error": {"name": "<hung>"}}
        except UnreadableError as error:
            self._problems.append(f"{spec['call']}: the runner cannot read the call: {error}")
            return {"result": None, "error": {"name": "<unreadable>"}}
        except Exception as error:
            try:
                return {"result": None, "error": error_of(error)}
            except UnmappedError as unmapped:
                self._problems.append(f"{spec['call']} raised an unmapped error: {unmapped}")
                return {"result": None, "error": {"name": "<unmapped>"}}
        if "as" in spec:
            self._labels[spec["as"]] = result_event_id(result)
        return {"result": dump(result), "error": None}

    async def idle(self) -> None:
        """Waits until the kit runs nothing more in the background.

        It reads the kit's internals (its lanes, its observers, its held
        tasks): the Python core offers no public wait for its background work.
        """
        quiet = 0
        try:
            async with asyncio.timeout(CALL_BOUND_S):
                while quiet < 3:
                    busy = await self._drain_once()
                    await asyncio.sleep(0)
                    quiet = 0 if busy else quiet + 1
        except TimeoutError:
            self._problems.append(f"the kit did not go idle within {CALL_BOUND_S} s")

    async def _drain_once(self) -> bool:
        busy = False
        for lane in list(self.kit._lanes._lanes.values()):
            if lane._indexed or lane._unindexed or lane._busy:
                busy = True
                await lane.drain()
        if self.kit.hook_engine.running_observers():
            busy = True
            await self.kit.hook_engine.drain_observers()
        tasks = [
            task
            for held in (self.kit._pending_hook_tasks, self.kit._decided_writes)
            for task in held
            if not task.done()
        ]
        tasks.extend(task for task in self._background.values() if not task.done())
        if tasks:
            busy = True
            await asyncio.wait(tasks, timeout=CALL_BOUND_S)
        return busy

    # -- references -------------------------------------------------------

    async def resolve(self, value: Any, scope: Mapping[str, str]) -> Any:
        if isinstance(value, dict):
            return {key: await self.resolve(item, scope) for key, item in value.items()}
        if isinstance(value, list):
            return [await self.resolve(item, scope) for item in value]
        if isinstance(value, str):
            return await self._reference(value, scope)
        return value

    async def _reference(self, value: str, scope: Mapping[str, str]) -> str:
        if value in ("$room", "$event"):
            return scope[value[1:]]
        if match := _CREATED_ROOM.match(value):
            return self._created_room(int(match[1]))
        if value.startswith("$") and value[1:] in self._labels:
            return self._labels[value[1:]] or value
        if match := _EVENT_ALIAS.match(value):
            room_id = await self._reference(match[1], scope)
            return await self._event_at(room_id, int(match[2])) or value
        return value

    def _created_room(self, number: int) -> str:
        created = [r for r in self.observed.created_rooms if r not in self.observed.literal_rooms]
        return created[number - 1] if number <= len(created) else f"$r{number}"

    async def _event_at(self, room_id: str, index: int) -> str | None:
        """The id of the room's event at *index*, read without lifecycle records."""
        timeline = await self._timeline(room_id)
        lifecycle = 0
        for event in timeline:
            if is_lifecycle_record(event):
                lifecycle += 1
            elif event["index"] - lifecycle == index:
                return event["id"]
        return None

    async def _timeline(self, room_id: str) -> list[dict[str, Any]]:
        events = await self.kit.store.list_events(
            room_id, limit=100_000, event_filter=EventFilter(include_blocked=True)
        )
        return sorted((e.model_dump(mode="json") for e in events), key=lambda e: e["index"])

    # -- observation and comparison ---------------------------------------

    async def _observe(self) -> None:
        store = self.kit.store
        for room in await store.list_rooms(limit=100_000):
            self.observed.timelines[room.id] = await self._timeline(room.id)
            self.observed.rooms[room.id] = room.model_dump(mode="json")
            self.observed.bindings[room.id] = {
                binding.channel_id: binding.model_dump(mode="json")
                for binding in await store.list_bindings(room.id)
            }
            self.observed.observations[room.id] = [
                o.model_dump(mode="json") for o in await store.list_observations(room.id)
            ]
            self.observed.tasks[room.id] = [
                t.model_dump(mode="json") for t in await store.list_tasks(room.id)
            ]
        self.observed.steps = self._step_outcomes
        self.observed.models = {
            channel_id: list(model.requests) for channel_id, model in self.models.items()
        }


def _literal_rooms(scenario: Mapping[str, Any]) -> set[str]:
    """The room ids a scenario writes itself; any other room a run creates is aliased."""
    found = {room["id"] for room in scenario.get("setup", {}).get("rooms", [])}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "room_id" and isinstance(item, str) and not item.startswith("$"):
                    found.add(item)
                if key == "room" and isinstance(item, dict) and isinstance(item.get("id"), str):
                    found.add(item["id"])
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(scenario.get("steps", []))
    walk(scenario.get("setup", {}).get("hooks", []))
    walk(scenario.get("setup", {}).get("channels", []))
    return found

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

from pydantic_core import to_jsonable_python

from roomkit import RoomKit, ThinkingEvent, current_turn_host
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import DeliveryHandle, DeliveryStatus
from roomkit.models.event import RoomEvent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.pending_input import PendingInputEvent
from roomkit.models.store_filter import EventFilter
from roomkit.models.task import Observation
from roomkit.models.tool_call import (
    AIGenerationEvent,
    AIResponseEvent,
    ToolCallEvent,
    ToolRoundEvent,
)
from roomkit.orchestration import HandoffHandler
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.status_bus import StatusEntry
from roomkit.realtime import EphemeralEvent, EphemeralEventType
from roomkit.store.base import ConversationStore
from roomkit.tasks import DelegatedTask
from tests.conformance._agent import ScriptedModel, tool_call_record
from tests.conformance._ai_hooks import generation_record, response_record, tool_round_record
from tests.conformance._audits import Auditors
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
from tests.conformance._delivery import (
    backend_busy,
    delivery_options,
    start_backend,
    wait_for_lock,
)
from tests.conformance._discussion import DISCUSSION_CALLS
from tests.conformance._handoff import HANDOFF_CALLS
from tests.conformance._identity import kit_options, register_identity_hook
from tests.conformance._inputs import INPUT_CALLS, request_record
from tests.conformance._locks import LOCK_EVENTS, lock_options, set_lock_releases
from tests.conformance._media import media_options
from tests.conformance._members import MEMBER_CALLS, store_identities
from tests.conformance._normalize import Observed, is_lifecycle_record
from tests.conformance._routing import ROUTING_CALLS
from tests.conformance._script import GATE_BOUND_S, Gates, ScriptedRouter
from tests.conformance._store import STORE_CALLS
from tests.conformance._strategies import (
    STRATEGY_CALLS,
    kit_orchestration,
    strategies,
)
from tests.conformance._tasks import TASK_CALLS
from tests.conformance._verification import (
    VERIFICATION_CALLS,
    VERIFICATION_EVENTS,
    opened_labels,
    verification_options,
)

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
    # Identity resolution's (Level 1, §8.2).
    "identity_resolved",
    "identity_timeout",
)

# The framework events of the Level 2 features a scenario exercises: the
# status bus's (§19.8) and a feedback's (§15.9).
LEVEL_2_EVENTS = ("status_posted", "feedback")

# The framework event of a human-input request's decision (§8.2, §9.3).
INPUT_EVENTS = ("user_input_required",)

# The framework events of an AI channel's turn (§8.2, Level 1), which
# ``expect.ai_events`` compares apart from ``framework_events``.
AI_EVENTS = ("before_ai_generation", "ai_response")

# A call that has not returned by then fails the scenario instead of hanging
# it; so does a kit that does not go idle.
CALL_BOUND_S = 10.0
_EVENT_ALIAS = re.compile(r"^(.+)#(\d+)$")
_CREATED_ROOM = re.compile(r"^\$r(\d+)$")
_REQUEST = re.compile(r"^\$p(\d+)$")


_THINKING_EVENTS = {
    EphemeralEventType.THINKING_START,
    EphemeralEventType.THINKING_DELTA,
    EphemeralEventType.THINKING_END,
}


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
        # The completion handles of deferred calls, by their call's label.
        self.handles: dict[str, DeliveryHandle] = {}
        self._background: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._after_turn: list[asyncio.Task[None]] = []
        self.tasks: dict[str, DelegatedTask] = {}
        self._checks: list[Check] = []
        self._step_outcomes: list[dict[str, Any]] = []
        self._problems: list[str] = []
        self.models: dict[str, ScriptedModel] = {}
        self.auditors = Auditors()
        self.handoff_handlers: dict[str, HandoffHandler] = {}
        setup = scenario.get("setup", {})
        self.strategies = strategies(setup, self)
        kit_setup = setup.get("kit", {})
        kwargs: dict[str, Any] = {}
        if "max_chain_depth" in kit_setup:
            kwargs["max_chain_depth"] = kit_setup["max_chain_depth"]
        if "process_timeout_ms" in kit_setup:
            kwargs["process_timeout"] = kit_setup["process_timeout_ms"] / 1000
        if store is not None:
            kwargs["store"] = store
        if "router" in kit_setup:
            router = kit_setup["router"]
            kwargs["inbound_router"] = ScriptedRouter(
                router["room_id"], router.get("gate"), self.gates
            )
        kwargs |= kit_options(kit_setup, self)
        kwargs |= delivery_options(kit_setup)
        kwargs |= lock_options(kit_setup)
        kwargs |= verification_options(kit_setup)
        kwargs |= media_options(kit_setup, self)
        kwargs |= kit_orchestration(kit_setup, self)
        self.kit = RoomKit(**kwargs)

    def strategy(self, strategy_id: str) -> Orchestration:
        """The scripted strategy *strategy_id* names (README, \"Setup\")."""
        if strategy_id not in self.strategies:
            raise UnreadableError(f"no scripted strategy {strategy_id!r}")
        return self.strategies[strategy_id]

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

    def record_room_callback(
        self, room_id: str, channel_id: str, call: str, binding: ChannelBinding | None
    ) -> None:
        """One room callback a scripted channel was called with (§6.1), also
        named in the hooks' order as ``<channel_id>.<call>``."""
        if not self._recording:
            return
        item: dict[str, Any] = {"call": call}
        if binding is not None:
            item["binding"] = binding.model_dump(mode="json")
        per_room = self.observed.room_callbacks.setdefault(room_id, {})
        per_room.setdefault(channel_id, []).append(item)
        self.observed.hook_order.append(f"{channel_id}.{call}")

    def record_hook(self, name: str, trigger: str, room_id: str, payload: Any) -> None:
        if not self._recording:
            return
        event = payload.model_dump(mode="json") if isinstance(payload, RoomEvent) else None
        call: dict[str, Any] = {"trigger": trigger, "room_id": room_id, "event": event}
        if isinstance(payload, ToolCallEvent):
            call["tool_call"] = tool_call_record(payload)
        if isinstance(payload, DeliveryStatus):
            call["delivery_status"] = payload.model_dump(mode="json")
        if isinstance(payload, StatusEntry):
            call["status_entry"] = payload.model_dump(mode="json")
        if isinstance(payload, Observation):
            call["observation"] = payload.model_dump(mode="json")
        if isinstance(payload, PendingInputEvent):
            self.saw_request(payload.pending_id)
            call["request"] = request_record(payload)
        if isinstance(payload, AIGenerationEvent):
            call["generation"] = generation_record(payload)
        if isinstance(payload, ToolRoundEvent):
            call["tool_round"] = tool_round_record(payload)
        if isinstance(payload, AIResponseEvent):
            call["response"] = response_record(payload)
        if isinstance(payload, ThinkingEvent):
            call["thinking"] = {
                "channel_id": payload.channel_id,
                "thinking": payload.thinking,
                "round_index": payload.round_index,
            }
        self.observed.hooks.setdefault(name, []).append(call)
        self.observed.hook_order.append(name)

    def record_resolver(self, room_id: str, message: Any) -> None:
        """One call of the scripted resolver (§11.1)."""
        if not self._recording:
            return
        call = {
            "room_id": room_id,
            "channel_id": message.channel_id,
            "sender_id": message.sender_id,
        }
        self.observed.resolver.append(call)

    def record_media_resolver(self, url: str, room_id: str) -> None:
        """One call of the kit's media resolver (§6.4)."""
        if not self._recording:
            return
        self.observed.media_resolver.append({"url": url, "room_id": room_id})

    def record_identity_hook(
        self, name: str, trigger: str, room_id: str, event: RoomEvent, id_result: Any
    ) -> None:
        """One call of a scripted identity hook (§11.2): its event and the
        IdentityResult it is handed."""
        if not self._recording:
            return
        call = {
            "trigger": trigger,
            "room_id": room_id,
            "event": event.model_dump(mode="json"),
            "identity_result": id_result.model_dump(mode="json"),
        }
        self.observed.hooks.setdefault(name, []).append(call)
        self.observed.hook_order.append(name)

    def saw_request(self, pending_id: str) -> None:
        """A human-input request the run shows: the next ``$p<n>``."""
        if pending_id not in self.observed.requests:
            self.observed.requests.append(pending_id)

    async def _on_framework_event(self, event: FrameworkEvent) -> None:
        if event.type == "room_created" and event.room_id is not None:
            self.observed.created_rooms.append(event.room_id)
        if event.type == "user_input_required":
            self.saw_request(str(event.data.get("pending_id")))
        if not self._recording:
            return
        # The members the RFC's Data column names, flat; `data` kept whole
        # too, for the rows that name `data.x` (§8.2). Read in the serialized
        # form: a status_posted's data is its entry's dump, its status an enum.
        data = to_jsonable_python(event.data, fallback=str)
        flat: dict[str, Any] = {"event": event.type, **data, "data": data}
        for key in ("room_id", "channel_id", "event_id"):
            value = getattr(event, key)
            if value is not None:
                flat[key] = value
        if event.type in AI_EVENTS:
            self.observed.ai_events.append(flat)
        else:
            self.observed.framework_events.append(flat)

    # -- the run ----------------------------------------------------------

    async def _subscribe_thinking(self, room_id: str) -> None:
        """Records the room's thinking events, for a scenario that expects them."""
        if "thinking_events" not in self.scenario.get("expect", {}):
            return

        async def record(event: EphemeralEvent) -> None:
            if event.type in _THINKING_EVENTS:
                data = event.data
                self.observed.thinking_events.setdefault(room_id, []).append(
                    {"type": event.type.value, **data}
                )

        await self.kit.realtime.subscribe_to_room(room_id, record)

    async def _thinking_read(self) -> None:
        """Lets the realtime backend's own tasks hand what it was published."""
        if "thinking_events" in self.scenario.get("expect", {}):
            for _ in range(20):
                await asyncio.sleep(0)

    async def run(self) -> list[str]:
        """Runs the scenario; returns every mismatch with its expectations."""
        try:
            await self._setup()
            self._recording = True
            for position, step in enumerate(self.scenario["steps"]):
                await self._step(step, f"steps[{position}]")
            await self.idle()
            await self._thinking_read()
            await self._observe()
        finally:
            await self.kit.close()
            self.auditors.close()
        return self._problems + compare(
            self.scenario.get("expect", {}), self.observed, self._checks
        )

    async def _setup(self) -> None:
        setup = self.scenario.get("setup", {})
        recorded = LEVEL_0_EVENTS + LEVEL_2_EVENTS + INPUT_EVENTS + VERIFICATION_EVENTS + AI_EVENTS
        recorded += LOCK_EVENTS
        for event_type in recorded:
            self.kit.on(event_type)(self._on_framework_event)
        self.auditors.set_up(setup.get("session_auditor"), self.kit)
        for channel in setup.get("channels", []):
            register_channel(self.kit, self, channel)
        await store_identities(self.kit, setup.get("identities", []))
        for room in setup.get("rooms", []):
            await create_room(self.kit, room)
            if "store_metadata" in room:
                await self.kit.store.patch_room_metadata(room["id"], room["store_metadata"])
            await self._subscribe_thinking(room["id"])
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
        for hook in setup.get("identity_hooks", []):
            register_identity_hook(self.kit, self, hook)
        await start_backend(self.kit)
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
        elif "wait_for_lock" in step:
            room_id, waiters = step["wait_for_lock"], step.get("waiters", 1)
            try:
                await wait_for_lock(self.kit, room_id, waiters)
            except TimeoutError:
                self._problems.append(f"{path}: {waiters} calls never waited for {room_id}'s lock")
        elif "lock_releases" in step:
            set_lock_releases(self.kit, step["lock_releases"])
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

    def leave_after_turn(self, actions: list[Mapping[str, Any]], scope: Mapping[str, str]) -> None:
        """Start the work a turn leaves for after it: a task in the turn's
        context, which makes *actions* in order once the turn ended."""
        host = current_turn_host()
        if host is None:
            self._problems.append("an after_turn reaction outside a turn: no turn's host")
            return
        self._after_turn.append(asyncio.ensure_future(self._after(host, actions, scope)))

    async def _after(
        self, host: Any, actions: list[Mapping[str, Any]], scope: Mapping[str, str]
    ) -> None:
        try:
            async with asyncio.timeout(GATE_BOUND_S):
                while host.open:
                    await asyncio.sleep(0.001)
                # The owner reads the turn's frames on its own: across a
                # backend, a call could overtake them. Work after the turn
                # is made once the kit's own work for it is over.
                quiet = 0
                while quiet < 3:
                    busy = await self._drain_once(after_turn=False)
                    await asyncio.sleep(0)
                    quiet = 0 if busy else quiet + 1
        except TimeoutError:
            self._problems.append(f"a turn did not end within {GATE_BOUND_S} s for its after_turn")
            return
        for action in actions:
            await self.run_action(action, scope)

    async def invoke(self, spec: Mapping[str, Any], scope: Mapping[str, str]) -> dict[str, Any]:
        """Makes one call; returns its outcome, its error named, never raising."""
        resolved = await self.resolve(dict(spec), scope)
        calls = CALLS | STORE_CALLS | TASK_CALLS | MEMBER_CALLS | INPUT_CALLS | STRATEGY_CALLS
        call = (calls | VERIFICATION_CALLS | ROUTING_CALLS | HANDOFF_CALLS | DISCUSSION_CALLS)[
            spec["call"]
        ]
        try:
            result = await asyncio.wait_for(call(self.kit, self, resolved), CALL_BOUND_S)
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
            # A delegation through the turn's host answers its task's id.
            named = result if spec["call"] == "host.delegate" else result_event_id(result)
            self._labels[spec["as"]] = named
            self._labels |= opened_labels(spec, result)
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

    async def _drain_once(self, *, after_turn: bool = True) -> bool:
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
            for held in (
                self.kit._pending_hook_tasks,
                self.kit._decided_writes,
                # A strategy's background runs (RFC §19.7.3): a run waits
                # for its task, then hands its outcome back.
                self.kit._background_runs,
            )
            for task in held
            if not task.done()
        ]
        tasks.extend(task for task in self._background.values() if not task.done())
        if after_turn:
            tasks.extend(task for task in self._after_turn if not task.done())
        if tasks:
            busy = True
            await asyncio.wait(tasks, timeout=CALL_BOUND_S)
        if await backend_busy(self.kit):
            busy = True
            await asyncio.sleep(0.005)
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

    def labelled(self, value: Any) -> Any:
        """*value* with each ``$<label>`` a call's ``as`` named read as what
        it names: a scripted model's call arguments, as the model writes
        the call (README, "Scripted agents")."""
        if isinstance(value, dict):
            return {key: self.labelled(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.labelled(item) for item in value]
        if isinstance(value, str) and value.startswith("$") and self._labels.get(value[1:]):
            return self._labels[value[1:]]
        return value

    async def _reference(self, value: str, scope: Mapping[str, str]) -> str:
        if value in ("$room", "$event"):
            return scope[value[1:]]
        if value == "$request" and "request" in scope:
            return scope["request"]
        if match := _REQUEST.match(value):
            number = int(match[1])
            requests = self.observed.requests
            return requests[number - 1] if number <= len(requests) else value
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
            self.observed.participants[room.id] = [
                p.model_dump(mode="json") for p in await store.list_participants(room.id)
            ]
        self.observed.steps = self._step_outcomes
        self.observed.models = {
            channel_id: list(model.requests) for channel_id, model in self.models.items()
        }
        tools, sessions, problems = self.auditors.observe()
        self.observed.tool_audit, self.observed.session_audit = tools, sessions
        self._problems.extend(problems)


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

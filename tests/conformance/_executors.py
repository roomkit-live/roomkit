"""A scenario's turn executors, beside its kit (roomkit-specs
``conformance/README.md``, ``executors``).

The kit holds a channel the setup marks ``declared`` by its declaration alone,
and the executors that host it build their own copy, on the turn backend the
kit is built with: in memory, or across Redis in the suite's Redis mode. The
processes of a setup, and an executor in a process of its own, are nodes of
one backend across processes: on a broker and a registry kept in this
process, or on Redis in the suite's Redis mode.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit import AIChannel, RoomKit, current_turn_id
from roomkit.channels.base import Channel
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.realtime import EphemeralEvent, EphemeralEventType, InMemoryRealtime
from roomkit.scoring import MockScorer, ScoringHook
from roomkit.telemetry import Span, SpanKind
from roomkit.telemetry.config import TelemetryConfig
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.turns import (
    DistributedTurnBackend,
    ExecutorInfo,
    InMemoryTurnBackend,
    InMemoryTurnBroker,
    InMemoryTurnRegistry,
    TurnExecutor,
    TurnTimeouts,
)
from roomkit.turns.base import ExecutorLink, TurnBackend
from tests import redis_turn_mode
from tests.conformance._calls import Call, UnreadableError, read, scripted_channel
from tests.conformance._held_ends import HeldEnds
from tests.conformance._media import media_options
from tests.conformance._own_process import Apart, apart_problems
from tests.conformance._owner_kill import OwnerKill
from tests.conformance._owner_nodes import (
    FaultingPresence,
    OnTheContractAlone,
    OwnerFaults,
    OwnerRealtime,
    WithoutCatalogue,
)
from tests.conformance._script import GATE_BOUND_S, hook_handler
from tests.conformance._turn_errors import ERROR_CLASSES, inspect_failure

if TYPE_CHECKING:
    from roomkit.turns import AgentPresence
    from tests.conformance._harness import Harness


class _CutLink(ExecutorLink):
    """A turn's link that carries nothing once its executor was killed."""

    def __init__(self, link: ExecutorLink, killed: Callable[[], bool]) -> None:
        self.request = link.request
        self._link = link
        self._killed = killed

    async def receive(self) -> Any:
        control = await self._link.receive()
        if self._killed():
            await asyncio.Event().wait()
        return control

    async def send(self, frame: Any) -> int:
        return 0 if self._killed() else await self._link.send(frame)

    def turn_context(self) -> Any:
        return self._link.turn_context()

    def turn_ended(self) -> None:
        self._link.turn_ended()

    def stream_context(self) -> Any:
        return self._link.stream_context()


class _ScenarioExecutor(TurnExecutor):
    """An executor a step can kill or pause, or that never answers a turn it
    is given, *taken* told of each."""

    def __init__(
        self,
        build: Callable[[str], Channel | None],
        answers: bool,
        taken: Callable[[], None],
        **options: Any,
    ):
        super().__init__(build, **options)
        self._answers = answers
        self._taken = taken
        self.killed = False
        self.paused = False

    def declares(self, channel_id: str) -> bool:
        return not self.killed and not self.paused and super().declares(channel_id)

    def take(self, link: ExecutorLink) -> None:
        if self._answers:
            super().take(_CutLink(link, lambda: self.killed))
        else:
            self._taken()


class Executors:
    """The executors of one run: the turn backend they serve on, and each one."""

    def __init__(self, harness: Harness) -> None:
        setup = harness.scenario.get("setup", {})
        self._harness = harness
        self.telemetry: dict[str, MockTelemetryProvider] = {}
        self.captured_errors: dict[str, Exception] = {}
        self._bounds = _bounds(setup.get("kit", {}).get("turn_timeouts"))
        self._specs: dict[str, Mapping[str, Any]] = {
            e["id"]: e for e in setup.get("executors", [])
        }
        self._settings = setup.get("kit", {})
        self._distributed = bool(self._specs and setup.get("processes"))
        # An executor in a process of its own reaches the kit across processes.
        self._across = self._distributed or any(
            spec.get("own_process") for spec in self._specs.values()
        )
        self._prefix = f"{redis_turn_mode.SESSION_PREFIX}:{uuid4().hex}"
        # The services the run's nodes share, out of the Redis mode.
        self._pair = (InMemoryTurnBroker(), InMemoryTurnRegistry())
        self._nodes: dict[str | None, TurnBackend] = {}
        self._process_ids = {p["id"] for p in setup.get("processes", [])}
        if self._distributed and "main" in self._process_ids:
            raise UnreadableError("main names the run's own process")
        self.faults = OwnerFaults(harness.gates.reach)
        self._shared_realtime = (
            InMemoryRealtime() if self._settings.get("shared_realtime") else None
        )
        self._channels = {channel["id"]: channel for channel in setup.get("channels", [])}
        self._builds: dict[tuple[str, str], int] = {}
        self._running: dict[str, _ScenarioExecutor] = {}
        self._backend: TurnBackend | None = None
        self.calls: dict[str, list[dict[str, Any]]] = {}
        """What each executor did, in order (README, ``executors``)."""
        self._published = asyncio.Semaphore(0)
        self.held_ends = HeldEnds(setup.get("channels", []), harness.gates.hold)
        """The channels whose turns hold their end once cut (README, ``holds_end``)."""
        self.owner_kill = OwnerKill()
        """The kit killed as its turns' owner (README, ``owner.kill``)."""
        self._kills_owner = any(
            step.get("call") == "owner.kill" for step in harness.scenario.get("steps", [])
        )
        self._apart: dict[str, Apart] = {}
        self.reserved: ExecutorInfo | None = None
        """The executor the kit's turns are submitted to, while a step
        reserved one (README, ``executor.reserve``)."""
        self.reserved_turns: dict[str, list[dict[str, Any]]] = {}
        """Per executor reserved, each turn submitted to it (README,
        ``reserved_turns``)."""

    def site_options(self, site: str, spec: Mapping[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if spec.get("telemetry"):
            recorder = MockTelemetryProvider()
            self.telemetry[site] = recorder
            options["telemetry"] = (
                TelemetryConfig(provider=recorder) if site == "owner" else recorder
            )
        if "error_classes" in spec:
            options["error_classes"] = [ERROR_CLASSES[name] for name in spec["error_classes"]]
        return options

    def kit_options(self) -> dict[str, Any]:
        """The turn backend the kit is built with, when the scenario has
        executors of its own, bounds a turn, holds a turn's end or kills the
        owner; the kit's default otherwise."""
        if not takes_a_transport(self._harness.scenario):
            return {}
        self._backend = self._node("main")
        self._submit_reserved(self._backend)
        if self.held_ends:
            self.held_ends.wrap(self._backend)
        if self._kills_owner:
            settings = self._settings
            if settings.get("turn_backend_faults") or settings.get("executor_catalogue") is False:
                raise UnreadableError("owner.kill takes no presence fixture of the kit")
            self.owner_kill.wrap(self._backend)
        return self.process_options("main")

    def _submit_reserved(self, backend: TurnBackend) -> None:
        """Has *backend* submit each of the kit's turns to the executor a
        step reserved, while one is (README, ``executor.reserve``): what a
        host that places its turns by its own rules does."""
        submit = backend.submit

        async def submitting(request: Any) -> Any:
            reserved = self.reserved
            if reserved is None:
                return await submit(request)
            entry: dict[str, Any] = {"room_id": request.room_id, "error": None}
            self.reserved_turns.setdefault(reserved.executor_id, []).append(entry)
            try:
                return await backend.submit_to(request, reserved)
            except Exception as error:
                # Its reason, as a refusal names it; any other failure's text.
                entry["error"] = getattr(error, "reason", None) or str(error)
                raise
            finally:
                entry["handed"] = request.handed

        target: Any = backend
        target.submit = submitting

    def node_id(self, process: str | None) -> str | None:
        """The node a step's ``process`` names on the run's distributed turn
        backend; none for a step that names no process."""
        if process is None:
            return None
        if not self._distributed or (process != "main" and process not in self._process_ids):
            raise UnreadableError(f"no process {process!r} on a distributed turn backend")
        node: Any = self._node(process)
        while isinstance(node, WithoutCatalogue):
            # Behind the runner's own transport, the node is the one it hands to.
            node = node.inner
        node_id: str = node.node_id
        return node_id

    def _node(self, name: str | None) -> TurnBackend:
        if name not in self._nodes:
            # The kit's own node too, out of the Redis mode, when an executor
            # apart meets it there.
            backend: TurnBackend
            if self._distributed or (self._across and not redis_turn_mode.ENABLED):
                backend = self._node_across()
            else:
                backend = _backend(self._bounds, self._prefix)
            if self._settings.get("turn_backend_faults"):
                backend = FaultingPresence(backend, self.faults)
            elif self._settings.get("turn_backend_contract_only"):
                backend = OnTheContractAlone(backend)
            if self._settings.get("executor_catalogue") is False:
                backend = WithoutCatalogue(backend)
            self._nodes[name] = backend
        return self._nodes[name]

    def process_options(self, name: str) -> dict[str, Any]:
        if name != "main" and not self._distributed:
            return {}
        options: dict[str, Any] = {"turn_backend": self._node(name)}
        if self._shared_realtime is not None or self._settings.get("turn_backend_faults"):
            options["realtime"] = OwnerRealtime(
                self._shared_realtime or InMemoryRealtime(), self.faults
            )
        return options

    async def close_process_workers(self, name: str) -> None:
        for worker, spec in self._specs.items():
            if spec.get("kit") == name and worker in self._running:
                await self.close(worker)

    def forget_process(self, name: str) -> None:
        self._nodes.pop(name, None)

    async def start_all(self) -> None:
        """Serves the executors the setup starts."""
        if self._settings.get("scoring"):
            ScoringHook(scorers=[MockScorer()]).attach(self._harness.kit)
        self._count_publications()
        for executor_id, spec in self._specs.items():
            if spec.get("started", True):
                await self.start(executor_id)

    def _count_publications(self) -> None:
        """Has the room owner count what it took of a turn's work after the
        turn: across a backend it arrives on its own, which no idle wait sees."""
        owner: Any = self._harness.kit._turns
        taken = owner.room_publication

        async def counting(message: Any) -> None:
            try:
                await taken(message)
            finally:
                self._published.release()

        owner.room_publication = counting

    async def publication_taken(self) -> None:
        """Waits until the room owner took the next publication after a turn."""
        await asyncio.wait_for(self._published.acquire(), GATE_BOUND_S)

    async def start(self, executor_id: str) -> None:
        """Serves one executor of the setup on the kit's backend, or as in a
        process of its own (README, an executor's ``own_process``)."""
        if executor_id not in self._specs or self._backend is None:
            raise UnreadableError(f"no executor {executor_id!r} in setup.executors")
        spec = self._specs[executor_id]
        if spec.get("own_process"):
            await self._start_apart(executor_id, spec)
            return
        executor = self._executor(executor_id, spec)
        name = spec.get("kit")
        if name is not None and name != "main" and name not in self._process_ids:
            raise UnreadableError(f"no process {name!r} for executor {executor_id!r}")
        node = self._node(name) if self._distributed else self._backend
        await node.serve(executor)
        self._running[executor_id] = executor

    def _node_across(self) -> DistributedTurnBackend:
        """One more node of the run's backend across processes: on its
        in-memory pair, or on Redis under the run's prefix in the Redis mode."""
        if redis_turn_mode.ENABLED:
            from roomkit.turns.redis import RedisTurnBackend

            return RedisTurnBackend(
                os.environ["REDIS_URL"], key_prefix=self._prefix, timeouts=self._bounds
            )
        return DistributedTurnBackend(*self._pair, timeouts=self._bounds)

    async def _start_apart(self, executor_id: str, spec: Mapping[str, Any]) -> None:
        """Serves one executor as in a process of its own, over a node of its
        own of the backend across processes the kit's meets."""
        if problems := apart_problems(spec, self._channels.values()):
            raise UnreadableError("; ".join(problems))
        make = partial(self._executor, executor_id, spec)
        self._apart[executor_id] = await Apart.start(make, self._node_across)

    def _executor(self, executor_id: str, spec: Mapping[str, Any]) -> _ScenarioExecutor:
        """The executor *spec* describes, its hooks registered."""
        bound = {"max_channels": spec["max_channels"]} if "max_channels" in spec else {}
        executor = _ScenarioExecutor(
            partial(self._build, executor_id),
            bool(spec.get("answers", True)),
            partial(self._harness.gates.reach, spec.get("gate", "")),
            hosts=(
                set(spec["host_predicate"]).__contains__
                if "host_predicate" in spec
                else spec.get("hosts")
            ),
            organizations=spec.get("organizations"),
            executor_id=executor_id,
            timeouts=self._bounds,
            **self.site_options(f"executor/{executor_id}", spec),
            **media_options(spec, self._harness),
            **bound,
        )
        self._add_hooks(executor, spec)
        return executor

    def _add_hooks(self, executor: TurnExecutor, spec: Mapping[str, Any]) -> None:
        """Registers the hooks and defines the room hooks of *spec* (§10.6.6)."""
        if spec.get("scoring"):
            ScoringHook(scorers=[MockScorer()]).attach(executor)
        for hook in spec.get("hooks", []):
            trigger, execution = _trigger(hook)
            handler = hook_handler(hook, self._harness)
            executor.hook(trigger, execution=execution, name=hook["name"])(handler)
        for hook in spec.get("room_hooks", []):
            trigger, execution = _trigger(hook)
            executor.room_hook(hook["name"], trigger, execution)(hook_handler(hook, self._harness))

    def _build(self, executor_id: str, channel_id: str) -> Channel | None:
        """The executor's own copy of a channel of the setup, whose calls it
        records, none for a channel outside what its factory ``builds``; a
        channel the kit holds live is built here only by a defect."""
        spec = self._channels.get(channel_id)
        builds = self._specs[executor_id].get("builds")
        if spec is None or (builds is not None and channel_id not in builds):
            return None
        harness = self._harness
        channel = scripted_channel(harness.kit, harness, spec, process=True)
        build = self._builds.get((executor_id, channel_id), 0) + 1
        self._builds[(executor_id, channel_id)] = build
        note = partial(harness.record_executor, executor_id, channel_id, build)
        note("built", None)
        turns: dict[str, dict[str, Any]] = {}
        _record_calls(channel, note, turns)
        _record_tools(channel, {tool["name"] for tool in spec.get("tools", [])}, turns)
        return channel

    def running(self, executor_id: str) -> _ScenarioExecutor:
        """The executor a step names, which must be serving here."""
        if executor_id in self._apart:
            raise UnreadableError(
                f"executor {executor_id!r} runs in a process of its own: steps start or close it"
            )
        if executor_id not in self._running:
            raise UnreadableError(f"executor {executor_id!r} is not running")
        return self._running[executor_id]

    async def close(self, executor_id: str) -> None:
        """Closes one executor: it takes no more turns and ends those it runs."""
        if executor_id in self._apart:
            await self._apart.pop(executor_id).close()
            return
        await self.running(executor_id).close()
        self._running.pop(executor_id)

    def kill(self, executor_id: str) -> None:
        """Stops one executor as a killed process stops: it says nothing more,
        and its turns are left to its owner's liveness checks."""
        self.running(executor_id).killed = True

    async def close_all(self) -> None:
        """Closes the executors a scenario left running, those in a process
        of their own whatever fails before, and stops reading for a killed
        owner."""
        try:
            for executor in self._running.values():
                await executor.close()
            self._running.clear()
            if None in self._nodes:
                await self._nodes[None].close()
            if self._shared_realtime is not None:
                await self._shared_realtime.close()
        finally:
            while self._apart:
                await self._apart.popitem()[1].close()
            await self.owner_kill.close()


def _trigger(hook: Mapping[str, Any]) -> tuple[HookTrigger, HookExecution]:
    """The trigger and the execution a scenario's hook names."""
    trigger = read(lambda: HookTrigger(hook["trigger"]))
    return trigger, read(lambda: HookExecution(hook.get("execution", "sync")))


def takes_a_transport(scenario: Mapping[str, Any]) -> bool:
    """Whether the runner builds the scenario's kit with a turn backend: the
    scenario has executors of its own, bounds a turn, holds a turn's end or
    kills the owner. Its kit takes its default otherwise."""
    setup = scenario.get("setup", {})
    return bool(
        setup.get("executors")
        or setup.get("kit", {}).get("turn_timeouts") is not None
        or any("holds_end" in channel for channel in setup.get("channels", []))
        or any(step.get("call") == "owner.kill" for step in scenario.get("steps", []))
    )


def _bounds(spec: Mapping[str, Any] | None) -> TurnTimeouts | None:
    """The bounds of ``setup.kit.turn_timeouts``, none when it is absent."""
    if spec is None:
        return None
    defaults = TurnTimeouts()
    return TurnTimeouts(
        accept=spec.get("accept_ms", defaults.accept * 1000) / 1000,
        heartbeat_interval=spec.get("heartbeat_interval_ms", defaults.heartbeat_interval * 1000)
        / 1000,
        heartbeat_timeout=spec.get("heartbeat_timeout_ms", defaults.heartbeat_timeout * 1000)
        / 1000,
        cancel=spec.get("cancel_ms", defaults.cancel * 1000) / 1000,
    )


def _backend(bounds: TurnTimeouts | None, key_prefix: str) -> TurnBackend:
    """The turn backend of the suite's mode (tests/redis_turn_mode.py), under
    *key_prefix* across Redis, where an executor in a process of its own
    meets it."""
    if redis_turn_mode.ENABLED:
        return redis_turn_mode.backend(bounds, key_prefix=key_prefix)
    backend = InMemoryTurnBackend()
    backend.timeouts = bounds
    return backend


def _record_calls(
    channel: Channel,
    note: Callable[..., dict[str, Any] | None],
    turns: dict[str, dict[str, Any]],
) -> None:
    """Has *channel* report each call its executor makes on it, as it starts;
    a detach from a room this build never saw let go of nothing, and is not
    reported (RFC §10.6.2). A turn's record, kept in *turns* by its id,
    gathers the calls its host's tool handler is handed (``tools``)."""
    seen: set[str | None] = set()

    def recorded(
        name: str, room_of: Callable[..., str | None], **details: Callable[..., Any]
    ) -> None:
        call = getattr(channel, name)

        async def reporting(*args: Any) -> Any:
            room_id = room_of(*args)
            if name != "on_room_detached" or room_id in seen:
                item = note(name, room_id, **{key: of(*args) for key, of in details.items()})
                if name == "on_event":
                    _keep_turn(item, turns)
            seen.add(room_id)
            return await call(*args)

        setattr(channel, name, reporting)

    recorded("on_event", lambda event, *_: event.room_id)
    recorded("on_room_attached", lambda room_id, *_: room_id)
    recorded("on_room_detached", lambda room_id, *_: room_id)
    recorded("channel_call", lambda room_id, *_: room_id, verb=lambda _, verb, *__: verb)
    recorded("close", lambda: None)


def _keep_turn(item: dict[str, Any] | None, turns: dict[str, dict[str, Any]]) -> None:
    """Keeps a turn's record under the turn's id, to gather its ``tools``."""
    if item is None:
        return
    item["tools"] = []
    turn_id = current_turn_id()
    if turn_id is not None:
        turns[turn_id] = item


def _record_tools(channel: Channel, scripted: set[str], turns: dict[str, dict[str, Any]]) -> None:
    """Has *channel*, a scripted agent, note each call of one of its own
    *scripted* tools its handler runs, under the turn it serves (README,
    ``executors``: ``tools``): the AI channel's ``tool_handler`` serves them."""
    if not isinstance(channel, AIChannel) or channel.tool_handler is None:
        return
    handler = channel.tool_handler

    async def noting(name: str, arguments: dict[str, Any]) -> Any:
        turn_id = current_turn_id()
        if name in scripted and turn_id is not None and turn_id in turns:
            turns[turn_id]["tools"].append(name)
        return await handler(name, arguments)

    channel.tool_handler = noting


async def _start(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await harness.executors.start(spec["executor"])


async def _close(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await harness.executors.close(spec["executor"])


async def _kill(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    harness.executors.kill(spec["executor"])


async def _kill_owner(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    harness.executors.owner_kill.kill()


async def _reserve(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    """Reserves an executor of the setup for the kit's turns, or none again."""
    executors = harness.executors
    executor_id = spec["executor"]
    if executor_id is None:
        executors.reserved = None
        return
    if executor_id not in executors._specs:
        raise UnreadableError(f"no executor {executor_id!r} in setup.executors")
    executors.reserved = ExecutorInfo(
        executor_id=executor_id, node=executors.node_id(spec.get("process"))
    )


async def _pause(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    harness.executors.running(spec["executor"]).paused = True


async def _resume(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    harness.executors.running(spec["executor"]).paused = False


async def _calls_so_far(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return harness.executor_calls(spec["executor"])


async def _publish(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    """Publishes on a channel's realtime as work its turn began would (§10.6.3)."""
    if "gate" in spec:
        await harness.gates.hold(spec["gate"])
    realtime = harness.executors.running(spec["executor"]).wiring.realtime
    event = read(
        lambda: EphemeralEvent(
            room_id=spec["room_id"],
            type=EphemeralEventType(spec["kind"]),
            user_id=spec["channel_id"],
            channel_id=spec["channel_id"],
            data=spec.get("data", {}),
        )
    )
    await realtime.publish_to_room(spec["room_id"], event)
    await harness.executors.publication_taken()


async def _channel_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    arguments = dict(spec.get("arguments", {}))
    if "organization_id" in spec:
        arguments["organization_id"] = spec["organization_id"]
    try:
        return await kit.channel_call(
            spec["channel_id"], spec["room_id"], spec["verb"], **arguments
        )
    except Exception as error:
        if "capture_error" not in spec:
            raise
        harness.executors.captured_errors[spec["capture_error"]] = error
        return inspect_failure(error)


async def _spans(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    """The completed spans of a kind at a site, those the call selects, in the
    order they ended (README, ``telemetry.spans``)."""
    spans = harness.executors.telemetry[spec["site"]].get_spans(SpanKind(spec["kind"]))
    attributes = spec.get("attributes", {})
    selected = [
        _span_record(span)
        for span in spans
        if spec.get("name", span.name) == span.name
        and all(
            key in span.attributes and span.attributes[key] == value
            for key, value in attributes.items()
        )
    ]
    if "distinct" not in spec:
        return selected
    answered: list[dict[str, Any]] = []
    for record in selected:
        kept = {member: record[member] for member in spec["distinct"]}
        if kept not in answered:
            answered.append(kept)
    return answered


async def _reset_spans(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """The site's recorder forgets the spans it recorded (README, ``telemetry.reset``)."""
    harness.executors.telemetry[spec["site"]].reset()


def _span_record(span: Span) -> dict[str, Any]:
    return {
        "name": span.name,
        "span_id": span.id,
        "parent_id": span.parent_id,
        "room_id": span.room_id,
        "channel_id": span.channel_id,
        "status": span.status,
        "attributes": dict(span.attributes),
        "error_message": span.error_message,
    }


async def _release_channel(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await kit.release_channel(spec["channel_id"])


async def _enable_room_hook(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await kit.enable_room_hook(
        spec["room_id"], spec["name"], organization_id=spec.get("organization_id")
    )


async def _executors(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    """The executors served on the kit's backend (README, ``executors``)."""
    return [
        {
            "executor_id": info.executor_id,
            "hosts": None if info.hosts is None else sorted(info.hosts),
            "organizations": (None if info.organizations is None else sorted(info.organizations)),
            "room_hooks": sorted(info.room_hooks),
            "online": info.online,
        }
        for info in await kit.executors()
    ]


async def _agent_presence(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    """Each of a room's agents, online or not (README, ``agent_presence``)."""
    arguments = {"organization_id": spec["organization_id"]} if "organization_id" in spec else {}

    async def present() -> list[AgentPresence]:
        while True:
            agents = await kit.agent_presence(spec["room_id"], **arguments)
            target = spec.get("wait_for_online")
            if target is None or any(
                agent.online and agent.channel_id == target for agent in agents
            ):
                return agents
            await asyncio.sleep(0.025)

    agents = (
        await asyncio.wait_for(present(), GATE_BOUND_S)
        if "wait_for_online" in spec
        else await present()
    )
    return [
        {
            "channel_id": presence.channel_id,
            "online": presence.online,
            "executors": sorted(presence.executors),
        }
        for presence in agents
    ]


async def _fault(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    if not harness.scenario["setup"]["kit"].get("turn_backend_faults"):
        raise UnreadableError("turn_backend.fault requires setup.kit.turn_backend_faults")
    harness.executors.faults.set(spec["operation"], spec["enabled"], spec.get("gate"))


EXECUTOR_CALLS: dict[str, Call] = {
    "telemetry.spans": _spans,
    "telemetry.reset": _reset_spans,
    "turn_backend.fault": _fault,
    "executor.start": _start,
    "executor.close": _close,
    "executor.kill": _kill,
    "owner.kill": _kill_owner,
    "executor.pause": _pause,
    "executor.resume": _resume,
    "executor.reserve": _reserve,
    "executor.calls": _calls_so_far,
    "channel.publish": _publish,
    "channel_call": _channel_call,
    "release_channel": _release_channel,
    "enable_room_hook": _enable_room_hook,
    "executors": _executors,
    "agent_presence": _agent_presence,
}

# Each call's arguments, beside `call` and `organization_id`.
EXECUTOR_ARGUMENTS: dict[str, set[str]] = {
    "telemetry.spans": {"site", "kind", "attributes", "name", "distinct"},
    "telemetry.reset": {"site"},
    "turn_backend.fault": {"operation", "enabled", "gate"},
    "executor.start": {"executor"},
    "executor.close": {"executor"},
    "executor.kill": {"executor"},
    "owner.kill": set(),
    "executor.pause": {"executor"},
    "executor.resume": {"executor"},
    "executor.reserve": {"executor", "process"},
    "executor.calls": {"executor"},
    "channel.publish": {"executor", "channel_id", "room_id", "kind", "data", "gate"},
    "channel_call": {"channel_id", "room_id", "verb", "arguments", "capture_error"},
    "release_channel": {"channel_id"},
    "enable_room_hook": {"room_id", "name"},
    "executors": set(),
    "agent_presence": {"room_id", "wait_for_online"},
}

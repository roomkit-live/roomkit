"""A scenario's turn executors, beside its kit (roomkit-specs
``conformance/README.md``, ``executors``).

The kit holds a channel the setup marks ``declared`` by its declaration alone,
and the executors that host it build their own copy, on the turn backend the
kit is built with: in memory, or across Redis in the suite's Redis mode.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.channels.base import Channel
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.realtime import EphemeralEvent, EphemeralEventType
from roomkit.turns import InMemoryTurnBackend, TurnExecutor, TurnTimeouts
from roomkit.turns.base import ExecutorLink, TurnBackend
from tests import redis_turn_mode
from tests.conformance._calls import Call, UnreadableError, read, scripted_channel
from tests.conformance._script import GATE_BOUND_S, hook_handler

if TYPE_CHECKING:
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
        self._bounds = _bounds(setup.get("kit", {}).get("turn_timeouts"))
        self._specs: dict[str, Mapping[str, Any]] = {
            e["id"]: e for e in setup.get("executors", [])
        }
        self._channels = {channel["id"]: channel for channel in setup.get("channels", [])}
        self._builds: dict[tuple[str, str], int] = {}
        self._running: dict[str, _ScenarioExecutor] = {}
        self._backend: TurnBackend | None = None
        self.calls: dict[str, list[dict[str, Any]]] = {}
        """What each executor did, in order (README, ``executors``)."""
        self._published = asyncio.Semaphore(0)

    def kit_options(self) -> dict[str, Any]:
        """The turn backend the kit is built with, when the scenario has
        executors of its own or bounds a turn; the kit's default otherwise."""
        if not self._specs and self._bounds is None:
            return {}
        self._backend = _backend(self._bounds)
        return {"turn_backend": self._backend}

    async def start_all(self) -> None:
        """Serves the executors the setup starts."""
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
        """Serves one executor of the setup on the kit's backend."""
        if executor_id not in self._specs or self._backend is None:
            raise UnreadableError(f"no executor {executor_id!r} in setup.executors")
        spec = self._specs[executor_id]
        executor = _ScenarioExecutor(
            partial(self._build, executor_id),
            bool(spec.get("answers", True)),
            partial(self._harness.gates.reach, spec.get("gate", "")),
            hosts=set(spec.get("hosts", [])),
            executor_id=executor_id,
            timeouts=self._bounds,
        )
        self._add_hooks(executor, spec)
        await self._backend.serve(executor)
        self._running[executor_id] = executor

    def _add_hooks(self, executor: TurnExecutor, spec: Mapping[str, Any]) -> None:
        """Registers the hooks and defines the room hooks of *spec* (§10.6.6)."""
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
        _record_calls(channel, note)
        return channel

    def running(self, executor_id: str) -> _ScenarioExecutor:
        """The executor a step names, which must be serving."""
        if executor_id not in self._running:
            raise UnreadableError(f"executor {executor_id!r} is not running")
        return self._running[executor_id]

    async def close(self, executor_id: str) -> None:
        """Closes one executor: it takes no more turns and ends those it runs."""
        await self.running(executor_id).close()
        self._running.pop(executor_id)

    def kill(self, executor_id: str) -> None:
        """Stops one executor as a killed process stops: it says nothing more,
        and its turns are left to its owner's liveness checks."""
        self.running(executor_id).killed = True

    async def close_all(self) -> None:
        """Closes the executors a scenario left running."""
        for executor in self._running.values():
            await executor.close()
        self._running.clear()


def _trigger(hook: Mapping[str, Any]) -> tuple[HookTrigger, HookExecution]:
    """The trigger and the execution a scenario's hook names."""
    trigger = read(lambda: HookTrigger(hook["trigger"]))
    return trigger, read(lambda: HookExecution(hook.get("execution", "sync")))


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


def _backend(bounds: TurnTimeouts | None) -> TurnBackend:
    """The turn backend of the suite's mode (tests/redis_turn_mode.py)."""
    if redis_turn_mode.ENABLED:
        return redis_turn_mode.backend(bounds)
    backend = InMemoryTurnBackend()
    backend.timeouts = bounds
    return backend


def _record_calls(channel: Channel, note: Callable[..., None]) -> None:
    """Has *channel* report each call its executor makes on it, as it starts;
    a detach from a room this build never saw let go of nothing, and is not
    reported (RFC §10.6.2)."""
    seen: set[str | None] = set()

    def recorded(
        name: str, room_of: Callable[..., str | None], **details: Callable[..., Any]
    ) -> None:
        call = getattr(channel, name)

        async def reporting(*args: Any) -> Any:
            room_id = room_of(*args)
            if name != "on_room_detached" or room_id in seen:
                note(name, room_id, **{key: reader(*args) for key, reader in details.items()})
            seen.add(room_id)
            return await call(*args)

        setattr(channel, name, reporting)

    recorded("on_event", lambda event, *_: event.room_id)
    recorded("on_room_attached", lambda room_id, *_: room_id)
    recorded("on_room_detached", lambda room_id, *_: room_id)
    recorded("channel_call", lambda room_id, *_: room_id, verb=lambda _, verb, *__: verb)
    recorded("close", lambda: None)


async def _start(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await harness.executors.start(spec["executor"])


async def _close(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await harness.executors.close(spec["executor"])


async def _kill(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    harness.executors.kill(spec["executor"])


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
    return await kit.channel_call(spec["channel_id"], spec["room_id"], spec["verb"], **arguments)


async def _release_channel(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await kit.release_channel(spec["channel_id"])


async def _enable_room_hook(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await kit.enable_room_hook(spec["room_id"], spec["name"])


EXECUTOR_CALLS: dict[str, Call] = {
    "executor.start": _start,
    "executor.close": _close,
    "executor.kill": _kill,
    "executor.pause": _pause,
    "executor.resume": _resume,
    "executor.calls": _calls_so_far,
    "channel.publish": _publish,
    "channel_call": _channel_call,
    "release_channel": _release_channel,
    "enable_room_hook": _enable_room_hook,
}

# Each call's arguments, beside `call` and `organization_id`.
EXECUTOR_ARGUMENTS: dict[str, set[str]] = {
    "executor.start": {"executor"},
    "executor.close": {"executor"},
    "executor.kill": {"executor"},
    "executor.pause": {"executor"},
    "executor.resume": {"executor"},
    "executor.calls": {"executor"},
    "channel.publish": {"executor", "channel_id", "room_id", "kind", "data", "gate"},
    "channel_call": {"channel_id", "room_id", "verb", "arguments"},
    "release_channel": {"channel_id"},
    "enable_room_hook": {"room_id", "name"},
}

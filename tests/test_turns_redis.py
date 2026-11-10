"""The Redis turn backend against a real Redis (RFC §10.6): set REDIS_URL.

A room owner and executors in other processes (subprocesses running
tests/redis_turn_executor.py), or several backends in this process standing
for several processes: placement, liveness, a lost executor, the work a turn
began reaching its owner.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from roomkit import (
    AgentIdentity,
    ChannelDeclaration,
    HookExecution,
    HookTrigger,
    RoomKit,
    current_turn_host,
)
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.steering import Cancel
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, StreamTextDelta
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from roomkit.tools.context import current_tool_room_id
from roomkit.turns import TurnClient, TurnExecutor, TurnTimeouts
from roomkit.turns._redis_beat import BeatThread, LiveExecutors
from roomkit.turns._redis_clients import call_bound, read_block, redis_clients
from roomkit.turns._redis_inbox import Envelope, Inbox
from roomkit.turns._redis_links import RedisExecutorLink
from roomkit.turns._redis_presence import Keys, Registry
from roomkit.turns._redis_routes import REFUSED
from roomkit.turns.redis import RedisTurnBackend
from roomkit.turns.wire import CommittedEvent, End, HostResult, RoomCall
from tests import redis_turn_mode
from tests.redis_silencing import Silencing, redis_address
from tests.test_framework import SimpleChannel
from tests.test_turns_channel_control import _Agent, _Recording, _turn
from tests.test_turns_own_executor import _Tagged

pytestmark = pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="REDIS_URL not set")

TIMEOUTS = TurnTimeouts(heartbeat_interval=0.2, heartbeat_timeout=1.0, accept=3.0, cancel=1.0)
EXECUTOR = Path(__file__).with_name("redis_turn_executor.py")


@pytest.fixture
def prefix() -> str:
    return f"roomkit:test:{uuid4().hex[:12]}"


@pytest.fixture
async def redis_client(prefix: str) -> AsyncIterator[Any]:
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    yield client
    keys = [key async for key in client.scan_iter(f"{prefix}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()


def _backend(prefix: str, timeouts: TurnTimeouts = TIMEOUTS) -> RedisTurnBackend:
    return RedisTurnBackend(os.environ["REDIS_URL"], key_prefix=prefix, timeouts=timeouts)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """The SQLite store the owner and the executor processes share."""
    return tmp_path / "rooms.db"


async def _owner(
    prefix: str,
    *agents: str,
    store: ConversationStore | None = None,
    timeouts: TurnTimeouts = TIMEOUTS,
) -> RoomKit:
    kit = RoomKit(store=store or InMemoryStore(), turn_backend=_backend(prefix, timeouts))
    kit.register_channel(SimpleChannel("tx"))
    for agent in agents:
        kit.register_channel(ChannelDeclaration(channel_id=agent, identity=AgentIdentity()))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    for agent in agents:
        await kit.attach_channel("r", agent, category=ChannelCategory.INTELLIGENCE)
    return kit


def _said(text: str) -> InboundMessage:
    return InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body=text))


async def _answers(kit: RoomKit, channel_id: str = "worker") -> list[str]:
    return [
        e.content.body
        for e in await kit.store.list_events("r")
        if e.source.channel_id == channel_id and e.type == EventType.MESSAGE
    ]


class _ExecutorProcess:
    """An executor serving ``worker`` in a subprocess."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process

    @classmethod
    async def start(cls, prefix: str, executor_id: str, database: Path) -> _ExecutorProcess:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(EXECUTOR),
            os.environ["REDIS_URL"],
            prefix,
            executor_id,
            str(database),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        started = cls(process)
        await started.wait_for("ready")
        return started

    async def wait_for(self, line: str) -> None:
        assert self.process.stdout is not None
        while True:
            got = await asyncio.wait_for(self.process.stdout.readline(), 20)
            assert got, f"the executor process ended before printing {line!r}"
            if got.decode().strip() == line:
                return

    async def stop(self) -> None:
        if self.process.returncode is None:
            assert self.process.stdin is not None
            self.process.stdin.write(b"stop\n")
            await self.process.stdin.drain()
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def kill(self) -> None:
        self.process.kill()
        await self.process.wait()


# -- An executor in another process -----------------------------------------------


async def test_a_turn_runs_on_an_executor_in_another_process(
    prefix: str, redis_client: Any, database: Path
) -> None:
    executor = await _ExecutorProcess.start(prefix, "host-1", database)
    kit = await _owner(prefix, "worker", store=SQLiteStore(database))
    try:
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert await _answers(kit) == ["Hello from host-1."]
        assert dict(result.response_metadata)["turns"]["worker"]["loop_end_reason"] == "completed"
    finally:
        await kit.close()
        await executor.stop()


async def test_a_turn_whose_cpu_stalls_its_executor_keeps_beating(
    prefix: str, redis_client: Any, database: Path
) -> None:
    """The executor's loop stops three times the heartbeat timeout: its
    thread beats on, so the turn is slow, never lost (RFC §10.6.8)."""
    executor = await _ExecutorProcess.start(prefix, "host-1", database)
    kit = await _owner(prefix, "worker", store=SQLiteStore(database))
    try:
        started = time.monotonic()
        result = await kit.process_inbound(_said("Please stall."), room_id="r")

        assert time.monotonic() - started >= 3.0
        assert result.error is None
        assert await _answers(kit) == ["Hello from host-1."]
    finally:
        await kit.close()
        await executor.stop()


async def test_a_killed_executor_fails_its_turn_and_the_room_goes_on(
    prefix: str, redis_client: Any, database: Path
) -> None:
    first = await _ExecutorProcess.start(prefix, "host-1", database)
    kit = await _owner(prefix, "worker", store=SQLiteStore(database))
    second: _ExecutorProcess | None = None
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Please hold."), room_id="r"))
        await first.wait_for("streaming")
        killed = time.monotonic()
        await first.kill()
        result = await asyncio.wait_for(turn, 10)

        assert time.monotonic() - killed < 3 * TIMEOUTS.heartbeat_timeout
        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "executor_lost"
        # Another executor takes the room's next turn.
        second = await _ExecutorProcess.start(prefix, "host-2", database)
        await kit.process_inbound(_said("Hi again."), room_id="r")
        assert (await _answers(kit))[-1] == "Hello from host-2."
    finally:
        await kit.close()
        if second is not None:
            await second.stop()


# -- Several processes, as backends of this one --------------------------------------


def _agent_executor(
    executor_id: str, store: ConversationStore, *, hosts: set[str], answer: str
) -> TurnExecutor:
    return TurnExecutor(
        lambda channel_id: Agent(
            channel_id, provider=MockAIProvider(responses=[answer] * 20, streaming=True)
        ),
        hosts=hosts,
        executor_id=executor_id,
        timeouts=TIMEOUTS,
    )


async def test_a_turn_no_executor_declares_fails_at_once(prefix: str, redis_client: Any) -> None:
    kit = await _owner(prefix, "nobody")
    try:
        started = time.monotonic()
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert time.monotonic() - started < TIMEOUTS.accept
        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "no_executor"
    finally:
        await kit.close()


@pytest.mark.parametrize(
    ("rule", "agent", "reason"),
    [("*", "org-sales", None), ("org-*", "org-sales", None), ("org-*", "other", "no_executor")],
)
async def test_an_executor_elsewhere_declaring_a_rule_takes_the_agents_it_covers(
    prefix: str, redis_client: Any, rule: str, agent: str, reason: str | None
) -> None:
    """An executor of another process declaring every agent, or a prefix,
    runs the turns of an agent it was never named; one outside its prefix
    has no executor (RFC §10.6.7)."""
    store = InMemoryStore()
    backend = _backend(prefix)
    executor = _agent_executor("pool", store, hosts={rule}, answer="From the pool.")
    await backend.serve(executor)
    kit = await _owner(prefix, agent, store=store)
    try:
        result = await kit.process_inbound(_said("Hi."), room_id="r")
        answers = await _answers(kit, agent)

        assert getattr(result.error, "reason", None) == reason
        assert answers == ([] if reason else ["From the pool."])
    finally:
        await kit.close()
        await backend.close()
        await executor.close()


async def test_a_room_s_turns_stay_on_the_executor_holding_it(
    prefix: str, redis_client: Any
) -> None:
    """Two executors take ``worker``: the room's turns stay on the one that
    took its first, until it stops (RFC §10.6.7)."""
    store = InMemoryStore()
    hosts = []
    executors = []
    for name in ("host-a", "host-b"):
        backend = _backend(prefix)
        executor = _agent_executor(name, store, hosts={"worker"}, answer=f"From {name}.")
        await backend.serve(executor)
        hosts.append(backend)
        executors.append(executor)
    kit = await _owner(prefix, "worker", store=store)
    try:
        for i in range(4):
            await kit.process_inbound(_said(f"Hi {i}."), room_id="r")
        answers = await _answers(kit)
        assert len(set(answers)) == 1
        holder = 0 if answers[0] == "From host-a." else 1
        await hosts[holder].close()
        await executors[holder].close()
        await kit.process_inbound(_said("Hi again."), room_id="r")

        assert (await _answers(kit))[-1] == f"From {['host-a', 'host-b'][1 - holder]}."
    finally:
        await kit.close()
        for backend, executor in zip(hosts, executors, strict=True):
            await backend.close()
            await executor.close()


async def test_an_executor_without_the_room_s_hook_is_not_given_its_turns(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()
    plain_backend, hooked_backend = _backend(prefix), _backend(prefix)
    plain = _agent_executor("plain", store, hosts={"worker"}, answer="From plain.")
    hooked = _agent_executor("hooked", store, hosts={"worker"}, answer="From hooked.")
    seen: list[str] = []

    @hooked.room_hook("audit", HookTrigger.ON_AI_RESPONSE)
    async def audit(event: Any, context: Any) -> None:
        seen.append("audited")

    await plain_backend.serve(plain)
    await hooked_backend.serve(hooked)
    kit = await _owner(prefix, "worker", store=store)
    try:
        await kit.enable_room_hook("r", "audit")
        for i in range(3):
            await kit.process_inbound(_said(f"Hi {i}."), room_id="r")

        assert await _answers(kit) == ["From hooked."] * 3
    finally:
        await kit.close()
        for backend, executor in ((plain_backend, plain), (hooked_backend, hooked)):
            await backend.close()
            await executor.close()


# -- Channel controls, work after a turn, steering ----------------------------------


class _Worker(Agent):
    """An agent with a per-room verb, that keeps the events handed to it."""

    def __init__(self, channel_id: str, provider: Any) -> None:
        super().__init__(channel_id, provider=provider)
        self.handed: list[str] = []

    async def channel_call(self, room_id: str, verb: str, arguments: dict[str, Any]) -> Any:
        return {"verb": verb, "room": room_id, **arguments}

    async def on_room_event(self, room_id: str, event: RoomEvent) -> None:
        assert isinstance(event.content, TextContent)
        self.handed.append(event.content.body)


async def _hosted(
    prefix: str,
    store: ConversationStore,
    channel: Any,
    name: str = "host-a",
    timeouts: TurnTimeouts = TIMEOUTS,
) -> tuple[RedisTurnBackend, TurnExecutor]:
    backend = _backend(prefix, timeouts)
    executor = TurnExecutor(
        lambda channel_id: channel if channel_id == channel.channel_id else None,
        hosts={channel.channel_id},
        executor_id=name,
        timeouts=timeouts,
    )
    await backend.serve(executor)
    return backend, executor


async def test_a_channel_call_and_a_committed_event_reach_the_executor_elsewhere(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    try:
        answer = await kit.channel_call("worker", "r", "config_options", key="model")
        event = RoomEvent(
            room_id="r",
            source=EventSource(channel_id="tx", channel_type=ChannelType.SMS),
            content=TextContent(body="A message for the memory."),
        )
        await kit._turns.backend.control(
            CommittedEvent(channel_id="worker", room_id="r", event=event)
        )
        async with asyncio.timeout(5):
            while not worker.handed:
                await asyncio.sleep(0.02)

        assert answer == {"verb": "config_options", "room": "r", "key": "model"}
        assert worker.handed == ["A message for the memory."]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_work_after_a_turn_reaches_the_owner_that_submitted_it(
    prefix: str, redis_client: Any
) -> None:
    """Two owners share the backend: a status the executor posts once its
    turn ended (its host no longer open) reaches the one whose turn began
    it, through the owner port, never the other (RFC §10.6.3)."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    later_work: list[asyncio.Task[None]] = []
    open_when_posted: list[bool] = []

    async def post_later(turn_host: Any) -> None:
        await asyncio.sleep(0.3)  # the turn has returned
        open_when_posted.append(turn_host.open)
        await turn_host.post_status("worker", "thought", "info", detail="after the turn")

    @executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def later(event: Any, context: Any) -> None:
        later_work.append(asyncio.get_running_loop().create_task(post_later(current_turn_host())))

    asking, other = await _owner(prefix, "worker", store=store), await _owner(prefix, "worker")
    posted: dict[str, list[str]] = {"asking": [], "other": []}

    async def on(name: str, entry: Any) -> None:
        posted[name].append(entry.detail)

    await asking.status_bus.subscribe(lambda entry: on("asking", entry))
    await other.status_bus.subscribe(lambda entry: on("other", entry))
    try:
        await asking.process_inbound(_said("Hi."), room_id="r")
        async with asyncio.timeout(5):
            while not posted["asking"]:
                await asyncio.sleep(0.02)

        assert open_when_posted == [False]
        assert posted == {"asking": ["after the turn"], "other": []}
    finally:
        await asking.close()
        await other.close()
        await host.close()
        await executor.close()


async def test_a_steering_cancel_reaches_a_turn_elsewhere(prefix: str, redis_client: Any) -> None:
    store = InMemoryStore()
    started = asyncio.Event()

    class _Slow(MockAIProvider):
        """Streams a long answer, a word every 50 ms: the loop reads the
        cancel at its next delta."""

        async def generate_structured_stream(self, context: Any) -> Any:
            started.set()
            for i in range(200):
                yield StreamTextDelta(text=f"w{i} ")
                await asyncio.sleep(0.05)

    worker = _Worker("worker", _Slow(streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Hi."), room_id="r"))
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.sleep(0.1)
        reached = await kit.steer(Cancel(), "worker", room_id="r")
        result = await asyncio.wait_for(turn, 5)

        assert reached == 1
        assert dict(result.response_metadata)["turns"]["worker"]["loop_end_reason"] == "cancelled"
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_turn_for_an_executor_that_closed_fails_at_once(
    prefix: str, redis_client: Any
) -> None:
    """Its backend still runs and its beat is not stale yet: the request
    reaches its process, which answers at once that it no longer serves."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    try:
        await executor.close()
        started = time.monotonic()
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert time.monotonic() - started < TIMEOUTS.accept
        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "no_executor"
    finally:
        await kit.close()
        await host.close()


# -- What the review of 1b7af07f found -----------------------------------------------


@pytest.mark.parametrize("frame", ["output", "end"])
async def test_a_frame_redis_refuses_fails_the_turn_rather_than_waits(
    prefix: str, redis_client: Any, frame: str
) -> None:
    """The executor's process still beats, but no longer runs the turn: its
    owner gives it up within the heartbeat window, never waits for good."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    post = host._outbox.post
    refused: list[str] = []

    def refusing(node: str, envelope: Any, failed: Any = None, **kwargs: Any) -> None:
        if envelope.kind == "frame" and f'"type":"{frame}"' in envelope.body and not refused:
            refused.append(frame)
            failed(ConnectionError("OOM command not allowed"))
            return
        post(node, envelope, failed, **kwargs)

    host._outbox.post = refusing  # type: ignore[method-assign]
    kit = await _owner(prefix, "worker", store=store)
    try:
        result = await asyncio.wait_for(
            kit.process_inbound(_said("Hi."), room_id="r"), 6 * TIMEOUTS.heartbeat_timeout
        )

        assert refused == [frame]
        if frame == "output":
            assert isinstance(result.error, TurnLinkError)
        else:  # its answer was read whole: only its end is missing
            assert await _answers(kit) == ["Hi."]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_many_rooms_at_once_keep_every_turn_s_order(prefix: str, redis_client: Any) -> None:
    """Sixty rooms' turns at once, owner and executor writing concurrently:
    every message arrives in its turn's order, no connection runs out."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 100, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = RoomKit(store=store, turn_backend=_backend(prefix))
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(ChannelDeclaration(channel_id="worker", identity=AgentIdentity()))
    rooms = [f"r{i}" for i in range(60)]
    for room in rooms:
        await kit.create_room(room_id=room)
        await kit.attach_channel(room, "tx")
        await kit.attach_channel(room, "worker", category=ChannelCategory.INTELLIGENCE)
    try:
        results = await asyncio.gather(
            *(kit.process_inbound(_said("Hi."), room_id=room) for room in rooms)
        )

        assert [r.error for r in results] == [None] * len(rooms)
        assert host._routes._executor_links == {}
        assert kit._turns.backend._routes._owner_links == {}
    finally:
        await kit.close()
        await host.close()
        await executor.close()


class _SlowWrites:
    """A Redis client whose pipelines take *delay* to write (2 ms: a cut
    lands while a frame is being written)."""

    def __init__(self, client: Any, delay: float = 0.002) -> None:
        self._client = client
        self._delay = delay

    def pipeline(self, **kwargs: Any) -> Any:
        pipe = self._client.pipeline(**kwargs)
        execute = pipe.execute

        async def slow() -> Any:
            await asyncio.sleep(self._delay)
            return await execute()

        pipe.execute = slow
        return pipe

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


async def test_a_cut_while_frames_are_written_leaves_no_hole(
    prefix: str, redis_client: Any
) -> None:
    """Cancelled deliveries whose frames were being written each end with
    their turn's end: nothing is reported incomplete (RFC §10.6.8)."""
    store = InMemoryStore()
    started = asyncio.Event()

    class _Words(MockAIProvider):
        async def generate_structured_stream(self, context: Any) -> Any:
            started.set()
            for i in range(500):
                yield StreamTextDelta(text=f"w{i} ")
                await asyncio.sleep(0.001)

    worker = _Worker("worker", _Words(streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    host._outbox._client = _SlowWrites(host._outbox._client)
    kit = await _owner(prefix, "worker", store=store)
    try:
        for _ in range(10):
            started.clear()
            result = await kit.process_inbound(_said("Talk."), room_id="r", defer_delivery=True)
            await asyncio.wait_for(started.wait(), 5)
            await asyncio.sleep(0.05)
            assert result.delivery is not None
            await asyncio.wait_for(result.delivery.cancel(), 10)
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_an_executor_restarted_under_its_id_does_not_hold_the_dead_one_s_turn(
    prefix: str, redis_client: Any, database: Path
) -> None:
    """A supervisor restarts the killed process under the same executor id:
    the turn the dead process ran is lost, not kept alive by the new one's
    beats."""
    first = await _ExecutorProcess.start(prefix, "host-1", database)
    kit = await _owner(prefix, "worker", store=SQLiteStore(database))
    second: _ExecutorProcess | None = None
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Please hold."), room_id="r"))
        await first.wait_for("streaming")
        await first.kill()
        second = await _ExecutorProcess.start(prefix, "host-1", database)
        result = await asyncio.wait_for(turn, 6 * TIMEOUTS.heartbeat_timeout)

        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "executor_lost"
    finally:
        await kit.close()
        if second is not None:
            await second.stop()


async def test_a_declaration_lost_from_redis_is_written_again(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    try:
        await redis_client.delete(f"{prefix}:executors")  # a restart without persistence
        await asyncio.sleep(3 * TIMEOUTS.heartbeat_interval)
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert result.error is None
        assert await _answers(kit) == ["Hi."]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_room_hook_defined_after_serving_is_declared(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)

    @executor.room_hook("audit", HookTrigger.ON_AI_RESPONSE)
    async def audit(event: Any, context: Any) -> None: ...

    kit = await _owner(prefix, "worker", store=store)
    await kit.enable_room_hook("r", "audit")
    try:
        await asyncio.sleep(3 * TIMEOUTS.heartbeat_interval)
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert result.error is None
    finally:
        await kit.close()
        await host.close()
        await executor.close()


class _Remembers(Agent):
    """An agent whose attach and memory take time, logging what it does."""

    def __init__(self, channel_id: str, log: list[str]) -> None:
        super().__init__(channel_id, provider=MockAIProvider(responses=["ok"] * 5, streaming=True))
        self._log = log

    async def on_room_attached(self, room_id: str, binding: Any) -> None:
        await asyncio.sleep(0.2)
        self._log.append("attached")

    async def on_room_event(self, room_id: str, event: RoomEvent) -> None:
        await asyncio.sleep(0.2)  # a memory store's ingest
        assert isinstance(event.content, TextContent)
        self._log.append(f"memory:{event.content.body}")

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        assert isinstance(event.content, TextContent)
        self._log.append(f"turn:{event.content.body}")
        return await super().on_event(event, binding, context)


async def test_channel_controls_apply_before_the_turns_that_follow(
    prefix: str, redis_client: Any
) -> None:
    """As in memory: the attach, then a committed event, are applied on the
    executor before the room's next turn runs (RFC §10.6.3)."""
    store = InMemoryStore()
    log: list[str] = []
    host, executor = await _hosted(prefix, store, _Remembers("worker", log))
    kit = await _owner(prefix, "worker", store=store)
    event = RoomEvent(
        room_id="r",
        source=EventSource(channel_id="tx", channel_type=ChannelType.SMS),
        content=TextContent(body="earlier"),
    )
    try:
        await kit._turns.backend.control(
            CommittedEvent(channel_id="worker", room_id="r", event=event)
        )
        await kit.process_inbound(_said("later"), room_id="r")

        assert log == ["attached", "memory:earlier", "turn:later"]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_room_whose_executor_closed_moves_to_another(
    prefix: str, redis_client: Any
) -> None:
    """Its holder's executor closed, its process still running: the room's
    next turn goes to another executor, as in memory."""
    store = InMemoryStore()
    backends = []
    executors = []
    for name in ("host-a", "host-b"):
        backend = _backend(prefix)
        executor = _agent_executor(name, store, hosts={"worker"}, answer=f"From {name}.")
        await backend.serve(executor)
        backends.append(backend)
        executors.append(executor)
    kit = await _owner(prefix, "worker", store=store)
    try:
        await kit.process_inbound(_said("Hi."), room_id="r")
        holder = 0 if (await _answers(kit))[0] == "From host-a." else 1
        await executors[holder].close()
        result = await kit.process_inbound(_said("Hi again."), room_id="r")

        assert result.error is None
        assert (await _answers(kit))[-1] == f"From {['host-a', 'host-b'][1 - holder]}."
    finally:
        await kit.close()
        for backend, executor in zip(backends, executors, strict=True):
            await backend.close()
            await executor.close()


async def test_a_request_redis_refuses_leaves_no_link(prefix: str, redis_client: Any) -> None:
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    backend = kit._turns.backend
    post = backend._outbox.post

    def refusing(node: str, envelope: Any, failed: Any = None, **kwargs: Any) -> None:
        if envelope.kind == "request":
            failed(ConnectionError("READONLY You can't write against a read only replica."))
            return
        post(node, envelope, failed, **kwargs)

    backend._outbox.post = refusing
    try:
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert isinstance(result.error, TurnLinkError)
        assert backend._routes._owner_links == {}
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_an_executor_s_turn_its_owner_gave_up_leaves_no_link(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()

    class _Holds(MockAIProvider):
        async def generate_structured_stream(self, context: Any) -> Any:
            yield StreamTextDelta(text="Hello ")
            await asyncio.sleep(60)

    host, executor = await _hosted(prefix, store, _Worker("worker", _Holds(streaming=True)))
    kit = await _owner(prefix, "worker", store=store)
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Hi."), room_id="r"))
        async with asyncio.timeout(5):
            while not host._routes._executor_links:
                await asyncio.sleep(0.02)
        await kit.close()  # the owner goes: the executor cuts the turn
        await asyncio.gather(turn, return_exceptions=True)
        async with asyncio.timeout(6 * TIMEOUTS.heartbeat_timeout):
            while host._routes._executor_links:
                await asyncio.sleep(0.05)

        assert host._routes.running == frozenset()
    finally:
        await host.close()
        await executor.close()


# -- What the review of dd6c8bc6 found -----------------------------------------------


class _CutAfterWrite:
    """A TCP proxy to Redis that, once, forwards a write carrying every one
    of *marks* and drops the connection before Redis's reply: a client that
    retries then writes it again."""

    def __init__(self, marks: list[bytes]) -> None:
        self._marks = marks
        self.cut = 0
        self._server: asyncio.Server | None = None

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._relay, "127.0.0.1", 0)
        return int(self._server.sockets[0].getsockname()[1])

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()

    async def _relay(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        host, port = redis_address()
        server_reader, server_writer = await asyncio.open_connection(host, port)
        written = asyncio.Event()

        async def up() -> None:
            while data := await reader.read(65536):
                if not self.cut and all(mark in data for mark in self._marks):
                    self.cut += 1
                    written.set()
                server_writer.write(data)
                await server_writer.drain()

        async def down() -> None:
            try:
                while data := await server_reader.read(65536):
                    if written.is_set():
                        return  # Redis executed it; its reply never comes
                    writer.write(data)
                    await writer.drain()
            finally:
                writer.close()
                server_writer.close()

        await asyncio.gather(up(), down(), return_exceptions=True)


async def test_a_request_a_client_writes_twice_runs_its_turn_once(
    prefix: str, redis_client: Any
) -> None:
    """A client given as ``client=`` retries a pipeline whose connection
    dropped after Redis executed it: the request lands twice, the turn runs
    once (RFC §10.6.7)."""
    redis = pytest.importorskip("redis.asyncio")
    store = InMemoryStore()
    runs: list[str] = []

    class _Counts(MockAIProvider):
        async def generate_structured_stream(self, context: Any) -> Any:
            runs.append("run")
            yield StreamTextDelta(text="Hello once.")

    host, executor = await _hosted(prefix, store, _Worker("worker", _Counts(streaming=True)))
    proxy = _CutAfterWrite([b"XADD", b"request"])
    client = redis.Redis(host="127.0.0.1", port=await proxy.start())
    backend = RedisTurnBackend(
        os.environ["REDIS_URL"], client=client, key_prefix=prefix, timeouts=TIMEOUTS
    )
    kit = RoomKit(store=store, turn_backend=backend)
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(ChannelDeclaration(channel_id="worker", identity=AgentIdentity()))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "worker", category=ChannelCategory.INTELLIGENCE)
    try:
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert proxy.cut == 1
        assert result.error is None
        assert runs == ["run"]
        assert await _answers(kit) == ["Hello once."]
    finally:
        await kit.close()
        await client.aclose()
        await proxy.close()
        await host.close()
        await executor.close()


async def test_what_every_process_reads_of_the_others_costs_the_same_however_many_turns_run(
    prefix: str, redis_client: Any
) -> None:
    """Fifty processes running a thousand turns each: an owner's read of
    who is alive, made each interval and at each placement, costs what it
    costs with none (Redis runs nothing else meanwhile)."""
    backend = _backend(prefix)
    nodes = [uuid4().hex[:12] for _ in range(50)]
    now = int(time.time() * 1000)
    pipe = redis_client.pipeline(transaction=False)
    for node in nodes:
        pipe.zadd(f"{prefix}:alive", {f"{node}:w": now})
        declared = f'{{"node":"{node}","executor_id":"w","channels":["worker"],"room_hooks":[]}}'
        pipe.hset(f"{prefix}:executors", f"{node}:w", declared)
    await pipe.execute()

    async def read() -> float:
        times = []
        for _ in range(15):
            started = time.perf_counter()
            await backend._live.current()
            times.append(time.perf_counter() - started)
        return sorted(times)[len(times) // 2]

    try:
        idle = await read()
        pipe = redis_client.pipeline(transaction=False)
        for node in nodes:
            pipe.sadd(f"{prefix}:running:{node}", *(uuid4().hex for _ in range(1000)))
        await pipe.execute()
        busy = await read()

        assert busy < 5 * idle + 0.002, f"{busy * 1000:.1f} ms against {idle * 1000:.1f} ms"
    finally:
        await backend.close()


async def test_an_executor_served_again_under_its_id_takes_the_turns(
    prefix: str, redis_client: Any
) -> None:
    """A process reloads its executor: it closes one and serves another of
    the same id, which owners elsewhere then reach."""
    store = InMemoryStore()
    host = _backend(prefix)
    first = _agent_executor("host-a", store, hosts={"worker"}, answer="From the first.")
    await host.serve(first)
    await first.close()
    second = _agent_executor("host-a", store, hosts={"worker"}, answer="From the second.")
    await host.serve(second)
    await asyncio.sleep(3 * TIMEOUTS.heartbeat_interval)  # the beat thread's beats
    kit = await _owner(prefix, "worker", store=store)
    try:
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert result.error is None
        assert await _answers(kit) == ["From the second."]
    finally:
        await kit.close()
        await host.close()
        await second.close()


async def test_a_turn_s_end_still_queued_when_its_executor_s_process_closes_reaches_its_owner(
    prefix: str, redis_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executor's process closes right after its turn ended, its writes
    slow (a remote Redis): the end queued is written before it goes."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    host._outbox._client = _SlowWrites(host._outbox._client, delay=0.05)
    ended = asyncio.Event()
    send = RedisExecutorLink.send

    async def sending(self: RedisExecutorLink, frame: Any) -> int:
        seq = await send(self, frame)
        if isinstance(frame, End):
            ended.set()
        return seq

    monkeypatch.setattr(RedisExecutorLink, "send", sending)
    kit = await _owner(prefix, "worker", store=store)
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Hi."), room_id="r"))
        await asyncio.wait_for(ended.wait(), 5)
        await host.close()
        result = await asyncio.wait_for(turn, 6 * TIMEOUTS.heartbeat_timeout)

        assert result.error is None
        assert await _answers(kit) == ["Hi."]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_turn_whose_control_redis_refused_stops_on_its_executor_at_once(
    prefix: str, redis_client: Any
) -> None:
    """The owner could not write a control and gave the turn up: its
    executor is told, rather than finding its owner silent a heartbeat
    timeout later."""
    store = InMemoryStore()
    holding = asyncio.Event()
    stopped: list[float] = []

    class _Holds(MockAIProvider):
        async def generate_structured_stream(self, context: Any) -> Any:
            yield StreamTextDelta(text="Hello ")
            holding.set()
            try:
                await asyncio.sleep(60)
            finally:
                stopped.append(time.monotonic())

    host, executor = await _hosted(prefix, store, _Worker("worker", _Holds(streaming=True)))
    kit = await _owner(prefix, "worker", store=store)
    backend = kit._turns.backend
    post = backend._outbox.post
    refusing = {"armed": False}

    def refused(node: str, envelope: Any, failed: Any = None, **kwargs: Any) -> None:
        if refusing["armed"] and envelope.kind == "control" and failed is not None:
            refusing["armed"] = False
            failed(ConnectionError("OOM command not allowed"))
            return
        post(node, envelope, failed, **kwargs)

    backend._outbox.post = refused
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Hi."), room_id="r"))
        await asyncio.wait_for(holding.wait(), 5)
        refusing["armed"] = True  # the owner's next control (a heartbeat) is refused
        result = await asyncio.wait_for(turn, 6 * TIMEOUTS.heartbeat_timeout)
        gave_up = time.monotonic()
        async with asyncio.timeout(3 * TIMEOUTS.heartbeat_timeout):
            while not stopped:
                await asyncio.sleep(0.01)

        assert isinstance(result.error, TurnLinkError)
        assert stopped[0] - gave_up < TIMEOUTS.heartbeat_timeout / 2
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_malformed_member_in_redis_does_not_stop_the_turns(
    prefix: str, redis_client: Any
) -> None:
    """An entry no process of the backend wrote, with no node: the others
    are still read, and turns still placed."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    await redis_client.zadd(f"{prefix}:alive", {":w": int(time.time() * 1000)})
    kit = await _owner(prefix, "worker", store=store)
    try:
        result = await kit.process_inbound(_said("Hi."), room_id="r")

        assert result.error is None
        assert await _answers(kit) == ["Hi."]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_closed_process_is_neither_an_owner_nor_an_executor_in_redis(
    prefix: str, redis_client: Any
) -> None:
    """Closing withdraws the process as an owner and as its executors, and
    no later beat writes them again: the others stop counting it alive at
    their next read."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    await kit.process_inbound(_said("Hi."), room_id="r")
    owner_node = kit._turns.backend.node_id
    await kit.close()
    await host.close()
    await executor.close()
    await asyncio.sleep(3 * TIMEOUTS.heartbeat_interval)

    assert await redis_client.zscore(f"{prefix}:owners", owner_node) is None
    assert await redis_client.zrange(f"{prefix}:alive", 0, -1) == []


async def test_the_redis_mode_deletes_only_its_own_session_s_keys(redis_client: Any) -> None:
    """Sessions sharing one Redis: one ending leaves the others' keys."""
    other = f"roomkit:test:{uuid4().hex[:12]}:holder:worker:r"
    await redis_client.set(other, "kept", ex=60)
    try:
        await asyncio.to_thread(redis_turn_mode.clean_up)

        assert await redis_client.get(other) == b"kept"
    finally:
        await redis_client.delete(other)


# -- What the review of 0e328a19 found -----------------------------------------------


def _slow_beat_stop(backend: RedisTurnBackend, delay: float) -> None:
    """Widen the window in which *backend* closes but is still a member."""
    stop = backend._beat.stop

    def slow() -> None:
        time.sleep(delay)
        stop()

    backend._beat.stop = slow  # type: ignore[method-assign]


async def test_a_turn_placed_on_a_closing_process_from_an_older_view_goes_elsewhere(
    prefix: str, redis_client: Any
) -> None:
    """The owner read who is alive just before the room's holder began
    closing: the closing process refuses the turn, which another executor
    takes (RFC §10.6.7)."""
    store = InMemoryStore()
    closing_host = _backend(prefix)
    first = _agent_executor("host-a", store, hosts={"worker"}, answer="From A.")
    await closing_host.serve(first)
    other_host = _backend(prefix)
    second = _agent_executor("host-b", store, hosts={"worker"}, answer="From B.")
    await other_host.serve(second)
    kit = await _owner(prefix, "worker", store=store)
    await redis_client.set(f"{prefix}:holder:worker:r", f"{closing_host.node_id}:host-a")
    live = kit._turns.backend._live
    older = await live.current()
    try:
        _slow_beat_stop(closing_host, 0.5)
        closing = asyncio.ensure_future(closing_host.close())
        await asyncio.sleep(0.05)

        async def as_read_before() -> Any:
            return older

        live.current = as_read_before  # type: ignore[method-assign]
        result = await kit.process_inbound(_said("Hi."), room_id="r")
        await closing

        assert result.error is None
        assert await _answers(kit) == ["From B."]
    finally:
        await kit.close()
        await closing_host.close()
        await other_host.close()
        await first.close()
        await second.close()


async def test_a_channel_call_being_applied_when_its_process_closes_is_answered(
    prefix: str, redis_client: Any
) -> None:
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    control = executor.control
    applying = asyncio.Event()

    async def slow_control(message: Any) -> Any:
        applying.set()
        await asyncio.sleep(0.6)  # within cancel_timeout: answered, not given up
        return await control(message)

    executor.control = slow_control  # type: ignore[method-assign]
    try:
        call = asyncio.ensure_future(kit.channel_call("worker", "r", "set_mode", mode="fast"))
        await asyncio.wait_for(applying.wait(), 5)
        closing = asyncio.ensure_future(host.close())
        answer = await asyncio.wait_for(call, 5)
        await closing

        assert answer == {"verb": "set_mode", "room": "r", "mode": "fast"}
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_work_after_a_turn_asking_from_a_closing_process_is_answered_at_once(
    prefix: str, redis_client: Any
) -> None:
    """An executor's process closes while the work its turn began asks the
    owner: the work learns it at once, rather than wait on a view no
    longer read."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    try:
        await kit.process_inbound(_said("Hi."), room_id="r")
        turn_id = next(iter(host._routes._turn_owners))
        _slow_beat_stop(host, 0.5)
        closing = asyncio.ensure_future(host.close())
        await asyncio.sleep(0.05)
        call = RoomCall(
            channel_id="worker", room_id="r", executor_id="host-a", turn_id=turn_id, verb="send"
        )
        result = await asyncio.wait_for(host._routes.port.room_call(call), 1)
        await closing

        assert result.error is not None
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_beat_built_before_an_executor_was_served_again_does_not_withdraw_it(
    prefix: str, redis_client: Any
) -> None:
    """The beat thread built its beat from the closed executor's set just
    before the new one of its id was served, and wrote it after."""
    registry = Registry(redis_client, Keys(prefix))
    member = "node-a:host-a"
    declared = '{"node":"node-a","executor_id":"host-a","channels":["worker"],"room_hooks":[]}'

    def beat(generation: int, *, gone: bool) -> str:
        return json.dumps(
            {
                "node": "node-a",
                "generation": generation,
                "declared": {} if gone else {member: declared},
                "undeclared": [],
                "gone": [member] if gone else [],
                "running": [],
                "owner": False,
            }
        )

    await registry.beat("node-a", beat(2, gone=False), 60_000)
    await registry.beat("node-a", beat(1, gone=True), 60_000)

    assert await redis_client.zscore(f"{prefix}:alive", member) is not None


# -- What the review of 1f40351d found -----------------------------------------------


class _SlowOwner:
    """A room owner whose room calls take *delay*, the calls it ran kept."""

    def __init__(self, inner: Any, delay: float) -> None:
        self._inner = inner
        self._delay = delay
        self.ran: list[str] = []

    async def room_call(self, message: RoomCall) -> HostResult:
        await asyncio.sleep(self._delay)
        self.ran.append(message.verb)
        return HostResult(call_id=message.call_id, result="sent")

    async def room_publication(self, message: Any) -> None:
        await self._inner.room_publication(message)


async def test_a_room_call_its_owner_runs_while_closing_is_answered(
    prefix: str, redis_client: Any
) -> None:
    """The owner's process closes while it runs a room call of work after
    a turn: the work reads the call's result, not a lost owner it would
    retry (a message sent twice)."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    owner_backend = kit._turns.backend
    try:
        await kit.process_inbound(_said("Hi."), room_id="r")
        turn_id = next(iter(host._routes._turn_owners))
        slow = _SlowOwner(owner_backend._routes.owner, delay=0.6)
        owner_backend._routes.owner = slow
        call = RoomCall(
            channel_id="worker", room_id="r", executor_id="host-a", turn_id=turn_id, verb="send"
        )
        asking = asyncio.ensure_future(host._routes.port.room_call(call))
        await asyncio.sleep(0.05)
        closing = asyncio.ensure_future(owner_backend.close())
        result = await asyncio.wait_for(asking, 5)
        await closing

        assert slow.ran == ["send"]
        assert result.error is None and result.result == "sent"
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_work_reaching_a_closing_owner_is_refused_at_once(
    prefix: str, redis_client: Any
) -> None:
    """While the owner's process closes, answering the room call it was
    running, another room call reaches it: refused at once, not left to
    wait out the close."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    kit = await _owner(prefix, "worker", store=store)
    owner_backend = kit._turns.backend
    try:
        await kit.process_inbound(_said("Hi."), room_id="r")
        turn_id = next(iter(host._routes._turn_owners))
        owner_backend._routes.owner = _SlowOwner(owner_backend._routes.owner, delay=0.8)

        def call(verb: str) -> RoomCall:
            return RoomCall(
                channel_id="worker", room_id="r", executor_id="host-a", turn_id=turn_id, verb=verb
            )

        running = asyncio.ensure_future(host._routes.port.room_call(call("send")))
        await asyncio.sleep(0.05)
        closing = asyncio.ensure_future(owner_backend.close())
        await asyncio.sleep(0.05)
        started = time.monotonic()
        late = await asyncio.wait_for(host._routes.port.room_call(call("send_again")), 5)
        refused_after = time.monotonic() - started
        await running
        await closing

        assert late.error is not None
        assert refused_after < TIMEOUTS.heartbeat_interval * 2
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_channel_call_and_an_event_for_a_closing_holder_go_to_another_executor(
    prefix: str, redis_client: Any
) -> None:
    """Placed from a view read just before the room's holder began closing:
    the closing process applies neither and refuses both, which the other
    executor applies (RFC §10.6.3)."""
    store = InMemoryStore()
    workers = {
        name: _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
        for name in ("host-a", "host-b")
    }
    closing_host, first = await _hosted(prefix, store, workers["host-a"], "host-a")
    other_host, second = await _hosted(prefix, store, workers["host-b"], "host-b")
    kit = await _owner(prefix, "worker", store=store)
    await redis_client.set(f"{prefix}:holder:worker:r", f"{closing_host.node_id}:host-a")
    live = kit._turns.backend._live
    older = await live.current()
    try:
        _slow_beat_stop(closing_host, 0.5)
        closing = asyncio.ensure_future(closing_host.close())
        await asyncio.sleep(0.05)

        async def as_read_before() -> Any:
            return older

        live.current = as_read_before  # type: ignore[method-assign]
        answer = await kit.channel_call("worker", "r", "set_mode", mode="fast")
        event = RoomEvent(
            room_id="r",
            source=EventSource(channel_id="tx", channel_type=ChannelType.SMS),
            content=TextContent(body="Remember this."),
        )
        await kit._turns.backend.control(
            CommittedEvent(channel_id="worker", room_id="r", event=event)
        )
        await closing

        assert answer == {"verb": "set_mode", "room": "r", "mode": "fast"}
        assert workers["host-a"].handed == [] and workers["host-b"].handed == ["Remember this."]
    finally:
        await kit.close()
        await closing_host.close()
        await other_host.close()
        await first.close()
        await second.close()


# -- What the review of ff1faca9 found -----------------------------------------------


class _LoopThread:
    """An event loop on a thread of its own: a process of its own as far as
    the test's loop can tell, its stalls not the test's."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    async def run[T](self, work: Coroutine[Any, Any, T]) -> T:
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(work, self.loop))

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(5)
        self.loop.close()


@pytest.mark.parametrize("stall", ["as_its_end_is_sent", "once_its_task_ended"])
async def test_a_turn_that_ended_is_not_lost_to_a_stall_before_its_end_is_written(
    prefix: str, redis_client: Any, monkeypatch: pytest.MonkeyPatch, stall: str
) -> None:
    """The executor's process stalls past the heartbeat timeout (another
    turn's CPU) just after a turn sent its end, its writes slow (a remote
    Redis): its beat names the turn until the end is written, so the owner
    reads a turn that completed, not a lost one (RFC §10.6.8)."""
    store = InMemoryStore()
    process = _LoopThread()
    send = RedisExecutorLink.send

    async def sending(self: RedisExecutorLink, frame: Any) -> int:
        seq = await send(self, frame)
        if isinstance(frame, End):
            loop, stalled = asyncio.get_running_loop(), 2.5 * TIMEOUTS.heartbeat_timeout
            if stall == "as_its_end_is_sent":
                loop.call_soon(time.sleep, stalled)
            else:
                loop.call_later(0.01, time.sleep, stalled)
        return seq

    monkeypatch.setattr(RedisExecutorLink, "send", sending)

    async def serve() -> tuple[RedisTurnBackend, TurnExecutor]:
        worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
        host, executor = await _hosted(prefix, store, worker)
        host._outbox._client = _SlowWrites(host._outbox._client, delay=0.05)
        return host, executor

    host, executor = await process.run(serve())
    kit = await _owner(prefix, "worker", store=store)
    try:
        result = await asyncio.wait_for(kit.process_inbound(_said("Hi."), room_id="r"), 15)

        assert result.error is None
        assert await _answers(kit) == ["Hi."]
    finally:
        await kit.close()
        await process.run(host.close())
        await process.run(executor.close())
        process.stop()


@pytest.mark.parametrize("first", ["backend", "executor"])
async def test_an_executor_s_process_closing_mid_turn_ends_the_turn_within_the_cancel_bound(
    prefix: str, redis_client: Any, first: str
) -> None:
    """The executor's process closes while a turn waits on its model: it
    cuts the turn and writes its end before it goes, so the owner fails the
    turn at once, not on its silence a heartbeat timeout later (RFC
    §10.6.8)."""
    timeouts = TurnTimeouts(heartbeat_interval=0.2, heartbeat_timeout=3.0, accept=3.0, cancel=1.0)
    store = InMemoryStore()
    holding = asyncio.Event()

    class _Holds(MockAIProvider):
        async def generate_structured_stream(self, context: Any) -> Any:
            yield StreamTextDelta(text="Hello ")
            holding.set()
            await asyncio.sleep(60)

    worker = _Worker("worker", _Holds(streaming=True))
    host, executor = await _hosted(prefix, store, worker, timeouts=timeouts)
    kit = await _owner(prefix, "worker", store=store, timeouts=timeouts)
    try:
        turn = asyncio.ensure_future(kit.process_inbound(_said("Hi."), room_id="r"))
        await asyncio.wait_for(holding.wait(), 5)
        started = time.monotonic()
        closing = (host, executor) if first == "backend" else (executor, host)
        for closed in closing:
            await closed.close()
        result = await asyncio.wait_for(turn, 6 * timeouts.heartbeat_timeout)
        waited = time.monotonic() - started

        assert waited < timeouts.heartbeat_timeout / 2, result.error
        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "executor_lost"
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_turn_whose_placement_failed_is_not_the_room_s_previous_turn(
    prefix: str, redis_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(prefix)
    agent = _Agent()
    executor = _Recording(agent, "a")
    await backend.serve(executor)
    client = TurnClient(backend)
    try:
        await _turn(client)
        place = backend._placement.place
        failures = [ConnectionError("Redis did not answer")]

        async def placed_once_failing(*args: Any, **kwargs: Any) -> Any:
            if failures:
                raise failures.pop()
            return await place(*args, **kwargs)

        monkeypatch.setattr(backend._placement, "place", placed_once_failing)
        # The read that places the turn fails: no executor was given it.
        with pytest.raises(ConnectionError):
            await _turn(client)
        await _turn(client)

        first, second = executor.requests
        assert second.previous_turn is not None
        assert second.previous_turn.turn_id == first.turn_id
        assert ("detached", "room") not in agent.log
    finally:
        await client.close()


class _TakingAcross(TurnExecutor):
    """An executor other processes reach for one channel, keeping the
    requests it took."""

    def __init__(self, channel: _Agent) -> None:
        super().__init__(
            lambda channel_id: channel if channel_id == channel.channel_id else None,
            hosts={channel.channel_id},
            executor_id="across",
        )
        self.requests: list[Any] = []

    def take(self, link: Any) -> None:
        self.requests.append(link.request)
        super().take(link)


async def test_a_turn_refused_then_not_placed_again_is_not_the_room_s_previous_turn(
    prefix: str, redis_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, host = _backend(prefix), _backend(prefix)
    agent = _Agent()
    executor = _TakingAcross(agent)
    await host.serve(executor)
    await asyncio.sleep(3 * TIMEOUTS.heartbeat_interval)  # the beat thread's beats
    client = TurnClient(owner)
    try:
        await _turn(client)
        # The host's process refuses the next request unapplied (as one
        # closing does), then the owner's read placing it again fails.
        take = host._routes._routes["request"]
        refusals = [REFUSED]

        def refusing_once(envelope: Any) -> None:
            if refusals:
                host._routes._answer_back(envelope, refusals.pop())
                return
            take(envelope)

        monkeypatch.setitem(host._routes._routes, "request", refusing_once)
        place = owner._placement.place
        places: list[int] = []

        async def placed_then_failing(*args: Any, **kwargs: Any) -> Any:
            places.append(1)
            if len(places) == 2:  # the read placing it again, past the refusal
                raise ConnectionError("Redis did not answer")
            return await place(*args, **kwargs)

        monkeypatch.setattr(owner._placement, "place", placed_then_failing)
        with pytest.raises(ConnectionError):
            await _turn(client)
        await _turn(client)

        first, last = executor.requests
        assert last.previous_turn is not None
        assert last.previous_turn.turn_id == first.turn_id
        assert ("detached", "room") not in agent.log
    finally:
        await client.close()
        await host.close()
        await executor.close()


async def test_a_request_that_cannot_be_written_leaves_no_owner_link(
    prefix: str, redis_client: Any
) -> None:
    owner, host = _backend(prefix), _backend(prefix)
    agent = _Agent()
    executor = _TakingAcross(agent)
    await host.serve(executor)
    await asyncio.sleep(3 * TIMEOUTS.heartbeat_interval)  # the beat thread's beats
    client = TurnClient(owner)
    try:
        await _turn(client)
        await asyncio.sleep(0.2)  # the first turn's end read
        before = set(owner._routes._owner_links)
        # Bytes that are not UTF-8 have no text to fall back on: the request
        # cannot be written, and nothing waits for its turn.
        with pytest.raises(ValueError):
            await _turn(client, raw=b"\xff\xfe")

        assert set(owner._routes._owner_links) == before
    finally:
        await client.close()
        await host.close()
        await executor.close()


async def test_a_live_agent_stays_on_the_kit_s_executor_whoever_holds_the_room(
    prefix: str, redis_client: Any
) -> None:
    """RMK-837: the room's holder names a host's executor in another process,
    which ran the agent while the kit held it by its declaration; once the kit
    holds the agent live, its turns, channel calls and committed events go to
    the kit's own executor."""
    seen: list[str] = []
    store = InMemoryStore()
    host_backend = _backend(prefix)
    host = TurnExecutor(
        lambda channel_id: _Tagged(channel_id, "host", seen),
        hosts={"ai"},
        executor_id="host",
        timeouts=TIMEOUTS,
    )
    await host_backend.serve(host)
    kit = RoomKit(store=store, turn_backend=_backend(prefix))
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(ChannelDeclaration(channel_id="ai"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)

    async def say(body: str) -> None:
        message = InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body=body))
        await kit.process_inbound(message, room_id="r")

    await say("first")
    kit.unregister_channel("ai")
    kit.register_channel(_Tagged("ai", "kit", seen))
    await say("second")
    answer = await kit.channel_call("ai", "r", "whoami")
    events = await store.list_events("r")
    await kit._turns.control(CommittedEvent(channel_id="ai", room_id="r", event=events[0]))
    said = [e.content.body for e in await store.list_events("r") if e.source.channel_id == "ai"]
    await kit.close()
    await host_backend.close()
    await host.close()

    assert said == ["from host", "from kit"]
    assert answer == "kit"
    assert seen == ["call:kit", "event:kit"]


# Short silence bounds for the half-open tests, the heartbeat interval under
# half of each: the example's, the suite's, and one with room to spare.
_HALF_OPEN = [
    TurnTimeouts(heartbeat_interval=0.5, heartbeat_timeout=2.0),
    TurnTimeouts(heartbeat_interval=0.2, heartbeat_timeout=1.0),
    TurnTimeouts(heartbeat_interval=0.5, heartbeat_timeout=4.0),
]

_BEAT = json.dumps(
    {
        "node": "n",
        "generation": 1,
        "declared": {"n:e": ""},
        "undeclared": [],
        "gone": [],
        "running": [],
        "owner": True,
    }
)
"""What the half-open tests' process beats: one executor, an owner."""


def _beat_thread(url: str, keys: Keys, timeouts: TurnTimeouts) -> BeatThread:
    clients = redis_clients(url, timeouts)
    return BeatThread(
        clients.beat,
        keys.beat("n"),
        timeouts.heartbeat_interval,
        int(timeouts.heartbeat_timeout * 3000),
        lambda: _BEAT,
        bound=call_bound(timeouts),
        timeout_errors=clients.timeout_errors,
    )


@pytest.mark.parametrize(
    "timeouts", _HALF_OPEN, ids=lambda t: f"{t.heartbeat_interval}s-of-{t.heartbeat_timeout}s"
)
async def test_a_half_open_beat_connection_takes_nobody_for_lost(
    prefix: str, redis_client: Any, timeouts: TurnTimeouts
) -> None:
    """RMK-859: the beat's one connection goes half-open: its call gives up
    within half the silence bound, the beat goes on at once on a new
    connection, and the other processes never see this one lost."""
    proxy = Silencing()
    keys = Keys(prefix)
    beat = _beat_thread(await proxy.url(), keys, timeouts)
    live = LiveExecutors(
        Registry(redis_client, keys), timeouts.heartbeat_interval, timeouts.heartbeat_timeout, dict
    )
    beat.start()
    await asyncio.sleep(2 * timeouts.heartbeat_interval)
    live.start()
    await asyncio.sleep(2 * timeouts.heartbeat_interval)
    assert live.alive("n:e")

    proxy.silence()
    seen_lost = False
    for _ in range(round(15 * timeouts.heartbeat_timeout)):
        await asyncio.sleep(0.1)
        seen_lost = seen_lost or not live.alive("n:e")

    await asyncio.to_thread(beat.stop)
    await live.close()
    await proxy.close()
    assert not seen_lost


@pytest.mark.parametrize(
    "timeouts", _HALF_OPEN, ids=lambda t: f"{t.heartbeat_interval}s-of-{t.heartbeat_timeout}s"
)
async def test_a_half_open_inbox_reads_within_the_silence_bound(
    prefix: str, redis_client: Any, timeouts: TurnTimeouts
) -> None:
    """RMK-859: the inbox's connection goes half-open just as a read starts:
    an entry written then is handed on within heartbeat_timeout, read again
    at once on a new connection."""
    proxy = Silencing()
    handed: list[float] = []
    key = f"{prefix}:inbox:n"
    clients = redis_clients(await proxy.url(), timeouts)
    reading = asyncio.Event()
    read = clients.inbox.xread

    async def xread(*args: Any, **kwargs: Any) -> Any:
        reading.set()
        return await read(*args, **kwargs)

    clients.inbox.xread = xread
    inbox = Inbox(
        clients.inbox,
        key,
        lambda envelope: handed.append(time.monotonic()),
        owns_client=True,
        block=read_block(timeouts),
        timeout_errors=clients.timeout_errors,
    )
    inbox.start()
    await asyncio.sleep(2 * timeouts.heartbeat_interval)
    reading.clear()
    await reading.wait()

    proxy.silence()
    written = time.monotonic()
    await redis_client.xadd(key, Envelope("control", "n", "{}").fields())
    async with asyncio.timeout(3 * timeouts.heartbeat_timeout):
        while not handed:
            await asyncio.sleep(0.02)

    await inbox.close()
    await proxy.close()
    assert handed[0] - written < timeouts.heartbeat_timeout


async def test_a_read_cut_by_its_connection_counts_from_its_start(
    prefix: str, redis_client: Any
) -> None:
    """RMK-859: Redis freezes, then refuses: the read in flight, cut by its
    connection's bound, counts as one that hung, so the others are unread
    about a heartbeat timeout after the freeze, not that plus the bound."""
    timeouts = TurnTimeouts(heartbeat_interval=0.25, heartbeat_timeout=2.0)
    proxy = Silencing()
    keys = Keys(prefix)
    beat = _beat_thread(os.environ["REDIS_URL"], keys, timeouts)
    clients = redis_clients(await proxy.url(), timeouts)
    live = LiveExecutors(
        Registry(clients.pooled, keys),
        timeouts.heartbeat_interval,
        timeouts.heartbeat_timeout,
        dict,
        timeout_errors=clients.timeout_errors,
    )
    beat.start()
    await asyncio.sleep(0.5)
    live.start()
    await asyncio.sleep(0.5)
    assert live.alive("n:e")

    proxy.silence()
    proxy.refuse()
    await asyncio.sleep(timeouts.heartbeat_timeout + 3 * timeouts.heartbeat_interval)
    unread = not live.alive("n:e")

    await asyncio.to_thread(beat.stop)
    await live.close()
    await clients.pooled.aclose()
    await proxy.close()
    assert unread


async def test_a_unix_socket_url_reaches_redis_on_every_connection(prefix: str) -> None:
    """A unix:// connection refuses TCP's options (socket_keepalive): the
    backend's three connections reach Redis through one."""
    proxy = Silencing()
    path = os.path.join(tempfile.mkdtemp(), "redis.sock")
    clients = redis_clients(await proxy.unix_url(path), TIMEOUTS)
    beat = clients.beat()

    answers = [
        await clients.pooled.ping(),
        await clients.inbox.ping(),
        await asyncio.to_thread(beat.ping),
    ]

    beat.close()
    await clients.pooled.aclose()
    await clients.inbox.aclose()
    await proxy.close()
    assert answers == [True, True, True]


def _otel(exporter: Any, service: str) -> Any:
    """A process's OpenTelemetry provider, exporting to *exporter*."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from roomkit.telemetry.opentelemetry import OpenTelemetryProvider

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenTelemetryProvider(tracer_provider=tracer_provider, service_name=service)


async def test_a_turn_elsewhere_shows_in_the_owner_s_trace(
    prefix: str, redis_client: Any, database: Path
) -> None:
    """RMK-836: a turn an executor runs in another process (another backend,
    its own telemetry) shows as one trace with the owner's spans: its
    generation is the owner's broadcast's child, and its tool call the
    generation's (RFC §10.6.9)."""
    memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    exporter = memory.InMemorySpanExporter()
    kit = RoomKit(
        store=SQLiteStore(database),
        turn_backend=_backend(prefix),
        telemetry=_otel(exporter, "owner"),
    )
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(ChannelDeclaration(channel_id="worker", identity=AgentIdentity()))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "worker", category=ChannelCategory.INTELLIGENCE)
    backend = _backend(prefix)
    executor = TurnExecutor(
        _weather_agent,
        hosts={"worker"},
        executor_id="host-a",
        timeouts=TIMEOUTS,
        telemetry=_otel(exporter, "executor"),
    )
    await backend.serve(executor)

    result = await kit.process_inbound(_said("Hi."), room_id="r")
    await kit.close()
    await backend.close()
    await executor.close()

    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    generate = next(s for s in spans if s.name == "roomkit.llm.generate")
    tool = next(s for s in spans if s.name == "roomkit.tool.get_weather")
    assert result.error is None
    assert {s.context.trace_id for s in spans} == {generate.context.trace_id}
    assert generate.parent is not None and tool.parent is not None
    assert by_id[generate.parent.span_id].name == "roomkit.framework.broadcast"
    assert by_id[generate.parent.span_id].instrumentation_scope.name == "owner"
    assert tool.parent.span_id == generate.context.span_id


def _weather_agent(channel_id: str) -> Agent:
    """An agent that calls one tool, then answers."""

    async def weather(name: str, arguments: dict[str, Any]) -> str:
        return '{"temp": 22}'

    call = AIToolCall(id="t1", name="get_weather", arguments={})
    return Agent(
        channel_id,
        provider=MockAIProvider(
            ai_responses=[AIResponse(content="", tool_calls=[call]), AIResponse(content="Sunny.")]
        ),
        tools=[AITool(name="get_weather", description="The weather.")],
        tool_handler=weather,
    )


async def test_a_verb_the_host_declared_runs_on_the_owner_from_an_executor_elsewhere(
    prefix: str, redis_client: Any
) -> None:
    """A host tool on an executor reaches what only the owner's process holds
    through a verb the owner's host declared (RFC §10.6.5): the verb runs
    there, for the turn's room, its arguments and its result crossing Redis."""
    store = InMemoryStore()
    served: list[tuple[str | None, dict[str, Any]]] = []
    tool = AITool(name="navigate", description="Open a page.", parameters={"type": "object"})

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        host = current_turn_host()
        assert host is not None
        answer = await host.call("app.navigate", {"page": "/boards"})
        return json.dumps(answer)

    calling = AIResponse(
        content="", tool_calls=[AIToolCall(id="c1", name="navigate", arguments={})]
    )
    agent = Agent(
        "worker",
        provider=MockAIProvider(ai_responses=[calling, AIResponse(content="Opened.")]),
        tools=[tool],
        tool_handler=handler,
    )
    host, executor = await _hosted(prefix, store, agent)
    kit = await _owner(prefix, "worker", store=store)

    async def navigate(arguments: dict[str, Any]) -> dict[str, Any]:
        served.append((current_tool_room_id(), arguments))
        return {"opened": arguments["page"]}

    kit.declare_host_verb("app.navigate", navigate)
    try:
        await kit.process_inbound(_said("Open my boards."), room_id="r")

        assert served == [("r", {"page": "/boards"})]
        assert await _answers(kit) == ["Opened."]
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_released_channel_is_let_go_by_its_executor_elsewhere(
    prefix: str, redis_client: Any
) -> None:
    """The owner's release reaches an executor of another process that holds
    the channel built (RFC §10.6.3): its next turn runs on a new build."""
    store = InMemoryStore()
    builds: list[_Worker] = []

    def build(channel_id: str) -> _Worker | None:
        if channel_id != "worker":
            return None
        built = _Worker("worker", MockAIProvider(responses=[f"Build {len(builds)}."] * 3))
        builds.append(built)
        return built

    host = _backend(prefix)
    executor = TurnExecutor(
        build,
        hosts={"worker"},
        executor_id="host-a",
    )
    await host.serve(executor)
    kit = await _owner(prefix, "worker", store=store)
    try:
        await kit.process_inbound(_said("Hi."), room_id="r")
        await kit.release_channel("worker")
        await kit.process_inbound(_said("Again."), room_id="r")

        assert await _answers(kit) == ["Build 0.", "Build 1."]
        assert len(builds) == 2
    finally:
        await kit.close()
        await host.close()
        await executor.close()


async def test_a_verb_declared_between_turns_reaches_the_owner_after_a_turn_elsewhere(
    prefix: str, redis_client: Any
) -> None:
    """An asynchronous hook on an executor of another process runs a verb the
    owner declared ``between_turns`` once its turn ended (RFC §10.6.5): the
    owner serves it for the turn's room."""
    store = InMemoryStore()
    worker = _Worker("worker", MockAIProvider(responses=["Hi."] * 5, streaming=True))
    host, executor = await _hosted(prefix, store, worker)
    later_work: list[asyncio.Task[Any]] = []
    served: list[tuple[str | None, dict[str, Any]]] = []

    async def write_later(turn_host: Any) -> None:
        await asyncio.sleep(0.3)  # the turn has returned
        await turn_host.call("app.card", {"verdict": "ok"})

    @executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def later(event: Any, context: Any) -> None:
        later_work.append(asyncio.get_running_loop().create_task(write_later(current_turn_host())))

    kit = await _owner(prefix, "worker", store=store)

    async def card(arguments: dict[str, Any]) -> None:
        served.append((current_tool_room_id(), arguments))

    kit.declare_host_verb("app.card", card, between_turns=True)
    try:
        await kit.process_inbound(_said("Hi."), room_id="r")
        async with asyncio.timeout(5):
            while not served:
                await asyncio.sleep(0.02)

        assert served == [("r", {"verdict": "ok"})]
    finally:
        await kit.close()
        await host.close()
        await executor.close()

"""A turn submitted to one executor a host chose (RFC §10.6.7, RMK-2098).

A host that reserves a worker by its own rules wraps the turn backend its kit
is given, and hands each turn to the worker it reserved
(``TurnBackend.submit_to``): that worker alone takes it, one that does not
take it refuses before anything is handed, and one whose answer is lost
leaves the turn's outcome unknown. Each rule is checked in memory and, with
REDIS_URL set, across a real Redis where every worker is a process of its own.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import AgentIdentity, ChannelDeclaration, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import TextContent
from roomkit.models.steering import Cancel
from roomkit.providers.ai.base import StreamTextDelta
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.turns import (
    ChannelControl,
    ChannelResult,
    ExecutorInfo,
    InMemoryTurnBackend,
    RoomOwner,
    TurnBackend,
    TurnExecutor,
    TurnLink,
    TurnRequest,
    TurnTimeouts,
)
from roomkit.turns._redis_routes import REFUSED
from roomkit.turns.distributed import DistributedTurnBackend
from tests.test_framework import SimpleChannel
from tests.turn_nodes import ACROSS, Nodes

BACKENDS = ["memory", *ACROSS]
# The accept bound is what a lost answer waits for: kept short.
TIMEOUTS = TurnTimeouts(heartbeat_interval=0.02, heartbeat_timeout=2.0, accept=0.5, cancel=1.0)


class _Reserving(TurnBackend):
    """A host's backend: each turn goes to the executor it reserved, when it
    reserved one, and is placed as usual otherwise."""

    def __init__(self, inner: TurnBackend) -> None:
        self.inner = inner
        self.timeouts = inner.timeouts
        self.reserved: ExecutorInfo | None = None
        self.requests: list[TurnRequest] = []

    async def submit(self, request: TurnRequest) -> TurnLink:
        self.requests.append(request)
        if self.reserved is None:
            return await self.inner.submit(request)
        return await self.inner.submit_to(request, self.reserved)

    async def serve(self, executor: TurnExecutor) -> None:
        await self.inner.serve(executor)

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        return await self.inner.control(message)

    async def attend(self, owner: RoomOwner) -> None:
        await self.inner.attend(owner)

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        return await self.inner.claim_presence(channel_id, room_id, online)

    async def executors(self) -> list[ExecutorInfo]:
        return await self.inner.executors()

    async def close(self) -> None:
        await self.inner.close()


class _Pool:
    """An owner kit behind a reserving backend, and workers declaring
    ``worker``: all on one in-memory backend, or each on a node of its own
    of the backend across processes, a process of its own."""

    def __init__(self, kind: str) -> None:
        self._nodes = Nodes(kind)
        self._memory = InMemoryTurnBackend() if kind == "memory" else None
        self.store = InMemoryStore()
        self.backend = _Reserving(self._node())
        self.hosts: dict[str, TurnBackend] = {}
        self.executors: dict[str, TurnExecutor] = {}
        self.kit = RoomKit(store=self.store, turn_backend=self.backend)

    def _node(self) -> TurnBackend:
        if self._memory is not None:
            return self._memory
        return self._nodes.node(TIMEOUTS)

    async def open(self) -> None:
        kit = self.kit
        kit.register_channel(SimpleChannel("tx"))
        kit.register_channel(ChannelDeclaration(channel_id="worker", identity=AgentIdentity()))
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "tx")
        await kit.attach_channel("r", "worker", category=ChannelCategory.INTELLIGENCE)

    async def serve(
        self,
        name: str,
        *,
        executor_id: str | None = None,
        room_hook: str | None = None,
        provider: MockAIProvider | None = None,
        here: bool = False,
        hosts: str = "worker",
    ) -> None:
        """Serve a worker answering ``From <name>.``, defining *room_hook*
        first so its first beat declares it; on the owner's own backend when
        *here*, else on one of its own."""
        model = provider or MockAIProvider(responses=[f"From {name}."] * 20, streaming=True)
        executor = TurnExecutor(
            lambda channel_id: Agent(channel_id, provider=model),
            hosts={hosts},
            executor_id=executor_id or name,
            timeouts=TIMEOUTS,
        )
        if room_hook is not None:
            executor.room_hook(room_hook, HookTrigger.ON_AI_RESPONSE)(_nothing)
        host = self.backend.inner if here else self._node()
        await host.serve(executor)
        self.hosts[name], self.executors[name] = host, executor

    def info(self, name: str, *, node: bool = True) -> ExecutorInfo:
        """*name* as the backend lists it, or by its id alone."""
        host = self.hosts[name]
        return ExecutorInfo(
            executor_id=self.executors[name].executor_id,
            node=host.node_id if node and isinstance(host, DistributedTurnBackend) else None,
        )

    async def say(self, text: str, reserved: ExecutorInfo | None = None) -> Any:
        """Process one message, its turn submitted to *reserved* when given."""
        self.backend.reserved = reserved
        try:
            message = InboundMessage(
                channel_id="tx", sender_id="u", content=TextContent(body=text)
            )
            return await self.kit.process_inbound(message, room_id="r")
        finally:
            self.backend.reserved = None

    async def answers(self) -> list[str]:
        return [
            event.content.body
            for event in await self.store.list_events("r")
            if event.source.channel_id == "worker" and event.type == EventType.MESSAGE
        ]

    async def stop(self, name: str) -> None:
        host, executor = self.hosts.pop(name), self.executors.pop(name)
        if host is not self._memory and host is not self.backend.inner:
            await host.close()
        await executor.close()

    async def close(self) -> None:
        await self.kit.close()
        for name in list(self.hosts):
            await self.stop(name)
        if self._memory is None:
            await self._nodes.forget()


async def _nothing(event: Any, context: Any) -> None:
    return None


@pytest.fixture(params=BACKENDS)
async def pool(request: pytest.FixtureRequest) -> Any:
    pool = _Pool(request.param)
    await pool.open()
    yield pool
    await pool.close()


async def test_a_turn_submitted_to_an_executor_runs_on_it_alone_and_the_room_follows(
    pool: _Pool,
) -> None:
    """Another executor holds the room: the turn submitted to the reserved one
    runs there once, and the room's next turns, placed as usual, stay on it."""
    await pool.serve("a")
    await pool.say("First.")
    await pool.serve("b")
    result = await pool.say("Second.", pool.info("b"))
    await pool.say("Third.")

    assert result.error is None
    assert pool.backend.requests[1].handed
    assert await pool.answers() == ["From a.", "From b.", "From b."]


@pytest.mark.parametrize("how", ["closed", "without the room's hook", "another node"])
async def test_an_executor_that_does_not_take_the_turn_refuses_it_before_it_is_handed(
    pool: _Pool, how: str
) -> None:
    """The reserved executor closed, lacks the room's hook, or is named in a
    process that does not serve it: ``no_executor``, the request not handed,
    no other executor given the turn, and the room still on its holder."""
    await pool.serve("a", room_hook="audit")
    await pool.say("First.")
    await pool.serve("b")
    reserved = pool.info("b")
    if how == "closed":
        await pool.stop("b")
    elif how == "without the room's hook":
        await pool.kit.enable_room_hook("r", "audit")
    else:
        reserved = reserved.model_copy(update={"node": "another"})
    result = await pool.say("Second.", reserved)
    await pool.say("Third.")

    assert isinstance(result.error, TurnLinkError)
    assert result.error.reason == "no_executor"
    assert not pool.backend.requests[1].handed
    assert await pool.answers() == ["From a.", "From a."]


async def test_a_turn_of_a_channel_the_kit_holds_live_is_submitted_to_no_other_executor(
    pool: _Pool,
) -> None:
    """The kit's own executor alone runs the channels the kit holds live: a
    turn of one submitted to another executor is refused (RFC §10.6.7)."""
    live = Agent("live", provider=MockAIProvider(responses=["From the kit."]))
    pool.kit.register_channel(live)
    await pool.kit.attach_channel("r", "live", category=ChannelCategory.INTELLIGENCE)
    await pool.serve("b")
    result = await pool.say("First.", pool.info("b"))
    handed = {request.channel_id: request.handed for request in pool.backend.requests}

    assert isinstance(result.error, TurnLinkError)
    assert result.error.reason == "no_executor"
    assert handed == {"worker": True, "live": False}


async def test_the_kit_s_own_executor_named_for_a_turn_it_does_not_take_refuses_by_its_name(
    pool: _Pool,
) -> None:
    """The kit's own executor is named as any other: for a turn of a channel
    it holds live in a room whose hook it does not define, the refusal names
    it, as the refusal of any executor that does not take the turn."""
    pool.kit.register_channel(Agent("live", provider=MockAIProvider(responses=["Hi."])))
    await pool.kit.detach_channel("r", "worker")
    await pool.kit.attach_channel("r", "live", category=ChannelCategory.INTELLIGENCE)
    await pool.kit.enable_room_hook("r", "audit")
    own = ExecutorInfo(executor_id=pool.kit.turn_executor.executor_id)
    result = await pool.say("First.", own)

    assert isinstance(result.error, TurnLinkError)
    assert result.error.reason == "no_executor"
    assert str(result.error) == (
        f"Executor {own.executor_id} does not take channel 'live' with room hooks ['audit']"
    )
    assert not pool.backend.requests[0].handed


async def test_a_turn_submitted_to_an_executor_is_cancelled_through_its_link(pool: _Pool) -> None:
    """The turn an executor was submitted streams and is cut as any other:
    a steering cancel reaches it on the executor that took it."""
    started = asyncio.Event()

    class _Slow(MockAIProvider):
        async def generate_structured_stream(self, context: Any) -> Any:
            started.set()
            for i in range(200):
                yield StreamTextDelta(text=f"w{i} ")
                await asyncio.sleep(0.05)

    await pool.serve("a")
    await pool.serve("b", provider=_Slow(streaming=True))
    turn = asyncio.ensure_future(pool.say("Hi.", pool.info("b")))
    await asyncio.wait_for(started.wait(), 5)
    reached = await pool.kit.steer("r", "worker", Cancel())
    result = await asyncio.wait_for(turn, 5)

    assert reached == 1
    assert dict(result.response_metadata)["turns"]["worker"]["loop_end_reason"] == "cancelled"


async def test_a_backend_that_names_no_executor_refuses_the_turn_unhanded() -> None:
    """A host's backend that keeps the default ``submit_to`` submits to no
    executor a caller names: ``no_executor``, nothing handed."""

    class _Plain(_Reserving):
        submit_to = TurnBackend.submit_to

    pool = _Pool("memory")
    pool.backend = _Plain(pool.backend)
    pool.kit = RoomKit(store=pool.store, turn_backend=pool.backend)
    await pool.open()
    try:
        await pool.serve("a")
        result = await pool.say("First.", pool.info("a"))

        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "no_executor"
        assert not pool.backend.requests[0].handed
    finally:
        await pool.close()


@pytest.mark.parametrize("services", ACROSS)
async def test_an_executor_of_the_owner_s_own_process_is_named_as_any_other(services: str) -> None:
    """A worker served on the owner's own backend, whose turns stay in the
    process: named as the backend lists it or by its id alone it takes the
    turn, the room follows it, and closed it refuses before the hand-over."""
    pool = _Pool(services)
    await pool.open()
    try:
        await pool.serve("a")
        await pool.say("First.")
        await pool.serve("here", here=True)
        listed = await pool.say("Second.", pool.info("here"))
        by_id = await pool.say("Third.", pool.info("here", node=False))
        await pool.say("Fourth.")
        await pool.executors["here"].close()
        refused = await pool.say("Fifth.", pool.info("here"))

        assert listed.error is None and by_id.error is None
        assert getattr(refused.error, "reason", None) == "no_executor"
        assert not pool.backend.requests[4].handed
        assert await pool.answers() == ["From a."] + ["From here."] * 3
    finally:
        await pool.close()


@pytest.mark.parametrize("services", ACROSS)
async def test_an_executor_id_several_processes_serve_needs_its_node(services: str) -> None:
    """Two processes serve the id, one replacing the other: named without a
    node it is refused, named with its node it takes the turn, and alone
    again it is named by its id."""
    pool = _Pool(services)
    await pool.open()
    try:
        await pool.serve("old", executor_id="worker-1")
        await pool.serve("new", executor_id="worker-1")
        refused = await pool.say("First.", pool.info("new", node=False))
        taken = await pool.say("Second.", pool.info("new"))
        await pool.stop("old")
        alone = await pool.say("Third.", pool.info("new", node=False))

        assert getattr(refused.error, "reason", None) == "no_executor"
        assert not pool.backend.requests[0].handed
        assert taken.error is None and alone.error is None
        assert await pool.answers() == ["From new."] * 2
    finally:
        await pool.close()


@pytest.mark.parametrize("services", ACROSS)
async def test_an_executor_id_names_the_one_process_that_would_take_the_turn(
    services: str,
) -> None:
    """Two processes serve the id and one of them alone declares the channel:
    named by its id, that one takes the turn, as it alone would placed."""
    pool = _Pool(services)
    await pool.open()
    try:
        await pool.serve("coder", executor_id="worker-1")
        await pool.serve("other", executor_id="worker-1", hosts="another-channel")
        result = await pool.say("First.", pool.info("coder", node=False))

        assert result.error is None
        assert await pool.answers() == ["From coder."]
    finally:
        await pool.close()


@pytest.mark.parametrize("services", ACROSS)
async def test_an_executor_whose_process_refuses_the_hand_over_leaves_the_turn_with_no_other(
    monkeypatch: pytest.MonkeyPatch, services: str
) -> None:
    """The reserved executor's process, closing, refuses the request it is
    handed: ``no_executor``, the request no longer handed, and the turn is
    placed on no other executor, as one placed as usual would be."""
    pool = _Pool(services)
    await pool.open()
    try:
        await pool.serve("a")
        await pool.say("First.")
        await pool.serve("b")
        host = pool.hosts["b"]
        assert isinstance(host, DistributedTurnBackend)
        refuse = host._routes._answer_back
        with monkeypatch.context() as refusing:
            refusing.setitem(
                host._routes._routes, "request", lambda envelope: refuse(envelope, REFUSED)
            )
            result = await pool.say("Second.", pool.info("b"))
        # b takes requests again: the room is still a's, which it never left.
        await pool.say("Third.")

        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "no_executor"
        assert not pool.backend.requests[1].handed
        assert await pool.answers() == ["From a.", "From a."]
    finally:
        await pool.close()


@pytest.mark.parametrize("services", ACROSS)
async def test_a_turn_taken_whose_holder_cannot_be_recorded_is_released(
    monkeypatch: pytest.MonkeyPatch, services: str
) -> None:
    """The reserved executor took the turn and Redis refuses the record of
    it holding the room: the turn is let go on its executor, not left running
    for a room that would not follow it, and the failure is the caller's."""
    pool = _Pool(services)
    await pool.open()
    try:
        await pool.serve("b")
        owner = pool.backend.inner
        assert isinstance(owner, DistributedTurnBackend)

        async def refused(*args: Any) -> None:
            raise ConnectionError("Redis did not answer")

        monkeypatch.setattr(owner._placement, "held_by", refused)
        result = await pool.say("First.", pool.info("b"))

        assert isinstance(result.error, ConnectionError)
        assert pool.backend.requests[0].handed
        assert not owner._routes._owner_links
        assert await pool.answers() == []
    finally:
        await pool.close()


@pytest.mark.parametrize("services", ACROSS)
@pytest.mark.wallclock
async def test_an_executor_whose_answer_to_the_hand_over_is_lost_leaves_the_turn_with_no_other(
    monkeypatch: pytest.MonkeyPatch, services: str
) -> None:
    """The reserved executor's process is handed the request and its answer
    never comes: ``executor_lost`` at the accept bound, the request handed (it
    may have begun), and no other executor given the turn."""
    pool = _Pool(services)
    await pool.open()
    try:
        await pool.serve("a")
        await pool.serve("b")
        host = pool.hosts["b"]
        assert isinstance(host, DistributedTurnBackend)
        monkeypatch.setitem(host._routes._routes, "request", lambda envelope: None)
        result = await pool.say("First.", pool.info("b"))

        assert isinstance(result.error, TurnLinkError)
        assert result.error.reason == "executor_lost"
        assert pool.backend.requests[0].handed
        assert await pool.answers() == []
    finally:
        await pool.close()

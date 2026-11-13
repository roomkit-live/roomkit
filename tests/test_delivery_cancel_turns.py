"""Cancelling a message's delivery stops its turns, never its delivery (RFC
§10.1, *Explicit delivery cancellation*): a turn not started yet never starts,
one handed to its executor is cut as it arrives, a running one is cut, and the
other turns of the room go on. The message still reaches the room's other
channels and fires its AFTER_BROADCAST hooks. A caller names the delivery by a
key of its own (``InboundMessage.delivery_key``), so code without its handle
cancels it: ``kit.cancel_delivery(key)``.

Every test runs on the kinds of intelligence channel a turn can run on: an
``Agent`` (a tool loop), the same agent taking a room's turns one at a time
(``serialize_room_turns``), an ACP agent (its own process, one prompt at a time
per room) and a channel with no steering at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from acp.schema import PromptResponse

from roomkit import Agent, ChannelCategory, HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.base import Channel
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import ChannelMediaType, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import InjectedEvent
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from roomkit.turns import InMemoryTurnBackend
from roomkit.turns.base import TurnLink
from roomkit.turns.wire import TurnRequest
from tests.test_channels.test_acp import _channel as acp_channel
from tests.test_framework import SimpleChannel

WAIT = 2.0


@dataclass
class Seen:
    """What the agent did with each message it was asked about: started
    working on it, and finished."""

    started: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    gates: dict[str, asyncio.Event] = field(default_factory=dict)

    def gate(self, body: str) -> asyncio.Event:
        return self.gates.setdefault(body, asyncio.Event())

    async def work(self, body: str) -> None:
        self.started.append(body)
        await self.gate(body).wait()
        self.finished.append(body)

    async def until_started(self, body: str) -> None:
        async with asyncio.timeout(WAIT):
            while body not in self.started:
                await asyncio.sleep(0.01)

    async def until_finished(self, body: str) -> None:
        async with asyncio.timeout(WAIT):
            while body not in self.finished:
                await asyncio.sleep(0.01)


def _body(context: AIContext) -> str:
    content = context.messages[-1].content
    return (
        content if isinstance(content, str) else "".join(getattr(p, "text", "") for p in content)
    )


class _Working(MockAIProvider):
    def __init__(self, seen: Seen) -> None:
        super().__init__()
        self._seen = seen

    async def generate(self, context: AIContext) -> AIResponse:
        await self._seen.work(_body(context))
        return AIResponse(content="done")


class _Steerless(Channel):
    """An intelligence channel with no ``steer()``: its owner cuts its turns."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, seen: Seen) -> None:
        super().__init__("agent")
        self._seen = seen

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(media_types=[ChannelMediaType.TEXT])

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.type != EventType.MESSAGE or event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        assert isinstance(event.content, TextContent)
        await self._seen.work(event.content.body)
        return ChannelOutput.empty()


def _agent(seen: Seen, tmp_path: Path) -> Channel:
    return Agent("agent", provider=_Working(seen))


TASKS = ("task A", "task B", "task C")


def _acp(seen: Seen, tmp_path: Path) -> Channel:
    channel, connection, _ = acp_channel(tmp_path, emit_updates=False)
    prompting: dict[str, str] = {}
    cut: set[str] = set()

    async def prompt(session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        text = "".join(getattr(block, "text", "") for block in prompt)
        # The prompt catches the agent up on the room first: its task is the last one named.
        body = max((t for t in TASKS if t in text), key=text.rfind)
        prompting[session_id] = body
        seen.started.append(body)
        await seen.gate(body).wait()
        if body in cut:
            return PromptResponse(stop_reason="cancelled")
        seen.finished.append(body)
        return PromptResponse(stop_reason="end_turn")

    async def cancel(session_id: str) -> None:
        # The agent stops the prompt it runs, which then returns "cancelled".
        body = prompting.get(session_id)
        if body is not None and body not in seen.finished:
            cut.add(body)
            seen.gate(body).set()

    connection.prompt = prompt  # type: ignore[method-assign]
    connection.cancel = cancel  # type: ignore[method-assign]
    return channel


def _steerless(seen: Seen, tmp_path: Path) -> Channel:
    return _Steerless(seen)


def _serial(seen: Seen, tmp_path: Path) -> Channel:
    return Agent("agent", provider=_Working(seen), serialize_room_turns=True)


PATHS = {"agent": _agent, "acp": _acp, "steerless": _steerless, "serial": _serial}


class _Transport(SimpleChannel):
    """A transport whose delivery of a message waits while it is held."""

    def __init__(self) -> None:
        super().__init__("ws")
        self.held: dict[str, asyncio.Event] = {}

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if isinstance(event.content, TextContent) and event.content.body in self.held:
            await self.held[event.content.body].wait()
        return await super().deliver(event, binding, context)


@dataclass
class Room:
    kit: RoomKit
    seen: Seen
    ws: _Transport
    broadcast: list[str]

    async def send(
        self, body: str, key: str | None = None, *, visibility: str = "all"
    ) -> InboundResult:
        message = InboundMessage(
            channel_id="sms",
            sender_id="user",
            content=TextContent(body=body),
            delivery_key=key,
            visibility=visibility,
        )
        return await self.kit.process_inbound(message, room_id="r", defer_delivery=True)

    def delivered(self) -> list[str]:
        return [e.content.body for e in self.ws.delivered if isinstance(e.content, TextContent)]

    async def until_broadcast(self, body: str) -> None:
        async with asyncio.timeout(WAIT):
            while body not in self.broadcast:
                await asyncio.sleep(0.01)


async def _room(
    path: str,
    tmp_path: Path,
    *,
    turn_backend: InMemoryTurnBackend | None = None,
    sms: Channel | None = None,
    organization_id: str | None = None,
    **options: Any,
) -> Room:
    seen = Seen()
    kit = RoomKit(turn_backend=turn_backend, **options)
    ws = _Transport()
    kit.register_channel(sms or SimpleChannel("sms"))
    kit.register_channel(ws)
    agent = PATHS[path](seen, tmp_path)
    kit.register_channel(agent)
    await kit.create_room(room_id="r", organization_id=organization_id)
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", agent.channel_id, category=ChannelCategory.INTELLIGENCE)
    broadcast: list[str] = []

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def record(event: RoomEvent, ctx: Any) -> None:
        if event.type == EventType.MESSAGE and isinstance(event.content, TextContent):
            broadcast.append(event.content.body)

    return Room(kit, seen, ws, broadcast)


@pytest.fixture(params=sorted(PATHS))
def path(request: pytest.FixtureRequest) -> str:
    return str(request.param)


async def _waiting_on_the_room_lock(room: Room) -> None:
    """Until a turn waits on the ACP channel's lock of the room, behind the
    turn that holds it."""
    channel = next(c for c in room.kit.channels.values() if hasattr(c, "_room_locks"))
    async with asyncio.timeout(WAIT):
        while True:
            lock = channel._room_locks.get("r")  # noqa: SLF001
            if lock is not None and any(not w.done() for w in lock._waiters or ()):  # noqa: SLF001
                return
            await asyncio.sleep(0.005)


async def _waiting_its_place(room: Room) -> None:
    """Until a turn of the agent has its place in the room, behind the turn
    that holds it."""
    lines = room.kit._get_router()._room_turns._lines  # noqa: SLF001
    async with asyncio.timeout(WAIT):
        while len(lines.get(("r", "agent"), ())) < 2:
            await asyncio.sleep(0.005)


async def _closed(room: Room) -> None:
    for gate in room.seen.gates.values():
        gate.set()
    await room.kit.close()


# -- queued: it never starts ----------------------------------------------------------


async def test_a_turn_queued_behind_the_room_s_turn_never_starts(
    path: str, tmp_path: Path
) -> None:
    room = await _room(path, tmp_path)
    held = room.ws.held["held"] = asyncio.Event()
    try:
        a = await room.send("task A", "a")
        await room.seen.until_started("task A")
        if path == "agent":
            # An agent runs a room's turns side by side: B waits in the
            # room's delivery lane, behind a message a transport still holds.
            await room.send("held", visibility="transport")
        b = await room.send("task B", "b")
        if path == "acp":
            # An ACP room's turns hold its session lock: B leaves the lane and
            # waits on that lock, which is where the cancel must find it.
            await _waiting_on_the_room_lock(room)
        if path == "serial":
            # An agent that takes a room's turns one at a time: B leaves the
            # lane and waits its place in its stream, behind A.
            await _waiting_its_place(room)
        assert room.seen.started == ["task A"]  # B queued behind A

        assert await room.kit.cancel_delivery("b") == 1
        assert a.delivery is not None and b.delivery is not None
        # The cancel returns at once; B's turns are stopped once its handle
        # reads them stopped (a turn backend elsewhere is told over its link).
        done_b = await asyncio.wait_for(b.delivery.wait(), WAIT)
        room.seen.gate("task A").set()
        held.set()
        done_a = await asyncio.wait_for(a.delivery.wait(), WAIT)
        await room.until_broadcast("task B")

        # B's agent never started; A, the room's other turn, ran to its end.
        assert room.seen.started == ["task A"]
        assert room.seen.finished == ["task A"]
        assert done_a.cancellation_reason is None
        assert done_b.cancellation_reason == "caller_cancelled"
        # B itself reached the room: its transport, its AFTER_BROADCAST.
        assert "task B" in room.delivered()
    finally:
        await _closed(room)


# -- running: it is cut, the queued one runs after -------------------------------


async def test_a_running_turn_is_cut_and_the_room_s_next_turn_runs(
    path: str, tmp_path: Path
) -> None:
    room = await _room(path, tmp_path)
    try:
        a = await room.send("task A", "a")
        await room.seen.until_started("task A")
        b = await room.send("task B", "b")
        if path == "serial":
            await _waiting_its_place(room)
            assert room.seen.started == ["task A"]

        assert await room.kit.cancel_delivery("a") == 1
        assert a.delivery is not None and b.delivery is not None
        done_a = await asyncio.wait_for(a.delivery.wait(), WAIT)
        await room.seen.until_started("task B")
        room.seen.gate("task B").set()
        done_b = await asyncio.wait_for(b.delivery.wait(), WAIT)
        await room.until_broadcast("task A")

        assert "task A" not in room.seen.finished
        assert "task B" in room.seen.finished
        assert done_a.cancellation_reason == "caller_cancelled"
        assert done_b.cancellation_reason is None
        assert room.delivered()[:2] == ["task A", "task B"]
    finally:
        await _closed(room)


# -- handed: cut as it arrives, never lost --------------------------------------------


class _HeldHandOver(InMemoryTurnBackend):
    """Holds the hand-over of the turns of one message, as a distributed
    backend waits for its executor's answer: the executor has them, their
    owner does not hold them yet."""

    def __init__(self, body: str) -> None:
        super().__init__()
        self._body = body
        self.handing = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, request: TurnRequest) -> TurnLink:
        link = await super().submit(request)
        content = request.event.content
        if isinstance(content, TextContent) and content.body == self._body:
            self.handing.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                # As a distributed backend does: a turn its owner gave up
                # while handed is released, and its executor cuts it.
                link.release()
                raise
        return link


async def test_a_turn_being_handed_over_is_cut_as_it_arrives(path: str, tmp_path: Path) -> None:
    backend = _HeldHandOver("task B")
    room = await _room(path, tmp_path, turn_backend=backend)
    try:
        b = await room.send("task B", "b")
        await asyncio.wait_for(backend.handing.wait(), WAIT)
        c = await room.send("task C", "c")

        assert await room.kit.cancel_delivery("b") == 1
        backend.release.set()
        room.seen.gate("task C").set()
        assert b.delivery is not None and c.delivery is not None
        done_b = await asyncio.wait_for(b.delivery.wait(), WAIT)
        done_c = await asyncio.wait_for(c.delivery.wait(), WAIT)

        assert "task B" not in room.seen.finished
        assert "task C" in room.seen.finished
        assert done_b.cancellation_reason == "caller_cancelled"
        assert done_c.cancellation_reason is None
        assert "task B" in room.delivered()
    finally:
        backend.release.set()
        await _closed(room)


# -- the key ----------------------------------------------------------------------------


async def test_an_awaited_call_reads_its_cancelled_end(tmp_path: Path) -> None:
    room = await _room("agent", tmp_path)
    try:
        message = InboundMessage(
            channel_id="sms",
            sender_id="user",
            content=TextContent(body="task A"),
            delivery_key="a",
        )
        call: asyncio.Task[InboundResult] = asyncio.create_task(
            room.kit.process_inbound(message, room_id="r")
        )
        await room.seen.until_started("task A")

        assert await room.kit.cancel_delivery("a", reason="stopped") == 1
        result = await asyncio.wait_for(call, WAIT)

        assert result.cancellation_reason == "stopped"
        assert room.seen.finished == []
    finally:
        await _closed(room)


async def test_a_key_names_every_delivery_in_flight_and_nothing_after(tmp_path: Path) -> None:
    room = await _room("agent", tmp_path)
    try:
        assert await room.kit.cancel_delivery("never sent") == 0
        a = await room.send("task A", "task")
        await room.seen.until_started("task A")
        b = await room.send("task B", "task")

        assert await room.kit.cancel_delivery("task") == 2
        # Already cancelled: nothing more to cancel.
        assert await room.kit.cancel_delivery("task") == 0
        assert a.delivery is not None and b.delivery is not None
        await asyncio.wait_for(a.delivery.wait(), WAIT)
        await asyncio.wait_for(b.delivery.wait(), WAIT)
        room.seen.gate("task C").set()
        c = await room.send("task C", "c")
        assert c.delivery is not None
        await asyncio.wait_for(c.delivery.wait(), WAIT)

        # Over: a key names nothing any more, and is held by nothing.
        assert await room.kit.cancel_delivery("c") == 0
        await _settled(lambda: not room.kit._delivery_keys._named)
        assert room.seen.finished == ["task C"]
    finally:
        await _closed(room)


async def test_a_cancel_under_the_room_lock_returns_at_once(path: str, tmp_path: Path) -> None:
    """A sync hook runs under the room lock, where waiting on the room's
    delivery would never end: the cancel returns at once, and the turn it
    names never starts."""
    room = await _room(path, tmp_path)
    cancelled: list[int] = []

    @room.kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def stop(event: RoomEvent, ctx: Any) -> HookResult:
        if isinstance(event.content, TextContent) and event.content.body == "stop B":
            cancelled.append(await room.kit.cancel_delivery("b"))
        return HookResult.allow()

    try:
        a = await room.send("task A", "a")
        await room.seen.until_started("task A")
        b = await room.send("task B", "b")
        await room.send("stop B")
        assert a.delivery is not None and b.delivery is not None
        done_b = await asyncio.wait_for(b.delivery.wait(), WAIT)
        room.seen.gate("task A").set()
        room.seen.gate("stop B").set()

        assert cancelled == [1]
        assert "task B" not in room.seen.started
        assert done_b.cancellation_reason == "caller_cancelled"
    finally:
        await _closed(room)


async def _no_turn_follows() -> None:
    """A turn the broadcast message would start begins a few steps after it:
    a short bounded wait, as nothing can be awaited for a turn that must
    not start."""
    await asyncio.sleep(0.02)


async def _settled(ready: Callable[[], Any]) -> None:
    async with asyncio.timeout(WAIT):
        while not ready():
            await asyncio.sleep(0.01)


async def test_a_task_delegated_from_a_delivery_is_not_its_turn_work(tmp_path: Path) -> None:
    """A task an AFTER_BROADCAST hook delegates runs on its own: cancelling
    the delivery that fired the hook stops the delivery's turns, never the
    delegated task's (RFC §10.1: no task detached by application code is
    owned)."""
    room = await _room("agent", tmp_path)
    worker = Seen()
    room.kit.register_channel(Agent("worker", provider=_Working(worker)))
    delegated: list[Any] = []

    @room.kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def delegate(event: RoomEvent, ctx: Any) -> None:
        if isinstance(event.content, TextContent) and event.content.body == "task A":
            delegated.append(await room.kit.delegate("r", "worker", "child task"))

    try:
        a = await room.send("task A", "a")
        await room.seen.until_started("task A")
        await _settled(lambda: worker.started)

        assert await room.kit.cancel_delivery("a") == 1
        assert a.delivery is not None
        done_a = await asyncio.wait_for(a.delivery.wait(), WAIT)
        worker.gate(worker.started[0]).set()
        result = await asyncio.wait_for(delegated[0].wait(), WAIT)

        assert done_a.cancellation_reason == "caller_cancelled"
        assert "task A" not in room.seen.finished
        assert worker.finished == worker.started
        assert result.output == "done"
    finally:
        for gate in worker.gates.values():
            gate.set()
        await _closed(room)


# -- what goes on after a cancel: the delivery, and nothing of the turns -----------------


async def test_a_cancel_returns_at_once_while_a_transport_still_delivers(
    tmp_path: Path,
) -> None:
    """The cancel joins the turns' work, not the delivery going on: a slow
    transport neither holds the cancel nor makes it report its cleanup
    incomplete, on the handle and on an awaited call alike."""
    room = await _room("steerless", tmp_path)
    room.ws.held["task A"] = asyncio.Event()
    room.ws.held["task B"] = asyncio.Event()
    try:
        a = await room.send("task A", "a")
        await room.seen.until_started("task A")
        assert a.delivery is not None
        done_a = await asyncio.wait_for(a.delivery.cancel(timeout=0.5), WAIT)
        assert done_a.cancellation_reason == "caller_cancelled"
        room.ws.held["task A"].set()
        await room.until_broadcast("task A")

        message = InboundMessage(
            channel_id="sms", sender_id="user", content=TextContent(body="task B")
        )
        call = asyncio.create_task(room.kit.process_inbound(message, room_id="r"))
        await room.seen.until_started("task B")
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 1.0)

        room.ws.held["task B"].set()
        await room.until_broadcast("task B")
        assert room.seen.finished == []
        assert room.delivered() == ["task A", "task B"]
    finally:
        await _closed(room)


async def test_a_cancelled_send_event_is_still_delivered(tmp_path: Path) -> None:
    room = await _room("steerless", tmp_path)
    try:
        call = asyncio.create_task(room.kit.send_event("r", "sms", TextContent(body="task A")))
        await room.seen.until_started("task A")
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, WAIT)
        await room.until_broadcast("task A")

        assert room.seen.finished == []
        assert room.delivered() == ["task A"]
    finally:
        await _closed(room)


async def test_an_event_injected_by_a_cancelled_delivery_starts_no_turn(
    path: str, tmp_path: Path
) -> None:
    """A hook's injected event is delivered by the plan of the message that
    fired it: once that delivery is cancelled, its turn never starts either."""
    room = await _room(path, tmp_path)
    held = room.ws.held["held"] = asyncio.Event()
    agent_id = "acp-agent" if path == "acp" else "agent"

    @room.kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def inject(event: RoomEvent, ctx: Any) -> HookResult:
        if isinstance(event.content, TextContent) and event.content.body == "task B":
            injected = RoomEvent(
                room_id="r",
                source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
                content=TextContent(body="task C"),
            )
            return HookResult.allow([InjectedEvent(event=injected, target_channel_ids=[agent_id])])
        return HookResult.allow()

    try:
        room.seen.gate("task C").set()
        await room.send("held", visibility="transport")
        b = await room.send("task B", "b")
        assert await room.kit.cancel_delivery("b") == 1
        held.set()
        assert b.delivery is not None
        await asyncio.wait_for(b.delivery.wait(), WAIT)
        await room.until_broadcast("task B")
        await _no_turn_follows()

        assert room.seen.started == []
    finally:
        await _closed(room)


# -- the key: held while the delivery is in flight, scoped to its organization --------


@pytest.mark.parametrize("work", [0, 0.05], ids=["yield", "work"])
@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_a_reentrant_call_s_key_names_its_delivery_until_it_runs(
    store: str, work: float, tmp_path: Path
) -> None:
    """A call made under the room's lock (a sync hook) returns before its
    delivery starts, which waits in the lane behind the hook's body, whatever
    the body awaits after the call: its key still names it, and the agent's
    turn on it never starts once cancelled (RFC §9.5.1 Reentrance). The
    message itself is delivered all the same, ahead of the hook's event."""
    sqlite = SQLiteStore(tmp_path / "rooms.db") if store == "sqlite" else None
    room = await _room("agent", tmp_path, store=sqlite)
    counts: list[int] = []

    @room.kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def post(event: RoomEvent, ctx: Any) -> HookResult:
        if isinstance(event.content, TextContent) and event.content.body == "lead":
            inner = InboundMessage(
                channel_id="sms",
                sender_id="user",
                content=TextContent(body="task B"),
                delivery_key="b",
            )
            await room.kit.process_inbound(inner, room_id="r")
            await asyncio.sleep(work)  # the hook's own work, e.g. an HTTP call
            counts.append(await room.kit.cancel_delivery("b"))
        return HookResult.allow()

    try:
        room.seen.gate("lead").set()
        lead = await room.send("lead")
        assert lead.delivery is not None
        await asyncio.wait_for(lead.delivery.wait(), WAIT)
        await room.until_broadcast("task B")
        await _no_turn_follows()

        assert (counts, room.seen.started) == ([1], ["lead"])
        assert room.delivered()[:2] == ["task B", "lead"]
        await _settled(lambda: not room.kit._delivery_keys._named)
    finally:
        await _closed(room)


async def test_a_cancel_for_an_organization_reaches_only_its_rooms(tmp_path: Path) -> None:
    """Two tenants that chose the same key never cancel each other's."""
    seen = Seen()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(_Steerless(seen))
    for room_id, organization in (("r1", "org-1"), ("r2", "org-2")):
        await kit.create_room(room_id=room_id, organization_id=organization)
        await kit.attach_channel(room_id, "sms")
        await kit.attach_channel(room_id, "agent", category=ChannelCategory.INTELLIGENCE)

    def task(body: str) -> InboundMessage:
        return InboundMessage(
            channel_id="sms", sender_id="user", content=TextContent(body=body), delivery_key="t"
        )

    try:
        one = await kit.process_inbound(task("task A"), room_id="r1", defer_delivery=True)
        two = await kit.process_inbound(task("task B"), room_id="r2", defer_delivery=True)
        await seen.until_started("task A")
        await seen.until_started("task B")

        assert await kit.cancel_delivery("t", organization_id="org-2") == 1
        assert two.delivery is not None and one.delivery is not None
        await asyncio.wait_for(two.delivery.wait(), WAIT)
        seen.gate("task A").set()
        done_one = await asyncio.wait_for(one.delivery.wait(), WAIT)

        assert seen.finished == ["task A"]
        assert done_one.cancellation_reason is None
        assert await kit.cancel_delivery("t", organization_id="org-1") == 0
    finally:
        for gate in seen.gates.values():
            gate.set()
        await kit.close()


# -- before its commit: the key names the delivery from the call's start --------------


@dataclass
class _Pause:
    """A point before the commit (RFC §10.1 steps 1-5) where the next call
    waits, once armed, until let go."""

    armed: bool = False
    reached: asyncio.Event = field(default_factory=asyncio.Event)
    go: asyncio.Event = field(default_factory=asyncio.Event)

    async def here(self) -> None:
        if not self.armed:
            return
        self.armed = False
        self.reached.set()
        await self.go.wait()

    async def until_reached(self) -> None:
        async with asyncio.timeout(WAIT):
            await self.reached.wait()


class _RoutingStore(InMemoryStore):
    """Routing (step 2) asks the store whether the channel is bound."""

    def __init__(self, pause: _Pause) -> None:
        super().__init__()
        self._pause = pause

    async def binding_exists(self, room_id: str, channel_id: str) -> bool:
        await self._pause.here()
        return await super().binding_exists(room_id, channel_id)


class _ContextStore(InMemoryStore):
    """The context build (step 3) reads the room, its bindings and its
    participants."""

    def __init__(self, pause: _Pause) -> None:
        super().__init__()
        self._pause = pause

    async def load_room_context(self, room_id: str) -> Any:
        await self._pause.here()
        return await super().load_room_context(room_id)


class _ParsingChannel(SimpleChannel):
    """The channel's own parsing of the message (step 4)."""

    def __init__(self, pause: _Pause) -> None:
        super().__init__("sms")
        self._pause = pause

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        await self._pause.here()
        return await super().handle_inbound(message, context)


class _IdentifyingResolver(MockIdentityResolver):
    """Identity resolution (step 5): the sender stays unknown, so every
    message asks again."""

    def __init__(self, pause: _Pause) -> None:
        super().__init__()
        self._pause = pause

    async def resolve(self, message: InboundMessage, context: RoomContext) -> Any:
        await self._pause.here()
        return await super().resolve(message, context)


PAUSES: dict[str, Callable[[_Pause], dict[str, Any]]] = {
    "routing": lambda pause: {"store": _RoutingStore(pause)},
    "context": lambda pause: {"store": _ContextStore(pause)},
    "parsing": lambda pause: {"sms": _ParsingChannel(pause)},
    "identity": lambda pause: {"identity_resolver": _IdentifyingResolver(pause)},
}


@pytest.fixture(params=sorted(PAUSES))
def step(request: pytest.FixtureRequest) -> str:
    return str(request.param)


async def _paused(path: str, tmp_path: Path, step: str, **options: Any) -> tuple[Room, _Pause]:
    pause = _Pause()
    return await _room(path, tmp_path, **PAUSES[step](pause), **options), pause


def _task(body: str, key: str) -> InboundMessage:
    return InboundMessage(
        channel_id="sms", sender_id="user", content=TextContent(body=body), delivery_key=key
    )


async def test_a_delivery_cancelled_before_its_commit_starts_no_turn(
    path: str, step: str, tmp_path: Path
) -> None:
    """A cancel that reaches the call while it routes its message, reads its
    room, parses or identifies it is the delivery's: the message is committed
    and delivered all the same, and its turn never starts."""
    room, pause = await _paused(path, tmp_path, step)
    try:
        pause.armed = True
        call = asyncio.create_task(room.send("task A", "a"))
        await pause.until_reached()

        assert await room.kit.cancel_delivery("a", reason="stopped") == 1
        pause.go.set()
        result = await asyncio.wait_for(call, WAIT)
        assert result.delivery is not None
        done = await asyncio.wait_for(result.delivery.wait(), WAIT)
        await room.until_broadcast("task A")

        assert done.cancellation_reason == "stopped"
        assert room.seen.started == []
        assert room.delivered() == ["task A"]
        await _settled(lambda: not room.kit._delivery_keys._named)
    finally:
        await _closed(room)


async def test_an_awaited_call_cancelled_before_its_commit_waits_for_no_turn(
    step: str, tmp_path: Path
) -> None:
    """Queued behind the room's running turn, an awaited call cancelled
    before its commit returns at once, as one cancelled just after it does:
    its plan waiting in the lane is not its turns' work."""
    room, pause = await _paused("steerless", tmp_path, step)
    try:
        await room.send("task A", "a")
        await room.seen.until_started("task A")
        pause.armed = True
        call = asyncio.create_task(room.kit.process_inbound(_task("task B", "b"), room_id="r"))
        await pause.until_reached()

        assert await room.kit.cancel_delivery("b", reason="stopped") == 1
        pause.go.set()
        result = await asyncio.wait_for(call, WAIT)

        assert result.cancellation_reason == "stopped"
        assert room.seen.finished == []
        room.seen.gate("task A").set()
        await room.until_broadcast("task B")
        assert room.seen.started == ["task A"]
    finally:
        await _closed(room)


@pytest.mark.parametrize(
    ("step", "call_scope", "cancel_scope", "counted", "stopped"),
    [
        pytest.param("routing", "org-1", "org-1", 1, True, id="named-its-organization"),
        pytest.param("routing", None, "org-1", 0, True, id="room-not-read-yet"),
        pytest.param("context", None, "org-1", 0, True, id="room-being-read"),
        pytest.param("parsing", None, "org-1", 1, True, id="room-read"),
        pytest.param("identity", None, "org-1", 1, True, id="room-read-identifying"),
        pytest.param("routing", None, "org-2", 0, False, id="unread-other-organization"),
        pytest.param("parsing", None, "org-2", 0, False, id="read-other-organization"),
        pytest.param("routing", "org-1", "org-2", 0, False, id="named-other-organization"),
    ],
)
async def test_a_scoped_cancel_before_the_commit_reaches_only_its_organization(
    step: str,
    call_scope: str | None,
    cancel_scope: str,
    counted: int,
    stopped: bool,
    tmp_path: Path,
) -> None:
    """A call that names its organization is that organization's from its
    start, so a cancel for it is counted at once. One that does not is
    located once it reads its room: a cancel for its room's organization
    arriving before is applied then, uncounted, since until then it is no
    organization's; another organization's never reaches it."""
    room, pause = await _paused("steerless", tmp_path, step, organization_id="org-1")
    try:
        pause.armed = True
        call = asyncio.create_task(
            room.kit.process_inbound(
                _task("task A", "t"),
                room_id="r",
                organization_id=call_scope,
                defer_delivery=True,
            )
        )
        await pause.until_reached()

        cancelled = await room.kit.cancel_delivery(
            "t", reason="stopped", organization_id=cancel_scope
        )
        assert cancelled == counted
        room.seen.gate("task A").set()
        pause.go.set()
        result = await asyncio.wait_for(call, WAIT)
        assert result.delivery is not None
        done = await asyncio.wait_for(result.delivery.wait(), WAIT)
        await room.until_broadcast("task A")

        assert done.cancellation_reason == ("stopped" if stopped else None)
        assert room.seen.started == ([] if stopped else ["task A"])
    finally:
        await _closed(room)


async def test_a_call_refused_before_its_commit_names_nothing_after(tmp_path: Path) -> None:
    """A room of another organization is not found to the call: its key
    names nothing once the call has raised."""
    room = await _room("steerless", tmp_path, organization_id="org-1")
    try:
        with pytest.raises(RoomNotFoundError):
            await room.kit.process_inbound(
                _task("task A", "t"), room_id="r", organization_id="org-2"
            )

        assert room.kit._delivery_keys._named == {}
        assert await room.kit.cancel_delivery("t") == 0
    finally:
        await _closed(room)


@pytest.mark.parametrize("budgeted", ["context", "parsing", "identity"])
async def test_a_cancelled_call_that_times_out_before_its_commit_names_nothing_after(
    budgeted: str, tmp_path: Path
) -> None:
    """A call refused before its commit (its pre-commit budget spent, steps
    3 to 5) reports its refusal, whatever cancel reached it: no delivery,
    and its key released."""
    room, pause = await _paused("steerless", tmp_path, budgeted, process_timeout=0.05)
    try:
        pause.armed = True
        call = asyncio.create_task(room.send("task A", "a"))
        await pause.until_reached()

        assert await room.kit.cancel_delivery("a") == 1
        result = await asyncio.wait_for(call, WAIT)

        assert result.blocked and result.reason == "process_timeout"
        assert result.delivery is None and result.cancellation_reason is None
        assert room.kit._delivery_keys._named == {}
        assert room.seen.started == []
    finally:
        pause.go.set()
        await _closed(room)


async def test_an_awaited_call_cancelled_while_it_waits_for_the_room_lock_waits_for_no_turn(
    tmp_path: Path,
) -> None:
    """Named by its cascade and still waiting for the room's lock, a call
    cancelled before its plan is queued returns at once once committed:
    the turn ahead of it in the lane is not its turns' work."""
    room = await _room("steerless", tmp_path)
    holding = asyncio.Event()
    let_go = asyncio.Event()

    @room.kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def hold(event: RoomEvent, ctx: Any) -> HookResult:
        if isinstance(event.content, TextContent) and event.content.body == "task A":
            holding.set()
            await let_go.wait()
        return HookResult.allow()

    try:
        a = asyncio.create_task(room.send("task A", "a"))
        await asyncio.wait_for(holding.wait(), WAIT)
        call = asyncio.create_task(room.kit.process_inbound(_task("task B", "b"), room_id="r"))
        named = room.kit._delivery_keys._named
        await _settled(lambda: "b" in named and named["b"][0].cascade is not None)

        assert await room.kit.cancel_delivery("b", reason="stopped") == 1
        let_go.set()
        await asyncio.wait_for(a, WAIT)
        await room.seen.until_started("task A")
        result = await asyncio.wait_for(call, WAIT)

        assert result.cancellation_reason == "stopped"
        assert room.seen.finished == []
        room.seen.gate("task A").set()
        await room.until_broadcast("task B")
        assert room.seen.started == ["task A"]
    finally:
        let_go.set()
        await _closed(room)

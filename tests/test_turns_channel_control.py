"""A channel's per-room state across the turn backend (RFC §10.6.2, §10.6.3,
§10.6.7, RMK-812).

Each request names the room's previous turn of its channel; an executor
drops a room's state another turn left; a room's turns stay on the executor
holding its state; channel control reaches executors outside any turn.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.models.steering import Cancel
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor
from roomkit.turns._rooms import room_state_key
from roomkit.turns.base import ExecutorLink, TurnLink
from roomkit.turns.wire import (
    ChannelCall,
    CommittedEvent,
    RoomAttached,
    RoomDetached,
    TurnRequest,
)


class _Agent(Channel):
    """An intelligence channel that records what its executor asks of it."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str = "agent") -> None:
        super().__init__(channel_id)
        self.log: list[tuple[str, str]] = []
        self.config: dict[str, dict[str, Any]] = {}
        self.restored: list[dict[str, Any]] = []
        self.release: asyncio.Event | None = None
        self.steered: list[Any] = []
        self.bindings: list[ChannelBinding] = []
        self.events: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.log.append(("turn", event.room_id))
        if self.release is not None:
            await self.release.wait()
        return ChannelOutput(responded=True)

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()

    async def on_room_attached(self, room_id: str, binding: ChannelBinding) -> None:
        self.log.append(("attached", room_id))
        self.bindings.append(binding)

    async def on_room_event(self, room_id: str, event: RoomEvent) -> None:
        self.events.append(event)

    async def on_room_detached(self, room_id: str) -> None:
        self.log.append(("detached", room_id))
        self.config.pop(room_id, None)

    async def channel_call(self, room_id: str, verb: str, arguments: dict[str, Any]) -> Any:
        self.log.append((verb, room_id))
        if verb == "set":
            self.config.setdefault(room_id, {}).update(arguments)
            return dict(self.config[room_id])
        if verb == "forget":
            return self.config.pop(room_id, None) is not None
        if verb == "read":
            return dict(self.config.get(room_id, {}))
        if verb == "unsendable":
            return object()
        return await super().channel_call(room_id, verb, arguments)

    def room_state(self, room_id: str) -> dict[str, Any]:
        config = self.config.get(room_id)
        return {"config": dict(config)} if config else {}

    def restore_room_state(self, room_id: str, state: dict[str, Any]) -> None:
        self.restored.append(state)
        if room_id not in self.config and state.get("config"):
            self.config[room_id] = dict(state["config"])

    async def steer(self, directive: Any, *, room_id: str) -> int:
        self.steered.append(directive)
        if self.release is not None:
            self.release.set()
        return 1


class _Recording(TurnExecutor):
    """An executor serving one channel that keeps the requests it took."""

    def __init__(self, channel: Channel, executor_id: str) -> None:
        self.declaring = True
        super().__init__(
            lambda channel_id: channel if channel_id == channel.channel_id else None,
            hosts=lambda channel_id: self.declaring and channel_id == channel.channel_id,
            executor_id=executor_id,
        )
        self.requests: list[TurnRequest] = []

    def take(self, link: ExecutorLink) -> None:
        self.requests.append(link.request)
        super().take(link)


def _event(room_id: str = "room") -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
        content=TextContent(body="hello"),
    )


def _binding(room_id: str = "room") -> ChannelBinding:
    return ChannelBinding(
        channel_id="agent",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _context(room_id: str = "room", metadata: dict[str, Any] | None = None) -> RoomContext:
    room = Room(id=room_id, metadata=metadata or {})
    return RoomContext(room=room, bindings=[_binding(room_id)])


async def _turn(client: TurnClient, room_id: str = "room", **metadata: Any) -> None:
    await client.run(_event(room_id), _binding(room_id), _context(room_id, metadata))


async def _two_executors() -> tuple[TurnClient, _Recording, _Recording, _Agent, _Agent]:
    """A client on a backend served by executors A, then B, each with its own
    copy of the channel."""
    backend = InMemoryTurnBackend()
    first, second = _Agent(), _Agent()
    a, b = _Recording(first, "a"), _Recording(second, "b")
    await backend.serve(a)
    await backend.serve(b)
    return TurnClient(backend), a, b, first, second


# -- Placement (RFC §10.6.7) ----------------------------------------------------


async def test_a_turn_no_executor_declares_fails_at_once() -> None:
    client = TurnClient(InMemoryTurnBackend())
    with pytest.raises(TurnLinkError) as raised:
        # Within a fraction of the 10 s accept bound: no executor could take it.
        await asyncio.wait_for(_turn(client), 1)
    assert raised.value.reason == "no_executor"


async def test_a_room_s_turns_stay_on_the_executor_holding_its_state() -> None:
    client, a, b, _first, _second = await _two_executors()
    a.declaring = False
    await _turn(client)
    a.declaring = True
    await _turn(client)

    # A is the first to declare the channel again, yet the room stays on B.
    assert (len(a.requests), len(b.requests)) == (0, 2)


async def test_a_room_whose_executor_stopped_moves_to_another() -> None:
    client, a, b, _first, _second = await _two_executors()
    await _turn(client)
    await a.close()
    await _turn(client)
    assert (len(a.requests), len(b.requests)) == (1, 1)


async def test_rooms_are_placed_one_by_one() -> None:
    client, a, b, _first, _second = await _two_executors()
    a.declaring = False
    await _turn(client, "r1")
    a.declaring = True
    await _turn(client, "r2")
    assert [r.room_id for r in a.requests] == ["r2"]
    assert [r.room_id for r in b.requests] == ["r1"]


# -- The previous turn (RFC §10.6.2) ----------------------------------------------


async def test_each_request_names_the_room_s_previous_turn() -> None:
    client, a, _b, _first, _second = await _two_executors()
    await _turn(client)
    await _turn(client)
    await _turn(client, "elsewhere")

    first, second, other = a.requests
    assert first.previous_turn is None
    assert second.previous_turn is not None
    assert (second.previous_turn.turn_id, second.previous_turn.executor_id) == (
        first.turn_id,
        "a",
    )
    assert other.previous_turn is None


async def test_a_detach_leaves_the_next_turn_no_previous_one() -> None:
    client, a, _b, _first, _second = await _two_executors()
    await _turn(client)
    await client.control(RoomDetached(channel_id="agent", room_id="room"))
    await _turn(client)
    assert a.requests[-1].previous_turn is None


async def test_an_executor_drops_a_room_s_state_a_turn_elsewhere_made_stale() -> None:
    client, a, b, first, second = await _two_executors()
    await _turn(client)  # on A
    await _turn(client)  # on A: its own previous turn, nothing dropped
    assert first.log == [("turn", "room"), ("turn", "room")]

    a.declaring = False
    await _turn(client)  # on B
    a.declaring = True
    await b.close()
    await _turn(client)  # back on A, whose state the turn on B made stale

    assert first.log == [
        ("turn", "room"),
        ("turn", "room"),
        ("detached", "room"),
        ("turn", "room"),
    ]
    # B had never held the room: dropping there let go of nothing.
    assert second.log == [("detached", "room"), ("turn", "room")]


async def test_an_executor_that_missed_a_detach_still_drops_the_room_s_state() -> None:
    backend = InMemoryTurnBackend()
    agent = _Agent()
    executor = _Recording(agent, "a")
    await backend.serve(executor)
    client = TurnClient(backend)
    await _turn(client)
    # The owner forgot the room's previous turn, as after a detach whose
    # message the executor never received.
    client._previous.pop("agent", "room")
    await _turn(client)
    assert agent.log == [("turn", "room"), ("detached", "room"), ("turn", "room")]


# -- Channel control (RFC §10.6.3) --------------------------------------------------


async def test_attach_and_detach_apply_where_the_channel_is_built_and_build_nothing() -> None:
    """Both executors declare the channel; A, served first, built it for a
    turn. A room's attach and detach reach both: A applies them, B, holding
    no build, has nothing to apply and builds nothing (RFC §10.6.3)."""
    backend = InMemoryTurnBackend()
    first, second = _Agent(), _Agent()
    built: list[str] = []

    def build_second(channel_id: str) -> _Agent:
        built.append(channel_id)
        return second

    await backend.serve(_Recording(first, "a"))
    await backend.serve(TurnExecutor(build_second, hosts={"agent"}, executor_id="b"))
    client = TurnClient(backend)
    await _turn(client, "other")

    await client.control(RoomAttached(channel_id="agent", room_id="room", binding=_binding()))
    await client.control(RoomDetached(channel_id="agent", room_id="room"))

    assert first.log == [("turn", "other"), ("attached", "room"), ("detached", "room")]
    assert second.log == [] and built == []


async def test_an_attach_during_the_channel_s_build_applies_once_it_is_built() -> None:
    """A turn is building the channel when a room's attach arrives: the
    attach waits for that build and applies on it (RFC §10.6.3)."""
    agent = _Agent()
    building, go = asyncio.Event(), asyncio.Event()

    async def build(channel_id: str) -> _Agent:
        building.set()
        await go.wait()
        return agent

    backend = InMemoryTurnBackend()
    executor = TurnExecutor(build, hosts={"agent"})
    await backend.serve(executor)
    client = TurnClient(backend)
    turn = asyncio.ensure_future(_turn(client, "other"))
    await asyncio.wait_for(building.wait(), 1)
    attach = asyncio.ensure_future(
        client.control(RoomAttached(channel_id="agent", room_id="room", binding=_binding()))
    )
    for _ in range(20):  # Let the attach reach the executor, which waits for the build.
        await asyncio.sleep(0)
    waited = not attach.done()
    go.set()
    # The first build in a process compiles the mark patterns, about a second.
    await asyncio.wait_for(asyncio.gather(turn, attach), 5)
    await executor.close()

    assert waited
    assert ("attached", "room") in agent.log


class _Opaque:
    """A value JSON has no form for."""


async def test_channel_control_carries_a_value_json_has_no_form_for_as_its_text() -> None:
    """An application in the owner's process keeps an object in a binding's
    or an event's metadata: the control message carries it as its text, as a
    turn's request does."""
    client, _a, b, first, second = await _two_executors()
    b.declaring = False
    await _turn(client)  # A builds the channel: the attach then applies there.
    b.declaring = True
    binding = _binding().model_copy(update={"metadata": {"handle": _Opaque()}})
    event = _event().model_copy(update={"metadata": {"handle": _Opaque()}})

    await client.control(RoomAttached(channel_id="agent", room_id="room", binding=binding))
    await client.control(CommittedEvent(channel_id="agent", room_id="room", event=event))

    assert first.bindings[0].metadata["handle"].startswith("<")
    [handed] = first.events + second.events
    assert handed.metadata["handle"].startswith("<")


async def test_a_channel_call_reaches_the_executor_holding_the_room() -> None:
    client, a, _b, first, second = await _two_executors()
    a.declaring = False
    await _turn(client)
    a.declaring = True
    call = ChannelCall(channel_id="agent", room_id="room", verb="set", arguments={"m": "x"})
    answer = await client.control(call)

    assert answer is not None and answer.result == {"m": "x"}
    assert ("set", "room") in second.log and ("set", "room") not in first.log


async def test_a_turn_follows_the_executor_a_channel_call_reached() -> None:
    client, a, b, _first, _second = await _two_executors()
    a.declaring = False
    await client.control(ChannelCall(channel_id="agent", room_id="room", verb="set"))
    a.declaring = True
    await _turn(client)
    assert (len(a.requests), len(b.requests)) == (0, 1)


async def test_a_channel_call_no_executor_declares_fails_no_executor() -> None:
    client = TurnClient(InMemoryTurnBackend())
    with pytest.raises(TurnLinkError) as raised:
        await client.control(ChannelCall(channel_id="agent", room_id="room", verb="set"))
    assert raised.value.reason == "no_executor"


async def test_a_channel_call_answers_the_room_s_state_when_the_verb_changed_it() -> None:
    client, _a, _b, _first, _second = await _two_executors()
    set_model = ChannelCall(
        channel_id="agent", room_id="room", verb="set", arguments={"model": "opus"}
    )
    changed = await client.control(set_model)
    assert changed is not None and changed.room_state == {"config": {"model": "opus"}}

    same = set_model.model_copy(update={"room_state": changed.room_state})
    unchanged = await client.control(same)
    assert unchanged is not None and unchanged.room_state is None

    forget = ChannelCall(
        channel_id="agent", room_id="room", verb="forget", room_state=changed.room_state
    )
    dropped = await client.control(forget)
    assert dropped is not None and dropped.room_state == {}


async def test_a_turn_hands_the_channel_what_the_room_keeps_for_it() -> None:
    client, _a, _b, first, _second = await _two_executors()
    await _turn(client, **{room_state_key("agent"): {"config": {"model": "opus"}}})
    assert first.restored == [{"config": {"model": "opus"}}]
    assert first.config == {"room": {"model": "opus"}}


async def test_an_async_steer_of_the_channel_is_awaited() -> None:
    client, _a, _b, first, _second = await _two_executors()
    first.release = asyncio.Event()
    turn = asyncio.create_task(_turn(client))
    while ("turn", "room") not in first.log:
        await asyncio.sleep(0.01)

    reached = await client.steer("room", "agent", Cancel(reason="hold"))

    assert reached == 1
    await asyncio.wait_for(turn, 5)
    assert [type(d) for d in first.steered] == [Cancel]


# -- Through the kit ----------------------------------------------------------------


async def _kit() -> tuple[RoomKit, _Agent]:
    kit = RoomKit()
    agent = _Agent()
    kit.register_channel(agent)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "agent")
    return kit, agent


async def test_the_kit_stores_a_channel_call_s_room_state_with_the_room() -> None:
    kit, agent = await _kit()
    result = await kit.channel_call("agent", "room", "set", model="opus")
    assert result == {"model": "opus"}
    room = await kit.get_room("room")
    assert room.metadata[room_state_key("agent")] == {"config": {"model": "opus"}}

    # The room's next executor finds it, here a channel that lost its copy.
    agent.config.clear()
    assert await kit.channel_call("agent", "room", "forget") is True
    room = await kit.get_room("room")
    assert room_state_key("agent") not in room.metadata
    await kit.close()


async def test_detaching_a_channel_forgets_what_the_room_kept_for_it() -> None:
    kit, agent = await _kit()
    await kit.channel_call("agent", "room", "set", model="opus")
    await kit.detach_channel("room", "agent")

    room = await kit.get_room("room")
    assert room_state_key("agent") not in room.metadata
    assert agent.log[-1] == ("detached", "room")
    await kit.close()


async def test_a_channel_call_across_a_detach_stores_nothing_with_the_room() -> None:
    kit, agent = await _kit()
    reached, release = asyncio.Event(), asyncio.Event()
    verb = agent.channel_call

    async def slow(room_id: str, name: str, arguments: dict[str, Any]) -> Any:
        reached.set()
        await release.wait()
        return await verb(room_id, name, arguments)

    agent.channel_call = slow  # type: ignore[method-assign]
    call = asyncio.create_task(kit.channel_call("agent", "room", "set", model="opus"))
    await asyncio.wait_for(reached.wait(), 5)
    await kit.detach_channel("room", "agent")
    release.set()
    assert await call == {"model": "opus"}

    # The detach forgot what the room kept; the call that ran across it
    # does not write it back beside the detach's tombstone.
    room = await kit.get_room("room")
    assert room_state_key("agent") not in room.metadata
    await kit.close()


async def test_a_verb_the_channel_lacks_raises_on_the_owner() -> None:
    kit, _agent = await _kit()
    with pytest.raises(ValueError, match="no per-room verb 'nope'"):
        await kit.channel_call("agent", "room", "nope")
    with pytest.raises(TypeError, match="cannot cross the turn backend"):
        await kit.channel_call("agent", "room", "unsendable")
    await kit.close()


async def test_an_intelligence_channel_s_attach_failure_is_logged_on_its_executor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Refusing(_Agent):
        async def on_room_attached(self, room_id: str, binding: ChannelBinding) -> None:
            raise RuntimeError("no room for you")

    kit = RoomKit()
    kit.register_channel(_Refusing())
    await kit.create_room(room_id="room")
    with caplog.at_level(logging.ERROR, logger="roomkit.turns"):
        # No result crosses back for an attachment (RFC §10.6.3).
        await kit.attach_channel("room", "agent")
    assert await kit.get_binding("room", "agent") is not None
    assert "no room for you" in caplog.text
    await kit.close()


# -- Review: what a channel call leaves, records that fall out, failed drops ------------


async def test_a_channel_call_drops_a_state_a_turn_elsewhere_made_stale() -> None:
    client, a, b, first, _second = await _two_executors()
    await client.control(
        ChannelCall(channel_id="agent", room_id="room", verb="set", arguments={"m": "sonnet"})
    )  # on A
    a.declaring = False
    await client.control(
        ChannelCall(channel_id="agent", room_id="room", verb="set", arguments={"m": "opus"})
    )  # on B
    a.declaring = True
    await b.close()
    answer = await client.control(
        ChannelCall(
            channel_id="agent",
            room_id="room",
            verb="read",
            room_state={"config": {"m": "opus"}},
        )
    )  # back on A, whose copy the call on B made stale

    assert first.log[-2:] == [("detached", "room"), ("read", "room")]
    # The read changed nothing: the room's opus is not overwritten.
    assert answer is not None and answer.room_state is None
    assert first.config == {"room": {"m": "opus"}}


async def test_a_turn_after_a_channel_call_names_it_and_keeps_its_state() -> None:
    client, a, _b, first, _second = await _two_executors()
    call = ChannelCall(channel_id="agent", room_id="room", verb="set", arguments={"m": "x"})
    answer = await client.control(call)
    await _turn(client)

    (request,) = a.requests
    assert request.previous_turn is not None
    assert request.previous_turn.turn_id == call.call_id
    assert answer is not None and request.previous_turn.executor_id == answer.executor_id == "a"
    assert ("detached", "room") not in first.log


async def test_a_channel_call_the_owner_stopped_waiting_for_is_the_previous_one() -> None:
    client, a, _b, first, _second = await _two_executors()
    await _turn(client)
    reached = asyncio.Event()

    async def hang(room_id: str, verb: str, arguments: dict[str, Any]) -> Any:
        reached.set()
        await asyncio.Event().wait()

    first.channel_call = hang  # type: ignore[method-assign]
    call = ChannelCall(channel_id="agent", room_id="room", verb="hang")
    waiting = asyncio.create_task(client.control(call))
    await asyncio.wait_for(reached.wait(), 5)
    # The call reached the executor, which recorded it as leaving the room's
    # state: the owner cancelled waiting for it, not the record.
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await _turn(client)

    request = a.requests[-1]
    assert request.previous_turn is not None
    assert request.previous_turn.turn_id == call.call_id
    assert ("detached", "room") not in first.log


async def test_a_channel_call_no_executor_took_leaves_the_previous_turn() -> None:
    backend = InMemoryTurnBackend()
    agent = _Agent()
    executor = _Recording(agent, "a")
    await backend.serve(executor)
    client = TurnClient(backend)
    await _turn(client)
    executor.declaring = False
    with pytest.raises(TurnLinkError) as raised:
        await client.control(ChannelCall(channel_id="agent", room_id="room", verb="set"))
    assert raised.value.reason == "no_executor"
    executor.declaring = True
    await _turn(client)

    first, second = executor.requests
    assert second.previous_turn is not None
    assert second.previous_turn.turn_id == first.turn_id
    assert ("detached", "room") not in agent.log


async def test_a_channel_call_refused_before_it_is_sent_leaves_the_previous_turn() -> None:
    backend = InMemoryTurnBackend()
    agent = _Agent()
    executor = _Recording(agent, "a")
    await backend.serve(executor)
    client = TurnClient(backend)
    await _turn(client)
    # An argument JSON has no form for: refused before it crosses.
    call = ChannelCall(channel_id="agent", room_id="room", verb="set", arguments={"x": object()})
    with pytest.raises(ValueError):
        await client.control(call)
    await _turn(client)

    first, second = executor.requests
    assert second.previous_turn is not None
    assert second.previous_turn.turn_id == first.turn_id
    assert ("detached", "room") not in agent.log


async def test_a_turn_refused_before_it_is_handed_over_leaves_the_previous_turn() -> None:
    backend = InMemoryTurnBackend()
    agent = _Agent()
    executor = _Recording(agent, "a")
    await backend.serve(executor)
    client = TurnClient(backend)
    await _turn(client)
    # Bytes that are not UTF-8 have no text to fall back on: the request
    # cannot be written, and no executor is given it.
    with pytest.raises(ValueError):
        await _turn(client, raw=b"\xff\xfe")
    await _turn(client)

    first, second = executor.requests
    assert second.previous_turn is not None
    assert second.previous_turn.turn_id == first.turn_id
    assert ("detached", "room") not in agent.log


class _AnswerAfterTaking(InMemoryTurnBackend):
    """Hands a request to its executor, then waits for the executor's answer,
    as a backend across processes does: a cancel there comes once the turn
    was taken."""

    def __init__(self) -> None:
        super().__init__()
        self.hold = False

    async def submit(self, request: TurnRequest) -> TurnLink:
        link = await super().submit(request)
        if self.hold:
            self.hold = False
            await asyncio.Event().wait()
        return link


async def test_a_turn_the_owner_stopped_submitting_once_taken_is_the_previous_one() -> None:
    backend = _AnswerAfterTaking()
    agent = _Agent()
    executor = _Recording(agent, "a")
    await backend.serve(executor)
    client = TurnClient(backend)
    await _turn(client)
    backend.hold = True
    submitting = asyncio.create_task(_turn(client))
    for _ in range(100):
        if agent.log.count(("turn", "room")) == 2:
            break
        await asyncio.sleep(0.01)
    # The executor began the turn, so it left the room's state: the owner
    # cancelled waiting for the answer, not the record.
    submitting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submitting
    await _turn(client)

    _first, taken, last = executor.requests
    assert last.previous_turn is not None
    assert last.previous_turn.turn_id == taken.turn_id
    assert ("detached", "room") not in agent.log


async def test_a_record_that_falls_out_drops_the_state_it_vouched_for() -> None:
    client, a, _b, first, _second = await _two_executors()
    a._rooms._left._kept = 1
    await _turn(client, "r1")
    await _turn(client, "r2")  # r1's record falls out
    async with asyncio.timeout(5):
        while ("detached", "r1") not in first.log:
            await asyncio.sleep(0.01)
    assert ("detached", "r2") not in first.log


async def test_a_failed_drop_is_tried_again_on_the_next_turn() -> None:
    client, a, b, first, _second = await _two_executors()
    await _turn(client)  # on A
    a.declaring = False
    await _turn(client)  # on B
    a.declaring = True
    await b.close()
    failing = [RuntimeError("store down")]
    detached = first.on_room_detached

    async def flaky(room_id: str) -> None:
        if failing:
            raise failing.pop()
        await detached(room_id)

    first.on_room_detached = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="store down"):
        await _turn(client)  # back on A: the drop fails, so does the turn
    await _turn(client)

    assert first.log[-2:] == [("detached", "room"), ("turn", "room")]


class _Yielding(InMemoryTurnBackend):
    """A backend whose submit yields, as one carried by a network does."""

    async def submit(self, request: TurnRequest) -> Any:
        await asyncio.sleep(0)
        return await super().submit(request)


async def test_a_turn_is_the_previous_one_from_the_moment_it_is_sent() -> None:
    backend = _Yielding()
    agent = _Agent()
    executor = _Recording(agent, "a")
    await backend.serve(executor)
    client = TurnClient(backend)
    await asyncio.gather(_turn(client), _turn(client))

    first, second = executor.requests
    assert second.previous_turn is not None
    assert second.previous_turn.turn_id == first.turn_id
    assert ("detached", "room") not in agent.log


async def test_a_turn_no_executor_took_is_no_previous_turn() -> None:
    backend = InMemoryTurnBackend()
    client = TurnClient(backend)
    with pytest.raises(TurnLinkError):
        await _turn(client)
    executor = _Recording(_Agent(), "a")
    await backend.serve(executor)
    await _turn(client)
    assert executor.requests[0].previous_turn is None


async def test_a_missing_hook_refusal_preserves_the_previous_turn() -> None:
    backend = InMemoryTurnBackend()
    executor = _Recording(_Agent(), "a")
    client = TurnClient(backend)
    await backend.serve(executor)
    try:
        await _turn(client)
        first = executor.requests[0].turn_id
        client.enable_room_hook("room", "missing")
        with pytest.raises(TurnLinkError) as raised:
            await _turn(client)
        assert raised.value.reason == "no_executor"
        assert client.disable_room_hook("room", "missing")
        await _turn(client)
        assert len(executor.requests) == 2
        assert executor.requests[1].previous_turn is not None
        assert executor.requests[1].previous_turn.turn_id == first
    finally:
        await executor.close()
        await backend.close()


async def test_detaching_an_ai_channel_drops_everything_it_kept_for_the_room() -> None:
    kit = RoomKit()
    agent = AIChannel("ai", provider=MockAIProvider(responses=["ok"]), enable_planning=True)
    kit.register_channel(agent)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "ai")
    assert agent._planner is not None
    await agent._planner.handle_plan_tasks(
        {"tasks": [{"title": "Read the logs", "status": "pending"}]}, room_id="room"
    )
    agent._tool_usage.record("room", "search", {"q": "x"}, "found")
    agent._skill_activation.activate("room", "triage")
    agent._refused_images.add("room", [("https://example.test/a.png", 0)])

    await kit.detach_channel("room", "ai")

    assert agent._planner.plan_for("room") is None
    assert agent._tool_usage.tool_names("room") == set()
    assert agent._skill_activation.active_names("room") == set()
    assert agent._refused_images.of("room") == frozenset()
    await kit.close()

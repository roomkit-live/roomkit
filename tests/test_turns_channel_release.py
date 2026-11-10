"""A declared channel's build is let go when its host says it is stale
(RFC §10.6.3, ``channel_released``).

An executor builds a declared channel when it first needs it and keeps it,
up to its bound (``max_channels``); when its host changes
what the channel is built from (a model, a provider), ``kit.release_channel``
has every executor holding a build let it go: the next turn builds it anew,
and the old one is closed once the turns running on it end.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ChannelDeclaration, RoomKit
from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.store.memory import InMemoryStore
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from roomkit.turns.wire import CommittedEvent
from tests.test_framework import SimpleChannel
from tests.test_turns_channel_control import _Agent


class _Built(_Agent):
    """The agent as one build of it: closing it, and the start of a turn or
    a channel call on it, are recorded; ``release``, when set, holds both."""

    def __init__(self, generation: int, channel_id: str = "agent") -> None:
        super().__init__(channel_id)
        self.generation = generation
        self.closed = False
        self.closing = asyncio.Event()
        self.busy = asyncio.Event()

    async def on_event(self, event: Any, binding: Any, context: Any) -> Any:
        self.busy.set()
        return await super().on_event(event, binding, context)

    async def channel_call(self, room_id: str, verb: str, arguments: dict[str, Any]) -> Any:
        self.busy.set()
        if self.release is not None:
            await self.release.wait()
        return await super().channel_call(room_id, verb, arguments)

    async def close(self) -> None:
        self.closed = True
        self.closing.set()
        await super().close()


class _Builds:
    """The host's factory: a new build of ``agent`` each time it is asked."""

    def __init__(self) -> None:
        self.built: list[_Built] = []

    def __call__(self, channel_id: str) -> _Built | None:
        if channel_id != "agent":
            return None
        built = _Built(len(self.built))
        self.built.append(built)
        return built


async def _kit() -> tuple[RoomKit, _Builds, TurnExecutor]:
    backend = InMemoryTurnBackend()
    builds = _Builds()
    executor = TurnExecutor(builds, hosts={"agent"})
    await backend.serve(executor)
    kit = RoomKit(store=InMemoryStore(), turn_backend=backend)
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(ChannelDeclaration(channel_id="agent"))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "ops")
    await kit.attach_channel("room", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, builds, executor


async def _say(kit: RoomKit) -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body="go")),
        room_id="room",
    )


def _turns(built: _Built) -> int:
    return sum(1 for entry in built.log if entry[0] == "turn")


async def test_a_released_channel_is_built_anew_for_the_next_turn_and_closed() -> None:
    kit, builds, executor = await _kit()
    try:
        await _say(kit)
        await kit.release_channel("agent")
        await _say(kit)

        first, second = builds.built
        assert (_turns(first), _turns(second)) == (1, 1)
        assert first.closed and not second.closed
    finally:
        await kit.close()
        await executor.close()


async def test_a_turn_running_on_a_released_channel_ends_on_it_before_it_closes() -> None:
    kit, builds, executor = await _kit()
    try:
        await _say(kit)
        first = builds.built[0]
        first.release = asyncio.Event()
        running = asyncio.create_task(_say(kit))
        for _ in range(200):
            if _turns(first) == 2:
                break
            await asyncio.sleep(0.01)

        await kit.release_channel("agent")
        await asyncio.sleep(0.05)
        assert not first.closed  # its turn still runs on it

        first.release.set()
        await asyncio.wait_for(running, 5)
        for _ in range(200):
            if first.closed:
                break
            await asyncio.sleep(0.01)

        assert first.closed
        assert _turns(first) == 2
        await _say(kit)
        assert _turns(builds.built[1]) == 1
    finally:
        await kit.close()
        await executor.close()


async def test_releasing_a_channel_never_built_builds_nothing() -> None:
    """Declared, never attached nor asked for a turn: there is nothing to let
    go, and a release is no reason to build it."""
    backend = InMemoryTurnBackend()
    builds = _Builds()
    executor = TurnExecutor(builds, hosts={"agent"})
    await backend.serve(executor)
    kit = RoomKit(store=InMemoryStore(), turn_backend=backend)
    kit.register_channel(ChannelDeclaration(channel_id="agent"))
    try:
        await kit.release_channel("agent")

        assert builds.built == []
    finally:
        await kit.close()
        await executor.close()


async def test_a_channel_the_kit_holds_live_is_its_registrys() -> None:
    """The kit's own executor reads the registry: releasing a live channel
    leaves it registered and open, its host replacing it by registering."""
    kit = RoomKit()
    live = _Built(0)
    kit.register_channel(live)
    try:
        await kit.release_channel("agent")

        assert kit.get_channel("agent") is live
        assert not live.closed
    finally:
        await kit.close()


async def test_releasing_an_unknown_channel_is_refused() -> None:
    kit = RoomKit()
    try:
        with pytest.raises(ChannelNotRegisteredError):
            await kit.release_channel("nobody")
    finally:
        await kit.close()


# -- A bound on the agents an executor keeps built (RFC §10.6.2) -------------------

# The first build in a process compiles the mark patterns, about a second.
_FIRST_BUILD_S = 5


class _Hold:
    """Holds one agent's turns and channel calls until released, and says
    when one started."""

    def __init__(self, agent: str) -> None:
        self.agent = agent
        self.release = asyncio.Event()
        self.started = asyncio.Event()


async def _pool(
    max_channels: int,
    *agents: str,
    hold: _Hold | None = None,
    slow: _Hold | None = None,
) -> tuple[RoomKit, list[_Built], TurnExecutor]:
    """An executor declaring every agent, keeping *max_channels* built, and
    a room per agent; *hold* holds an agent's turns and calls, *slow* its
    build."""
    backend = InMemoryTurnBackend()
    built: list[_Built] = []

    async def build(channel_id: str) -> _Built:
        if slow is not None and channel_id == slow.agent:
            slow.started.set()
            await slow.release.wait()
        agent = _Built(len(built), channel_id)
        if hold is not None and channel_id == hold.agent:
            agent.release, agent.busy = hold.release, hold.started
        built.append(agent)
        return agent

    executor = TurnExecutor(build, hosts={"*"}, max_channels=max_channels)
    await backend.serve(executor)
    kit = RoomKit(store=InMemoryStore(), turn_backend=backend)
    kit.register_channel(SimpleChannel("ops"))
    for agent in agents:
        kit.register_channel(ChannelDeclaration(channel_id=agent))
        await kit.create_room(room_id=agent)
        await kit.attach_channel(agent, "ops")
        await kit.attach_channel(agent, agent, category=ChannelCategory.INTELLIGENCE)
    return kit, built, executor


async def _say_to(kit: RoomKit, agent: str) -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body="go")),
        room_id=agent,
    )


async def test_an_executor_past_its_bound_lets_go_of_the_agent_used_least_recently() -> None:
    """It keeps two agents built. Alpha used again after beta, building
    gamma lets go of beta, and beta's next turn builds it anew, letting go
    of alpha: each closed when let go."""
    kit, built, executor = await _pool(2, "alpha", "beta", "gamma")
    try:
        for agent in ("alpha", "beta", "alpha", "gamma", "beta"):
            await _say_to(kit, agent)
        closes = asyncio.gather(built[0].closing.wait(), built[1].closing.wait())
        await asyncio.wait_for(closes, _FIRST_BUILD_S)
        closed = [(b.channel_id, b.generation, b.closed) for b in built]
    finally:
        await kit.close()
        await executor.close()

    assert closed == [
        ("alpha", 0, True),
        ("beta", 1, True),
        ("gamma", 2, False),
        ("beta", 3, False),
    ]


@pytest.mark.parametrize("on_alpha", ["turn", "channel_call"])
async def test_the_bound_never_lets_go_of_an_agent_something_runs_on(on_alpha: str) -> None:
    """Bound 1, beta built while a turn or a channel call runs on alpha:
    alpha is kept, and beta, idle once its turn ended, is let go instead."""
    hold = _Hold("alpha")
    kit, built, executor = await _pool(1, "alpha", "beta", hold=hold)
    running = asyncio.ensure_future(
        _say_to(kit, "alpha")
        if on_alpha == "turn"
        else kit.channel_call("alpha", "alpha", "set", mode="quiet")
    )
    try:
        await asyncio.wait_for(hold.started.wait(), _FIRST_BUILD_S)
        await _say_to(kit, "beta")
        await asyncio.wait_for(built[1].closing.wait(), _FIRST_BUILD_S)
        alpha_kept = not built[0].closed
        hold.release.set()
        await running
    finally:
        hold.release.set()
        await kit.close()
        await executor.close()

    assert alpha_kept
    assert [b.channel_id for b in built] == ["alpha", "beta"]


async def test_a_build_made_for_a_committed_event_alone_is_the_first_let_go() -> None:
    """Bound 1, alpha built for a turn: a committed event of beta's room
    builds beta and hands it the event, then the bound lets go of beta, not
    of alpha, which its room's turns keep using."""
    kit, built, executor = await _pool(1, "alpha", "beta")
    event = RoomEvent(
        room_id="beta",
        source=EventSource(channel_id="ops", channel_type=ChannelType.SMS),
        content=TextContent(body="earlier"),
    )
    try:
        await _say_to(kit, "alpha")
        await kit._turns.backend.control(
            CommittedEvent(channel_id="beta", room_id="beta", event=event)
        )
        await asyncio.wait_for(built[1].closing.wait(), _FIRST_BUILD_S)
        alpha_closed = built[0].closed
        handed = [e.content.body for e in built[1].events if isinstance(e.content, TextContent)]
    finally:
        await kit.close()
        await executor.close()

    assert not alpha_closed
    assert handed == ["earlier"]


async def test_a_committed_event_joining_a_turn_s_build_leaves_it_recent() -> None:
    """Bound 2: gamma's build, begun for a turn, is joined by a committed
    event of its room. The turn used gamma: delta's build then lets go of
    beta, used before it, never of gamma."""
    slow = _Hold("gamma")
    kit, built, executor = await _pool(2, "alpha", "beta", "gamma", "delta", slow=slow)
    event = RoomEvent(
        room_id="gamma",
        source=EventSource(channel_id="ops", channel_type=ChannelType.SMS),
        content=TextContent(body="earlier"),
    )
    try:
        for agent in ("alpha", "beta"):
            await _say_to(kit, agent)
        turn = asyncio.ensure_future(_say_to(kit, "gamma"))
        await asyncio.wait_for(slow.started.wait(), _FIRST_BUILD_S)
        handed = asyncio.ensure_future(
            kit._turns.backend.control(
                CommittedEvent(channel_id="gamma", room_id="gamma", event=event)
            )
        )
        for _ in range(20):  # Let the event reach the executor and join the build.
            await asyncio.sleep(0)
        slow.release.set()
        await asyncio.wait_for(asyncio.gather(turn, handed), _FIRST_BUILD_S)
        await _say_to(kit, "delta")
        await asyncio.wait_for(built[1].closing.wait(), _FIRST_BUILD_S)
        closed = {b.channel_id: b.closed for b in built}
    finally:
        slow.release.set()
        await kit.close()
        await executor.close()

    assert closed == {"alpha": True, "beta": True, "gamma": False, "delta": False}


def test_a_bound_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="max_channels"):
        TurnExecutor(lambda _: None, max_channels=0)

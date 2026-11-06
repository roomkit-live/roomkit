"""A declared channel's build is let go when its host says it is stale
(RFC §10.6.3, ``channel_released``).

An executor builds a declared channel once and keeps it; when its host changes
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
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.store.memory import InMemoryStore
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from tests.test_framework import SimpleChannel
from tests.test_turns_channel_control import _Agent


class _Built(_Agent):
    """The agent as one build of it: closing it is recorded."""

    def __init__(self, generation: int) -> None:
        super().__init__("agent")
        self.generation = generation
        self.closed = False

    async def close(self) -> None:
        self.closed = True
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

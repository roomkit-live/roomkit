"""A discussion served by several processes (RFC §19.7.5 rule 16).

Each "process" is a kit of its own, with its own agent objects; they share the
store and the lock manager, as processes share Postgres and its advisory
locks. One queue in the store, one process giving the turns under a lease.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import Discussion, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import TextContent
from roomkit.orchestration.strategies.discussion import _driver, _shared
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel

AGENTS = ("investigator", "sre", "comms")


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll and lease on a test's time scale."""
    monkeypatch.setattr(_driver, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(_shared, "LEASE_SECONDS", 0.6)
    monkeypatch.setattr(_shared, "_RENEW_BEFORE", 0.4)


class _Slow(MockAIProvider):
    """Answers after a while, and logs when it generates, across processes."""

    def __init__(self, name: str, log: list[str], delay: float = 0.15) -> None:
        super().__init__(ai_responses=[AIResponse(content=f"{name} answers")])
        self.who, self.log, self.delay = name, log, delay

    async def generate(self, context: Any) -> AIResponse:
        self.log.append(f"+{self.who}")
        await asyncio.sleep(self.delay)
        self.log.append(f"-{self.who}")
        return await super().generate(context)


class _Cluster:
    """Processes serving one room: one store, one lock manager."""

    def __init__(self) -> None:
        self.store = InMemoryStore()
        self.locks = InMemoryLockManager()
        self.log: list[str] = []
        self.kits: list[RoomKit] = []

    def strategy(self, delay: float = 0.15, **options: Any) -> Discussion:
        agents = [AIChannel(n, provider=_Slow(n, self.log, delay)) for n in AGENTS]
        return Discussion(agents, **options)

    async def process(self, *, install: bool, create: bool = False, **options: Any) -> RoomKit:
        kit = RoomKit(store=self.store, lock_manager=self.locks)
        kit.register_channel(SimpleChannel("ops"))
        self.kits.append(kit)
        if create:
            await kit.create_room(room_id="r1", orchestration=self.strategy(**options))
            await kit.attach_channel("r1", "ops")
        elif install:
            await self.strategy(**options).install(kit, "r1")
        return kit

    async def answers(self) -> list[str]:
        events = await self.store.list_events("r1", limit=100)
        return [
            e.source.channel_id
            for e in events
            if e.type == EventType.MESSAGE and e.source.channel_id in AGENTS
        ]

    def overlap(self) -> int:
        running = most = 0
        for line in self.log:
            running += 1 if line.startswith("+") else -1
            most = max(most, running)
        return most

    async def close(self) -> None:
        for kit in self.kits:
            await kit.close()


async def _say(kit: RoomKit, body: str, **extra: Any) -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body), **extra)
    )


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(0.02)


async def test_a_process_that_did_not_install_it_follows_the_discussion() -> None:
    cluster = _Cluster()
    await cluster.process(install=True, create=True)
    follower = await cluster.process(install=False)

    result = await _say(follower, "checkout is failing, what happened?")

    # The follower asks nobody at broadcast: no dogpile.
    assert result.response_events == []

    async def three() -> bool:
        return len(await cluster.answers()) >= 3

    await _until(three)
    await asyncio.sleep(0.3)
    assert sorted(await cluster.answers()) == sorted(AGENTS)
    assert cluster.overlap() == 1
    queue = follower.speak_queue("r1")
    assert queue is not None  # the follower knows the room holds a discussion
    await cluster.close()


async def test_two_processes_never_give_two_turns_at_once() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=True)

    await _say(a, "@investigator what do the logs say?")
    await asyncio.sleep(0.02)
    await _say(b, "@sre and the metrics?")

    async def both() -> bool:
        return sorted(await cluster.answers()) == ["investigator", "sre"]

    await _until(both)
    assert cluster.overlap() == 1
    await cluster.close()


async def test_a_process_that_stopped_hands_the_room_over_once_its_lease_expires() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=True)
    await _say(a, "@investigator hello")

    async def one() -> bool:
        return len(await cluster.answers()) == 1

    await _until(one)
    # A crashes: its driver dies without giving the lease back.
    held = a._discussions["r1"]
    assert held.driver is not None and held.driver._task is not None
    held.driver._task.cancel()
    held.closed = True

    await _say(b, "@sre are you there?")

    async def two() -> bool:
        return len(await cluster.answers()) == 2

    await _until(two)
    assert (await cluster.answers())[-1] == "sre"
    await cluster.close()


async def test_uninstalling_in_one_process_ends_the_discussion_in_the_others() -> None:
    cluster = _Cluster()
    strategy = cluster.strategy()
    a = RoomKit(store=cluster.store, lock_manager=cluster.locks)
    a.register_channel(SimpleChannel("ops"))
    cluster.kits.append(a)
    await a.create_room(room_id="r1", orchestration=strategy)
    await a.attach_channel("r1", "ops")
    follower = await cluster.process(install=False)
    await _say(follower, "@investigator hello")

    async def one() -> bool:
        return len(await cluster.answers()) == 1

    await _until(one)
    await strategy.uninstall(a, "r1")
    for agent in AGENTS:
        follower.register_channel(AIChannel(agent, provider=_Slow(agent, cluster.log, 0.0)))

    result = await _say(follower, "@sre plain room again?")

    # The follower left: no name is read any more and the room answers by its
    # policy again, the agents asked at broadcast.
    assert follower.speak_queue("r1") is None
    assert {e.source.channel_id for e in result.response_events} <= set(AGENTS)
    assert result.response_events
    await cluster.close()


async def test_an_instruction_moves_the_turns_to_the_process_holding_it() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True)
    b = await cluster.process(install=True)
    await _say(a, "@investigator hello")

    async def one() -> bool:
        return len(await cluster.answers()) == 1

    await _until(one)

    await _say(b, "summarise", addressed_to=["comms"], event_type=EventType.INSTRUCTION)

    async def two() -> bool:
        return len(await cluster.answers()) == 2

    await _until(two)
    assert (await cluster.answers())[-1] == "comms"
    queue = b.speak_queue("r1")
    state = b._discussions["r1"].state
    assert queue is not None and state.lease_holder == b._instance_id
    await cluster.close()


async def test_listen_only_in_one_process_cuts_the_turn_running_in_another() -> None:
    cluster = _Cluster()
    a = await cluster.process(install=True, create=True, delay=2.0)
    follower = await cluster.process(install=False)
    await _say(follower, "@investigator take your time")

    async def started() -> bool:
        return "+investigator" in cluster.log

    await _until(started)
    await follower.listen_only("r1", ["investigator"])

    async def ended() -> bool:
        queue = a.speak_queue("r1")
        return queue is not None and queue.speaking is None

    await _until(ended, timeout=3.0)
    assert await cluster.answers() == []
    await cluster.close()

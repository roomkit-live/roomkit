"""A discussion skips the turns of an agent no executor serves (RFC §19.7.5
rule 17, §10.6.7); the conformance scenario
``discussion/an-offline-agent-s-turn-is-skipped-not-failed`` covers the skip
and the agent's return."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import uuid4

import pytest

from roomkit import Agent, AgentIdentity, ChannelDeclaration, RoomKit
from roomkit.models.enums import EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.orchestration.strategies.discussion import Discussion, SpeakQueueChange
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import (
    InMemoryTurnBackend,
    RedisTurnBackend,
    TurnBackend,
    TurnExecutor,
    TurnTimeouts,
)
from tests.orchestration.test_discussion_edges import _People, _say
from tests.orchestration.test_strategy_discussion import _settle


class _Unlisting(InMemoryTurnBackend):
    """A turn backend that does not list its executors."""

    executors = TurnBackend.executors  # type: ignore[assignment]


class _ListFails(InMemoryTurnBackend):
    async def executors(self) -> Any:
        raise ConnectionError("registry unreachable")


async def _discussion(backend: TurnBackend) -> tuple[RoomKit, list[Any], list[str]]:
    """``ana`` held by its declaration, served by nobody; ``bob`` live."""
    kit = RoomKit(turn_backend=backend)
    changes: list[Any] = []
    errors: list[str] = []

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def _queue(event: Any, ctx: Any) -> None:
        changes.append((event.change, event.channel_ids))

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def _error(event: RoomEvent, ctx: Any) -> None:
        errors.append(event.source.channel_id)

    kit.register_channel(_People("chat"))
    kit.register_channel(ChannelDeclaration(channel_id="ana", identity=AgentIdentity()))
    bob = Agent("bob", provider=MockAIProvider(responses=["bob here."] * 4))
    await kit.create_room(room_id="r1", orchestration=Discussion(agents=["ana", bob]))
    await kit.attach_channel("r1", "chat")
    return kit, changes, errors


async def _until(predicate: Any) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("backend", [_Unlisting, _ListFails], ids=["unlisting", "list-fails"])
async def test_an_agent_the_kit_cannot_tell_is_offline_still_takes_its_turn(
    backend: type[InMemoryTurnBackend],
) -> None:
    kit, changes, errors = await _discussion(backend())
    try:
        await _say(kit, "Hello everyone.")
        await _settle(kit)
        await _until(lambda: errors and (SpeakQueueChange.TURN_ENDED, ("bob",)) in changes)
    finally:
        await kit.close()

    assert (SpeakQueueChange.TURN_GIVEN, ("ana",)) in changes
    assert all(change != SpeakQueueChange.TURN_SKIPPED for change, _ in changes)
    assert errors == ["ana"]  # its turn failed: no executor serves it


async def test_an_instruction_for_an_offline_agent_is_dropped_and_said_so() -> None:
    kit, changes, errors = await _discussion(InMemoryTurnBackend())
    try:
        await _say(kit, "summarise", addressed_to=["ana"], event_type=EventType.INSTRUCTION)
        await _settle(kit)
        await _until(lambda: (SpeakQueueChange.INSTRUCTION_DROPPED, ("ana",)) in changes)
        held = dict(kit._discussions["r1"].instructions)
    finally:
        await kit.close()

    told = [change for change, _ in changes]
    assert told[told.index(SpeakQueueChange.QUEUED) + 1 :] == [
        SpeakQueueChange.TURN_SKIPPED,
        SpeakQueueChange.INSTRUCTION_DROPPED,
    ]
    assert held == {}
    assert errors == []


@pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="REDIS_URL not set")
async def test_an_executor_started_since_the_owner_last_read_takes_the_turn() -> None:
    """Across Redis the owner reads the executors once per heartbeat interval:
    an agent whose executor started since then reads offline in that view, and
    is checked against the registry read now before its turn is dropped."""
    redis = pytest.importorskip("redis.asyncio")
    prefix = f"roomkit:test:{uuid4().hex[:12]}"
    timeouts = TurnTimeouts(heartbeat_interval=5.0, heartbeat_timeout=15.0, accept=3.0)
    owner, elsewhere = (
        RedisTurnBackend(os.environ["REDIS_URL"], key_prefix=prefix, timeouts=timeouts)
        for _ in range(2)
    )
    host = TurnExecutor(
        lambda cid: Agent(cid, provider=MockAIProvider(responses=["ana here."])),
        hosts={"ana"},
        executor_id="ana-host",
        timeouts=timeouts,
    )
    kit, changes, errors = await _discussion(owner)
    client = redis.from_url(os.environ["REDIS_URL"])
    try:
        assert [p.online for p in await kit.agent_presence("r1") if p.channel_id == "ana"] == [
            False
        ]
        await elsewhere.serve(host)
        await _say(kit, "are you there?", addressed_to=["ana"])
        await _until(lambda: any(c[1] == ("ana",) and c[0] in _ENDS for c in changes))
    finally:
        await kit.close()
        await host.close()
        await elsewhere.close()
        keys = [key async for key in client.scan_iter(f"{prefix}:*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()

    assert (SpeakQueueChange.TURN_GIVEN, ("ana",)) in changes
    assert (SpeakQueueChange.TURN_SKIPPED, ("ana",)) not in changes
    assert errors == []


_ENDS = {SpeakQueueChange.TURN_ENDED, SpeakQueueChange.TURN_SKIPPED}

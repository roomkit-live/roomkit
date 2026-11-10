"""A discussion skips the turns of an agent no executor serves (RFC §19.7.5
rule 17, §10.6.7); the conformance scenario
``discussion/an-offline-agent-s-turn-is-skipped-not-failed`` covers the skip
and the agent's return."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import Agent, AgentIdentity, ChannelDeclaration, RoomKit
from roomkit.models.enums import EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.orchestration.strategies.discussion import Discussion, SpeakQueueChange
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import InMemoryTurnBackend, TurnBackend
from tests.orchestration.test_discussion_edges import _People, _say
from tests.orchestration.test_strategy_discussion import _settle


class _Unlisting(InMemoryTurnBackend):
    """A turn backend that does not list its executors."""

    executors = TurnBackend.executors  # type: ignore[assignment]


class _ListFails(InMemoryTurnBackend):
    async def executors(self) -> Any:
        raise ConnectionError("registry unreachable")


async def _discussion(backend: InMemoryTurnBackend) -> tuple[RoomKit, list[Any], list[str]]:
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

    assert (SpeakQueueChange.TURN_SKIPPED, ("ana",)) in changes
    assert (SpeakQueueChange.TURN_GIVEN, ("ana",)) not in changes
    assert held == {}
    assert errors == []

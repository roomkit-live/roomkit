"""A discussion hands each committed message to its agents' memories through
the turn backend, where each agent runs (RFC §10.6.3, §19.7.5 rule 3)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from roomkit import Agent, AgentIdentity, ChannelDeclaration, Discussion, RoomKit
from roomkit.memory.mock import MockMemoryProvider
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking.mock import MockSpeakPolicy
from roomkit.store.memory import InMemoryStore
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from tests.test_framework import SimpleChannel


def _agent(channel_id: str, memory: MockMemoryProvider, **kwargs: object) -> Agent:
    provider = MockAIProvider(responses=[f"{channel_id} here."] * 4)
    return Agent(channel_id, provider=provider, memory=memory, **kwargs)  # type: ignore[arg-type]


def _ingested(memory: MockMemoryProvider) -> list[str]:
    return [
        call.event.content.body
        for call in memory.ingest_calls
        if isinstance(call.event.content, TextContent)
    ]


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


def _idle(kit: RoomKit) -> bool:
    """The discussion runs no turn and has none to give: every message it was
    going to hand on is committed, its hand-offs started."""
    queue = kit.speak_queue("r")
    return queue is not None and queue.speaking is None and set(queue.queue) <= queue.listening


async def _settled(kit: RoomKit, handed: Callable[[], bool]) -> None:
    """Until *handed* holds and the discussion is idle, then a moment for a
    hand-off the test says never comes (a second one) to have landed."""
    await _until(lambda: handed() and _idle(kit))
    await asyncio.sleep(0.02)


async def _say(kit: RoomKit, text: str) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="marie", content=TextContent(body=text)),
        room_id="r",
    )


@pytest.mark.parametrize("organization", [None, "acme"], ids=["no-organization", "acme"])
async def test_a_declared_agent_s_memory_is_handed_each_message_where_it_runs(
    organization: str | None,
) -> None:
    """In a room of an organization, each message goes to the executor serving
    it, as the room's turns do (RFC §10.6.7)."""
    memories = {"ana": MockMemoryProvider(), "bob": MockMemoryProvider()}
    agents = {cid: _agent(cid, memory) for cid, memory in memories.items()}
    store, backend = InMemoryStore(), InMemoryTurnBackend()
    executor = TurnExecutor(
        agents.get,
        hosts=lambda cid: cid in agents,
        organizations=None if organization is None else {organization},
    )
    await backend.serve(executor)
    kit = RoomKit(store=store, turn_backend=backend)
    kit.register_channel(SimpleChannel("sms"))
    for cid in agents:
        kit.register_channel(ChannelDeclaration(channel_id=cid, identity=AgentIdentity()))
    await kit.create_room(
        room_id="r", organization_id=organization, orchestration=Discussion(agents=["ana", "bob"])
    )
    await kit.attach_channel("r", "sms")

    await _say(kit, "Hello everyone.")
    await _settled(
        kit,
        lambda: (
            _ingested(memories["ana"]).count("bob here.") == 1
            and _ingested(memories["bob"]).count("ana here.") == 1
        ),
    )
    handed = {cid: _ingested(memory) for cid, memory in memories.items()}
    await kit.close()
    await executor.close()

    for cid, said in handed.items():
        other = "bob" if cid == "ana" else "ana"
        assert said.count("Hello everyone.") == 1
        assert f"{cid} here." not in said  # its own answer is never handed to it
        assert said.count(f"{other} here.") == 1


async def test_a_silent_discussion_turn_does_not_hand_the_message_again() -> None:
    quiet = MockMemoryProvider()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    ana = _agent("ana", quiet, speak_policy=MockSpeakPolicy(["silent"]))
    bob = _agent("bob", MockMemoryProvider())
    await kit.create_room(room_id="r", orchestration=Discussion(agents=[ana, bob]))
    await kit.attach_channel("r", "sms")

    await _say(kit, "@ana are you there?")
    await _settled(kit, lambda: bool(_ingested(quiet)))
    said = _ingested(quiet)
    await kit.close()

    assert said.count("@ana are you there?") == 1


class _MeetingMemory(MockMemoryProvider):
    """A memory whose ingest learns only once every agent's memory is
    learning too: handed one after the other, they would wait for good."""

    def __init__(self, meeting: asyncio.Barrier) -> None:
        super().__init__()
        self._meeting = meeting

    async def ingest(
        self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
    ) -> None:
        await self._meeting.wait()
        await super().ingest(room_id, event, channel_id=channel_id)


async def test_a_message_is_handed_to_every_agent_s_memory_at_once() -> None:
    """The message waits for the slowest memory, not for the sum of them."""
    meeting = asyncio.Barrier(2)
    memories = {cid: _MeetingMemory(meeting) for cid in ("ana", "bob")}
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    agents = [_agent(cid, memory) for cid, memory in memories.items()]
    await kit.create_room(room_id="r", orchestration=Discussion(agents=agents))
    await kit.attach_channel("r", "sms")
    said = asyncio.ensure_future(_say(kit, "@nobody hello."))
    try:
        await asyncio.wait({said}, timeout=2)
        handed = said.done()
    finally:
        await meeting.abort()
        await asyncio.wait({said})
        await kit.close()

    assert handed, "each memory waited for the one before"
    assert all(_ingested(memory) == ["@nobody hello."] for memory in memories.values())


async def test_an_agent_no_executor_runs_misses_its_memory_and_the_room_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An agent offline is handed nothing: no executor would take it."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    bob_memory = MockMemoryProvider()
    kit.register_channel(ChannelDeclaration(channel_id="ana"))
    bob = _agent("bob", bob_memory)
    await kit.create_room(room_id="r", orchestration=Discussion(agents=["ana", bob]))
    await kit.attach_channel("r", "sms")

    await asyncio.wait_for(_say(kit, "Hello."), 2)
    await _settled(kit, lambda: bool(_ingested(bob_memory)))
    said = _ingested(bob_memory)
    await kit.close()

    assert said.count("Hello.") == 1
    assert not [r for r in caplog.records if "memory was not handed" in r.getMessage()]

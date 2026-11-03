"""A strategy that takes an agent's turns runs on the room's owner, and each
generation it asks of the agent is a strategy turn through the turn backend
(RFC §10.6.1, §19.7.3)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels._strategy_turn import in_strategy_turn
from roomkit.channels.agent import Agent
from roomkit.channels.websocket import WebSocketChannel
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    DeleteType,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import (
    DeleteContent,
    EditContent,
    EventContent,
    EventSource,
    RoomEvent,
    TextContent,
)
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.orchestration import Loop, Orchestration
from roomkit.orchestration.strategies.supervisor import Supervisor, _install_auto
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

PASS1 = "Formulate the task for the workers."


class ObservingAgent(Agent):
    """Wraps ``on_event`` as the subclassing docs show: every turn the
    channel runs, with the room it runs for and whether a strategy asked it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.turns: list[tuple[str, bool, str]] = []

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        task = asyncio.current_task()
        room_id = context.room.id if context.room else event.room_id
        self.turns.append((room_id, in_strategy_turn(), task.get_name() if task else ""))
        return await super().on_event(event, binding, context)


class CountingMemory(SlidingWindowMemory):
    """Remembers who each event it was handed came from."""

    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    async def ingest(
        self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
    ) -> None:
        self.sources.append(event.source.channel_id)


def _agent(name: str, *answers: str, cls: type[Agent] = Agent, **kwargs: Any) -> Any:
    responses = [AIResponse(content=a) for a in answers] or [AIResponse(content=f"{name} here")]
    return cls(name, provider=MockAIProvider(ai_responses=responses), role=name, **kwargs)


def _supervisor(boss: Agent, worker: Agent, *, refine_task: bool = True) -> Supervisor:
    return Supervisor(
        supervisor=boss,
        workers=[worker],
        strategy="parallel",
        auto_delegate=True,
        refine_task=refine_task,
        refine_instruction=PASS1,
    )


async def _room(kit: RoomKit, *agents: Agent) -> None:
    kit.register_channel(SimpleChannel("ops"))
    for agent in agents:
        kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")


async def _say(kit: RoomKit, body: str) -> InboundResult:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body)),
        room_id="r1",
    )


def _answers(result: InboundResult) -> list[str]:
    return [e.content.body for e in result.response_events if isinstance(e.content, TextContent)]


async def test_a_supervisors_passes_are_strategy_turns_its_on_event_sees() -> None:
    boss = _agent(
        "boss",
        "Look into it.",
        "Here is what we found.",
        cls=ObservingAgent,
        system_prompt="You are Boss.",
    )
    worker = _agent("worker", "worker result")
    kit = RoomKit()
    await _room(kit, boss, worker)
    await kit.install_strategy("r1", _supervisor(boss, worker))

    result = await _say(kit, "Why?")

    assert _answers(result) == ["Here is what we found."]
    # Two generations in the room (the task-formulation pass, the
    # presentation), each a strategy turn an executor runs, both entering
    # through the channel's on_event.
    in_room = [turn for turn in boss.turns if turn[0] == "r1"]
    assert len(in_room) == 2
    assert all(strategy and task.startswith("roomkit-turn-") for _, strategy, task in in_room)
    # The pass-1 instruction follows the prompt the turn resolved, and the
    # presentation does not read it.
    prompts = [call.system_prompt or "" for call in boss.provider.calls]
    assert len(prompts) == 2
    assert prompts[0].startswith("You are Boss.")
    assert PASS1 in prompts[0]
    assert PASS1 not in prompts[1]
    await kit.close()


async def test_a_supervisors_runner_runs_on_the_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    boss = _agent("boss", "Look into it.", "Presented.")
    worker = _agent("worker", "worker result")
    kit = RoomKit()
    await _room(kit, boss, worker)
    tasks: list[str] = []
    run = _install_auto._DelegatingTurns.run

    async def spy(self: Any, *args: Any) -> Any:
        task = asyncio.current_task()
        tasks.append(task.get_name() if task else "")
        return await run(self, *args)

    monkeypatch.setattr(_install_auto._DelegatingTurns, "run", spy)
    await kit.install_strategy("r1", _supervisor(boss, worker))

    await _say(kit, "Why?")

    # The runner took the turn on the owner's delivery, not in a turn an
    # executor runs.
    assert len(tasks) == 1
    assert not tasks[0].startswith("roomkit-turn-")
    await kit.close()


async def test_an_injected_event_never_runs_the_strategy() -> None:
    boss = _agent("boss", "Look into it.", "Presented.", "unread")
    worker = _agent("worker", "worker result", "never asked")
    kit = RoomKit()
    await _room(kit, boss, worker)
    await kit.install_strategy("r1", _supervisor(boss, worker))

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.SYNC)
    async def inject(event: RoomEvent, ctx: Any) -> HookResult:
        if event.source.channel_id != "ops":
            return HookResult.allow()
        note = RoomEvent(
            room_id="r1",
            type=EventType.MESSAGE,
            source=EventSource(channel_id="ops", channel_type=ChannelType.WEBSOCKET),
            content=TextContent(body="injected note"),
        )
        return HookResult.allow([InjectedEvent(event=note, target_channel_ids=["boss"])])

    await _say(kit, "Why?")

    # The delegation ran once, for the message; the injected note is
    # delivered to the supervisor and never answered, so no pass runs for it.
    assert len(worker.provider.calls) == 1
    assert len(boss.provider.calls) == 2
    await kit.close()


@pytest.mark.parametrize("refine_task", [True, False], ids=["two-pass", "one-pass"])
async def test_the_workers_results_reach_the_supervisors_memory_once(refine_task: bool) -> None:
    memory = CountingMemory()
    answers = ("Look into it.", "Presented.") if refine_task else ("Presented.",)
    boss = _agent("boss", *answers, memory=memory)
    worker = _agent("worker", "worker result")
    kit = RoomKit()
    await _room(kit, boss, worker)
    await kit.install_strategy("r1", _supervisor(boss, worker, refine_task=refine_task))

    await _say(kit, "Why?")

    # The results stand in for the event the supervisor answers; the turn
    # that presents them hands them to memory, nothing else does.
    assert memory.sources.count("system") == 1
    await kit.close()


def _strategy(name: str, lead: Agent, helper: Agent) -> Orchestration:
    if name == "loop":
        return Loop(agent=lead, reviewers=[helper], max_iterations=1)
    return _supervisor(lead, helper, refine_task=name == "two-pass")


def _change(kind: EventType, target: RoomEvent) -> EventContent:
    if kind == EventType.EDIT:
        return EditContent(target_event_id=target.id, new_content=TextContent(body="Why not?"))
    if kind == EventType.DELETE:
        return DeleteContent(target_event_id=target.id, delete_type=DeleteType.SENDER)
    return TextContent(body="The order shipped")


async def _from_ws(kit: RoomKit, kind: EventType, content: EventContent) -> InboundResult:
    return await kit.process_inbound(
        InboundMessage(channel_id="ws", sender_id="alice", event_type=kind, content=content),
        room_id="r1",
    )


@pytest.mark.parametrize(
    "kind", [EventType.EDIT, EventType.DELETE, EventType.SYSTEM], ids=["edit", "delete", "system"]
)
@pytest.mark.parametrize("name", ["loop", "one-pass", "two-pass"])
async def test_a_strategy_takes_no_turn_on_what_opens_none(name: str, kind: EventType) -> None:
    """A strategy that takes the agent's turns asks none on a change (RFC
    §10.3, RMK-1226) nor on an event that is no message nor instruction
    (RFC §6.4, RMK-1257), as the agent's own channel answers none."""
    memory = CountingMemory()
    lead = _agent("lead", *["lead answer"] * 4, memory=memory)
    helper = _agent("helper", *["APPROVED"] * 4)
    kit = RoomKit()
    for channel in (WebSocketChannel("ws"), lead, helper):
        kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ws")
    await kit.install_strategy("r1", _strategy(name, lead, helper))
    said = (await _from_ws(kit, EventType.MESSAGE, TextContent(body="Why?"))).event
    assert said is not None
    asked = (len(lead.provider.calls), len(helper.provider.calls))
    ingested = len(memory.sources)

    result = await _from_ws(kit, kind, _change(kind, said))

    assert not result.blocked
    assert (len(lead.provider.calls), len(helper.provider.calls)) == asked
    assert result.response_events == []
    # The lead's memory still learns it.
    assert memory.sources[ingested:] == ["ws"]
    await kit.close()


async def test_an_agent_with_no_strategy_answers_as_itself() -> None:
    agent = _agent("solo", "hello", cls=ObservingAgent)
    kit = RoomKit()
    await _room(kit, agent)
    await kit.attach_channel("r1", "solo", category=ChannelCategory.INTELLIGENCE)
    assert kit._turn_runners.get("r1", "solo") is None

    result = await _say(kit, "hi")

    assert _answers(result) == ["hello"]
    assert [strategy for _, strategy, _ in agent.turns] == [False]
    await kit.close()

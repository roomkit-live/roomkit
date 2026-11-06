"""Code on a turn's executor reaches a room through the turn's host (RFC §10.6.5).

A host's tool reads ``current_turn_host()``; each verb crosses the turn
backend and runs on the room owner, in a task of its own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import ValidationError

from roomkit import RoomKit, TurnHost, current_turn_host
from roomkit.channels.agent import Agent
from roomkit.core._turn_host import KitHost
from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.task import Observation
from roomkit.models.tool_call import ToolCallEvent
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking.always import AlwaysSpeak
from roomkit.tools.context import (
    _current_turn_chain_depth,
    current_tool_actor_id,
    current_tool_call,
    current_tool_room_id,
)
from tests.test_framework import SimpleChannel

USE = AITool(name="use_host", description="Use the host.", parameters={"type": "object"})

HostUse = Callable[[TurnHost], Awaitable[Any]]


def _call(call_id: str = "c1") -> AIResponse:
    return AIResponse(
        content="", tool_calls=[AIToolCall(id=call_id, name="use_host", arguments={})]
    )


async def _kit(
    use: HostUse, *more: Agent, then: tuple[str, ...] = ("done",)
) -> tuple[RoomKit, Agent, dict[str, Any]]:
    """A kit whose agent's one host tool runs *use* with the turn's host,
    its answers after the call *then*."""
    seen: dict[str, Any] = {}

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        host = current_turn_host()
        task = asyncio.current_task()
        seen["host"] = host
        seen["tool_task"] = task.get_name() if task else ""
        assert host is not None
        seen["result"] = await use(host)
        return "used"

    provider = MockAIProvider(ai_responses=[_call(), *(AIResponse(content=a) for a in then)])
    agent = Agent("agent", provider=provider, role="agent", tools=[USE], tool_handler=handler)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    for other in more:
        kit.register_channel(other)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, agent, seen


async def _say(kit: RoomKit, body: str = "go") -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body)),
        room_id="r1",
    )


@pytest.fixture
def verbs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Each verb the owner ran: its name, the task that ran it."""
    seen: list[tuple[str, str]] = []
    call = KitHost.call

    async def spy(self: KitHost, request: Any, verb: str, arguments: dict[str, Any]) -> Any:
        task = asyncio.current_task()
        seen.append((verb, task.get_name() if task else ""))
        return await call(self, request, verb, arguments)

    monkeypatch.setattr(KitHost, "call", spy)
    return seen


def _on_owner(verbs: list[tuple[str, str]], verb: str) -> bool:
    return any(v == verb and task.startswith("roomkit-owner-call-") for v, task in verbs)


# -- Where the host is ----------------------------------------------------------


async def test_a_host_tool_reads_the_turns_host_and_it_ends_with_the_turn() -> None:
    async def use(host: TurnHost) -> Any:
        return host.open

    kit, _, seen = await _kit(use)
    assert current_turn_host() is None

    await _say(kit)

    host = seen["host"]
    assert seen["result"] is True
    assert not host.open
    # A verb that needs its turn is refused after it (RFC §10.6.5).
    with pytest.raises(RuntimeError, match="needs a turn"):
        await host.room_tasks()
    await kit.close()


async def test_a_hook_the_executor_fires_during_a_turn_reads_its_host() -> None:
    async def use(host: TurnHost) -> Any:
        return None

    kit, _, _ = await _kit(use)
    hosts: list[TurnHost | None] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
    async def report(event: ToolCallEvent, ctx: Any) -> None:
        hosts.append(current_turn_host())

    await _say(kit)

    assert len(hosts) == 1
    assert hosts[0] is not None
    await kit.close()


# -- The verbs --------------------------------------------------------------------


async def test_send_event_runs_on_the_owner_and_the_room_moves_on(
    verbs: list[tuple[str, str]],
) -> None:
    async def use(host: TurnHost) -> Any:
        return await host.send_event(
            "r1", "ops", TextContent(body="from the tool"), addressed_to=[]
        )

    kit, _, seen = await _kit(use, then=("done", "again"))

    await asyncio.wait_for(_say(kit), 5)

    event = seen["result"]
    assert event.content.body == "from the tool"
    assert await kit.store.get_event(event.id) is not None
    assert _on_owner(verbs, "send_event")
    # The room's lane went on: a later message is answered.
    result = await asyncio.wait_for(_say(kit, "next"), 5)
    assert [e.content.body for e in result.response_events] == ["again"]
    await kit.close()


async def test_delegate_answers_a_task_id_that_wait_task_waits(
    verbs: list[tuple[str, str]],
) -> None:
    async def use(host: TurnHost) -> Any:
        task_id = await host.delegate("r1", "worker", "Look it up")
        return task_id, await host.wait_task(task_id, timeout=5)

    worker = Agent(
        "worker",
        provider=MockAIProvider(ai_responses=[AIResponse(content="found")]),
        role="worker",
    )
    kit, _, seen = await _kit(use, worker)

    await asyncio.wait_for(_say(kit), 10)

    task_id, result = seen["result"]
    assert isinstance(task_id, str)
    assert result is not None
    assert result.task_id == task_id
    assert result.output == "found"
    assert _on_owner(verbs, "delegate")
    assert _on_owner(verbs, "wait_task")
    await kit.close()


async def test_wait_task_answers_none_for_a_task_it_does_not_know() -> None:
    async def use(host: TurnHost) -> Any:
        return await host.wait_task("no-such-task", timeout=0.1)

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert seen["result"] is None
    await kit.close()


async def test_room_metadata_status_and_observations_are_written_on_the_owner(
    verbs: list[tuple[str, str]],
) -> None:
    async def use(host: TurnHost) -> Any:
        room = await host.update_room_metadata("r1", {"stage": "billing"})
        await host.post_status("agent", "lookup", "info", detail="looked")
        await host.add_observation(
            Observation(id="o1", room_id="r1", channel_id="agent", content="noted")
        )
        return room

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert seen["result"].metadata["stage"] == "billing"
    assert (await kit.get_room("r1")).metadata["stage"] == "billing"
    [entry] = await kit.status_bus.recent(1, agent_id="agent")
    assert entry.action == "lookup"
    assert [o.content for o in await kit.store.list_observations("r1")] == ["noted"]
    for verb in ("update_room_metadata", "post_status", "add_observation"):
        assert _on_owner(verbs, verb)
    await kit.close()


async def test_a_status_post_names_its_turn_s_room() -> None:
    async def use(host: TurnHost) -> Any:
        await host.post_status("agent", "lookup", "info")
        try:
            await host.post_status("agent", "spoof", "info", metadata={"room_id": "elsewhere"})
        except ValueError as exc:
            return exc
        return None

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert isinstance(seen["result"], ValueError)
    [entry] = await kit.status_bus.recent(1, agent_id="agent")
    assert (entry.action, (entry.metadata or {}).get("room_id")) == ("lookup", "r1")
    await kit.close()


async def test_a_turn_s_host_posts_no_worker_run_entry() -> None:
    """A worker run's terminal entry ends the line of the task it names,
    whatever its action: a turn forging one would end a running task for
    task_status and cancel_task (RMK-875 review)."""

    async def use(host: TurnHost) -> Any:
        try:
            await host.post_status(
                "agent", "iteration", "failed", metadata={"task_id": "t1", "task_run": True}
            )
        except ValueError as exc:
            return exc
        return None

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert isinstance(seen["result"], ValueError)
    assert await kit.status_bus.recent(5, agent_id="agent") == []
    await kit.close()


async def test_a_turn_s_host_posts_no_task_status() -> None:
    """The room's task tools read the bus's task statuses to know which tasks
    are the room's: a forged one would let cancel_task reach another room's."""

    async def use(host: TurnHost) -> Any:
        try:
            await host.post_status(
                "agent", "task", "pending", metadata={"task_id": "someone-else-s"}
            )
        except ValueError as exc:
            return exc
        return None

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert isinstance(seen["result"], ValueError)
    assert await kit.status_bus.recent(5, agent_id="agent") == []
    await kit.close()


async def test_a_verb_that_raises_raises_in_the_tool_with_its_class() -> None:
    async def use(host: TurnHost) -> Any:
        try:
            await host.delegate("r1", "nobody", "Look it up")
        except ChannelNotRegisteredError as exc:
            return exc
        return None

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert isinstance(seen["result"], ChannelNotRegisteredError)
    await kit.close()


@pytest.mark.parametrize(
    "verb",
    [
        lambda host: host.send_event("r2", "ops", TextContent(body="elsewhere")),
        lambda host: host.delegate("r2", "worker", "Look"),
        lambda host: host.update_room_metadata("r2", {"x": 1}),
        lambda host: host.update_room_metadata("r1", {"x": 1}, organization_id="other"),
        lambda host: host.add_observation(
            Observation(id="o1", room_id="r2", channel_id="agent", content="noted")
        ),
    ],
    ids=["send_event", "delegate", "metadata", "organization", "observation"],
)
async def test_a_host_acts_only_for_its_turns_room_and_organization(verb: Any) -> None:
    async def use(host: TurnHost) -> Any:
        try:
            await verb(host)
        except ValueError as exc:
            return exc
        return None

    kit, _, seen = await _kit(use)
    await kit.create_room(room_id="r2")

    await _say(kit)

    assert isinstance(seen["result"], ValueError)
    assert (await kit.get_room("r2")).metadata == {}
    await kit.close()


async def test_an_option_a_verb_does_not_name_is_refused() -> None:
    async def use(host: TurnHost) -> Any:
        try:
            await host.send_event("r1", "ops", TextContent(body="x"), addresed_to=[])
        except ValidationError as exc:
            return exc
        return None

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert isinstance(seen["result"], ValidationError)
    await kit.close()


async def test_a_delegation_through_the_host_keeps_its_turns_chain_depth() -> None:
    depths: list[int] = []
    kit_holder: list[RoomKit] = []

    async def use(host: TurnHost) -> Any:
        task_id = await host.delegate("r1", "worker", "Look it up")
        return await host.wait_task(task_id, timeout=5)

    worker = Agent(
        "worker",
        provider=MockAIProvider(ai_responses=[AIResponse(content="found")]),
        role="worker",
    )
    kit, _, seen = await _kit(use, worker)
    kit_holder.append(kit)
    run_background = type(kit)._run_background

    async def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        depths.append(_current_turn_chain_depth())
        return await run_background(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(kit), "_run_background", spy)
        await asyncio.wait_for(_say(kit), 10)

    # The agent answers a human message: its turn's response is at depth 1,
    # and so is what is delivered later on its behalf.
    assert depths == [1]
    assert seen["result"] is not None
    await kit.close()


async def _hook_kit(*more: Agent) -> RoomKit:
    """A kit whose agent reports its speak decisions: a hook its executor
    fires during the turn, before the tool loop sets the turn's depth."""
    agent = Agent(
        "agent",
        provider=MockAIProvider(ai_responses=[AIResponse(content="hi")]),
        role="agent",
        speak_policy=AlwaysSpeak(),
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    for other in more:
        kit.register_channel(other)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit


async def test_an_event_a_hook_sends_before_the_tool_loop_is_as_deep_as_the_response() -> None:
    kit = await _hook_kit()
    sent: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def note(event: Any, ctx: Any) -> None:
        host = current_turn_host()
        assert host is not None and host.open
        event = await host.send_event("r1", "ops", TextContent(body="noted"), addressed_to=[])
        sent.set_result(event)

    # The agent answers an event at depth 3: its response is at depth 4.
    await asyncio.wait_for(kit.send_event("r1", "ops", TextContent(body="go"), chain_depth=3), 10)

    assert (await asyncio.wait_for(sent, 5)).chain_depth == 4
    await kit.close()


async def test_a_delegation_a_hook_makes_before_the_tool_loop_keeps_the_turns_depth() -> None:
    worker = Agent(
        "worker", provider=MockAIProvider(ai_responses=[AIResponse(content="found")]), role="w"
    )
    kit = await _hook_kit(worker)
    depths: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    run_background = type(kit)._run_background

    async def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not depths.done():
            depths.set_result(_current_turn_chain_depth())
        return await run_background(self, *args, **kwargs)

    @kit.hook(HookTrigger.ON_SPEAK_DECISION, execution=HookExecution.ASYNC)
    async def note(event: Any, ctx: Any) -> None:
        host = current_turn_host()
        assert host is not None and host.open
        await host.delegate("r1", "worker", "Look it up")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(kit), "_run_background", spy)
        await asyncio.wait_for(
            kit.send_event("r1", "ops", TextContent(body="go"), chain_depth=3), 10
        )
        assert await asyncio.wait_for(depths, 5) == 4
    await kit.close()


async def test_wait_task_answers_none_past_its_timeout_and_the_result_after() -> None:
    release = asyncio.Event()

    async def slow(event: Any, binding: Any, context: Any) -> Any:
        await release.wait()
        return await original(event, binding, context)

    async def use(host: TurnHost) -> Any:
        task_id = await host.delegate("r1", "worker", "Look it up")
        early = await host.wait_task(task_id, timeout=0.1)
        release.set()
        late = await host.wait_task(task_id, timeout=5)
        again = await host.wait_task(task_id, timeout=5)
        return early, late, again

    worker = Agent(
        "worker",
        provider=MockAIProvider(ai_responses=[AIResponse(content="found")]),
        role="worker",
    )
    original = worker.on_event
    worker.on_event = slow  # type: ignore[method-assign]
    kit, _, seen = await _kit(use, worker)

    await asyncio.wait_for(_say(kit), 10)

    early, late, again = seen["result"]
    assert early is None
    assert late is not None and late.output == "found"
    # A result stays readable.
    assert again is not None and again.task_id == late.task_id
    await kit.close()


async def test_a_verb_on_its_way_when_the_turn_ends_is_answered_first() -> None:
    spawned: list[asyncio.Task[Any]] = []

    async def use(host: TurnHost) -> Any:
        observation = Observation(id="o1", room_id="r1", channel_id="agent", content="audited")
        # A task of the tool's own, whose verb is still on its way when the
        # turn has nothing left to do.
        spawned.append(asyncio.create_task(host.add_observation(observation)))
        return None

    kit, _, _ = await _kit(use)
    add_observation = kit.store.add_observation

    async def slow_write(observation: Observation) -> None:
        await asyncio.sleep(0.05)  # longer than what is left of the turn: a few ms
        await add_observation(observation)

    kit.store.add_observation = slow_write  # type: ignore[method-assign]

    await _say(kit)
    await asyncio.wait_for(spawned[0], 5)

    assert [o.content for o in await kit.store.list_observations("r1")] == ["audited"]
    await kit.close()


async def test_the_owner_refuses_a_verb_or_arguments_it_does_not_know() -> None:
    kit = RoomKit()
    host = KitHost(kit)
    request: Any = None

    with pytest.raises(ValueError, match="No verb"):
        await host.call(request, "close_kit", {})
    with pytest.raises(ValueError, match="not valid"):
        await host.call(request, "send_event", {"room_id": "r1"})
    await kit.close()


async def test_send_event_from_a_turn_the_lane_itself_reads_returns_detached() -> None:
    """A supervisor's first pass is read inside the room's lane, by its
    runner: an event its host tool sends into the room comes back detached,
    not awaited on the delivery the lane is busy with (RFC §10.1 step 18)."""
    sent: list[Any] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        host = current_turn_host()
        assert host is not None
        # Addressed to no agent: its delivery set, queued behind the one the
        # lane runs, holds no turn. Awaited, it would wait for the lane.
        sent.append(
            await host.send_event("r1", "ops", TextContent(body="posted"), addressed_to=[])
        )
        return "posted"

    boss = Agent(
        "boss",
        provider=MockAIProvider(
            ai_responses=[
                _call(),
                AIResponse(content="Look it up."),
                AIResponse(content="Presented."),
            ]
        ),
        role="boss",
        tools=[USE],
        tool_handler=handler,
    )
    worker = Agent(
        "worker", provider=MockAIProvider(ai_responses=[AIResponse(content="found")]), role="w"
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(boss)
    kit.register_channel(worker)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.install_strategy(
        "r1",
        Supervisor(
            supervisor=boss,
            workers=[worker],
            strategy="parallel",
            auto_delegate=True,
            refine_task=True,
        ),
    )

    result = await asyncio.wait_for(_say(kit, "Why?"), 10)

    assert [e.content.body for e in sent] == ["posted"]
    texts = [e.content.body for e in result.response_events if isinstance(e.content, TextContent)]
    assert texts == ["Presented."]
    await kit.close()


# -- The verbs a host declares ----------------------------------------------------


async def test_a_declared_verb_runs_on_the_owner_under_what_the_turn_gave_the_call(
    verbs: list[tuple[str, str]],
) -> None:
    """A host's code that needs the owner's process reaches it by name: the
    owner runs it with the turn's room, actor and tool call, its arguments
    and its result data (RFC §10.6.5)."""
    on_executor: dict[str, Any] = {}

    async def use(host: TurnHost) -> Any:
        on_executor["actor"] = current_tool_actor_id()
        on_executor["call"] = current_tool_call()
        return await host.call("app.where", {"page": "/boards"})

    async def where(arguments: dict[str, Any]) -> dict[str, Any]:
        call = current_tool_call()
        return {
            "room": current_tool_room_id(),
            "actor": current_tool_actor_id(),
            "tool_call_id": call.tool_call_id if call else None,
            "arguments": arguments,
        }

    kit, _, seen = await _kit(use)
    kit.declare_host_verb("app.where", where)

    await kit.send_event(
        room_id="r1", channel_id="ops", content=TextContent(body="go"), participant_id="alice"
    )

    assert on_executor["actor"] == "alice"
    assert seen["result"] == {
        "room": "r1",
        "actor": "alice",
        "tool_call_id": on_executor["call"].tool_call_id,
        "arguments": {"page": "/boards"},
    }
    assert _on_owner(verbs, "app.where")
    await kit.close()


async def test_a_declared_verb_that_raises_raises_in_the_tool_with_its_class() -> None:
    async def use(host: TurnHost) -> Any:
        try:
            await host.call("app.refuse")
        except PermissionError as exc:
            return str(exc)
        return None

    async def refuse(arguments: dict[str, Any]) -> None:
        raise PermissionError("not this member")

    kit, _, seen = await _kit(use)
    kit.declare_host_verb("app.refuse", refuse)

    await _say(kit)

    assert seen["result"] == "not this member"
    await kit.close()


async def test_a_verb_no_host_declared_is_refused_by_the_owner() -> None:
    async def use(host: TurnHost) -> Any:
        try:
            await host.call("app.missing", {})
        except ValueError as exc:
            return str(exc)
        return None

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert "No verb 'app.missing'" in seen["result"]
    await kit.close()


async def test_a_declared_verb_needs_its_turn() -> None:
    ran: list[dict[str, Any]] = []

    async def use(host: TurnHost) -> Any:
        return None

    async def record(arguments: dict[str, Any]) -> None:
        ran.append(arguments)

    kit, _, seen = await _kit(use)
    kit.declare_host_verb("app.record", record)

    await _say(kit)

    # Sent after its turn, and refused there by the owner: it was not
    # declared to be served between turns.
    with pytest.raises(ValueError, match="needs a turn"):
        await seen["host"].call("app.record", {"late": True})
    assert ran == []
    await kit.close()


async def test_a_verb_declared_between_turns_runs_after_its_turn_for_its_room() -> None:
    """Work a turn began (an asynchronous hook's) still runs a verb declared
    ``between_turns`` once the turn is over: for the turn's room and channel,
    with no actor nor tool call, the turn being over."""
    ran: list[dict[str, Any]] = []

    async def use(host: TurnHost) -> Any:
        return None

    async def record(arguments: dict[str, Any]) -> str:
        call = current_tool_call()
        ran.append(
            {
                "arguments": arguments,
                "room": current_tool_room_id(),
                "channel": call.channel_id if call else None,
                "actor": current_tool_actor_id(),
            }
        )
        return "recorded"

    kit, _, seen = await _kit(use)
    kit.declare_host_verb("app.record", record, between_turns=True)

    await _say(kit)
    answer = await seen["host"].call("app.record", {"verdict": "ok"})

    assert answer == "recorded"
    assert ran == [
        {"arguments": {"verdict": "ok"}, "room": "r1", "channel": "agent", "actor": None}
    ]
    await kit.close()


async def test_a_roomkit_verb_is_sent_by_its_own_method_only() -> None:
    async def use(host: TurnHost) -> Any:
        try:
            await host.call("update_room_metadata", {"room_id": "r1", "metadata": {"x": 1}})
        except ValueError as exc:
            return str(exc)
        return None

    kit, _, seen = await _kit(use)

    await _say(kit)

    assert "TurnHost.update_room_metadata" in seen["result"]
    assert (await kit.get_room("r1")).metadata == {}
    await kit.close()


@pytest.mark.parametrize("name", ["", "send_event", "app.twice"])
async def test_a_verb_name_already_held_or_empty_is_refused(name: str) -> None:
    async def verb(arguments: dict[str, Any]) -> None:
        return None

    kit = RoomKit()
    kit.declare_host_verb("app.twice", verb)

    with pytest.raises(ValueError):
        kit.declare_host_verb(name, verb)
    await kit.close()

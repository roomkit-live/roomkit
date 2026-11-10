"""An executor wires the channels it runs, the kit's own as a host's (RFC
§10.6.2, §10.6.3, §10.6.6, RMK-821).

A host builds an executor with a factory; its channels fire its hooks,
against a room context asked of the room's owner, and what they publish
and emit during a turn reaches the room's owner as frames. Room hooks on the
triggers a turn fires are defined by name on the executors and enabled per
room on the owner.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any

import pytest

from roomkit import HookResult, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import TurnLinkError
from roomkit.core.hooks import HookRegistration
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.scoring import MockScorer, ScoringHook
from roomkit.store.memory import InMemoryStore
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.context import get_current_span
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor, memory
from roomkit.turns._owner_turn import _of_the_turn
from roomkit.turns.wire import FrameworkEventFrame
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="Look it up.", parameters={"type": "object"})


def _agent(*, calls: bool = True) -> Agent:
    """An agent that calls ``lookup`` once, then answers ``done``."""
    first = (
        AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})])
        if calls
        else AIResponse(content="done")
    )

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    provider = MockAIProvider(ai_responses=[first, AIResponse(content="done")])
    return Agent("agent", provider=provider, role="agent", tools=[LOOKUP], tool_handler=handler)


async def _kit(
    backend: InMemoryTurnBackend | None = None, *, declared: bool = False, **kit: Any
) -> RoomKit:
    """A kit with the agent, live or by its *declared* declaration alone (a
    host's executor then runs it), in room ``r1``."""
    room = RoomKit(turn_backend=backend, **kit)
    room.register_channel(SimpleChannel("ops"))
    room.register_channel(_agent().declaration() if declared else _agent())
    await room.create_room(room_id="r1")
    await room.attach_channel("r1", "ops")
    await room.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    return room


async def _say(kit: RoomKit, body: str = "go") -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body)),
        room_id="r1",
    )


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _elsewhere(store: InMemoryStore, **options: Any) -> tuple[InMemoryTurnBackend, Any]:
    """A backend served by an executor a host built, which builds the agent a
    kit holds by its declaration: the room's turns run there."""
    backend = InMemoryTurnBackend()
    built: list[Agent] = []

    def factory(channel_id: str) -> Agent | None:
        if channel_id != "agent":
            return None
        built.append(_agent())
        return built[-1]

    executor = TurnExecutor(
        factory, hosts=lambda c: c == "agent", executor_id="elsewhere", **options
    )
    executor.built = built  # ty: ignore[unresolved-attribute]
    await backend.serve(executor)
    return backend, executor


# -- An executor a host built -------------------------------------------------------


async def test_a_host_s_executor_fires_its_own_hooks() -> None:
    store = InMemoryStore()
    backend, executor = await _elsewhere(store)
    kit = await _kit(backend, store=store, declared=True)
    seen: list[str] = []

    @executor.hook(HookTrigger.BEFORE_TOOL_USE)
    async def gate(event: Any, context: Any) -> Any:
        seen.append(f"gate:{context.room.id}")

    @executor.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def report(event: Any, context: Any) -> None:
        seen.append(f"tool:{event.name}")

    @executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def response(event: Any, context: Any) -> None:
        seen.append(f"response:{event.response_content}")

    await _say(kit)
    await _settle()

    assert seen == ["gate:r1", "tool:lookup", "response:done"]
    await kit.close()
    await executor.close()


async def test_a_host_s_executor_builds_a_channel_once_and_closes_it() -> None:
    store = InMemoryStore()
    backend, executor = await _elsewhere(store)
    kit = await _kit(backend, store=store, declared=True)
    await _say(kit, "one")
    await _say(kit, "two")

    (agent,) = executor.built
    closed: list[str] = []
    original = agent.close

    async def close() -> None:
        closed.append(agent.channel_id)
        await original()

    agent.close = close  # ty: ignore[invalid-assignment]
    await executor.close()
    assert closed == ["agent"]
    await kit.close()


async def test_what_a_turn_elsewhere_publishes_reaches_the_owner() -> None:
    store = InMemoryStore()
    backend, executor = await _elsewhere(store)
    kit = await _kit(backend, store=store, declared=True)
    published: list[str] = []
    emitted: list[str] = []

    async def on_ephemeral(event: Any) -> None:
        published.append(str(event.type))

    await kit.realtime.subscribe_to_room("r1", on_ephemeral)

    @kit.on("tool_call")
    async def on_tool_call(event: Any) -> None:
        emitted.append(f"tool_call:{event.data['tool_name']}")

    @kit.on("ai_response")
    async def on_response(event: Any) -> None:
        emitted.append(f"ai_response:{event.channel_id}")

    await _say(kit)
    await _settle()

    assert "tool_call_start" in published and "tool_call_end" in published
    assert emitted == ["tool_call:lookup", "ai_response:agent"]
    await kit.close()
    await executor.close()


async def test_the_kit_s_own_turns_publish_through_the_backend_too() -> None:
    kit = await _kit()
    frames: list[str] = []
    executor = kit.turn_executor
    original = executor.take

    def take(link: Any) -> None:
        send = link.send

        async def recorded(frame: Any) -> int:
            frames.append(frame.type)
            return await send(frame)

        link.send = recorded
        original(link)

    executor.take = take  # ty: ignore[invalid-assignment]
    await _say(kit)
    await _settle()

    assert "ephemeral" in frames and "framework_event" in frames
    await kit.close()


async def test_a_turn_in_a_context_of_its_own_continues_the_owner_s_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An executor in another process starts a turn in a fresh context: the
    request's trace is what parents the turn's spans to the broadcast's."""
    monkeypatch.setattr(memory._ExecutorLink, "turn_context", lambda self: contextvars.Context())
    telemetry = MockTelemetryProvider()
    kit = await _kit(telemetry=telemetry)
    requests: list[Any] = []
    take = kit.turn_executor.take

    def record(link: Any) -> None:
        requests.append(link.request)
        take(link)

    kit.turn_executor.take = record  # ty: ignore[invalid-assignment]
    await _say(kit)

    (request,) = requests
    assert request.trace is not None
    span_id = request.trace["span_id"]
    assert span_id in {span.id for span in telemetry.get_spans(SpanKind.BROADCAST)}
    generations = telemetry.get_spans(SpanKind.LLM_GENERATE)
    assert generations and all(span.parent_id == span_id for span in generations)
    await kit.close()


async def test_a_turn_s_work_runs_under_the_broadcast() -> None:
    """The work of a turn reads the broadcast as its span, as an executor in
    another process does: what a tool asks of the kit sits under it (RFC
    §10.6.9, §15.7)."""
    telemetry = MockTelemetryProvider()
    seen: list[str | None] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        seen.append(get_current_span())
        return "42"

    calling = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
    )
    ai = AIChannel(
        "agent",
        provider=MockAIProvider(ai_responses=[calling, AIResponse(content="Done.")]),
        tool_handler=lookup,
    )
    room = RoomKit(telemetry=telemetry)
    room.register_channel(SimpleChannel("ops"))
    room.register_channel(ai)
    await room.create_room(room_id="r1")
    await room.attach_channel("r1", "ops")
    await room.attach_channel(
        "r1",
        "agent",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "tools": [{"name": "lookup", "description": "Look.", "parameters": {"type": "object"}}]
        },
    )
    await _say(room)

    broadcasts = {span.id for span in telemetry.get_spans(SpanKind.BROADCAST)}
    assert seen and all(span_id in broadcasts for span_id in seen)
    await room.close()


async def test_a_scoring_hook_attaches_to_a_host_s_executor() -> None:
    store = InMemoryStore()
    backend, executor = await _elsewhere(store)
    kit = await _kit(backend, store=store, declared=True)
    hook = ScoringHook(scorers=[MockScorer()])
    hook.attach(executor)

    await _say(kit)
    await _settle()

    assert hook.recent_scores
    assert await store.list_observations("r1")
    await kit.close()
    await executor.close()


def test_a_scoring_hook_needs_no_store_on_a_host_s_executor() -> None:
    """Scores reach the owner through the turn's host (RFC §10.6.5): an
    executor with no store takes a scorer."""
    executor = TurnExecutor(lambda channel_id: None)
    ScoringHook(scorers=[MockScorer()]).attach(executor)


# -- Room hooks by name (RFC §10.6.6) -------------------------------------------------


async def test_a_room_hook_runs_for_the_rooms_that_enable_it() -> None:
    kit = await _kit()
    await kit.create_room(room_id="r2")
    await kit.attach_channel("r2", "ops")
    await kit.attach_channel("r2", "agent", category=ChannelCategory.INTELLIGENCE)
    seen: list[str] = []

    @kit.turn_executor.room_hook("audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)
    async def audit(event: Any, context: Any) -> None:
        seen.append(context.room.id)

    await kit.enable_room_hook("r1", "audit")
    await _say(kit)
    await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body="go")),
        room_id="r2",
    )
    await _settle()
    assert seen == ["r1"]

    assert await kit.disable_room_hook("r1", "audit") is True
    assert await kit.disable_room_hook("r1", "audit") is False
    await _say(kit)
    await _settle()
    assert seen == ["r1"]  # disabled: the next turn of r1 runs without it
    await kit.close()


async def test_a_room_s_turn_goes_to_an_executor_defining_its_hooks() -> None:
    store = InMemoryStore()
    backend, elsewhere = await _elsewhere(store)
    auditor = TurnExecutor(
        lambda channel_id: _agent() if channel_id == "agent" else None,
        hosts={"agent"},
        executor_id="auditor",
    )
    await backend.serve(auditor)
    kit = await _kit(backend, store=store, declared=True)
    seen: list[str] = []

    @auditor.room_hook("audit", HookTrigger.ON_AI_RESPONSE, HookExecution.ASYNC)
    async def audit(event: Any, context: Any) -> None:
        seen.append("auditor")

    taken: list[str] = []
    take = elsewhere.take

    def record(link: Any) -> None:
        taken.append(link.request.turn_id)
        take(link)

    elsewhere.take = record  # ty: ignore[invalid-assignment]
    await kit.enable_room_hook("r1", "audit")
    await _say(kit)
    await _settle()

    # The executor served first does not define "audit": the one that does takes it.
    assert seen == ["auditor"]
    assert taken == []
    await kit.close()
    await elsewhere.close()
    await auditor.close()


@pytest.mark.in_process("reads the in-memory backend's placement")
async def test_a_room_hook_no_executor_defines_leaves_the_room_unanswered() -> None:
    kit = await _kit()
    await kit.enable_room_hook("r1", "missing")

    await _say(kit)

    assert kit._turns.room_hook_names("r1") == ["missing"]
    timeline = await kit.store.list_events("r1")
    assert not [e for e in timeline if e.source.channel_id == "agent"]
    with pytest.raises(TurnLinkError, match="room hooks"):
        kit._turns.backend._place("agent", "r1", ["missing"])
    await kit.close()


def test_a_room_hook_on_an_executor_trigger_is_refused_by_callable() -> None:
    executor = TurnExecutor(lambda channel_id: None)

    async def fn(event: Any, context: Any) -> None:
        return None

    registration = HookRegistration(
        trigger=HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, fn=fn, name="x"
    )
    with pytest.raises(ValueError, match="room_hook"):
        executor.hooks.add_room_hook("r1", registration)


async def _hold_audit(kit: RoomKit, room_id: str, how: str, trigger: HookTrigger) -> None:
    """Give *room_id* a hook named ``audit``: the kit's room hook on
    *trigger*, or the executors' room hook of that name enabled."""
    if how == "kit":
        await kit.add_room_hook(room_id, trigger, HookExecution.ASYNC, _noop_hook, name="audit")
    else:
        await kit.enable_room_hook(room_id, "audit")


@pytest.mark.parametrize(
    "trigger",
    [HookTrigger.AFTER_BROADCAST, HookTrigger.ON_AI_RESPONSE],
    ids=["owner trigger", "kit executor trigger"],
)
@pytest.mark.parametrize(("first", "then"), [("kit", "enabled"), ("enabled", "kit")])
async def test_a_room_never_holds_two_hooks_of_one_name(
    trigger: HookTrigger, first: str, then: str
) -> None:
    """RFC §9.1, §10.6.6: a name the room enables joins the room's scope, so
    enabling it where a room hook holds it, or registering a room hook under
    an enabled name, is refused."""
    kit = RoomKit()
    kit.turn_executor.room_hook("audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)(_noop_hook)
    await _hold_audit(kit, "r1", first, trigger)

    with pytest.raises(ValueError, match="audit"):
        await _hold_audit(kit, "r1", then, trigger)

    assert kit._turns.room_hook_names("r1") == (["audit"] if first == "enabled" else [])


async def test_a_name_another_room_holds_is_enabled() -> None:
    kit = RoomKit()
    kit.turn_executor.room_hook("audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)(_noop_hook)
    await _hold_audit(kit, "r1", "kit", HookTrigger.AFTER_BROADCAST)

    await _hold_audit(kit, "r2", "enabled", HookTrigger.AFTER_BROADCAST)

    assert kit._turns.room_hook_names("r2") == ["audit"]


async def _noop_hook(event: Any, context: Any) -> None:
    return None


# -- Review: what the owner does with a turn's publications ------------------------


ASK = AITool(
    name="ask_user",
    description="Ask the user.",
    parameters={"type": "object", "properties": {"question": {"type": "string"}}},
)


async def test_a_framework_handler_may_answer_the_input_its_turn_asks() -> None:
    """The handler runs off the turn's frame reader: the answer's ``input_taken``
    is read while it waits."""
    call = AIResponse(
        content="",
        tool_calls=[AIToolCall(id="c1", name="ask_user", arguments={"question": "Color?"})],
    )
    provider = MockAIProvider(ai_responses=[call, AIResponse(content="Thanks.")])
    human = HumanInputToolHandler(tool_names={"ask_user"}, timeout=5, tool_definitions=[ASK])
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(AIChannel("agent", provider=provider, human_input_handler=human))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    said: list[bool] = []

    @kit.on("user_input_required")
    async def auto_answer(event: Any) -> None:
        said.append(await kit.answer_input(event.data["pending_id"], "Blue"))

    await asyncio.wait_for(_say(kit), 4)
    assert said == [True]
    await kit.close()


async def test_the_owner_publishes_a_turn_s_events_for_its_room_and_channel_only() -> None:
    request = type("R", (), {"room_id": "r1", "channel_id": "agent"})()
    assert _of_the_turn(request, "r1", "agent")
    assert _of_the_turn(request, None, None)
    assert not _of_the_turn(request, "r2", "agent")
    assert not _of_the_turn(request, "r1", "other")


async def test_a_frame_for_another_room_or_of_another_kind_is_dropped() -> None:
    kit = await _kit()
    emitted: list[Any] = []

    @kit.on("video_vision_result")
    async def vision(event: Any) -> None:
        emitted.append(event)

    @kit.on("tool_call")
    async def tool_call(event: Any) -> None:
        emitted.append(event)

    executor = kit.turn_executor
    take = executor.take

    def forging(link: Any) -> None:
        send = link.send

        async def forged(frame: Any) -> int:
            if frame.type == "accepted":
                seq = await send(frame)
                await send(FrameworkEventFrame(event_type="video_vision_result", room_id="r1"))
                await send(FrameworkEventFrame(event_type="tool_call", room_id="elsewhere"))
                await send(FrameworkEventFrame(event_type="tool_call", room_id=None))
                return seq
            return await send(frame)

        link.send = forged
        take(link)

    executor.take = forging  # ty: ignore[invalid-assignment]
    await _say(kit)
    await _settle()

    # The forged kind and room are dropped; a room-less one is the turn's room.
    forged = [e for e in emitted if e.type == "video_vision_result" or e.room_id != "r1"]
    assert forged == []
    await kit.close()


# -- Review: room hooks reach every door of the kit's executor ------------------------


async def test_a_room_hook_gates_a_call_outside_a_turn_as_soon_as_it_is_enabled() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")

    async def deny(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("needs approval")

    kit.turn_executor.room_hook("approval", HookTrigger.BEFORE_TOOL_USE)(deny)
    event = ToolCallEvent(
        channel_id="voice",
        channel_type=ChannelType.REALTIME_VOICE,
        tool_call_id="t1",
        name="transfer_money",
        arguments={},
        room_id="r1",
    )
    await kit.enable_room_hook("r1", "approval")
    # What the realtime voice tool gate calls, with no turn of the room yet.
    denied, _ = await kit._decide_before_tool_use(event, "voice")
    await kit.disable_room_hook("r1", "approval")
    allowed, _ = await kit._decide_before_tool_use(event, "voice")

    assert (denied.allowed, allowed.allowed) == (False, True)
    await kit.close()


async def _gate_denies(event: Any, ctx: Any) -> HookResult:
    return HookResult.block("needs approval")


async def _voice_call_allowed(kit: RoomKit) -> bool:
    """Whether the realtime voice tool gate lets a call in room r1 run."""
    event = ToolCallEvent(
        channel_id="voice",
        channel_type=ChannelType.REALTIME_VOICE,
        tool_call_id="t1",
        name="transfer_money",
        arguments={},
        room_id="r1",
    )
    decision, _ = await kit._decide_before_tool_use(event, "voice")
    return decision.allowed


async def test_a_room_hook_enabled_before_it_is_defined_gates_a_call_outside_a_turn() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    # A room's configuration restored before the host's hooks are loaded.
    await kit.enable_room_hook("r1", "approval")
    kit.turn_executor.room_hook("approval", HookTrigger.BEFORE_TOOL_USE)(_gate_denies)

    allowed = await _voice_call_allowed(kit)

    await kit.close()
    assert not allowed


async def test_a_room_hook_defined_again_after_its_removal_gates_a_call_outside_a_turn() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    kit.turn_executor.room_hook("approval", HookTrigger.BEFORE_TOOL_USE)(_gate_denies)
    await kit.enable_room_hook("r1", "approval")
    kit.turn_executor.remove_room_hook("approval")
    removed = await _voice_call_allowed(kit)
    kit.turn_executor.room_hook("approval", HookTrigger.BEFORE_TOOL_USE)(_gate_denies)

    allowed = await _voice_call_allowed(kit)

    await kit.close()
    assert (removed, allowed) == (True, False)


def test_a_room_hook_defined_again_on_a_trigger_replaces_the_first() -> None:
    executor = TurnExecutor(lambda channel_id: None)

    async def first(event: Any, ctx: Any) -> None: ...
    async def second(event: Any, ctx: Any) -> None: ...

    executor.room_hook("audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)(first)
    executor.room_hook("audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)(second)
    executor.room_hook("audit", HookTrigger.ON_AI_RESPONSE, HookExecution.ASYNC)(first)

    defined = executor.hooks._named_room_hooks["audit"]
    assert [(h.trigger, h.fn) for h in defined] == [
        (HookTrigger.ON_TOOL_CALL, second),
        (HookTrigger.ON_AI_RESPONSE, first),
    ]


def test_a_fail_closed_room_hook_must_run_in_line() -> None:
    executor = TurnExecutor(lambda channel_id: None)

    async def fn(event: Any, ctx: Any) -> None: ...

    with pytest.raises(ValueError):
        executor.room_hook(
            "audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC, fail_closed=True
        )(fn)


async def test_an_executor_refuses_a_turn_whose_room_hooks_it_does_not_define() -> None:
    backend = InMemoryTurnBackend()
    executor = TurnExecutor(
        lambda channel_id: _agent(),
    )
    await backend.serve(executor)
    client = TurnClient(backend)
    client._room_hooks["r1"] = ["missing"]
    # A backend that places on the channel alone (not the in-memory one).
    backend._place = lambda channel_id, room_id, room_hooks=(): executor  # type: ignore[method-assign]
    event = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="ops", channel_type=ChannelType.SMS),
        content=TextContent(body="go"),
    )
    binding = ChannelBinding(
        channel_id="agent",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    with pytest.raises(TurnLinkError) as raised:
        await client.run(event, binding, RoomContext(room=Room(id="r1"), bindings=[binding]))
    assert raised.value.reason == "no_executor"
    await executor.close()


# -- Review: the channels an executor builds ------------------------------------------


async def test_concurrent_first_turns_share_one_build_and_a_close_waits_for_it() -> None:
    release = asyncio.Event()
    built: list[Agent] = []

    async def factory(channel_id: str) -> Agent:
        await release.wait()
        built.append(_agent())
        return built[-1]

    executor = TurnExecutor(factory)
    first = asyncio.create_task(executor._channels.get("agent"))
    second = asyncio.create_task(executor._channels.get("agent"))
    await asyncio.sleep(0)
    closing = asyncio.create_task(executor.close())
    await asyncio.sleep(0)
    release.set()
    await closing

    results = await asyncio.gather(first, second, return_exceptions=True)
    assert len(built) == 1
    # The build finished after the close began: it is closed, never held.
    assert all(isinstance(r, RuntimeError) for r in results)
    assert executor._channels._built == {}


async def test_a_factory_error_is_its_turn_s_and_the_next_build_tries_again() -> None:
    calls: list[int] = []

    def factory(channel_id: str) -> Agent:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("factory down")
        return _agent()

    executor = TurnExecutor(factory)
    with pytest.raises(RuntimeError, match="factory down"):
        await executor._channels.get("agent")
    assert await executor._channels.get("agent") is not None
    await executor.close()


async def test_a_host_s_channel_gets_the_host_s_media_readers() -> None:
    async def resolve(reference: str) -> Any:
        return None

    executor = TurnExecutor(lambda channel_id: _agent(), media_resolver=resolve)
    channel = await executor._channels.get("agent")
    assert channel is not None and channel._media_resolver is resolve  # type: ignore[attr-defined]
    await executor.close()


async def test_a_host_s_executor_drops_what_is_published_outside_a_turn() -> None:
    executor = TurnExecutor(lambda channel_id: _agent())
    event = EphemeralEvent(room_id="r1", type=EphemeralEventType.CUSTOM, user_id="agent")
    await executor.wiring.realtime.publish_to_room("r1", event)  # nothing raises
    await executor.wiring.emit("tool_call", room_id="r1")


async def test_a_view_that_cannot_be_read_costs_the_turn_its_view_only() -> None:
    agent = _agent(calls=False)

    async def failing(context: Any) -> str | None:
        raise RuntimeError("owner down")

    agent._room_vision_loader = failing
    context = RoomContext(room=Room(id="r1"))
    assert await agent._vision_notes(context, standalone=False) == []


async def test_a_turn_runs_the_room_hooks_its_request_named_and_a_disable_holds() -> None:
    """A turn built before a disable runs the hooks its request named; the
    room's set, which doors outside a turn read, stays the owner's."""
    kit = await _kit()
    seen: list[str] = []

    @kit.turn_executor.room_hook("audit", HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)
    async def audit(event: Any, context: Any) -> None:
        seen.append("audit")

    await kit.enable_room_hook("r1", "audit")
    executor = kit.turn_executor
    take = executor.take

    def disable_once_taken(link: Any) -> None:
        take(link)
        # The kit's trusted, synchronous level: no await between the take
        # and the disable.
        kit._turns.disable_room_hook("r1", "audit")

    executor.take = disable_once_taken  # ty: ignore[invalid-assignment]
    await _say(kit)
    await _settle()

    assert seen == ["audit"]  # the turn ran what its request named
    assert executor.hooks.room_hook_names("r1") == []  # the room's set stays off
    await kit.close()

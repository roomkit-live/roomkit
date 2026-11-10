"""Work a turn began reaches the room owner after the turn ended (RFC
§10.6.3, §10.6.5): the host verbs that write the room, at the turn's chain
depth, and the channel's publications, from the executor of the room's last
turn only."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import (
    Agent,
    AIChannel,
    ChannelDeclaration,
    Discussion,
    MockSpeakPolicy,
    MockThinker,
    RoomKit,
    Thought,
    ThoughtEvent,
    current_turn_host,
)
from roomkit.memory.mock import MockMemoryProvider
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.task import Observation
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.store.memory import InMemoryStore
from roomkit.tasks.status import TASK_ACTION
from roomkit.turns import InMemoryTurnBackend, RoomCall, TurnExecutor
from roomkit.turns._outlet import emit_framework_event
from tests.test_framework import SimpleChannel

LATER = AITool(name="later", description="Starts work that outlives the turn.", parameters={})


class _Later:
    """A host tool starting work that waits for the test to let it go, after
    the turn ended, then runs *work* with the host the turn left it."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.done: asyncio.Future[Any] | None = None
        self.work: Any = None

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        async def run() -> Any:
            await self.release.wait()
            return await self.work()

        self.done = asyncio.ensure_future(run())
        return "started"


def _calls_later() -> AIResponse:
    call = AIToolCall(id="c1", name="later", arguments={})
    return AIResponse(content="", finish_reason="tool_calls", tool_calls=[call])


class _Host:
    def __init__(self, later: _Later) -> None:
        self.store = InMemoryStore()
        self.backend = InMemoryTurnBackend()
        provider = MockAIProvider(
            ai_responses=[_calls_later(), AIResponse(content="On it.")]
            + [AIResponse(content="Again.")] * 4
        )
        self.agent = Agent("ana", provider=provider, tools=[LATER], tool_handler=later.handler)
        self.executor = TurnExecutor(
            {"ana": self.agent}.get,
            hosts=lambda cid: cid == "ana",
        )

    async def kit(self) -> RoomKit:
        await self.backend.serve(self.executor)
        kit = RoomKit(store=self.store, turn_backend=self.backend)
        kit.register_channel(SimpleChannel("sms"))
        kit.register_channel(ChannelDeclaration(channel_id="ana"))
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "sms")
        await kit.attach_channel("r", "ana", category=ChannelCategory.INTELLIGENCE)
        return kit

    async def close(self, kit: RoomKit) -> None:
        await kit.close()
        await self.executor.close()


async def _say(kit: RoomKit, text: str = "Go.") -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="marie", content=TextContent(body=text)),
        room_id="r",
    )


async def _until(predicate: Callable[[], Any]) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


async def _after_the_turn(later: _Later) -> Any:
    assert later.done is not None
    later.release.set()
    return await asyncio.wait_for(later.done, 5)


async def test_work_a_turn_began_writes_the_room_after_the_turn_ended() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()
    posted: list[Any] = []

    async def on_status(entry: Any) -> None:
        posted.append(entry)

    await kit.status_bus.subscribe(on_status)

    async def work() -> Any:
        turn_host = current_turn_host()
        assert turn_host is not None and not turn_host.open
        await turn_host.post_status("ana", "thought", "info", detail="Still thinking")
        return await turn_host.send_event(
            room_id="r", channel_id="ana", content=TextContent(body="A thought, later.")
        )

    later.work = work
    await _say(kit)
    sent = await _after_the_turn(later)
    stored = [e for e in await kit.store.list_events("r") if e.id == sent.id]
    await host.close(kit)

    assert [
        (p.action, p.detail, p.metadata.get("room_id")) for p in posted if p.action == "thought"
    ] == [("thought", "Still thinking", "r")]
    # Work a turn began starts fresh once the turn is over (RFC §8.3).
    assert [e.chain_depth for e in stored] == [0]


async def test_a_hook_s_body_sends_after_the_turn_at_its_trigger_s_depth() -> None:
    """RMK-1004: the body of an ASYNC hook on the executor that sends once
    its turn is over sends at least as deep as the turn was when the trigger
    fired, as a hook's body does on the room owner's side: the owner floors
    nothing between turns, so the verb carries the depth (RFC §8.3)."""
    later = _Later()
    later.work = lambda: asyncio.sleep(0, True)
    host = _Host(later)
    fired: list[bool] = []
    sent: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    @host.executor.hook(HookTrigger.BEFORE_AI_GENERATION, execution=HookExecution.ASYNC)
    async def remind(event: Any, ctx: Any) -> None:
        if fired:
            return
        fired.append(True)
        turn_host = current_turn_host()
        assert turn_host is not None
        await _until(lambda: not turn_host.open)
        reminder = TextContent(body="A reminder, once the turn is over.")
        event = await turn_host.send_event(room_id="r", channel_id="ana", content=reminder)
        sent.set_result(event)

    kit = await host.kit()
    await _say(kit)
    await _after_the_turn(later)
    event = await asyncio.wait_for(sent, 5)
    stored = [e for e in await kit.store.list_events("r") if e.id == event.id]
    await host.close(kit)

    assert [e.chain_depth for e in stored] == [1]


async def test_a_verb_that_needs_its_turn_is_refused_after_it() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()

    async def work() -> Any:
        turn_host = current_turn_host()
        assert turn_host is not None
        with pytest.raises(RuntimeError, match="needs a turn"):
            await turn_host.delegate("r", "ana", "More.")
        return True

    later.work = work
    await _say(kit)
    refused = await _after_the_turn(later)
    await host.close(kit)

    assert refused is True


async def test_work_of_a_turn_that_is_no_longer_the_room_s_last_is_refused() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()

    async def work() -> Any:
        turn_host = current_turn_host()
        assert turn_host is not None
        with pytest.raises(PermissionError, match="no longer the room's last"):
            await turn_host.post_status("ana", "stale", "info")
        return True

    later.work = work
    await _say(kit)
    await _say(kit, "And again.")  # a later turn of ana in r
    refused = await _after_the_turn(later)
    await host.close(kit)

    assert refused is True


async def test_a_publication_after_the_turn_reaches_the_owner_for_its_room_only() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()
    seen: list[str] = []

    async def on_event(event: EphemeralEvent) -> None:
        if event.type is EphemeralEventType.TYPING_START:
            seen.append(f"{event.room_id}:{event.type}")

    await kit.realtime.subscribe_to_room("r", on_event)
    await kit.realtime.subscribe_to_room("elsewhere", on_event)

    async def work() -> Any:
        realtime = host.agent._realtime
        for room_id in ("r", "elsewhere"):
            await realtime.publish_to_room(
                room_id,
                EphemeralEvent(
                    room_id=room_id,
                    type=EphemeralEventType.TYPING_START,
                    user_id="ana",
                    channel_id="ana",
                ),
            )
        return True

    later.work = work
    await _say(kit)
    await _after_the_turn(later)
    await _until(lambda: seen)
    await asyncio.sleep(0.02)  # the other room's, published after it, would be in by now
    await host.close(kit)

    assert seen == ["r:typing_start"]


# -- Review of 99cf3e83/d204befd --


def _listener(thinker: MockThinker) -> AIChannel:
    return AIChannel(
        "nova",
        provider=MockAIProvider(responses=["x"] * 8),
        speak_policy=MockSpeakPolicy(["silent"]),
        thinker=thinker,
        think_wait=0.0,
    )


async def _thoughts_posted(
    kit: RoomKit, messages: list[str], until: Callable[[list[str]], bool]
) -> tuple[list[str], list[str]]:
    """The thoughts posted and refused once *messages* were said and the posts
    satisfy *until*; the caller closes the kit, which lets a hook still running
    end, before it reads them."""
    posted: list[str] = []
    refused: list[str] = []

    @kit.hook(HookTrigger.ON_THOUGHT, execution=HookExecution.ASYNC)
    async def post(event: ThoughtEvent, ctx: Any) -> None:
        host = current_turn_host()
        assert host is not None
        try:
            await host.post_status("nova", "thought", "info", detail=event.thought.text)
        except PermissionError as exc:
            refused.append(str(exc))

    async def on_status(entry: Any) -> None:
        if entry.action == "thought":
            posted.append(entry.detail)

    await kit.status_bus.subscribe(on_status)
    for text in messages:
        await _say(kit, text)
    async with asyncio.timeout(5):
        while not until(posted):
            await asyncio.sleep(0.01)
    return posted, refused


async def _kit_with(channel: AIChannel) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", channel.channel_id, category=ChannelCategory.INTELLIGENCE)
    return kit


async def test_a_thought_at_once_reaches_the_room_before_the_turn_s_output() -> None:
    """The owner knows the turn's executor from its acceptance: a thought that
    lands before the turn's output is read is the turn's own work."""
    kit = await _kit_with(_listener(MockThinker([Thought(text="Quick.")])))

    posted, refused = await _thoughts_posted(kit, ["One."], lambda posted: bool(posted))
    await kit.close()

    assert (posted, refused) == (["Quick."], [])


async def test_a_thinker_running_across_turns_reports_under_the_latest_turn() -> None:
    # Each thought takes long enough for the second turn to be sent before the first lands.
    thinker = MockThinker([Thought(text="First."), Thought(text="Latest.")], delay=0.15)
    kit = await _kit_with(_listener(thinker))

    # The first thought began before the latest one: it has landed once the latest is in.
    posted, refused = await _thoughts_posted(
        kit, ["One.", "Two."], lambda posted: "Latest." in posted
    )
    await kit.close()

    # The thought on the first event lands once the second turn was sent: that
    # earlier turn's work is refused; the thought on the latest event is not.
    assert posted == ["Latest."]
    assert len(refused) <= 1


async def test_the_write_verbs_between_turns_keep_their_refusals() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()

    async def work() -> Any:
        turn_host = current_turn_host()
        assert turn_host is not None
        # A key only the framework writes is dropped there, the rest written.
        room = await turn_host.update_room_metadata(
            "r", {"thought": "kept", "roomkit:channel_state:ana": {}, "author_register": {}}
        )
        await turn_host.add_observation(
            Observation(id="o1", room_id="r", channel_id="ana", content="seen", category="note")
        )
        refusals = []
        for attempt in (
            turn_host.post_status("ana", TASK_ACTION, "pending"),
            turn_host.post_status("ana", "x", "info", metadata={"room_id": "other"}),
            turn_host.send_event(room_id="other", channel_id="ana", content=TextContent(body="x")),
        ):
            with pytest.raises(ValueError):
                await attempt
            refusals.append(True)
        sent = await turn_host.send_event(
            room_id="r", channel_id="ana", content=TextContent(body="Later."), chain_depth=0
        )
        return room, refusals, sent

    later.work = work
    await _say(kit)
    room, refusals, sent = await _after_the_turn(later)
    observations = await kit.store.list_observations("r")
    await host.close(kit)

    assert room.metadata == {"thought": "kept"}
    assert [o.id for o in observations] == ["o1"]
    assert refusals == [True] * 3
    # Work a turn began starts fresh once the turn is over (RFC §8.3).
    assert sent.chain_depth == 0


async def test_a_framework_event_after_the_turn_reaches_the_owner() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()
    seen: list[str] = []

    @kit.on("tool_call")
    async def on_tool_call(event: Any) -> None:
        if event.data.get("later"):
            seen.append(f"{event.room_id}:{event.channel_id}")

    async def work() -> Any:
        await emit_framework_event(
            None, "tool_call", room_id="r", channel_id="ana", data={"later": True}
        )
        await emit_framework_event(
            None, "room_created", room_id="r", channel_id="ana", data={"later": True}
        )
        return True

    later.work = work
    await _say(kit)
    await _after_the_turn(later)
    await _until(lambda: seen)
    await asyncio.sleep(0.02)  # a second delivery of it would be in by now
    await host.close(kit)

    assert seen == ["r:ana"]


async def test_a_publication_from_an_executor_no_longer_holding_the_room_is_dropped() -> None:
    later = _Later()
    store, backend = InMemoryStore(), InMemoryTurnBackend()
    takes_a = [True]
    provider_a = MockAIProvider(ai_responses=[_calls_later(), AIResponse(content="A.")])
    agent_a = Agent("ana", provider=provider_a, tools=[LATER], tool_handler=later.handler)
    agent_b = Agent("ana", provider=MockAIProvider(responses=["B."] * 4))
    first = TurnExecutor(
        {"ana": agent_a}.get,
        hosts=lambda cid: takes_a[0],
    )
    second = TurnExecutor(
        {"ana": agent_b}.get,
        hosts=lambda cid: cid == "ana",
    )
    await backend.serve(first)
    await backend.serve(second)
    kit = RoomKit(store=store, turn_backend=backend)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(ChannelDeclaration(channel_id="ana"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ana", category=ChannelCategory.INTELLIGENCE)
    seen: list[str] = []

    async def on_event(event: EphemeralEvent) -> None:
        if event.type is EphemeralEventType.TYPING_START:
            seen.append(event.room_id)

    await kit.realtime.subscribe_to_room("r", on_event)

    async def work() -> Any:
        await agent_a._realtime.publish_to_room(
            "r",
            EphemeralEvent(
                room_id="r", type=EphemeralEventType.TYPING_START, user_id="ana", channel_id="ana"
            ),
        )
        return True

    later.work = work
    await _say(kit)
    takes_a[0] = False
    await _say(kit, "Now B.")  # the room's next turn of ana goes to the second executor
    await _after_the_turn(later)
    await asyncio.sleep(0.05)  # nothing to await: its publication would reach the owner by now
    await kit.close()
    await first.close()
    await second.close()

    assert seen == []


async def test_work_after_the_owner_closed_is_refused() -> None:
    later = _Later()
    host = _Host(later)
    kit = await host.kit()

    async def work() -> Any:
        turn_host = current_turn_host()
        assert turn_host is not None
        with pytest.raises(RuntimeError, match="closed"):
            await turn_host.post_status("ana", "late", "info")
        return True

    later.work = work
    await _say(kit)
    await kit.close()
    refused = await _after_the_turn(later)
    await host.executor.close()

    assert refused is True


async def test_a_committed_event_goes_where_the_room_s_turns_would() -> None:
    """Two executors declare the agent; only one defines the room's hook, so
    only it takes the room's turns, and only it is handed the room's events."""
    memories = {"a": MockMemoryProvider(), "b": MockMemoryProvider()}
    store, backend = InMemoryStore(), InMemoryTurnBackend()
    executors = {}
    for name, memory in memories.items():
        agent = Agent("ana", provider=MockAIProvider(responses=["x"] * 4), memory=memory)
        executors[name] = TurnExecutor(
            {"ana": agent}.get,
            hosts=lambda cid: cid == "ana",
        )
        await backend.serve(executors[name])

    @executors["b"].room_hook("audit", HookTrigger.ON_AI_RESPONSE, HookExecution.ASYNC)
    async def audit(event: Any, ctx: Any) -> None:
        return None

    kit = RoomKit(store=store, turn_backend=backend)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(ChannelDeclaration(channel_id="ana"))
    kit.register_channel(ChannelDeclaration(channel_id="bob"))
    await kit.create_room(room_id="r", orchestration=Discussion(agents=["ana", "bob"]))
    await kit.attach_channel("r", "sms")
    await kit.enable_room_hook("r", "audit")

    await _say(kit, "@bob hello")
    await _until(lambda: memories["b"].ingest_calls)
    await asyncio.sleep(0.02)  # the other executor, handed it too, would have ingested it by now
    handed = {name: len(memory.ingest_calls) for name, memory in memories.items()}
    await kit.close()
    for executor in executors.values():
        await executor.close()

    assert handed == {"a": 0, "b": 1}


async def test_a_second_owner_on_one_in_memory_backend_is_told(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = InMemoryTurnBackend()
    first = RoomKit(turn_backend=backend)
    second = RoomKit(turn_backend=backend)
    for kit in (first, second):
        await kit._turns._serve_own()

    assert "second room owner" in caplog.text
    await first.close()


async def test_a_room_call_after_the_turn_sends_at_the_depth_it_names() -> None:
    """Work a turn began starts fresh once the turn is over (RFC §8.3): a
    room call's event is at the depth its arguments name, whatever depth
    the call carries for its turn and the owner recorded."""
    later = _Later()
    host = _Host(later)
    kit = await host.kit()
    later.work = lambda: asyncio.sleep(0, True)
    await _say(kit)
    record = kit._turns._previous.get("ana", "r")
    assert record is not None and record.executor_id is not None
    forged = RoomCall(
        channel_id="ana",
        room_id="r",
        executor_id=record.executor_id,
        turn_id=record.turn_id,
        chain_depth=1,
        verb="send_event",
        arguments={"room_id": "r", "channel_id": "ana", "content": {"type": "text", "body": "x"}},
    )
    answer = await kit._turns.room_call(forged)
    await _after_the_turn(later)
    await host.close(kit)

    assert answer.error is None
    assert answer.result["chain_depth"] == 0

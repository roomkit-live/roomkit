"""An executor has no store: the room sends what a turn reads (RFC §10.6.3, §10.6.6)."""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any

import pytest

from roomkit import (
    Agent,
    ChannelCategory,
    ChannelDeclaration,
    HookResult,
    HookTrigger,
    InboundMessage,
    InMemoryTurnBackend,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.core.exceptions import RoomNotFoundError, TurnLinkError
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.room import Room
from roomkit.models.steering import Cancel
from roomkit.providers.ai import AITool, AIToolCall
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.turns import TurnExecutor
from roomkit.turns._wiring_env import TurnWiringEnv, read_channel_rows, set_turn_request
from roomkit.turns.host import set_turn_host
from roomkit.turns.wire import TurnRequest

LOOKUP = AITool(name="lookup", description="Looks up.", parameters={})
CALL = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})])
SECRET = "the code is 4242"
EXCHANGES = 30
"""More exchanges than the history a turn is sent holds."""


class _Seen(InMemoryTurnBackend):
    """Keeps the requests it carries."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[TurnRequest] = []

    async def submit(self, request: TurnRequest) -> Any:
        self.requests.append(request)
        return await super().submit(request)


def _agent(channel_id: str, *answers: AIResponse, handler: Any = None) -> Agent:
    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return SECRET

    return Agent(
        channel_id,
        provider=MockAIProvider(ai_responses=list(answers)),
        tools=[LOOKUP],
        tool_handler=handler or lookup,
    )


async def _kit(backend: InMemoryTurnBackend, *channels: Any) -> RoomKit:
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(WebSocketChannel("you"))
    for channel in channels:
        kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "you")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit


async def _say(kit: RoomKit, text: str, answers: int) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="you", sender_id="you", content=TextContent(body=text)),
        room_id="r",
    )
    async with asyncio.timeout(5):
        while len(await _answers(kit)) < answers:
            await asyncio.sleep(0.01)


async def _answers(kit: RoomKit) -> list[Any]:
    events = await kit.store.list_events("r", limit=500)
    return [e for e in events if e.source.channel_id == "agent" and e.type == EventType.MESSAGE]


async def test_a_turn_on_an_executor_is_sent_its_rows_each_once() -> None:
    backend = _Seen()
    kit = await _kit(backend, ChannelDeclaration(channel_id="agent"))
    answers = (CALL, AIResponse(content="done"), AIResponse(content="ok"))
    agent = _agent("agent", *answers)
    executor = TurnExecutor(lambda _c: agent, hosts={"agent"}, executor_id="elsewhere")
    await backend.serve(executor)

    await _say(kit, "find it", 1)
    await _say(kit, "and now?", 2)

    rows = backend.requests[-1].channel_rows
    indexes = [row.index for row in rows]
    assert {row.type for row in rows} == {EventType.TOOL_CALL_START, EventType.TOOL_CALL_END}
    assert indexes == sorted(set(indexes)), "oldest first, each once"
    await kit.close()
    await executor.close()


async def test_a_turn_on_the_kit_s_own_executor_is_sent_none() -> None:
    backend = _Seen()
    answers = (CALL, AIResponse(content="done"), AIResponse(content="ok"))
    kit = await _kit(backend, _agent("agent", *answers))

    await _say(kit, "find it", 1)
    await _say(kit, "and now?", 2)

    assert backend.requests[-1].channel_rows == []
    await kit.close()


async def test_a_deleted_row_never_leaves_the_room() -> None:
    backend = _Seen()
    kit = await _kit(backend, ChannelDeclaration(channel_id="agent"))
    answers = (CALL, AIResponse(content="done"), AIResponse(content="ok"))
    agent = _agent("agent", *answers)
    executor = TurnExecutor(lambda _c: agent, hosts={"agent"}, executor_id="elsewhere")
    await backend.serve(executor)
    await _say(kit, "find it", 1)
    events = await kit.store.list_events("r", limit=50)
    assert SECRET in "".join(e.model_dump_json() for e in events), "the row holds the result"
    ended = next(e for e in events if e.type == EventType.TOOL_CALL_END)
    deleted = ended.model_copy(update={"metadata": {**ended.metadata, "deleted": True}})
    await kit.store.update_event(deleted)

    await _say(kit, "and now?", 2)

    assert SECRET not in backend.requests[-1].model_dump_json()
    await kit.close()
    await executor.close()


async def test_a_hook_whose_context_the_room_cannot_build_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InMemoryTurnBackend()
    kit = await _kit(backend, ChannelDeclaration(channel_id="agent"))
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ran"

    agent = _agent("agent", CALL, AIResponse(content="done"), handler=handler)
    executor = TurnExecutor(lambda _c: agent, hosts={"agent"}, executor_id="elsewhere")

    async def allow(_event: Any, _context: Any) -> HookResult:
        return HookResult.allow()

    executor.hook(HookTrigger.BEFORE_TOOL_USE, name="allow")(allow)
    await backend.serve(executor)

    async def broken(_scope: Any, _args: Any) -> None:
        raise OSError("the store is down")

    monkeypatch.setitem(kit._turns._host._verbs, "room_context", broken)

    await _say(kit, "find it", 1)

    assert ran == [], "a hook that cannot read the room does not let the call through"
    await kit.close()
    await executor.close()


class _CutHost:
    """A turn's host whose owner no longer answers."""

    reachable = True

    def __init__(self, error: Exception, *, is_open: bool = True) -> None:
        self._error = error
        self.open = is_open

    async def room_context(self) -> dict[str, Any]:
        raise self._error


def _request(kit_context: Any, event: Any, binding: Any) -> TurnRequest:
    return TurnRequest(
        turn_id="t",
        room_id="r",
        channel_id="agent",
        event=event,
        binding=binding,
        context=kit_context,
    )


async def test_a_hook_in_a_cut_turn_reads_the_context_the_turn_started_with() -> None:
    backend = _Seen()
    kit = await _kit(backend, ChannelDeclaration(channel_id="agent"))
    agent = _agent("agent", AIResponse(content="done"))
    executor = TurnExecutor(lambda _c: agent, hosts={"agent"}, executor_id="elsewhere")
    await backend.serve(executor)
    await _say(kit, "hello", 1)
    started = backend.requests[-1]
    env = TurnWiringEnv()

    async def read(error: Exception) -> Any:
        set_turn_request(started)
        set_turn_host(_CutHost(error))  # type: ignore[arg-type]
        return await env.context("r")

    cut = await asyncio.create_task(
        read(TurnLinkError("cancelled")), context=contextvars.copy_context()
    )
    assert cut == started.context

    async def over() -> Any:
        set_turn_request(started)
        set_turn_host(_CutHost(OSError("never asked"), is_open=False))  # type: ignore[arg-type]
        return await env.context("r")

    assert await asyncio.create_task(over(), context=contextvars.copy_context()) == started.context
    with pytest.raises(OSError):
        await asyncio.create_task(read(OSError("down")), context=contextvars.copy_context())
    await kit.close()
    await executor.close()


async def test_a_hook_outside_any_turn_reads_the_executor_s_last_turn_in_the_room() -> None:
    backend = _Seen()
    kit = await _kit(
        backend, ChannelDeclaration(channel_id="agent"), ChannelDeclaration(channel_id="other")
    )
    await kit.attach_channel("r", "other", category=ChannelCategory.INTELLIGENCE)
    agents = {cid: _agent(cid, AIResponse(content="done")) for cid in ("agent", "other")}
    executor = TurnExecutor(agents.get, hosts=set(agents), executor_id="elsewhere")
    await backend.serve(executor)
    await _say(kit, "hello", 1)
    async with asyncio.timeout(5):
        while {r.channel_id for r in backend.requests} != set(agents):
            await asyncio.sleep(0.01)
    env = executor.wiring.env

    async def outside(room_id: str) -> Any:
        return await asyncio.create_task(env.context(room_id), context=contextvars.Context())

    assert await outside("r") == backend.requests[-1].context
    with pytest.raises(RoomNotFoundError):
        await outside("no-turn-ran-there")
    # While a channel whose turns ran in the room stays attached, its hooks
    # keep reading the room; once none is, a room created again under the
    # id reads nothing of the earlier conversation (RMK-1645).
    await kit.detach_channel("r", "agent")
    assert await outside("r") == backend.requests[-1].context
    await kit.detach_channel("r", "other")
    with pytest.raises(RoomNotFoundError):
        await outside("r")
    await kit.close()
    await executor.close()


async def test_a_hook_reads_the_room_s_history_whatever_the_channel_s_window() -> None:
    backend = InMemoryTurnBackend()
    kit = await _kit(backend, ChannelDeclaration(channel_id="agent", recent_events_window=0))
    answers = (AIResponse(content="one"), AIResponse(content="two"), CALL, AIResponse(content="d"))
    agent = _agent("agent", *answers)
    executor = TurnExecutor(lambda _c: agent, hosts={"agent"}, executor_id="elsewhere")
    seen: list[int] = []

    async def hook(_event: Any, context: Any) -> HookResult:
        seen.append(len(context.recent_events))
        return HookResult.allow()

    executor.hook(HookTrigger.BEFORE_TOOL_USE, name="count")(hook)
    await backend.serve(executor)

    await _say(kit, "first", 1)
    await _say(kit, "second", 2)
    await _say(kit, "third", 3)

    assert seen and seen[0] >= 5, "the two exchanges and the message that asked"
    await kit.close()
    await executor.close()


async def test_a_steer_during_the_room_s_read_waits_for_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InMemoryTurnBackend()
    kit = await _kit(backend, ChannelDeclaration(channel_id="agent"))
    agent = _agent("agent", AIResponse(content="never shown"))
    executor = TurnExecutor(lambda _c: agent, hosts={"agent"}, executor_id="elsewhere")
    await backend.serve(executor)
    host = kit._turns._host
    original = host.channel_rows
    reading, go_on = asyncio.Event(), asyncio.Event()

    async def held(*args: Any) -> Any:
        reading.set()
        await go_on.wait()
        return await original(*args)

    monkeypatch.setattr(host, "channel_rows", held)
    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="you", sender_id="you", content=TextContent(body="go")),
            room_id="r",
        )
    )
    await asyncio.wait_for(reading.wait(), 5)
    steered = asyncio.create_task(kit.steer("r", "agent", Cancel()))
    await asyncio.sleep(0.01)
    assert not steered.done(), "a steer waits for a turn still being handed over"
    go_on.set()
    await asyncio.wait_for(steered, 5)
    await asyncio.wait_for(turn, 5)
    await kit.close()
    await executor.close()


async def test_a_store_that_fails_costs_the_turn_its_rows_not_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Seen()
    kit = await _kit(backend, ChannelDeclaration(channel_id="agent"))
    agent = _agent("agent", AIResponse(content="still here"))
    executor = TurnExecutor(lambda _c: agent, hosts={"agent"}, executor_id="elsewhere")
    await backend.serve(executor)

    async def broken(*_args: Any) -> Any:
        raise OSError("the store is down")

    monkeypatch.setattr(kit._turns._host, "channel_rows", broken)

    await _say(kit, "hello", 1)

    assert (await _answers(kit))[0].content.body == "still here"
    assert backend.requests[-1].channel_rows == []
    await kit.close()
    await executor.close()


async def test_a_new_executor_keeps_the_latest_calls_and_replays_only_the_history() -> None:
    backend = InMemoryTurnBackend()
    kit = await _kit(backend, ChannelDeclaration(channel_id="agent"))
    answers = [CALL, AIResponse(content="done"), *[AIResponse(content="a")] * EXCHANGES]
    first = _agent("agent", *answers)
    one = TurnExecutor(lambda _c: first, hosts={"agent"}, executor_id="one")
    await backend.serve(one)
    for n in range(EXCHANGES + 1):
        await _say(kit, f"message {n}", n + 1)
    await one.close()
    provider = MockAIProvider(ai_responses=[AIResponse(content="ok")])
    second = Agent("agent", provider=provider, tools=[LOOKUP], tool_handler=first.tool_handler)
    two = TurnExecutor(lambda _c: second, hosts={"agent"}, executor_id="two")
    await backend.serve(two)

    await _say(kit, "what now?", EXCHANGES + 2)

    assert SECRET not in provider.calls[-1].model_dump_json(), "older than the turn's history"
    assert "lookup" in second._tool_usage.tool_names("r"), "the latest calls still rebuild"
    await kit.close()
    await two.close()


async def test_the_room_sends_what_the_channel_read_just_before_the_history() -> None:
    store = InMemoryStore()
    await store.create_room(Room(id="r"))

    async def add(kind: EventType, channel: str) -> RoomEvent:
        source = EventSource(channel_id=channel, channel_type=ChannelType.AI)
        event = RoomEvent(room_id="r", type=kind, source=source, content=TextContent(body="x"))
        return await store.add_event_auto_index("r", event)

    await add(EventType.MESSAGE, "you")
    read = await add(EventType.INSTRUCTION_READ, "agent")
    first = await add(EventType.MESSAGE, "you")

    rows = await read_channel_rows(store, "r", "agent", first.index)

    assert [row.id for row in rows] == [read.id]


async def test_the_room_reads_from_where_the_turn_s_history_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InMemoryTurnBackend()
    kit = await _kit(backend, ChannelDeclaration(channel_id="agent"))
    agent = _agent("agent", AIResponse(content="done"))
    executor = TurnExecutor(lambda _c: agent, hosts={"agent"}, executor_id="elsewhere")
    await backend.serve(executor)
    asked: list[int | None] = []

    async def channel_rows(_room: str, _channel: str, first: int | None) -> list[RoomEvent]:
        asked.append(first)
        return []

    monkeypatch.setattr(kit._turns._host, "channel_rows", channel_rows)
    monkeypatch.setattr("roomkit.turns.owner.history_start", lambda _event, _context: 7)

    await _say(kit, "hello", 1)

    assert asked == [7]
    await kit.close()
    await executor.close()

"""The room owner controls a turn across the turn backend (RFC §10.6.8):
its cancellation says why, and steering reaches the turn's loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import RoomKit, current_turn_id
from roomkit.channels.agent import Agent
from roomkit.channels.base import Channel
from roomkit.core.exceptions import TurnLinkError
from roomkit.core.task_utils import cancel_reason
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.models.steering import Cancel, InjectMessage, SteeringDirective
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor, TurnTimeouts
from roomkit.turns._error_codec import ROOMKIT_ERRORS
from roomkit.turns._executor_turn import ExecutorTurn
from roomkit.turns._owner_turn import OwnerTurn
from roomkit.turns.owner import turn_request
from tests.test_framework import SimpleChannel

WAIT = AITool(name="wait_here", description="Wait.", parameters={"type": "object"})


class Paused:
    """A host tool that waits until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.started.set()
        await self.release.wait()
        return "waited"


async def _kit(tool: Paused, *answers: str) -> tuple[RoomKit, Agent]:
    call = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="wait_here", arguments={})])
    provider = MockAIProvider(ai_responses=[call, *(AIResponse(content=a) for a in answers)])
    agent = Agent("agent", provider=provider, role="agent", tools=[WAIT], tool_handler=tool)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, agent


def _message(body: str = "go") -> InboundMessage:
    return InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body))


# -- Cancellation ---------------------------------------------------------------


async def test_a_cancelled_delivery_tells_the_executor_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasons: list[str] = []
    cut = ExecutorTurn._cut_turn

    def spy(self: ExecutorTurn, reason: str) -> None:
        reasons.append(reason)
        cut(self, reason)

    monkeypatch.setattr(ExecutorTurn, "_cut_turn", spy)
    tool = Paused()
    kit, _ = await _kit(tool, "never")
    first = await kit.process_inbound(_message(), room_id="r1", defer_delivery=True)
    await asyncio.wait_for(tool.started.wait(), 2)

    assert first.delivery is not None
    await first.delivery.cancel(reason="superseded")

    assert "superseded" in reasons
    await kit.close()


# -- Steering through the kit ---------------------------------------------------


async def test_a_directive_for_a_room_reaches_its_turns_loop() -> None:
    tool = Paused()
    kit, agent = await _kit(tool, "done")
    turn = asyncio.create_task(kit.process_inbound(_message(), room_id="r1"))
    await asyncio.wait_for(tool.started.wait(), 2)

    reached = await kit.steer("r1", "agent", InjectMessage(content="Mind the budget."))
    tool.release.set()
    await asyncio.wait_for(turn, 5)

    assert reached == 1
    # The next round reads the injected message.
    second = agent.provider.calls[1]
    assert any("Mind the budget." in str(m.content) for m in second.messages)
    await kit.close()


async def test_a_cancel_for_a_room_ends_its_turn_before_another_round() -> None:
    tool = Paused()
    kit, agent = await _kit(tool, "never asked")
    turn = asyncio.create_task(kit.process_inbound(_message(), room_id="r1"))
    await asyncio.wait_for(tool.started.wait(), 2)

    reached = await kit.steer("r1", "agent", Cancel(reason="stop"))
    tool.release.set()
    await asyncio.wait_for(turn, 5)

    assert reached == 1
    assert len(agent.provider.calls) == 1
    await kit.close()


async def test_a_directive_for_a_room_with_no_turn_reaches_nothing() -> None:
    kit, _ = await _kit(Paused())

    assert await kit.steer("r1", "agent", Cancel()) == 0
    assert await kit.steer("r1", "agent", Cancel(), turn_id="t1") == 0
    await kit.close()


# -- Steering one turn, on a channel with or without steering ------------------


class Held(Channel):
    """An intelligence channel whose stream waits until released; with
    ``steers``, a steerable one recording what reached it."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self) -> None:
        super().__init__("agent")
        self.release = asyncio.Event()
        self.steered: list[tuple[SteeringDirective, str | None]] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(responded=True, response_stream=self._stream())

    async def _stream(self) -> AsyncIterator[str]:
        yield "first"
        await self.release.wait()
        yield "second"

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class Steerable(Held):
    def steer(self, directive: SteeringDirective, *, turn_id: str | None = None) -> int:
        self.steered.append((directive, turn_id))
        return 1


def _client(channel: Channel) -> TurnClient:
    channels = {channel.channel_id: channel}
    return TurnClient(
        InMemoryTurnBackend(),
        executor=TurnExecutor(channels.get, hosts=channels.__contains__),
    )


async def _start(client: TurnClient) -> ChannelOutput:
    event = RoomEvent(
        room_id="room",
        source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
        content=TextContent(body="hello"),
    )
    binding = ChannelBinding(
        channel_id="agent",
        room_id="room",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    return await client.run(event, binding, RoomContext(room=Room(id="room")))


async def test_a_directive_for_a_turn_reaches_that_turns_loop_even_unread() -> None:
    channel = Steerable()
    client = _client(channel)
    output = await _start(client)
    turn_id = client._started[0].request.turn_id

    # Nobody reads the stream: the owner still hears the turn's answer.
    reached = await asyncio.wait_for(
        client.steer("room", "agent", InjectMessage(content="x"), turn_id=turn_id), 2
    )

    assert reached == 1
    assert channel.steered == [(InjectMessage(content="x"), turn_id)]
    channel.release.set()
    assert output.response_stream is not None
    assert "".join([item async for item in output.response_stream]) == "firstsecond"
    await client.close()


async def test_a_turn_client_acting_for_an_organization_reaches_no_turn_outside_it() -> None:
    """RFC §17.2: given an organization, a turn client reaches only the
    turns of that organization's rooms; a turn of a room outside it reads as
    one that does not run."""
    channel = Steerable()
    client = _client(channel)
    output = await _start(client)

    assert await client.steer("room", "agent", Cancel(), organization_id="globex") == 0

    assert channel.steered == []
    channel.release.set()
    assert output.response_stream is not None
    assert "".join([item async for item in output.response_stream]) == "firstsecond"
    await client.close()


async def test_a_cancel_for_a_channel_without_steering_cuts_its_turn() -> None:
    channel = Held()
    client = _client(channel)
    output = await _start(client)
    assert output.response_stream is not None
    stream = output.response_stream
    assert await anext(stream) == "first"

    reached = await asyncio.wait_for(client.steer("room", "agent", Cancel()), 2)
    # The turn ends as a closed stream would: no error for its reader.
    rest = [item async for item in stream]

    assert reached == 1
    assert rest == []
    await client.close()


class Finalizing(Held):
    """A channel without steering whose stream's cleanup takes a while."""

    def __init__(self) -> None:
        super().__init__()
        self.unwinding = asyncio.Event()
        self.finished = asyncio.Event()

    async def _stream(self) -> AsyncIterator[str]:
        try:
            yield "first"
            await asyncio.Future()
        finally:
            self.unwinding.set()
            await self.release.wait()
            self.finished.set()


async def test_a_close_after_a_cut_lets_the_channel_finish_its_cleanup() -> None:
    channel = Finalizing()
    client = _client(channel)
    output = await _start(client)
    assert output.response_stream is not None
    stream = output.response_stream
    assert await anext(stream) == "first"

    await asyncio.wait_for(client.steer("room", "agent", Cancel()), 2)
    await asyncio.wait_for(channel.unwinding.wait(), 2)
    # A close reaching the executor while the cut's cleanup runs does not
    # cancel the cleanup again: the channel's finalizer runs to its end.
    closing = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0.05)
    channel.release.set()
    await asyncio.wait_for(closing, 2)

    assert channel.finished.is_set()
    await client.close()


class FinalizingAfterMarker(Held):
    """A channel without steering whose stream ends its reads on a marker,
    then takes a while to clean up."""

    def __init__(self) -> None:
        super().__init__()
        self.unwinding = asyncio.Event()
        self.finished = asyncio.Event()

    async def _stream(self) -> AsyncIterator[Any]:
        try:
            yield ToolCallStartMarker(tool_name="t", tool_id="c1")
            yield "more"
        finally:
            self.unwinding.set()
            await self.release.wait()
            self.finished.set()


async def test_a_cut_after_a_close_lets_the_channel_finish_its_cleanup() -> None:
    channel = FinalizingAfterMarker()
    client = _client(channel)
    output = await _start(client)
    assert output.response_stream is not None
    stream = output.response_stream
    await anext(stream)

    # The close finds no read under way: it starts the channel's cleanup.
    closing = asyncio.create_task(stream.aclose())
    await asyncio.wait_for(channel.unwinding.wait(), 2)
    await asyncio.wait_for(client.steer("room", "agent", Cancel()), 2)
    await asyncio.sleep(0.05)
    channel.release.set()
    await asyncio.wait_for(closing, 2)

    assert channel.finished.is_set()
    await client.close()


async def test_a_directive_other_than_cancel_reaches_no_channel_without_steering() -> None:
    channel = Held()
    client = _client(channel)
    output = await _start(client)

    assert await client.steer("room", "agent", InjectMessage(content="x")) == 0
    channel.release.set()
    assert output.response_stream is not None
    assert "".join([item async for item in output.response_stream]) == "firstsecond"
    await client.close()


# -- What the review of the first steering commit found -------------------------


class Rendering(SimpleChannel):
    """A streaming transport that renders everything it is handed."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self, stream: Any, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        async for _ in stream:
            pass
        return ChannelOutput.empty()


async def test_a_cancelled_delivery_streamed_to_a_transport_tells_the_executor_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasons: list[str] = []
    cut = ExecutorTurn._cut_turn

    def spy(self: ExecutorTurn, reason: str) -> None:
        reasons.append(reason)
        cut(self, reason)

    monkeypatch.setattr(ExecutorTurn, "_cut_turn", spy)
    tool = Paused()
    call = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="wait_here", arguments={})])
    provider = MockAIProvider(ai_responses=[call, AIResponse(content="never")])
    agent = Agent("agent", provider=provider, role="agent", tools=[WAIT], tool_handler=tool)
    kit = RoomKit()
    kit.register_channel(Rendering("ops"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    first = await kit.process_inbound(_message(), room_id="r1", defer_delivery=True)
    await asyncio.wait_for(tool.started.wait(), 2)

    assert first.delivery is not None
    await first.delivery.cancel(reason="superseded")

    assert reasons and set(reasons) == {"superseded"}
    await kit.close()


async def test_a_turn_addressed_by_the_id_its_tool_reads() -> None:
    tool = Paused()
    turn_ids: list[str | None] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        turn_ids.append(current_turn_id())
        return await tool(name, arguments)

    kit, agent = await _kit(tool, "done")
    agent._user_tool_handler = handler  # type: ignore[assignment]
    turn = asyncio.create_task(kit.process_inbound(_message(), room_id="r1"))
    await asyncio.wait_for(tool.started.wait(), 2)

    [turn_id] = turn_ids
    assert turn_id is not None
    # A turn is addressed within its room: named with another, it is not reached.
    assert await kit.steer("r2", "agent", InjectMessage(content="For r2."), turn_id=turn_id) == 0
    reached = await kit.steer(
        "r1", "agent", InjectMessage(content="Mind the budget."), turn_id=turn_id
    )
    tool.release.set()
    await asyncio.wait_for(turn, 5)

    assert reached == 1
    assert any("Mind the budget." in str(m.content) for m in agent.provider.calls[1].messages)
    await kit.close()


async def test_a_cancel_reaches_every_turn_of_the_room_another_directive_the_latest() -> None:
    channel = Steerable()
    client = _client(channel)
    first = await _start(client)
    second = await _start(client)
    turn_ids = [turn.request.turn_id for turn in client._started]

    assert await client.steer("room", "agent", InjectMessage(content="x")) == 1
    assert channel.steered[-1][1] == turn_ids[1]
    assert await client.steer("room", "agent", Cancel()) == 2
    channel.release.set()
    for output in (first, second):
        assert output.response_stream is not None
        await output.response_stream.aclose()
    await client.close()


class OldSteerable(Held):
    """A steer() of the channel's own, from before turns were addressed."""

    def steer(
        self,
        directive: SteeringDirective,
        *,
        loop_id: str | None = None,
        room_id: str | None = None,
    ) -> int:
        self.steered.append((directive, room_id))
        return 1


class BrokenSteerable(Held):
    def steer(self, directive: SteeringDirective, *, turn_id: str | None = None) -> int:
        raise RuntimeError("steering broke")


async def test_a_steer_of_the_channels_own_older_shape_takes_the_turns_room() -> None:
    channel = OldSteerable()
    client = _client(channel)
    output = await _start(client)

    assert await asyncio.wait_for(client.steer("room", "agent", Cancel()), 2) == 1
    assert channel.steered == [(Cancel(), "room")]
    # The turn's controls still run: its stream reads to its end.
    channel.release.set()
    assert output.response_stream is not None
    assert "".join([item async for item in output.response_stream]) == "firstsecond"
    await client.close()


async def test_a_steer_that_fails_reaches_nothing_and_the_turn_goes_on() -> None:
    channel = BrokenSteerable()
    client = _client(channel)
    output = await _start(client)

    assert await asyncio.wait_for(client.steer("room", "agent", Cancel()), 2) == 0
    channel.release.set()
    assert output.response_stream is not None
    assert "".join([item async for item in output.response_stream]) == "firstsecond"
    await client.close()


class HeldInACall(Held):
    """A stream that starts a call, then waits."""

    async def _stream(self) -> AsyncIterator[Any]:
        yield ToolCallStartMarker(tool_name="lookup", tool_id="c1", arguments={"q": "x"})
        await self.release.wait()


async def test_a_call_a_steering_cancel_left_open_ends_cancelled() -> None:
    channel = HeldInACall()
    client = _client(channel)
    output = await _start(client)
    assert output.response_stream is not None
    stream = output.response_stream
    start = await anext(stream)

    assert await asyncio.wait_for(client.steer("room", "agent", Cancel()), 2) == 1
    rest = [item async for item in stream]

    [end] = rest
    assert isinstance(end, ToolCallEndMarker)
    assert (end.tool_id, end.outcome, end.status) == ("c1", "cancelled", "failed")
    assert start.ended is end
    await client.close()


class _BrokenLink:
    """A link whose transport breaks on the first read."""

    def __init__(self, request: Any) -> None:
        self.request = request
        self.taken = True

    async def receive(self) -> Any:
        raise RuntimeError("transport broke")

    async def send(self, message: Any) -> None:
        return None

    def release(self) -> None:
        return None


async def test_a_turn_whose_frames_cannot_be_read_fails_rather_than_hangs() -> None:
    request = turn_request(
        RoomEvent(
            room_id="room",
            source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
            content=TextContent(body="hello"),
        ),
        ChannelBinding(
            channel_id="agent",
            room_id="room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
        RoomContext(room=Room(id="room")),
    )
    turn = OwnerTurn(_BrokenLink(request), TurnTimeouts(), ROOMKIT_ERRORS)  # type: ignore[arg-type]

    with pytest.raises(TurnLinkError) as raised:
        await asyncio.wait_for(turn.output(), 2)

    assert raised.value.reason == "frame_invalid"


async def test_a_call_left_open_ends_cancelled_for_a_reader_that_comes_late() -> None:
    channel = HeldInACall()
    client = _client(channel)
    output = await _start(client)
    assert output.response_stream is not None
    stream = output.response_stream
    start = await anext(stream)

    assert await asyncio.wait_for(client.steer("room", "agent", Cancel()), 2) == 1
    async with asyncio.timeout(2):
        while not client._started[0].over:
            await asyncio.sleep(0.01)
    # The reader was busy elsewhere while the turn ended.
    rest = [item async for item in stream]

    [end] = rest
    assert isinstance(end, ToolCallEndMarker)
    assert start.ended is end
    await client.close()


async def test_a_turn_cut_while_its_channel_is_built_never_runs() -> None:
    built = asyncio.Event()
    ran: list[str] = []

    class Counting(Held):
        async def on_event(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            ran.append("on_event")
            return await super().on_event(event, binding, context)

    channel = Counting()

    async def build(channel_id: str) -> Channel | None:
        await built.wait()
        return channel

    client = TurnClient(
        InMemoryTurnBackend(),
        executor=TurnExecutor(build, hosts={"agent"}.__contains__),
    )
    started = asyncio.create_task(_start(client))
    async with asyncio.timeout(2):
        while not client._started:
            await asyncio.sleep(0.01)

    assert await asyncio.wait_for(client.steer("room", "agent", Cancel()), 2) == 1
    built.set()
    output = await asyncio.wait_for(started, 2)

    assert ran == []
    assert output.response_stream is None
    await client.close()


async def test_the_turns_own_work_reads_why_it_was_cancelled() -> None:
    reasons: list[str | None] = []
    begun = asyncio.Event()

    class Waiting(Held):
        async def on_event(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            begun.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cut:
                reasons.append(cancel_reason(cut))
                raise
            return ChannelOutput.empty()

    client = _client(Waiting())
    turn = asyncio.create_task(_start(client))
    await asyncio.wait_for(begun.wait(), 2)

    turn.cancel("superseded")
    await asyncio.gather(turn, return_exceptions=True)

    assert reasons == ["superseded"]
    await client.close()


class _AckedLate(InMemoryTurnBackend):
    """A backend that, as a distributed one, hands a turn over (*handed*) and
    then waits for its acknowledgement (*acked*): the turn already runs
    meanwhile."""

    def __init__(self) -> None:
        super().__init__()
        self.handed, self.acked = asyncio.Event(), asyncio.Event()

    async def submit(self, request: Any) -> Any:
        link = await super().submit(request)
        self.handed.set()
        await self.acked.wait()
        return link


async def test_a_steer_reaches_a_turn_still_being_handed_over() -> None:
    channel = Steerable()
    channels = {channel.channel_id: channel}
    backend = _AckedLate()
    client = TurnClient(backend, executor=TurnExecutor(channels.get, hosts=channels.__contains__))
    output = asyncio.ensure_future(_start(client))
    await asyncio.wait_for(backend.handed.wait(), 2)  # handed over, not yet acknowledged

    steering = asyncio.ensure_future(client.steer("room", "agent", InjectMessage(content="x")))
    for _ in range(5):  # the steer under way while the handover waits
        await asyncio.sleep(0)
    backend.acked.set()
    reached = await asyncio.wait_for(steering, 2)

    assert reached == 1
    channel.release.set()
    await output
    await client.close()

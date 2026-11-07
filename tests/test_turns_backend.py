"""The turn backend carries a turn as serialized messages (RFC §10.6).

These drive a turn through :class:`TurnClient`, the in-memory backend and an
executor directly: what the owner reads is the channel's turn, rebuilt from
frames, never one of the channel's own objects.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit.channels._strategy_turn import in_strategy_turn, strategy_turn
from roomkit.channels._tool_registry import (
    _TURN_OWNER_TOOLS,
    TurnOwnerTools,
    set_turn_owner_tools,
)
from roomkit.channels.base import Channel
from roomkit.core.exceptions import RemoteTurnError, ToolTimeoutError, TurnLinkError
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.room import Room
from roomkit.models.streaming import (
    InstructionReadMarker,
    LoopEndMarker,
    SegmentBreakMarker,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.providers.ai.base import AIImagePart, AITextPart, AITool
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor, TurnTimeouts
from roomkit.turns._codec import decode_item, encode_item, request_json
from roomkit.turns._error_codec import ROOMKIT_ERRORS, error_record
from roomkit.turns._owner_stream import StreamCopies
from roomkit.turns.base import MessageOrder, OwnerTools, TurnBackend
from roomkit.turns.owner import turn_request
from roomkit.turns.wire import CommittedEvent, OwnerTool, TurnRequest
from tests.turn_errors import same_error


class _ForeignError(Exception):
    """An exception class the owner does not rebuild by type."""


class ScriptedChannel(Channel):
    """An intelligence channel whose stream runs a script given per test."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, script: Any = None, *, raises: BaseException | None = None) -> None:
        super().__init__("agent")
        self.script = script
        self.raises = raises
        self.record = ResponseMetadata()
        self.yielded: list[Any] = []
        self.stream_started = False
        self.stream_closed = False
        self.on_event_task: asyncio.Task[Any] | None = None

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.on_event_task = asyncio.current_task()
        if self.raises is not None:
            raise self.raises
        if self.script is None:
            return ChannelOutput(responded=True, response_metadata=self.record)
        return ChannelOutput(
            responded=True, response_stream=self._stream(), response_metadata=self.record
        )

    async def _stream(self) -> AsyncIterator[Any]:
        self.stream_started = True
        try:
            async for item in self.script(self):
                self.yielded.append(item)
                yield item
        finally:
            self.stream_closed = True

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


def _event() -> RoomEvent:
    return RoomEvent(
        room_id="room",
        source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
        content=TextContent(body="hello"),
    )


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="agent",
        room_id="room",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _context() -> RoomContext:
    return RoomContext(room=Room(id="room"), bindings=[_binding()])


def _client(channel: Channel | None, timeouts: TurnTimeouts | None = None) -> TurnClient:
    channels = {} if channel is None else {channel.channel_id: channel}
    return TurnClient(
        InMemoryTurnBackend(),
        executor=TurnExecutor(channels.get, hosts=channels.__contains__, timeouts=timeouts),
        timeouts=timeouts,
    )


async def _run(client: TurnClient) -> ChannelOutput:
    return await client.run(_event(), _binding(), _context())


async def _read_all(output: ChannelOutput) -> list[Any]:
    return [item async for item in output.response_stream]


# -- The wire ---------------------------------------------------------------


@pytest.mark.parametrize(
    "item",
    [
        "a text delta",
        InstructionReadMarker(text="Introduce yourself."),
        ThinkingDeltaMarker(thinking="hmm"),
        ToolCallStartMarker(tool_name="search", tool_id="c1", arguments={"q": "x"}),
        ToolCallStartMarker(
            tool_name="search", tool_id="c1", arguments={"q": "x"}, intent="I search."
        ),
        ToolCallEndMarker(
            tool_name="search",
            tool_id="c1",
            arguments={"q": "x"},
            result=[AITextPart(text="found"), AIImagePart(url="data:image/png;base64,AA")],
            status="completed",
            duration_ms=12,
            structured_content={"hits": 1},
            outcome="served",
            refused_but_ran=True,
            intent="I search.",
        ),
        SegmentBreakMarker(),
        LoopEndMarker(reason="max_rounds", rounds=3, usage={"input_tokens": 5}, max_rounds=3),
    ],
)
def test_every_stream_item_crosses_as_a_new_equal_object(item: Any) -> None:
    rebuilt = decode_item(encode_item(item))
    assert rebuilt == item
    if not isinstance(item, str):
        assert rebuilt is not item
    if getattr(item, "intent", "") is None:
        # A call without an intent has no ``intent`` key on the wire (RFC §6.4).
        assert "intent" not in encode_item(item)


def test_a_call_its_turn_left_open_ends_cancelled_as_it_started() -> None:
    """The owner closes it with its arguments and its intent (RFC §6.4)."""
    copies = StreamCopies()
    start = ToolCallStartMarker(
        tool_name="search", tool_id="c1", arguments={"q": "x"}, intent="I search."
    )
    copies.item(1, encode_item(start))

    [end] = copies.open_call_ends()

    assert (end.arguments, end.intent, end.outcome) == ({"q": "x"}, "I search.", "cancelled")


def test_a_stream_item_the_protocol_cannot_carry_is_refused() -> None:
    with pytest.raises(TypeError, match="cannot carry"):
        encode_item(object())


def test_a_roomkit_error_is_rebuilt_by_type_with_its_attributes_and_causes() -> None:
    cause = ValueError("bad argument")
    error = ToolTimeoutError("search", 3.0)
    error.__cause__ = cause
    rebuilt = ROOMKIT_ERRORS.rebuild(error_record(error))
    assert same_error(rebuilt, error)
    assert type(rebuilt.__cause__) is ValueError
    assert str(rebuilt.__cause__) == "bad argument"


def test_an_error_of_a_foreign_class_comes_back_naming_it() -> None:
    rebuilt = ROOMKIT_ERRORS.rebuild(error_record(_ForeignError("boom")))
    assert isinstance(rebuilt, RemoteTurnError)
    assert rebuilt.kind.endswith(":_ForeignError")
    assert str(rebuilt) == "boom"


def test_a_duplicate_is_dropped_and_a_gap_fails_the_turn() -> None:
    order = MessageOrder()
    assert order.admit(1)
    assert not order.admit(1)
    assert order.admit(2)
    with pytest.raises(TurnLinkError) as raised:
        order.admit(4)
    assert raised.value.reason == "frame_lost"


class _Opaque:
    """A value JSON has no form for."""


def _opaque_request() -> TurnRequest:
    """A request whose event and room each keep a value JSON has no form for,
    the event in the room's history too."""
    event = _event().model_copy(update={"metadata": {"handle": _Opaque()}})
    room = Room(id="room", metadata={"session": _Opaque()})
    context = RoomContext(room=room, bindings=[_binding()], recent_events=[event])
    return turn_request(event, _binding(), context)


def test_a_request_value_json_has_no_form_for_crosses_as_its_text() -> None:
    """An application in the owner's process keeps an object in a room's
    metadata or an event's: the request's wire form, which a distributed
    backend carries, writes it as its text, as the record does."""
    crossed = TurnRequest.model_validate_json(request_json(_opaque_request()))

    assert crossed.event.metadata["handle"].startswith("<")
    assert crossed.context.room.metadata["session"].startswith("<")
    assert crossed.context.recent_events[0].metadata["handle"].startswith("<")


# -- A turn through the in-memory backend -----------------------------------


class _ReadsTheRequest(ScriptedChannel):
    """Keeps what its turn read of the event and the room, and changes the
    room and the binding it was given."""

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> ChannelOutput:
        self.read = (event, context)
        context.room.metadata["changed"] = True
        binding.metadata["changed"] = True
        return await super().on_event(event, binding, context)


async def test_an_in_memory_turn_reads_the_request_s_events_as_they_are() -> None:
    """RMK-985: the event the turn answers and the room's history are records
    nothing changes once made (RFC §10.6.1, §14.4): in memory they reach the
    executor as the owner holds them, a value JSON has no form for included,
    and the rest of the request crosses in its wire form."""
    request = _opaque_request()
    channel = _ReadsTheRequest()

    output = await _client(channel).run(request.event, request.binding, request.context)

    read_event, read_context = channel.read
    assert output.responded
    assert read_event is request.event
    assert read_context.recent_events[0] is request.event
    assert read_context.room.metadata["session"].startswith("<")


async def test_the_executor_changes_its_own_copy_of_the_room_and_binding() -> None:
    """RMK-985: what crosses in the wire form is the executor's own: a channel
    that changes the room or the binding its turn read changes nothing of
    the owner's."""
    channel = _ReadsTheRequest()
    binding = _binding()
    context = RoomContext(room=Room(id="room"), bindings=[binding])

    await _client(channel).run(_event(), binding, context)

    assert (context.room.metadata, binding.metadata) == ({}, {})


async def test_the_owner_reads_new_objects_never_the_channels() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "Hello"
        yield ToolCallStartMarker(tool_name="t", tool_id="c1", arguments={})
        yield LoopEndMarker(reason="completed")

    channel = ScriptedChannel(script)
    output = await _run(_client(channel))
    items = await _read_all(output)
    assert items == channel.yielded
    assert all(a is not b for a, b in zip(items[1:], channel.yielded[1:], strict=True))
    assert output.response_metadata is not channel.record


async def test_a_turn_says_whether_a_strategy_asked_it_and_never_inherits_it() -> None:
    class Telling(ScriptedChannel):
        def __init__(self) -> None:
            super().__init__()
            self.asked: list[bool] = []

        async def on_event(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            self.asked.append(in_strategy_turn())
            return await super().on_event(event, binding, context)

    channel = Telling()
    client = _client(channel)
    await client.run(_event(), _binding(), _context(), strategy_turn=True)
    # A turn submitted from inside a strategy's generation runs in a copy of
    # that context, and is still a turn of its own.
    with strategy_turn(True):
        await _run(client)
    assert channel.asked == [True, False]


class _Declares(OwnerTools):
    """An owner that serves the tools it is told to name."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def declared(self, channel_id: str, room_id: str) -> list[OwnerTool]:
        return [OwnerTool(tool=AITool(name=n, description=n)) for n in self.names]

    async def serve(self, request: Any, call: Any) -> tuple[Any, None]:
        return "served", None


async def test_a_turn_reads_the_owner_tools_of_its_own_request() -> None:
    class Telling(ScriptedChannel):
        def __init__(self) -> None:
            super().__init__()
            self.declared: list[list[str]] = []

        async def on_event(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            tools = _TURN_OWNER_TOOLS.get()
            self.declared.append(sorted(tools.entries) if tools is not None else [])
            return await super().on_event(event, binding, context)

    channel = Telling()
    owner = _Declares()
    client = TurnClient(
        InMemoryTurnBackend(),
        executor=TurnExecutor({"agent": channel}.get, hosts={"agent"}.__contains__),
        tools=owner,
    )
    owner.names = ["probe"]
    await _run(client)
    # Submitted from inside another turn, whose context it copies.
    owner.names = []
    inside = contextvars.copy_context()
    inside.run(set_turn_owner_tools, TurnOwnerTools("agent", "room", {"stale": None}))  # type: ignore[dict-item]
    await asyncio.get_running_loop().create_task(_run(client), context=inside)
    assert channel.declared == [["probe"], []]


async def test_an_error_on_event_raised_is_raised_on_the_owner() -> None:
    channel = ScriptedChannel(raises=_ForeignError("provider down"))
    with pytest.raises(RemoteTurnError, match="provider down"):
        await _run(_client(channel))


async def test_a_stream_that_raises_raises_on_the_owner_after_its_items() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "partial"
        raise ValueError("cut")

    output = await _run(_client(ScriptedChannel(script)))
    stream = output.response_stream
    assert await anext(stream) == "partial"
    with pytest.raises(ValueError, match="cut"):
        await anext(stream)


async def test_a_stream_closed_unread_never_starts() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "never"

    channel = ScriptedChannel(script)
    output = await _run(_client(channel))
    await output.response_stream.aclose()
    assert not channel.stream_started


async def test_the_executor_reads_past_no_item_but_a_delta_until_the_owner_pulls() -> None:
    """A pull grants several items, and deltas run ahead of their reader; past
    an item that is no delta (a call's start), the executor waits for the
    reader to pull again (RFC §10.6.4)."""
    reached: list[str] = []

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        reached.append("first")
        yield ToolCallStartMarker(tool_name="t", tool_id="c1", arguments={})
        reached.append("second")
        yield "two"

    channel = ScriptedChannel(script)
    output = await _run(_client(channel))
    assert isinstance(await anext(output.response_stream), ToolCallStartMarker)
    await asyncio.sleep(0.05)
    assert reached == ["first"]
    await output.response_stream.aclose()
    assert channel.stream_closed
    assert reached == ["first"]


async def test_a_record_written_mid_stream_reaches_the_owner_before_the_next_item() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "before"
        channel.record["acp"] = {"stop_reason": None}
        channel.record["acp"]["stop_reason"] = "max_tokens"
        yield "after"

    output = await _run(_client(ScriptedChannel(script)))
    stream = output.response_stream
    await anext(stream)
    assert "acp" not in output.response_metadata
    await anext(stream)
    assert output.response_metadata["acp"] == {"stop_reason": "max_tokens"}


async def test_fixed_arguments_and_an_early_end_reach_the_owners_start() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        start = ToolCallStartMarker(tool_name="t", tool_id="c1", arguments={"q": "raw"})
        yield start
        start.ran_with = {"q": "fixed"}
        end = ToolCallEndMarker(tool_name="t", tool_id="c1", arguments={"q": "fixed"}, result="ok")
        start.ended = end
        yield "text after the round"
        yield end

    output = await _run(_client(ScriptedChannel(script)))
    stream = output.response_stream
    start = await anext(stream)
    await anext(stream)
    assert start.ran_with == {"q": "fixed"}
    assert start.ended is not None and start.ended.result == "ok"
    end = await anext(stream)
    assert end is start.ended


async def test_a_call_that_ended_before_a_close_keeps_its_end() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        start = ToolCallStartMarker(tool_name="t", tool_id="c1", arguments={})
        yield start
        start.ended = ToolCallEndMarker(tool_name="t", tool_id="c1", result="done")
        yield "more"

    output = await _run(_client(ScriptedChannel(script)))
    stream = output.response_stream
    start = await anext(stream)
    await anext(stream)
    await stream.aclose()
    assert start.ended is not None and start.ended.result == "done"


async def test_cancelling_the_owner_cancels_the_turns_work() -> None:
    started = asyncio.Event()

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        started.set()
        await asyncio.Event().wait()
        yield "never"

    channel = ScriptedChannel(script)
    output = await _run(_client(channel))
    reader = asyncio.create_task(anext(output.response_stream))
    await started.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    assert channel.stream_closed


async def test_a_turn_no_executor_takes_fails_no_executor() -> None:
    timeouts = TurnTimeouts(accept=0.2, heartbeat_interval=0.05, heartbeat_timeout=0.2)
    with pytest.raises(TurnLinkError) as raised:
        await _run(_client(None, timeouts))
    assert raised.value.reason == "no_executor"


async def test_an_event_no_executor_takes_names_the_room_hooks_it_needs(
    turn_backend: TurnBackend,
) -> None:
    """RMK-883: a committed event whose room hooks no executor defines is
    refused as a turn is, naming them, on either backend."""
    executor = TurnExecutor({"agent": None}.get, hosts={"agent"})
    await turn_backend.serve(executor)
    event = CommittedEvent(
        channel_id="agent", room_id="room", event=_event(), room_hooks=["audit"]
    )

    with pytest.raises(TurnLinkError) as raised:
        await turn_backend.control(event)
    await executor.close()

    assert raised.value.reason == "no_executor"
    assert str(raised.value) == "No executor serves channel 'agent' with room hooks ['audit']"


async def test_a_silent_executor_loses_its_turn() -> None:
    timeouts = TurnTimeouts(accept=0.05, heartbeat_interval=0.01, heartbeat_timeout=0.04)

    class Mute(TurnExecutor):
        def take(self, link: Any) -> None:
            pass  # takes the turn and never answers

    channels: dict[str, Channel] = {"agent": ScriptedChannel()}
    client = TurnClient(InMemoryTurnBackend(), executor=Mute(channels.get), timeouts=timeouts)
    with pytest.raises(TurnLinkError) as raised:
        await _run(client)
    assert raised.value.reason == "executor_lost"


async def test_a_turn_runs_on_a_task_of_its_own() -> None:
    channel = ScriptedChannel()
    caller = asyncio.current_task()
    await _run(_client(channel))
    assert channel.on_event_task is not None and channel.on_event_task is not caller

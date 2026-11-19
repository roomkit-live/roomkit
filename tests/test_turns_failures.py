"""How a turn behind the turn backend fails (RFC §10.6.8): what the owner
and the executor do when the other side goes, when a frame cannot cross, and
what an error keeps of where it was raised."""

from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import RemoteTurnError, TurnLinkError, error_type_name
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import InMemoryTurnBackend, OwnerHost, TurnClient, TurnExecutor, TurnTimeouts
from roomkit.turns._error_codec import ROOMKIT_ERRORS, error_record
from roomkit.turns.host import current_turn_host
from roomkit.turns.wire import ErrorRecord
from tests.test_framework import SimpleChannel
from tests.test_turns_backend import (
    ScriptedChannel,
    _client,
    _Opaque,
    _read_all,
    _run,
)


def _turn_tasks() -> list[asyncio.Task[Any]]:
    return [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and t.get_name().startswith("roomkit-turn-")
    ]


async def _forever(channel: ScriptedChannel) -> AsyncIterator[Any]:
    yield "first"
    await asyncio.Event().wait()
    yield "never"


async def test_a_stream_its_reader_drops_is_closed_on_the_executor() -> None:
    """The executor reads ahead past "first"; the reader's drop closes the
    turn, and the close cuts that read at once (RFC §10.6.4)."""
    channel = ScriptedChannel(_forever)
    output = await _run(_client(channel))
    assert await anext(output.response_stream) == "first"
    del output
    gc.collect()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if channel.stream_closed:
            break
    assert channel.stream_closed
    await asyncio.sleep(0.01)
    assert _turn_tasks() == []


async def test_a_close_cuts_the_read_ahead_of_its_reader_at_once() -> None:
    """Past a delta the executor reads on without a pull; a close that finds
    it reading cuts the read, as a local reader's close would have found the
    stream suspended, and never waits for the model's next item."""
    seen: list[str] = []

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "first"
            await asyncio.sleep(5)
            yield "second"
        except asyncio.CancelledError:
            seen.append("cancelled")
            raise

    channel = ScriptedChannel(script)
    output = await _run(_client(channel))
    stream = output.response_stream
    assert await anext(stream) == "first"
    await asyncio.sleep(0.01)
    await asyncio.wait_for(stream.aclose(), timeout=1.0)

    assert channel.stream_closed
    assert seen == ["cancelled"]
    assert channel.yielded == ["first"]


async def test_a_close_past_a_call_s_start_closes_the_stream_it_holds() -> None:
    """Past a call's start the executor reads nothing until a pull, so a
    close finds the stream suspended at that item and closes it: the channel
    sees a close, not a cancellation (RFC §10.6.4)."""
    seen: list[str] = []

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield ToolCallStartMarker(tool_name="t", tool_id="c1", arguments={})
            yield "never read"
        except GeneratorExit:
            seen.append("closed")
            raise
        except asyncio.CancelledError:
            seen.append("cancelled")
            raise

    channel = ScriptedChannel(script)
    output = await _run(_client(channel))
    stream = output.response_stream
    assert isinstance(await anext(stream), ToolCallStartMarker)
    await asyncio.sleep(0.01)
    await stream.aclose()

    assert channel.stream_closed
    assert seen == ["closed"]
    assert len(channel.yielded) == 1


async def test_closing_the_client_ends_every_turn_in_flight() -> None:
    channel = ScriptedChannel(_forever)
    client = _client(channel)
    output = await _run(client)
    await anext(output.response_stream)
    await client.close()
    await asyncio.sleep(0.01)
    assert channel.stream_closed
    assert _turn_tasks() == []


async def test_an_executor_whose_owner_goes_cuts_the_turn() -> None:
    channel = ScriptedChannel(_forever)
    client = _client(channel)
    output = await _run(client)
    assert await anext(output.response_stream) == "first"
    reader = asyncio.create_task(anext(output.response_stream))
    await asyncio.sleep(0.01)
    for turn in list(client._open):
        turn.abandon()  # the owner stops holding the turn
    with pytest.raises(TurnLinkError):
        await reader
    for _ in range(50):
        await asyncio.sleep(0.01)
        if channel.stream_closed:
            break
    assert channel.stream_closed


async def test_a_tool_result_the_wire_cannot_carry_fails_the_turn_naming_it() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield ToolCallStartMarker(tool_name="t", tool_id="c1", arguments={})
        yield ToolCallEndMarker(tool_name="t", tool_id="c1", result=_Opaque())

    channel = ScriptedChannel(script)
    output = await _run(_client(channel))
    stream = output.response_stream
    await anext(stream)
    with pytest.raises(TypeError, match="cannot cross the turn backend"):
        await anext(stream)
    assert channel.stream_closed


async def test_a_stream_item_of_an_unknown_kind_fails_the_turn_naming_it() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield _Opaque()

    output = await _run(_client(ScriptedChannel(script)))
    with pytest.raises(TypeError, match="stream item cannot cross"):
        await anext(output.response_stream)


async def test_a_record_value_json_has_no_form_for_crosses_as_its_text() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        channel.record["opaque"] = _Opaque()
        yield "text"

    channel = ScriptedChannel(script)
    channel.record["at_output"] = _Opaque()
    output = await _run(_client(channel))
    assert await anext(output.response_stream) == "text"
    assert output.response_metadata["at_output"].startswith("<")
    assert output.response_metadata["opaque"].startswith("<")


async def test_an_agent_answers_in_a_room_whose_metadata_json_has_no_form_for() -> None:
    agent = Agent(
        "agent", provider=MockAIProvider(ai_responses=[AIResponse(content="hello")]), role="a"
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(agent)
    await kit.create_room(room_id="room", metadata={"session": _Opaque()})
    await kit.attach_channel("room", "sms")
    await kit.attach_channel("room", "agent", category=ChannelCategory.INTELLIGENCE)
    message = InboundMessage(
        channel_id="sms", sender_id="+15550000001", content=TextContent(body="hi")
    )

    result = await kit.process_inbound(message, room_id="room")

    await kit.close()
    assert result.error is None
    assert [e.content.body for e in result.response_events] == ["hello"]


async def test_a_tool_blocking_the_loop_past_the_heartbeat_timeout_is_not_lost() -> None:
    """In memory the executor's CPU is the owner's stall: a tool that blocks
    the event loop past the heartbeat timeout stalls the owner's count too,
    so the turn ends as it would have, never lost (RFC §10.6.8)."""
    backend = InMemoryTurnBackend()
    backend.timeouts = TurnTimeouts(heartbeat_interval=0.005, heartbeat_timeout=0.025)
    tool = {"name": "lookup", "description": "Look up.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
            ),
            AIResponse(content="42", finish_reason="stop"),
        ]
    )

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        time.sleep(0.06)  # noqa: ASYNC251 - the turn's CPU: the event loop stops
        return "42"

    kit = RoomKit(turn_backend=backend)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("agent", provider=provider, tools=[tool], tool_handler=lookup))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "sms")
    await kit.attach_channel("room", "agent", category=ChannelCategory.INTELLIGENCE)
    message = InboundMessage(
        channel_id="sms", sender_id="+15550000001", content=TextContent(body="What is it?")
    )

    result = await kit.process_inbound(message, room_id="room")
    rows = [e.type for e in await kit.store.list_events("room") if e.source.channel_id == "agent"]

    await kit.close()
    assert result.error is None
    assert rows == [EventType.TOOL_CALL_START, EventType.TOOL_CALL_END, EventType.MESSAGE]


def _raise_deep() -> None:
    raise KeyError("missing")


async def test_an_error_keeps_where_the_executor_raised_it() -> None:
    try:
        _raise_deep()
    except KeyError as exc:
        rebuilt = ROOMKIT_ERRORS.rebuild(error_record(exc))
    notes = "\n".join(getattr(rebuilt, "__notes__", []))
    assert "_raise_deep" in notes


@pytest.mark.parametrize(
    "kind",
    [
        "builtins:SystemExit",
        "builtins:KeyboardInterrupt",
        "roomkit.core.framework:asyncio",
        "roomkit.core.exceptions:RoomKitError.__class__",
        "roomkit.not_loaded_module:Anything",
    ],
)
def test_a_name_off_the_wire_reaches_no_class_but_a_loaded_exception(kind: str) -> None:
    rebuilt = ROOMKIT_ERRORS.rebuild(ErrorRecord(kind=kind, message="m"))
    assert isinstance(rebuilt, RemoteTurnError)


def test_a_rebuilt_foreign_error_is_reported_under_its_own_class() -> None:
    assert (
        error_type_name(RemoteTurnError("m", kind="vendor.sdk.errors:RateLimited"))
        == "RateLimited"
    )
    assert error_type_name(ValueError("m")) == "ValueError"


class _AnswersAtOnce(ScriptedChannel):
    """Returns its stream at once for a person's message, nothing otherwise."""

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> ChannelOutput:
        if event.source.channel_id != "sms":
            return ChannelOutput.empty()
        return await super().on_event(event, binding, context)


class _NeverAnswers(ScriptedChannel):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> ChannelOutput:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_the_turns_of_a_cancelled_delivery_set_all_end() -> None:
    """A caller gives up while one agent still answers: that turn is cut,
    and the stream the other agent already returned is closed, not left
    open on its executor."""
    fast, slow = _AnswersAtOnce(_forever), _NeverAnswers()
    fast.channel_id, slow.channel_id = "fast", "slow"
    kit = RoomKit()
    for channel in (SimpleChannel("sms"), fast, slow):
        kit.register_channel(channel)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "sms")
    for agent in ("fast", "slow"):
        await kit.attach_channel("room", agent, category=ChannelCategory.INTELLIGENCE)
    message = InboundMessage(
        channel_id="sms", sender_id="+15550000001", content=TextContent(body="hi")
    )
    call = asyncio.create_task(kit.process_inbound(message, room_id="room"))
    await slow.entered.wait()
    await asyncio.sleep(0.05)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    for _ in range(100):
        if not _turn_tasks():
            break
        await asyncio.sleep(0.01)
    assert _turn_tasks() == []
    await kit.close()


# -- An executor closed while a turn's end waits for a verb ---------------------

FAST = TurnTimeouts(heartbeat_interval=0.05, heartbeat_timeout=0.2, cancel=0.3)


class _HeldHost(OwnerHost):
    """Holds each verb until released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def call(self, request: Any, verb: str, arguments: dict[str, Any]) -> Any:
        self.started.set()
        await self.release.wait()
        return None


class _PostsOnATask(ScriptedChannel):
    """Its turn posts a status from a task of its own (a tool's), then
    answers: the turn's end waits for that verb."""

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> ChannelOutput:
        host = current_turn_host()
        assert host is not None
        self.posting = asyncio.create_task(host.post_status("agent", "x", "running"))
        return await super().on_event(event, binding, context)


def _executor_turn_tasks() -> list[str]:
    """The tasks of an executor's turn still running."""
    names = [t.get_coro().__qualname__ for t in asyncio.all_tasks() if not t.done()]
    return [name for name in names if name.startswith("ExecutorTurn.")]


async def _waiting_for_the_verb(
    channel: ScriptedChannel, host: _HeldHost, *, own_executor: bool
) -> tuple[TurnClient, TurnExecutor, ChannelOutput]:
    """A client whose turn ended on its executor but for the verb *host*
    holds, the turn's end waiting for it."""
    channels = {channel.channel_id: channel}
    executor = TurnExecutor(channels.get, hosts=channels.__contains__, timeouts=FAST)
    backend = InMemoryTurnBackend()
    if not own_executor:
        await backend.serve(executor)
    client = TurnClient(
        backend, executor=executor if own_executor else None, host=host, timeouts=FAST
    )
    output = await _run(client)
    await asyncio.wait_for(host.started.wait(), 1)
    await asyncio.sleep(0.05)
    return client, executor, output


async def _turn_over(client: TurnClient) -> bool:
    for _ in range(100):
        if all(turn.over for turn in client._started) and not _executor_turn_tasks():
            return True
        await asyncio.sleep(0.01)
    return False


async def test_an_executor_closed_while_a_turn_s_end_waits_for_a_verb_ends_the_turn() -> None:
    channel, host = _PostsOnATask(), _HeldHost()
    client, executor, _ = await _waiting_for_the_verb(channel, host, own_executor=False)

    await asyncio.wait_for(executor.close(), 2)

    over, left = await _turn_over(client), _executor_turn_tasks()
    host.release.set()
    await client.close()
    assert left == []
    assert over
    # The verb on its way failed in its task: nothing answers it any more.
    with pytest.raises(TurnLinkError):
        await asyncio.wait_for(channel.posting, 1)


async def test_the_reader_of_a_turn_whose_executor_closed_while_its_end_waited_ends() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "hi"

    channel, host = _PostsOnATask(script), _HeldHost()
    client, executor, output = await _waiting_for_the_verb(channel, host, own_executor=False)
    reader = asyncio.create_task(_read_all(output))
    await asyncio.sleep(0.05)

    await asyncio.wait_for(executor.close(), 2)

    # Ten heartbeat timeouts: a silent executor would have failed the turn.
    done, _ = await asyncio.wait({reader}, timeout=2)
    reader.cancel()
    host.release.set()
    await asyncio.gather(reader, channel.posting, return_exceptions=True)
    await client.close()
    assert reader in done


async def test_an_owner_closed_while_its_turn_s_end_waits_for_a_verb_leaves_no_turn_task() -> None:
    channel, host = _PostsOnATask(), _HeldHost()
    client, _, _ = await _waiting_for_the_verb(channel, host, own_executor=True)

    await asyncio.wait_for(client.close(), 2)

    left = _executor_turn_tasks()
    host.release.set()
    await asyncio.gather(channel.posting, return_exceptions=True)
    assert left == []

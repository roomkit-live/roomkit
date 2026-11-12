"""An ACP agent across the turn backend (RFC §10.6.2, §10.6.3, §10.6.8, RMK-812).

The host's session verbs are channel calls reaching the executor that holds
the room's session, and the values it sets belong to the room; a lost
executor's room reopens its session on the next one, caught up and set up
again; a detach closes the room's session; a steering ``Cancel`` cancels the
prompt as the channel's own cancel does; each room's agent updates and
permission requests run under that room's turn, whichever opened the
connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import socket
from collections.abc import AsyncIterator, Callable
from typing import Any

import acp
import pytest
from acp.schema import PermissionOption, PromptResponse

from roomkit import ACPChannel, ChannelDeclaration, RoomKit, current_turn_host
from roomkit.core._turn_scope import enter_turn
from roomkit.core.exceptions import ChannelNotFoundError, RoomNotFoundError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.steering import Cancel
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.tools.external import ExternalToolHandler, ToolDecision
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from roomkit.turns._rooms import room_state_key
from tests.conftest import make_event
from tests.test_channels.test_acp import (
    _binding,
    _channel,
    _context,
    _FakeACPConnection,
    _FakeTransport,
    _InProcessEchoAgent,
    _sent,
    _SocketPairTransport,
)
from tests.test_channels.test_acp_discussion import _Hanging
from tests.test_channels.test_acp_instruction import _instruction
from tests.test_framework import SimpleChannel

ROOM = "room-1"
AGENT = "acp-agent"


async def _room(
    channel: ACPChannel | ChannelDeclaration, backend: InMemoryTurnBackend | None = None
) -> RoomKit:
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "sms", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, AGENT, category=ChannelCategory.INTELLIGENCE)
    return kit


async def _say(kit: RoomKit, body: str, room_id: str = ROOM) -> None:
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id="marie",
            content=TextContent(body=body),
            addressed_to=[AGENT],
        ),
        room_id=room_id,
    )
    await asyncio.sleep(0)


def _hanging(tmp_path: Any) -> tuple[ACPChannel, _Hanging]:
    connection = _Hanging()
    channel = ACPChannel(AGENT, transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    return channel, connection


async def _eventually(holds: Callable[[], bool]) -> None:
    """Wait for *holds*, done in the background, within 5 s."""
    async with asyncio.timeout(5):
        while not holds():
            await asyncio.sleep(0.01)


class _Unanswered(_FakeACPConnection):
    """An agent that never answers ``session/close``."""

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)
        await asyncio.Event().wait()


# -- The host's session verbs --------------------------------------------------------


async def test_a_host_s_session_values_belong_to_the_room(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)

    values = await kit.channel_call(
        AGENT, ROOM, "set_config_option", config_id="model", value="sonnet"
    )

    assert values == {"model": "sonnet"}
    assert await kit.channel_call(AGENT, ROOM, "session_config") == {"model": "sonnet"}
    (option,) = await kit.channel_call(AGENT, ROOM, "config_options")
    assert option["id"] == "model" and "sonnet" in str(option)
    room = await kit.get_room(ROOM)
    assert room.metadata[room_state_key(AGENT)] == {"config": {"model": "sonnet"}}

    assert await kit.channel_call(AGENT, ROOM, "close_session") is True
    room = await kit.get_room(ROOM)
    assert room_state_key(AGENT) not in room.metadata
    assert connection.closed_sessions == ["session-1"]
    await kit.close()


async def test_a_verb_an_acp_agent_lacks_is_refused(tmp_path: Any) -> None:
    channel, _connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)
    with pytest.raises(ValueError, match="set_config_option"):
        await kit.channel_call(AGENT, ROOM, "prompt", text="hi")
    await kit.close()


# -- A lost executor ---------------------------------------------------------------


async def test_a_lost_executor_s_room_reopens_its_session_on_the_next(tmp_path: Any) -> None:
    """The kit holds the agent by its declaration, and two executors of a host
    serve it: the room's session opens on the first, then on the next once the
    first is lost."""
    backend = InMemoryTurnBackend()
    remote, remote_connection, _ = _channel(tmp_path, emit_updates=False)
    local, local_connection, _ = _channel(tmp_path, emit_updates=False)
    elsewhere = TurnExecutor(
        lambda channel_id: remote if channel_id == AGENT else None,
        hosts={AGENT},
        executor_id="elsewhere",
    )
    following = TurnExecutor(
        lambda channel_id: local if channel_id == AGENT else None,
        hosts={AGENT},
        executor_id="following",
    )
    # Served first: the room's session opens there.
    await backend.serve(elsewhere)
    await backend.serve(following)
    kit = await _room(local.declaration(), backend)

    await kit.channel_call(AGENT, ROOM, "set_config_option", config_id="model", value="sonnet")
    await _say(kit, "first question")
    assert len(remote_connection.prompt_calls) == 1
    assert not local_connection.prompt_calls

    await elsewhere.close()
    await _say(kit, "second question")

    # The next executor opened the room's session anew: set to the value the
    # room keeps, and caught up on what the lost session had been told.
    assert len(local_connection.new_session_calls) == 1
    assert [(c["config_id"], c["value"]) for c in local_connection.set_config_calls] == [
        ("model", "sonnet")
    ]
    sent = _sent(local_connection)
    assert "first question" in sent and sent.endswith("second question")
    await kit.close()
    await following.close()
    await remote.close()


# -- Detaching -------------------------------------------------------------------------


async def test_detaching_an_acp_agent_closes_the_room_s_session(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)
    await _say(kit, "hello")
    session = channel.session_id(ROOM)
    assert session is not None

    await kit.detach_channel(ROOM, AGENT)

    # Off the detach, which never waits for the agent.
    await _eventually(lambda: connection.closed_sessions == [session])
    assert channel.session_id(ROOM) is None
    await kit.attach_channel(ROOM, AGENT, category=ChannelCategory.INTELLIGENCE)
    await _say(kit, "hello again")
    assert len(connection.new_session_calls) == 2
    await kit.close()


async def test_a_detach_never_waits_for_the_room_s_running_turn(tmp_path: Any) -> None:
    channel, connection = _hanging(tmp_path)
    kit = await _room(channel)
    said = asyncio.create_task(_say(kit, "read every file"))
    await asyncio.wait_for(connection.prompted.wait(), 5)
    session = channel.session_id(ROOM)

    await asyncio.wait_for(kit.detach_channel(ROOM, AGENT), 2)
    assert connection.closed_sessions == []

    connection.release.set()
    await asyncio.wait_for(said, 5)
    # The turn that ran across the detach closed its session at its end.
    assert connection.closed_sessions == [session]
    await kit.close()


async def test_a_detach_never_waits_for_an_agent_that_does_not_answer(tmp_path: Any) -> None:
    connection = _Unanswered(None, emit_updates=False)
    channel = ACPChannel(AGENT, transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    kit = await _room(channel)
    await _say(kit, "hello")

    await asyncio.wait_for(kit.detach_channel(ROOM, AGENT), 2)

    await _eventually(lambda: connection.closed_sessions == ["session-1"])
    await kit.close()


@pytest.mark.parametrize("fresh_session", [False, True])
async def test_a_turn_started_before_a_detach_never_prompts_after_it(
    tmp_path: Any, fresh_session: bool
) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    event = make_event(room_id="room-1", body="hello", index=0)
    output = await channel.on_event(event, _binding(), _context(event))
    assert output.response_stream is not None

    await channel.on_room_detached("room-1")
    if fresh_session:
        await channel.set_config_option("room-1", "model", "sonnet")
    session = channel.session_id("room-1")
    items = [item async for item in output.response_stream]

    assert items == [] and connection.prompt_calls == []
    assert len(connection.new_session_calls) == int(fresh_session)
    assert connection.closed_sessions == []
    assert channel.session_id("room-1") == session
    assert output.response_metadata is not None
    assert output.response_metadata["acp"]["interrupted"] is True
    await channel.close()


async def test_a_session_opened_by_a_turn_across_a_detach_is_closed(tmp_path: Any) -> None:
    connection = _SlowOpening()
    channel = ACPChannel(AGENT, transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    event = make_event(room_id="room-1", body="hello", index=0)
    output = await channel.on_event(event, _binding(), _context(event))
    assert output.response_stream is not None
    reading = asyncio.create_task(anext(output.response_stream, None))
    await asyncio.wait_for(connection.opening.wait(), 5)

    await channel.on_room_detached("room-1")
    connection.release.set()
    await asyncio.wait_for(reading, 5)

    # Opened for a room it was detached from: never prompted, and closed.
    assert connection.prompt_calls == []
    assert connection.closed_sessions == ["session-1"]
    assert channel.session_id("room-1") is None
    await channel.close()


# -- Steering ------------------------------------------------------------------------


async def test_a_steering_cancel_cancels_the_prompt_as_its_own_cancel_does(
    tmp_path: Any,
) -> None:
    channel, connection = _hanging(tmp_path)
    kit = await _room(channel)
    responses: list[Any] = []

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def observe(event: Any, ctx: Any) -> None:
        responses.append(event)

    said = asyncio.create_task(_say(kit, "read every file"))
    await asyncio.wait_for(connection.prompted.wait(), 5)

    reached = await kit.steer(ROOM, AGENT, Cancel(reason="hold"))
    await asyncio.wait_for(said, 5)
    await asyncio.sleep(0.05)

    assert reached == 1
    assert connection.cancelled_sessions == [channel.session_id(ROOM)]
    # The prompt returned: the turn ended as the agent ended it, not cut.
    assert len(responses) == 1
    await kit.close()


async def test_a_steering_cancel_reaches_no_idle_acp_agent(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)
    await _say(kit, "hello")
    assert await kit.steer(ROOM, AGENT, Cancel(reason="hold")) == 0
    assert connection.cancelled_sessions == []
    await kit.close()


class _SlowOpening(_FakeACPConnection):
    """An agent whose ``session/new`` waits until released."""

    def __init__(self) -> None:
        super().__init__(None, emit_updates=False)
        self.opening = asyncio.Event()
        self.release = asyncio.Event()

    async def new_session(self, **kwargs: Any) -> Any:
        self.opening.set()
        await self.release.wait()
        return await super().new_session(**kwargs)


async def test_a_steering_cancel_while_the_session_opens_stops_the_prompt(
    tmp_path: Any,
) -> None:
    connection = _SlowOpening()
    channel = ACPChannel(AGENT, transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    kit = await _room(channel)
    said = asyncio.create_task(_say(kit, "hello"))
    await asyncio.wait_for(connection.opening.wait(), 5)

    assert await kit.steer(ROOM, AGENT, Cancel(reason="hold")) == 1
    connection.release.set()
    await asyncio.wait_for(said, 5)

    # Nothing was prompted: the opening was stopped before a prompt could go.
    assert connection.prompt_calls == []
    assert not channel._prompting(ROOM)
    await kit.close()


async def test_a_steering_cancel_before_the_agent_is_reached_cuts_the_turn(
    tmp_path: Any,
) -> None:
    contributing = asyncio.Event()
    release = asyncio.Event()

    async def contributor(context: Any, trigger: Any) -> list[str]:
        contributing.set()
        await release.wait()
        return ["note"]

    channel, connection, _ = _channel(
        tmp_path, emit_updates=False, context_contributor=contributor
    )
    kit = await _room(channel)
    said = asyncio.create_task(_say(kit, "hello"))
    await asyncio.wait_for(contributing.wait(), 5)

    # No prompt of the turn is out yet: the owner cuts the turn itself.
    assert await kit.steer(ROOM, AGENT, Cancel(reason="hold")) == 1
    release.set()
    await asyncio.wait_for(said, 5)
    assert connection.prompt_calls == []
    await kit.close()


async def test_a_steering_cancel_reaches_the_turn_it_names_only(tmp_path: Any) -> None:
    channel, connection = _hanging(tmp_path)
    event = make_event(room_id="room-1", body="hello", index=0)
    under_turn = contextvars.copy_context()
    under_turn.run(enter_turn, "turn-1")

    async def prompt() -> None:
        output = await channel.on_event(event, _binding(), _context(event))
        assert output.response_stream is not None
        [chunk async for chunk in output.response_stream]

    running = asyncio.create_task(prompt(), context=under_turn)
    await asyncio.wait_for(connection.prompted.wait(), 5)

    assert await channel.steer(Cancel(reason="hold"), turn_id="turn-2") is None
    assert connection.cancelled_sessions == []
    assert await channel.steer(Cancel(reason="hold"), turn_id="turn-1") == 1
    await asyncio.wait_for(running, 5)
    assert connection.cancelled_sessions == ["session-1"]
    await channel.close()


# -- Who may call ---------------------------------------------------------------------


async def test_a_channel_call_needs_the_channel_attached_to_the_room(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)
    await kit.detach_channel(ROOM, AGENT)

    with pytest.raises(ChannelNotFoundError):
        await kit.channel_call(AGENT, ROOM, "set_config_option", config_id="model", value="opus")
    assert connection.new_session_calls == []
    room = await kit.get_room(ROOM)
    assert room_state_key(AGENT) not in room.metadata
    await kit.close()


async def test_a_channel_call_is_scoped_to_the_room_s_organization(tmp_path: Any) -> None:
    channel, _connection, _ = _channel(tmp_path, emit_updates=False)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id=ROOM, organization_id="acme")
    await kit.attach_channel(ROOM, AGENT, category=ChannelCategory.INTELLIGENCE)

    with pytest.raises(RoomNotFoundError):
        await kit.channel_call(AGENT, ROOM, "session_config", organization_id="globex")
    assert await kit.channel_call(AGENT, ROOM, "session_config", organization_id="acme") == {}
    await kit.close()


async def test_a_failed_detach_message_still_forgets_the_room_s_values(tmp_path: Any) -> None:
    channel, _connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)
    await kit.channel_call(AGENT, ROOM, "set_config_option", config_id="model", value="sonnet")
    sent = kit._turns.control

    async def refuse(message: Any) -> Any:
        if message.type == "room_detached":
            raise RuntimeError("backend down")
        return await sent(message)

    kit._turns.control = refuse  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="backend down"):
        await kit.detach_channel(ROOM, AGENT)
    room = await kit.get_room(ROOM)
    assert room_state_key(AGENT) not in room.metadata
    kit._turns.control = sent  # type: ignore[method-assign]
    await kit.close()


async def test_a_detach_during_a_standalone_turn_closes_the_room_s_session(
    tmp_path: Any,
) -> None:
    channel, connection = _hanging(tmp_path)
    connection.release.set()
    first = make_event(room_id="room-1", body="hello", index=0)
    output = await channel.on_event(first, _binding(), _context(first))
    assert output.response_stream is not None
    [chunk async for chunk in output.response_stream]
    room_session = channel.session_id("room-1")
    connection.release.clear()
    connection.prompted.clear()

    async def standalone() -> None:
        output = await channel.on_event(_instruction(standalone=True), _binding(), _context(first))
        assert output.response_stream is not None
        [chunk async for chunk in output.response_stream]

    running = asyncio.create_task(standalone())
    await asyncio.wait_for(connection.prompted.wait(), 5)
    await channel.on_room_detached("room-1")
    connection.release.set()
    await asyncio.wait_for(running, 5)

    # The standalone turn closed its own session; the room's closes after it.
    await _eventually(lambda: room_session in connection.closed_sessions)
    assert channel.session_id("room-1") is None
    await channel.close()


async def test_a_value_set_on_the_channel_object_is_not_stored_by_a_read(
    tmp_path: Any,
) -> None:
    channel, _connection, _ = _channel(tmp_path, emit_updates=False)
    kit = await _room(channel)
    await kit.channel_call(AGENT, ROOM, "set_config_option", config_id="model", value="sonnet")
    await channel.set_config_option(ROOM, "model", "opus")  # this process's copy

    assert (await kit.channel_call(AGENT, ROOM, "session_config"))["model"] == "opus"
    room = await kit.get_room(ROOM)
    assert room.metadata[room_state_key(AGENT)] == {"config": {"model": "sonnet"}}
    await kit.close()


# -- A channel call across a cancel or a detach --------------------------------------


class _HangingConfig(_FakeACPConnection):
    """An agent whose ``set_config_option`` never answers while ``hangs``."""

    def __init__(self) -> None:
        super().__init__(None, emit_updates=False)
        self.hangs = False
        self.setting = asyncio.Event()

    async def set_config_option(self, **kwargs: Any) -> Any:
        if self.hangs:
            self.setting.set()
            await asyncio.Event().wait()
        return await super().set_config_option(**kwargs)


async def test_a_cancelled_channel_call_keeps_the_room_s_session(tmp_path: Any) -> None:
    connection = _HangingConfig()
    channel = ACPChannel(AGENT, transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    kit = await _room(channel)
    await _say(kit, "first question")
    assert channel.session_id(ROOM) == "session-1"

    connection.hangs = True
    call = asyncio.create_task(
        kit.channel_call(AGENT, ROOM, "set_config_option", config_id="model", value="sonnet")
    )
    await asyncio.wait_for(connection.setting.wait(), 5)
    # As a host's timeout does: the call may have reached the executor.
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    connection.hangs = False

    await _say(kit, "second question")
    assert connection.closed_sessions == []
    assert len(connection.new_session_calls) == 1
    assert [c["session_id"] for c in connection.prompt_calls] == ["session-1", "session-1"]
    await kit.close()


async def test_a_channel_call_across_a_detach_keeps_no_session_and_no_value(
    tmp_path: Any,
) -> None:
    connection = _SlowOpening()
    channel = ACPChannel(AGENT, transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    kit = await _room(channel)
    call = asyncio.create_task(
        kit.channel_call(AGENT, ROOM, "set_config_option", config_id="model", value="sonnet")
    )
    await asyncio.wait_for(connection.opening.wait(), 5)

    await kit.detach_channel(ROOM, AGENT)
    connection.release.set()
    await asyncio.wait_for(call, 5)

    # The session it opened was the detached room's: closed, its value forgotten.
    assert channel.session_id(ROOM) is None
    assert connection.closed_sessions == ["session-1"]
    assert channel.room_state(ROOM) == {}
    room = await kit.get_room(ROOM)
    assert room_state_key(AGENT) not in room.metadata
    await kit.attach_channel(ROOM, AGENT, category=ChannelCategory.INTELLIGENCE)
    await _say(kit, "hello again")
    assert [c["session_id"] for c in connection.prompt_calls] == ["session-2"]
    await kit.close()


class _HangingAndSlowConfig(_Hanging):
    """An agent whose prompt runs until released, and whose
    ``set_config_option`` answers once ``answer`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.setting = asyncio.Event()
        self.answer = asyncio.Event()

    async def set_config_option(self, **kwargs: Any) -> Any:
        self.setting.set()
        await self.answer.wait()
        return await super().set_config_option(**kwargs)


async def test_a_value_set_across_a_detach_during_a_turn_is_not_kept(tmp_path: Any) -> None:
    connection = _HangingAndSlowConfig()
    channel = ACPChannel(AGENT, transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    kit = await _room(channel)
    said = asyncio.create_task(_say(kit, "read every file"))
    await asyncio.wait_for(connection.prompted.wait(), 5)
    call = asyncio.create_task(
        kit.channel_call(AGENT, ROOM, "set_config_option", config_id="model", value="sonnet")
    )
    await asyncio.wait_for(connection.setting.wait(), 5)

    # The running turn holds the room's turn lock: the detach's close waits.
    await kit.detach_channel(ROOM, AGENT)
    connection.answer.set()
    await asyncio.wait_for(call, 5)

    assert channel.room_state(ROOM) == {}
    room = await kit.get_room(ROOM)
    assert room_state_key(AGENT) not in room.metadata
    connection.release.set()
    await asyncio.wait_for(said, 5)
    assert connection.closed_sessions == ["session-1"]
    await kit.close()


# -- Each room's agent updates, under that room's turn -------------------------------


class _ReadingAgent(_InProcessEchoAgent):
    """A real ACP agent that thinks, asks to read a file, reads it and answers."""

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        await self.conn.session_update(
            session_id=session_id, update=acp.update_agent_thought_text("pondering")
        )
        await self.conn.session_update(
            session_id=session_id,
            update=acp.start_tool_call("read-1", "Read", kind="read", status="in_progress"),
        )
        await self.conn.request_permission(
            session_id=session_id,
            tool_call=acp.update_tool_call("read-1", title="Read"),
            options=[PermissionOption(option_id="allow", name="Allow", kind="allow_once")],
        )
        await self.conn.session_update(
            session_id=session_id, update=acp.update_tool_call("read-1", status="completed")
        )
        await self.conn.session_update(
            session_id=session_id, update=acp.update_agent_message_text("done")
        )
        return PromptResponse(stop_reason="end_turn")


@contextlib.asynccontextmanager
async def _socket_agent() -> AsyncIterator[_SocketPairTransport]:
    """A transport to a real ACP agent over a socket pair, served until exit."""
    agent_sock, client_sock = socket.socketpair()
    agent_reader, agent_writer = await asyncio.open_connection(sock=agent_sock)
    client_reader, client_writer = await asyncio.open_connection(sock=client_sock)
    serving = asyncio.create_task(acp.run_agent(_ReadingAgent(), agent_writer, agent_reader))
    try:
        yield _SocketPairTransport(client_reader, client_writer)
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        agent_writer.close()


async def _two_rooms(channel: ACPChannel, realtime: InMemoryRealtime | None = None) -> RoomKit:
    kit = RoomKit(realtime=realtime)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    for room_id in ("room-1", "room-2"):
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "sms", category=ChannelCategory.TRANSPORT)
        await kit.attach_channel(room_id, AGENT, category=ChannelCategory.INTELLIGENCE)
    return kit


class _PostingHandler(ExternalToolHandler):
    """Approves every call, posting a status through the turn's host, if any,
    and reports each end to ON_TOOL_CALL."""

    async def process_tool_call(
        self, tool_name: str, tool_input: dict[str, Any], **kwargs: Any
    ) -> ToolDecision:
        host = current_turn_host()
        if host is not None:
            await host.post_status(AGENT, "permission", "ok", detail=kwargs["room_id"])
        return ToolDecision(approved=True)

    async def on_tool_result(
        self, tool_name: str, tool_input: dict[str, Any], result: str, **kwargs: Any
    ) -> None:
        await self._fire_on_tool_hook(tool_name, tool_input, result, **kwargs)


async def test_each_room_s_agent_updates_reach_that_room(tmp_path: Any) -> None:
    realtime = InMemoryRealtime()
    seen: dict[str, list[EphemeralEvent]] = {"room-1": [], "room-2": []}
    async with _socket_agent() as transport:
        kit = await _two_rooms(ACPChannel(AGENT, transport=transport, cwd=tmp_path), realtime)
        for room_id, events in seen.items():

            async def capture(event: EphemeralEvent, events: list[Any] = events) -> None:
                events.append(event)

            await realtime.subscribe_to_room(room_id, capture)

        # Room-1's turn opens the connection; room-2's updates arrive on it.
        await _say(kit, "first room", "room-1")
        await _say(kit, "second room", "room-2")

        def thought(room_id: str) -> bool:
            return any(e.type is EphemeralEventType.THINKING_DELTA for e in seen[room_id])

        await _eventually(lambda: thought("room-1") and thought("room-2"))
        await kit.close()


@pytest.mark.parametrize(
    ("options", "reads_text"),
    [({}, False), ({"publish_thinking_text": True}, True)],
    ids=["private-by-default", "opted-in"],
)
async def test_a_subscriber_reads_an_agent_s_reasoning_only_when_opted_in(
    tmp_path: Any, options: dict[str, Any], reads_text: bool
) -> None:
    """The channel has no visibility gate: by default a subscriber reads what a
    live indicator needs, never the text (RMK-1052)."""
    realtime = InMemoryRealtime()
    seen: list[EphemeralEvent] = []

    async def capture(event: EphemeralEvent) -> None:
        if event.type is EphemeralEventType.THINKING_DELTA:
            seen.append(event)

    await realtime.subscribe_to_room("room-1", capture)
    async with _socket_agent() as transport:
        channel = ACPChannel(AGENT, transport=transport, cwd=tmp_path, **options)
        kit = await _two_rooms(channel, realtime)
        await _say(kit, "first room", "room-1")
        await _eventually(lambda: bool(seen))
        await kit.close()
    assert all(("thinking" in event.data) is reads_text for event in seen)
    assert all(event.data["thinking_length"] > 0 for event in seen)


async def test_each_room_s_tool_hooks_and_permissions_see_that_room_s_host(
    tmp_path: Any,
) -> None:
    async with _socket_agent() as transport:
        channel = ACPChannel(
            AGENT, transport=transport, cwd=tmp_path, external_tool_handler=_PostingHandler()
        )
        kit = await _two_rooms(channel)

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
        async def audit(event: Any, ctx: Any) -> None:
            host = current_turn_host()
            if host is not None:
                await host.post_status(AGENT, "tool", "ok", detail=event.room_id)

        await _say(kit, "first room", "room-1")
        await _say(kit, "second room", "room-2")

        async def posted() -> list[tuple[str, str, Any]]:
            entries = await kit.status_bus.recent(20)
            return [(e.action, e.detail, e.metadata.get("room_id")) for e in entries]

        async with asyncio.timeout(5):
            while len(await posted()) < 4:
                await asyncio.sleep(0.01)
        # Each posted for the room it ran for, through that room's turn.
        assert sorted(await posted()) == [
            ("permission", "room-1", "room-1"),
            ("permission", "room-2", "room-2"),
            ("tool", "room-1", "room-1"),
            ("tool", "room-2", "room-2"),
        ]
        await kit.close()

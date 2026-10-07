"""An INSTRUCTION reaching an ACP agent (RFC §10.1.1).

An ACP session holds its history inside the agent's process, so the rules an
in-process channel keeps by rebuilding its context each turn are kept here by
what the prompt says and by which session receives it.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest
from acp.schema import PromptResponse

from roomkit.channels._instruction import INSTRUCTION_MARKER
from roomkit.models.delivery import STANDALONE
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent
from tests.conftest import make_event
from tests.test_channels.test_acp import (
    _binding,
    _channel,
    _context,
    _model_option,
    _prompt,
    _sent,
)

ROOM = "room-1"


SUMMARY = "Summarize the meeting."


def _instruction(body: str = SUMMARY, *, standalone: bool = False) -> RoomEvent:
    # Never committed, so it carries no index of its own (index 0).
    return make_event(room_id=ROOM, body=body, index=0).model_copy(
        update={
            "type": EventType.INSTRUCTION,
            "metadata": {STANDALONE: True} if standalone else {},
        }
    )


async def test_an_instruction_moves_the_cursor_past_what_it_caught_up_on(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    alpha = make_event(room_id=ROOM, body="alpha said", index=0)
    beta = make_event(room_id=ROOM, body="beta said", index=1)
    await _prompt(channel, _instruction(), _context(alpha, beta))

    following = make_event(room_id=ROOM, body="next request", index=2)
    await _prompt(channel, following, _context(alpha, beta, following))

    assert "beta said" in _sent(connection, turn=0)
    assert _sent(connection, turn=1) == "next request"
    await channel.close()


async def test_the_agent_reads_it_as_the_applications_direction(tmp_path: Any) -> None:
    """Step 6: marked, and the reply records a fingerprint, never the text."""
    channel, connection, _ = _channel(tmp_path, emit_updates=False)

    output = await channel.on_event(_instruction(), _binding(), _context())
    [chunk async for chunk in output.response_stream]

    assert _sent(connection) == f"{INSTRUCTION_MARKER}\n{SUMMARY}"
    assert output.response_metadata["instruction"] == {
        "sha256": hashlib.sha256(SUMMARY.encode()).hexdigest(),
        "length": len(SUMMARY),
    }
    await channel.close()


async def _talked(tmp_path: Any) -> tuple[Any, Any, list[RoomEvent]]:
    """A channel whose room session already exchanged one prompt."""
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    first = make_event(room_id=ROOM, body="first request", index=0)
    await _prompt(channel, first, _context(first))
    assert channel.session_id(ROOM) == "session-1"
    return channel, connection, [first]


async def test_a_standalone_turn_runs_in_a_session_of_its_own(tmp_path: Any) -> None:
    """Step 7: the room's session cannot be emptied, so it is not the one asked."""
    channel, connection, history = await _talked(tmp_path)
    missed = make_event(room_id=ROOM, body="said while you were away", index=1)
    cursor = channel._prompted_index[ROOM]

    await _prompt(channel, _instruction(standalone=True), _context(*history, missed))

    turn = connection.prompt_calls[-1]
    assert turn["session_id"] == "session-2"
    assert _sent(connection, turn=1) == f"{INSTRUCTION_MARKER}\n{SUMMARY}"  # no catch-up
    assert connection.closed_sessions == ["session-2"]
    assert channel.session_id(ROOM) == "session-1"
    assert channel._prompted_index[ROOM] == cursor

    # The room's session still owes the catch-up it was never sent.
    following = make_event(room_id=ROOM, body="next request", index=2)
    await _prompt(channel, following, _context(*history, missed, following))
    assert connection.prompt_calls[-1]["session_id"] == "session-1"
    assert "said while you were away" in _sent(connection, turn=2)
    await channel.close()


async def test_a_standalone_turn_declares_its_session_as_the_turns(tmp_path: Any) -> None:
    """Step 7: a relay that files sessions by room can tell the two apart.

    Both name the room: filed under it, the turn would be answered from the
    room's session, which would then be closed as the turn's.
    """
    channel, connection, history = await _talked(tmp_path)

    await _prompt(channel, _instruction(standalone=True), _context(*history))

    room, turn = connection.new_session_calls
    assert room["roomkit.live/roomId"] == turn["roomkit.live/roomId"] == ROOM
    assert room["roomkit.live/sessionScope"] == "room"
    assert turn["roomkit.live/sessionScope"] == "turn"
    await channel.close()


async def test_a_relay_answering_the_turn_with_the_rooms_session_fails_the_turn(
    tmp_path: Any,
) -> None:
    """A relay that ignores the scope must not get the room's session prompted, then closed."""
    channel, connection, history = await _talked(tmp_path)
    room_session = connection.new_session

    async def files_by_room(**kwargs: Any) -> Any:
        response = await room_session(**kwargs)
        return response.model_copy(update={"session_id": "session-1"})

    connection.new_session = files_by_room  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="must file a 'turn' session apart"):
        await _prompt(channel, _instruction(standalone=True), _context(*history))

    assert len(connection.prompt_calls) == 1  # the room's session was not told
    assert connection.closed_sessions == []
    following = make_event(room_id=ROOM, body="next request", index=1)
    await _prompt(channel, following, _context(*history, following))
    assert connection.prompt_calls[-1]["session_id"] == "session-1"
    await channel.close()


async def test_a_standalone_turn_on_an_agent_without_session_close_is_never_prompted_again(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Step 7: the agent keeps the turn's session, and the channel says so, once per turn."""
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    connection.takes_session_close = False  # read at initialize, before the first prompt
    first = make_event(room_id=ROOM, body="first request", index=0)
    await _prompt(channel, first, _context(first))

    await _prompt(channel, _instruction(standalone=True), _context(first))
    following = make_event(room_id=ROOM, body="next request", index=1)
    await _prompt(channel, following, _context(first, following))

    assert connection.closed_sessions == []
    assert [call["session_id"] for call in connection.prompt_calls] == [
        "session-1",
        "session-2",
        "session-1",
    ]
    assert "standalone turn session session-2 stays open" in caplog.text
    await channel.close()


async def test_a_standalone_turn_takes_the_rooms_configuration(tmp_path: Any) -> None:
    channel, connection, history = await _talked(tmp_path)
    await channel.set_config_option(ROOM, "model", "sonnet")
    connection.config_options = [_model_option("opus")]  # what a new session starts on

    await _prompt(channel, _instruction(standalone=True), _context(*history))

    assert connection.set_config_calls[-1] == {
        "config_id": "model",
        "session_id": "session-2",
        "value": "sonnet",
    }
    assert channel.session_config(ROOM) == {"model": "sonnet"}
    await channel.close()


async def test_cancelling_the_room_stops_its_standalone_turn(tmp_path: Any) -> None:
    channel, connection, history = await _talked(tmp_path)
    started = asyncio.Event()

    async def never_finishes(session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
        started.set()
        await asyncio.sleep(3600)
        return PromptResponse(stop_reason="end_turn")

    connection.prompt = never_finishes  # type: ignore[method-assign]
    output = await channel.on_event(_instruction(standalone=True), _binding(), _context(*history))
    consumer = asyncio.create_task(anext(output.response_stream))
    await started.wait()

    assert await channel.cancel(ROOM)
    assert connection.cancelled_sessions == ["session-2"]
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)
    await output.response_stream.aclose()
    await channel.close()


async def test_the_rooms_session_catches_up_on_a_standalone_reply(tmp_path: Any) -> None:
    """Another session produced it, so the room's session must be told (step 7)."""
    channel, connection, history = await _talked(tmp_path)
    output = await channel.on_event(_instruction(standalone=True), _binding(), _context(*history))
    [chunk async for chunk in output.response_stream]
    assert output.response_metadata["acp"]["standalone"] is True

    reply = make_event(room_id=ROOM, body="The summary.", index=1, channel_id="acp-agent")
    reply = reply.model_copy(update={"metadata": dict(output.response_metadata)})
    following = make_event(room_id=ROOM, body="next request", index=2)
    await _prompt(channel, following, _context(*history, reply, following))

    assert "you (in a separate session): “The summary.”" in _sent(connection, turn=2)
    await channel.close()


async def test_a_standalone_turn_interrupted_during_setup_leaves_no_session(
    tmp_path: Any,
) -> None:
    channel, connection, history = await _talked(tmp_path)
    await channel.set_config_option(ROOM, "model", "sonnet")
    connection.config_options = [_model_option("opus")]
    copying = asyncio.Event()

    async def hangs(**kwargs: Any) -> Any:
        copying.set()
        await asyncio.sleep(3600)

    connection.set_config_option = hangs  # type: ignore[method-assign]
    output = await channel.on_event(_instruction(standalone=True), _binding(), _context(*history))
    consumer = asyncio.create_task(anext(output.response_stream))
    await copying.wait()
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    assert connection.closed_sessions == ["session-2"]
    assert channel._turn_sessions == {}
    assert await channel.cancel(ROOM)
    assert connection.cancelled_sessions == ["session-1"]  # the room's, not the dead one
    await channel.close()


async def test_a_tool_setting_config_inside_a_first_standalone_turn_does_not_deadlock(
    tmp_path: Any,
) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    original_prompt = connection.prompt

    async def sets_config(session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
        await asyncio.wait_for(channel.set_config_option(ROOM, "model", "sonnet"), timeout=2)
        return await original_prompt(session_id, prompt, **kwargs)

    connection.prompt = sets_config  # type: ignore[method-assign]
    await _prompt(channel, _instruction(standalone=True), _context())

    assert channel.session_id(ROOM) == "session-2"  # the turn's was session-1
    assert channel.session_config(ROOM) == {"model": "sonnet"}
    await channel.close()

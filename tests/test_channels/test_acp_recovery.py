"""A host-authorized pre-execution refusal gets one atomic session rebuild."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import acp
import pytest
from acp.schema import PermissionOption

from roomkit import ACPSessionInvalidatedError
from roomkit.models.delivery import STANDALONE
from roomkit.models.enums import EventType
from roomkit.providers.ai.base import ProviderError
from roomkit.turns._error_codec import error_record
from tests.conftest import make_event
from tests.test_channels.test_acp import _binding, _channel, _context, _prompt

ROOM = "room-1"


def test_invalidation_retains_provider_attributes() -> None:
    refusal = ACPSessionInvalidatedError("unusable")
    assert error_record(refusal).attributes == {
        "retryable": False,
        "provider": "acp",
        "status_code": None,
        "context_overflow": None,
        "before_request": False,
        "recovery_authorized": False,
    }


def _refusal(*, authorized: bool = True) -> ACPSessionInvalidatedError:
    return ACPSessionInvalidatedError("session no longer usable", recovery_authorized=authorized)


async def _output(channel: Any, **fields: Any) -> Any:
    event = make_event(room_id=ROOM, body="request", **fields)
    return await channel.on_event(event, _binding(), _context(event))


async def _consume(output: Any) -> list[Any]:
    return [item async for item in output.response_stream]


@pytest.mark.parametrize("terminal", ["second_refusal", "open_error", "unauthorized", "ordinary"])
async def test_recovery_is_bounded_and_requires_authorization(
    tmp_path: Any, terminal: str
) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    seen: list[str] = []
    error = (
        RuntimeError("ordinary")
        if terminal == "ordinary"
        else _refusal(authorized=terminal != "unauthorized")
    )
    opener = connection.new_session

    async def prompt(session_id: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(session_id)
        raise error

    async def open_session(**kwargs: Any) -> Any:
        if seen and terminal == "open_error":
            raise RuntimeError("opening failed")
        return await opener(**kwargs)

    with (
        patch.object(connection, "prompt", prompt),
        patch.object(connection, "new_session", open_session),
    ):
        output = await _output(channel)
        with pytest.raises((ProviderError, RuntimeError)):
            await _consume(output)
    attempts = 2 if terminal == "second_refusal" else 1
    assert len(seen) == attempts
    assert output.response_metadata["acp"]["interrupted"] is True
    assert channel._turns == {}
    assert channel._prompted_index == {}
    assert all(not lock.locked() for lock in channel._room_locks.values())
    if terminal in {"second_refusal", "open_error"}:
        assert channel._sessions == channel._session_rooms == channel._session_options == {}
    await channel.close()


async def test_close_after_reconstruction_finishes_prevents_replacement_prompt(
    tmp_path: Any,
) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    replace = channel._replace_session
    closing: asyncio.Task[None] | None = None
    attempts: list[str] = []

    async def prompt(session_id: str, *_args: Any, **_kwargs: Any) -> Any:
        attempts.append(session_id)
        raise _refusal()

    async def replace_then_close(room_id: str, conn: Any) -> str:
        nonlocal closing
        session_id = await replace(room_id, conn)
        # Schedule shutdown before the completed runner wakes its consumer.
        closing = asyncio.create_task(channel.close())
        return session_id

    with (
        patch.object(connection, "prompt", prompt),
        patch.object(channel, "_replace_session", replace_then_close),
    ):
        output = await _output(channel)
        assert await asyncio.wait_for(_consume(output), 1) == []
        assert closing is not None
        await asyncio.wait_for(closing, 1)

    assert attempts == ["session-1"]
    assert output.response_metadata["acp"]["interrupted"] is True
    assert channel.active_turns == 0
    assert channel._sessions == channel._session_rooms == channel._session_options == {}
    assert channel._turns == channel._prompted_index == channel._room_locks == {}


@pytest.mark.parametrize("stage", ["close", "open", "configured"])
async def test_close_stops_reconstruction_without_cancelling_the_invalid_remote_session(
    tmp_path: Any, stage: str
) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    reached = asyncio.Event()
    opener = connection.new_session
    closer = connection.close_session
    publisher = channel._publish_config_options
    remote_cancels: list[str] = []
    attempts: list[str] = []

    async def pause() -> None:
        reached.set()
        await asyncio.Event().wait()

    async def prompt(session_id: str, *_args: Any, **_kwargs: Any) -> Any:
        attempts.append(session_id)
        raise _refusal()

    async def open_session(**kwargs: Any) -> Any:
        if attempts and stage == "open":
            await pause()
        return await opener(**kwargs)

    async def close_session(session_id: str) -> None:
        if stage == "close" and session_id == "session-1":
            await pause()
        await closer(session_id)

    async def publish(session_id: str, options: Any, values: Any) -> None:
        if stage == "configured" and session_id == "session-2":
            await pause()
        await publisher(session_id, options, values)

    async def remote_cancel(session_id: str) -> None:
        remote_cancels.append(session_id)
        await asyncio.Event().wait()

    with (
        patch.object(connection, "prompt", prompt),
        patch.object(connection, "new_session", open_session),
        patch.object(connection, "close_session", close_session),
        patch.object(channel, "_publish_config_options", publish),
        patch.object(connection, "cancel", remote_cancel),
        patch("roomkit.channels.acp._SHUTDOWN_TIMEOUT", 0.01),
    ):
        output = await _output(channel)
        consumer = asyncio.create_task(_consume(output))
        await asyncio.wait_for(reached.wait(), 1)
        await asyncio.wait_for(channel.close(), 1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, 1)

    assert attempts == ["session-1"]
    assert remote_cancels == []
    assert output.response_metadata["acp"]["interrupted"] is True
    assert channel._sessions == channel._session_rooms == channel._session_options == {}
    assert channel._turns == channel._prompted_index == channel._room_locks == {}


@pytest.mark.parametrize("activity", ["text", "thinking", "tool", "plan", "permission"])
async def test_activity_makes_even_an_authorized_signal_terminal(
    tmp_path: Any, activity: str
) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    seen: list[str] = []

    async def prompt(session_id: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(session_id)
        updates = {
            "text": acp.update_agent_message_text("partial answer"),
            "thinking": acp.update_agent_thought_text("already working"),
            "tool": acp.start_tool_call("tool-1", "Write", kind="edit", status="in_progress"),
            "plan": acp.update_plan([acp.plan_entry("Working", status="in_progress")]),
        }
        if activity == "permission":
            await connection.client.request_permission(
                session_id,
                acp.update_tool_call("tool-1", title="Write"),
                [PermissionOption(option_id="no", name="No", kind="reject_once")],
            )
        else:
            await connection.client.session_update(session_id, updates[activity])
        raise _refusal()

    with patch.object(connection, "prompt", prompt):
        output = await _output(channel)
        with pytest.raises(ProviderError, match="activity"):
            await _consume(output)
    assert seen == ["session-1"]
    assert connection.closed_sessions == []
    assert channel._turns == {}
    assert output.response_metadata["acp"]["interrupted"] is True
    await channel.close()


async def test_queued_updates_are_drained_before_considering_recovery(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)

    async def prompt(*_args: Any, **_kwargs: Any) -> Any:
        raise _refusal()

    async def drain(session_id: str) -> None:
        await connection.client.session_update(session_id, acp.update_agent_message_text("queued"))

    with (
        patch.object(connection, "prompt", prompt),
        patch.object(channel, "_drain_session_updates", drain),
    ):
        output = await _output(channel)
        with pytest.raises(ProviderError, match="activity"):
            await _consume(output)
    assert len(connection.new_session_calls) == 1
    await channel.close()


async def test_standalone_never_rebuilds_or_changes_the_room_session(tmp_path: Any) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    first = make_event(room_id=ROOM, body="earlier", index=4)
    await _prompt(channel, first, _context(first))

    async def prompt(*_args: Any, **_kwargs: Any) -> Any:
        raise _refusal()

    with patch.object(connection, "prompt", prompt):
        output = await _output(channel, type=EventType.INSTRUCTION, metadata={STANDALONE: True})
        with pytest.raises(ACPSessionInvalidatedError):
            await _consume(output)
    assert len(connection.new_session_calls) == 2
    assert connection.closed_sessions == ["session-2"]
    assert channel.session_id(ROOM) == "session-1"
    assert channel._prompted_index == {ROOM: 4}
    assert channel._turn_sessions == channel._turns == {}
    await channel.close()


@pytest.mark.parametrize("stage", ["close", "open", "prompt", "abandon"])
async def test_cancel_or_abandon_recovery_cleans_state_and_releases_the_lock(
    tmp_path: Any, stage: str
) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    reached = asyncio.Event()
    attempts = 0
    opener = connection.new_session
    closer = connection.close_session

    async def hang() -> None:
        reached.set()
        await asyncio.Event().wait()

    async def prompt(session_id: str, *_args: Any, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _refusal()
        if stage == "abandon":
            await connection.client.session_update(
                session_id, acp.update_agent_message_text("partial")
            )
        await hang()

    async def open_session(**kwargs: Any) -> Any:
        if attempts and stage == "open":
            await hang()
        return await opener(**kwargs)

    async def close_session(session_id: str) -> None:
        if stage == "close":
            await hang()
        await closer(session_id)

    with (
        patch.object(connection, "prompt", prompt),
        patch.object(connection, "new_session", open_session),
        patch.object(connection, "close_session", close_session),
    ):
        output = await _output(channel)
        if stage == "abandon":
            assert await anext(output.response_stream) == "partial"
            await asyncio.wait_for(output.response_stream.aclose(), 1)
        else:
            consumer = asyncio.create_task(_consume(output))
            await asyncio.wait_for(reached.wait(), 1)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(consumer, 1)
    assert channel._sessions == channel._session_rooms == channel._session_options == {}
    assert channel._turns == channel._prompted_index == channel._room_locks == {}
    assert output.response_metadata["acp"]["interrupted"] is True
    # A following prompt obtains a fresh session; no orphaned lock or runner.
    await asyncio.wait_for(_consume(await _output(channel)), 1)
    await channel.close()


@pytest.mark.parametrize("stage", ["close", "open", "configured"])
async def test_public_cancel_stops_reconstruction_before_prompt(tmp_path: Any, stage: str) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    reached = asyncio.Event()
    attempts: list[str] = []
    opener = connection.new_session
    closer = connection.close_session
    publisher = channel._publish_config_options

    async def pause() -> None:
        reached.set()
        await asyncio.Event().wait()

    async def prompt(session_id: str, *_args: Any, **_kwargs: Any) -> Any:
        attempts.append(session_id)
        raise _refusal()

    async def open_session(**kwargs: Any) -> Any:
        if attempts and stage == "open":
            await pause()
        return await opener(**kwargs)

    async def close_session(session_id: str) -> None:
        if stage == "close" and session_id == "session-1":
            await pause()
        await closer(session_id)

    async def publish(session_id: str, options: Any, values: Any) -> None:
        if stage == "configured" and session_id == "session-2":
            await pause()
        await publisher(session_id, options, values)

    with (
        patch.object(connection, "prompt", prompt),
        patch.object(connection, "new_session", open_session),
        patch.object(connection, "close_session", close_session),
        patch.object(channel, "_publish_config_options", publish),
    ):
        output = await _output(channel)
        consumer = asyncio.create_task(_consume(output))
        await asyncio.wait_for(reached.wait(), 1)
        assert channel.active_turns == 1
        assert await channel.cancel(ROOM) is True
        assert await asyncio.wait_for(consumer, 1) == []

    assert attempts == ["session-1"]
    assert output.response_metadata["acp"]["interrupted"] is True
    assert channel.active_turns == 0
    assert channel._sessions == channel._session_rooms == channel._session_options == {}
    assert channel._turns == channel._prompted_index == channel._room_locks == {}
    await asyncio.wait_for(_consume(await _output(channel)), 1)
    await channel.close()


async def test_finished_preparation_is_not_cancelled_again_during_release(tmp_path: Any) -> None:
    """Cleanup holds the room after the cancelled preparation has left its registry."""
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    reached, release = asyncio.Event(), asyncio.Event()
    opener, closer = connection.new_session, connection.close_session
    opened = 0

    async def refuses(*args: Any, **kwargs: Any) -> Any:
        raise _refusal()

    async def opens(**kwargs: Any) -> Any:
        nonlocal opened
        result = await opener(**kwargs)
        opened += 1
        if opened == 2:
            assert await channel.cancel(ROOM) is True
        return result

    async def closes(session_id: str) -> None:
        if session_id == "session-2":
            reached.set()
            await release.wait()
        await closer(session_id)

    with (
        patch.object(connection, "prompt", refuses),
        patch.object(connection, "new_session", opens),
        patch.object(connection, "close_session", closes),
    ):
        output = await _output(channel)
        consuming = asyncio.create_task(_consume(output))
        await asyncio.wait_for(reached.wait(), 5)
        assert channel.session_id(ROOM) is None
        assert await channel.cancel(ROOM) is False
        release.set()
        assert await asyncio.wait_for(consuming, 5) == []
        assert output.response_metadata["acp"]["interrupted"] is True
    await channel.close()

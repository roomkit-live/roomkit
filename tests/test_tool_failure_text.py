"""A tool that raised reads the same on every channel, without its message (RMK-295).

RFC §9.3: the model reads ``{"error": "Tool 'x' failed
(<ExceptionClass>)"}``; the message, which can hold anything the failing code
held, goes to the log and to ON_TOOL_CALL's observers
(``ToolCallEvent.error_detail``), never to the model nor to the stored
TOOL_CALL_END. ``ToolRefusedError`` keeps its words.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookTrigger,
    RoomKit,
)
from roomkit.channels._skill_handlers import handle_run_script
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.mixins.delegation import _delegation_result_text
from roomkit.models.context import RoomContext
from roomkit.models.tool_call import ToolCallEvent
from roomkit.sandbox.executor import SandboxExecutor
from roomkit.tasks.models import DelegatedTaskResult, TaskStatus
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import _call, realtime_kit
from tests.test_realtime_fixed_tools import call
from tests.test_realtime_tool_recovery import _injected
from tests.test_unified_tool_call import _ai_room, _call_one_tool

SECRET = "postgres://admin:hunter2@db/internal"


def _leaks(value: Any) -> bool:
    return "hunter2" in json.dumps(value, default=str)


async def _raises(*_: Any) -> str:
    raise ConnectionError(f"cannot reach {SECRET}")


async def test_the_ai_channel_reads_the_class_and_observers_get_the_message(
    streaming: bool, caplog: pytest.LogCaptureFixture
) -> None:
    kit, ch, room_id, observed, _ = await _ai_room(streaming=streaming, tool_handler=_raises)

    run = await _call_one_tool(kit, ch, room_id, "get_weather")

    assert run.calls[0].result == '{"error": "Tool \'get_weather\' failed (ConnectionError)"}'
    # What the TOOL_CALL_END row is built from: the loop's record of the call.
    assert not _leaks([run.calls[0].result, run.calls[0].error])
    assert [e.error_detail for e in observed] == [f"ConnectionError: cannot reach {SECRET}"]
    assert SECRET in caplog.text
    await kit.close()


async def _realtime(**kwargs: Any) -> tuple[RoomKit, RealtimeVoiceChannel, Any, Any, list]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[{"name": "lookup", "description": "Look up", "parameters": {}}],
        tool_handler=_raises,
        **kwargs,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt")
    session = await channel.start_session(room.id, "u1", object())
    return kit, channel, provider, session, observed


async def test_a_realtime_call_reads_the_class_and_observers_get_the_message() -> None:
    kit, channel, provider, session, observed = await _realtime()
    try:
        result = await call(channel, provider, session, "lookup", {})
        await asyncio.sleep(0.05)
    finally:
        await kit.close()

    assert result == {"error": "Tool 'lookup' failed (ConnectionError)"}
    assert [e.error_detail for e in observed] == [f"ConnectionError: cannot reach {SECRET}"]


async def test_a_recovered_call_that_raised_is_reported_to_the_model_and_the_observers() -> None:
    kit, _channel, provider, session, observed = await _realtime()
    try:
        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
        await asyncio.sleep(0.1)
    finally:
        await kit.close()

    injected = _injected(provider)[-1]
    assert injected.startswith("[Tool lookup failed]\n<tool_result>\n")
    assert "hunter2" not in injected
    assert [e.error_detail for e in observed] == [f"ConnectionError: cannot reach {SECRET}"]


async def test_the_conference_reads_the_class_and_observers_get_the_message() -> None:
    provider = MockRealtimeProvider()

    async def handler(room_id: str, tool: str, args: dict[str, Any]) -> str:
        raise ConnectionError(f"cannot reach {SECRET}")

    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(
            provider=provider,
            tools=[{"name": "lookup", "description": "Look up", "parameters": {}}],
            tool_handler=handler,
        ),
    )
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    result = await _call(channel, provider, observed, "lookup", {})

    assert result == {"error": "Tool 'lookup' failed (ConnectionError)"}
    assert [e.error_detail for e in observed] == [f"ConnectionError: cannot reach {SECRET}"]
    await kit.close()


async def test_a_skill_script_that_raised_raises_on_to_the_channel() -> None:
    skills = MagicMock()
    skills.get_skill.return_value = MagicMock()
    executor = AsyncMock()
    executor.execute.side_effect = ConnectionError(f"cannot reach {SECRET}")

    with pytest.raises(ConnectionError):
        await handle_run_script({"skill_name": "s", "script_name": "run"}, skills, executor)


class _FailingSandbox(SandboxExecutor):
    async def execute(self, command: str, arguments: dict[str, Any] | None = None) -> Any:
        raise ConnectionError(f"cannot reach {SECRET}")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [{"name": "sandbox_bash", "description": "Run", "parameters": {"type": "object"}}]


async def test_a_sandbox_command_that_raised_is_a_failed_call(streaming: bool) -> None:
    """The channel reads it as any raised call: the marker, the class for the
    model, the message for the observers."""
    kit, ch, room_id, observed, _ = await _ai_room(streaming=streaming, tool_handler=_raises)
    ch._sandbox = _FailingSandbox()

    run = await _call_one_tool(kit, ch, room_id, "sandbox_bash")

    assert run.calls[0].failed
    assert run.calls[0].result == '{"error": "Tool \'sandbox_bash\' failed (ConnectionError)"}'
    assert [e.error_detail for e in observed] == [f"ConnectionError: cannot reach {SECRET}"]
    await kit.close()


async def test_hooks_that_raised_on_an_unserved_call_reach_the_observers(
    streaming: bool,
) -> None:
    async def unknown(name: str, arguments: dict[str, Any]) -> str:
        return json.dumps({"error": f"Unknown tool: {name}"})

    kit, ch, room_id, observed, _ = await _ai_room(streaming=streaming, tool_handler=unknown)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="server")
    async def server(event: ToolCallEvent, ctx: RoomContext) -> None:
        raise RuntimeError(f"cannot reach {SECRET}")

    run = await _call_one_tool(kit, ch, room_id, "get_weather")

    assert run.calls[0].result == '{"error": "No handler for tool get_weather"}'
    assert [e.error_detail for e in observed] == [f"server: cannot reach {SECRET}"]
    await kit.close()


def test_a_failed_delegated_task_reads_as_failed_without_its_error() -> None:
    result = DelegatedTaskResult(
        task_id="t1",
        child_room_id="c1",
        parent_room_id="p1",
        agent_id="worker",
        status=TaskStatus.FAILED,
        output=None,
        error=f"ConnectionError: cannot reach {SECRET}",
        duration_ms=1.0,
    )

    text = _delegation_result_text(result)

    assert text.startswith("[Background task from worker failed.")
    assert "hunter2" not in text


async def test_a_recovered_call_is_reported_even_when_the_model_cannot_be_told() -> None:
    kit, _channel, provider, session, observed = await _realtime()
    try:
        provider.inject_text = AsyncMock(side_effect=RuntimeError("socket closed"))
        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
        await asyncio.sleep(0.1)
    finally:
        await kit.close()

    assert [e.error_detail for e in observed] == [f"ConnectionError: cannot reach {SECRET}"]


async def test_realtime_hooks_that_raised_on_an_unserved_call_reach_the_observers() -> None:
    kit, channel, provider, session, observed = await _realtime()
    channel._tool_handler = None

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="server")
    async def server(event: ToolCallEvent, ctx: RoomContext) -> None:
        raise RuntimeError(f"cannot reach {SECRET}")

    try:
        result = await call(channel, provider, session, "lookup", {})
        await asyncio.sleep(0.05)
    finally:
        await kit.close()

    assert result == {"error": "No handler for tool lookup"}
    assert [e.error_detail for e in observed] == [f"server: cannot reach {SECRET}"]

"""A refused call reaches ON_TOOL_CALL's observers only, marked ``refused``,
on every door (RMK-432, RFC §9.3).

A gate's refusal, an external handler's refusal (through its
``on_tool_refused``), a call the external door refused itself and a rejected
ACP permission never reach a SYNC hook. A handler that raised is a failure
the channel reports with what failed. An ACP call reports the same body with
and without an external handler.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import acp
import pytest
from acp import PromptResponse

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.exceptions import ToolRefusedError
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import BeforeToolDecision, PolicyExternalToolHandler, ToolDecision
from roomkit.tools.result import cancelled_tool_error
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_channels.test_acp import _channel
from tests.test_framework import SimpleChannel

SECRET = "postgres://admin:hunter2@db"


class _Denying(PolicyExternalToolHandler):
    async def process_tool_call(self, tool_name: str, tool_input: Any, **kw: Any) -> ToolDecision:
        return ToolDecision(approved=False, reason="not on this host")


class _Raising(PolicyExternalToolHandler):
    async def process_tool_call(self, tool_name: str, tool_input: Any, **kw: Any) -> ToolDecision:
        raise RuntimeError(SECRET)


class _Heard:
    """What the SYNC hooks and the ASYNC observers of ON_TOOL_CALL received."""

    def __init__(self, kit: RoomKit) -> None:
        self.sync: list[ToolCallEvent] = []
        self.observed: list[ToolCallEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: Any) -> HookResult:
            self.sync.append(event)
            return HookResult.allow()

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: ToolCallEvent, ctx: Any) -> None:
            self.observed.append(event)


async def _room(kit: RoomKit, agent_id: str) -> None:
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "sms")
    await kit.attach_channel("room-1", agent_id, category=ChannelCategory.INTELLIGENCE)


async def _ask(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.2)


async def _external_door(handler: Any, call: AIToolCall) -> _Heard:
    kit = RoomKit()
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ]
    )
    kit.register_channel(AIChannel("ai1", provider=provider, external_tool_handler=handler))
    heard = _Heard(kit)
    await _room(kit, "ai1")
    await _ask(kit)
    await kit.close()
    return heard


BASH = AIToolCall(id="p1", name="Bash", arguments={"cmd": "ls"})
_BASH_TOOL = AITool(
    name="Bash", description="Runs a command.", parameters={"type": "object", "properties": {}}
)
CUT = AIToolCall(id="p1", name="Bash", arguments={"raw": "[1"}, partial=True)


@pytest.mark.parametrize(
    ("handler", "call"),
    [(_Denying(), BASH), (PolicyExternalToolHandler(), CUT)],
    ids=["handler-refuses", "channel-refuses-a-cut-call"],
)
async def test_an_external_door_refusal_reaches_the_observers_only(
    handler: Any, call: AIToolCall
) -> None:
    heard = await _external_door(handler, call)

    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused, event.cancelled) == (True, True, False)


async def test_an_external_handler_that_raises_is_a_failure_with_its_detail() -> None:
    heard = await _external_door(_Raising(), BASH)

    # The call never ran: no SYNC hook hears of it, as on the local door.
    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused) == (True, False)
    assert event.error_detail == f"RuntimeError: {SECRET}"
    assert SECRET not in str(event.result)


async def _acp(handler: Any, *, status: str, raw_output: Any) -> _Heard:
    async def prompt(connection: Any, session_id: str, *a: Any, **k: Any) -> PromptResponse:
        await connection.client.session_update(
            session_id,
            acp.start_tool_call("tool-1", "Read file", kind="read", status="in_progress"),
        )
        await connection.client.session_update(
            session_id, acp.update_tool_call("tool-1", status=status, raw_output=raw_output)
        )
        return PromptResponse(stop_reason="end_turn")

    with tempfile.TemporaryDirectory() as tmp:
        kit = RoomKit()
        channel, connection, _ = _channel(Path(tmp), handler=handler, emit_updates=False)
        connection.prompt = lambda *a, **k: prompt(connection, *a, **k)  # type: ignore[method-assign]
        kit.register_channel(channel)
        heard = _Heard(kit)
        await _room(kit, "acp-agent")
        await _ask(kit)
        await kit.close()
        return heard


async def _acp_cut(handler: Any) -> _Heard:
    """An ACP turn that stops while its tool runs: the call is cancelled."""

    async def start_then_stop(
        connection: Any, session_id: str, *a: Any, **k: Any
    ) -> PromptResponse:
        await connection.client.session_update(
            session_id,
            acp.start_tool_call("tool-1", "Write", kind="edit", status="in_progress"),
        )
        return PromptResponse(stop_reason="cancelled")

    with tempfile.TemporaryDirectory() as tmp:
        kit = RoomKit()
        channel, connection, _ = _channel(Path(tmp), handler=handler, emit_updates=False)
        connection.prompt = lambda *a, **k: start_then_stop(connection, *a, **k)  # type: ignore[method-assign]
        kit.register_channel(channel)
        heard = _Heard(kit)
        await _room(kit, "acp-agent")
        await _ask(kit)
        await kit.close()
        return heard


async def _acp_permission(handler: Any) -> _Heard:
    with tempfile.TemporaryDirectory() as tmp:
        kit = RoomKit()
        channel, connection, _ = _channel(Path(tmp), handler=handler)
        connection.tool_status = "failed"
        connection.tool_raw_output = {"error": "permission rejected"}
        kit.register_channel(channel)
        heard = _Heard(kit)
        await _room(kit, "acp-agent")
        await _ask(kit)
        await kit.close()
        return heard


@pytest.mark.parametrize("handler", [None, _Denying()], ids=["no-handler", "denying-handler"])
async def test_a_rejected_acp_permission_reaches_the_observers_only(handler: Any) -> None:
    heard = await _acp_permission(handler)

    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused) == (True, True)


async def test_an_acp_handler_that_raises_is_a_failure_with_its_detail() -> None:
    heard = await _acp_permission(_Raising())

    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused) == (True, False)
    assert event.error_detail == f"RuntimeError: {SECRET}"


@pytest.mark.parametrize(
    "raw_output",
    [[{"type": "image", "mimeType": "image/png", "data": "A" * (600 * 1024)}], None],
    ids=["image", "nothing"],
)
async def test_an_acp_failure_reports_one_body_with_or_without_a_handler(raw_output: Any) -> None:
    bodies = []
    for handler in (None, PolicyExternalToolHandler()):
        heard = await _acp(handler, status="failed", raw_output=raw_output)
        [event] = [*heard.sync, *heard.observed][:1]
        bodies.append(event.result)

    assert bodies[0] == bodies[1]
    assert len(str(bodies[0])) < 1000


async def test_a_gate_refusal_is_marked_refused_on_the_local_door() -> None:
    kit = RoomKit()
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[BASH]),
            AIResponse(content="done"),
        ]
    )
    kit.register_channel(AIChannel("ai1", provider=provider, tool_handler=lambda *a: "ran"))
    heard = _Heard(kit)
    await _room(kit, "ai1")
    await _ask(kit)
    await kit.close()

    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused, event.cancelled) == (True, True, False)


async def test_a_gate_refusal_is_marked_refused_on_a_realtime_session() -> None:
    kit = RoomKit()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[{"name": "lookup", "description": "lookup", "parameters": {"type": "object"}}],
        tool_handler=lambda *a: "ran",
    )
    kit.register_channel(channel)
    heard = _Heard(kit)
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "rt")
    session = await channel.start_session("room-1", "u", "ws")

    await provider.simulate_tool_call(session, "c1", "nope", {})
    for _ in range(100):
        if heard.observed:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused) == (True, True)


class _Recording(PolicyExternalToolHandler):
    """Approves with an input ACP cannot apply, and records what reaches it."""

    def __init__(self) -> None:
        super().__init__()
        self.refused: list[str] = []
        self.results: list[str] = []

    async def process_tool_call(self, tool_name: str, tool_input: Any, **kw: Any) -> ToolDecision:
        return ToolDecision(approved=True, modified_input={"path": "/elsewhere"})

    async def on_tool_result(self, tool_name: str, *args: Any, **kwargs: Any) -> None:
        self.results.append(tool_name)

    async def on_tool_refused(
        self, tool_name: str, tool_input: Any, reason: str, **kwargs: Any
    ) -> None:
        self.refused.append(tool_name)
        await super().on_tool_refused(tool_name, tool_input, reason, **kwargs)


async def test_a_call_the_external_door_cut_is_the_channel_s_to_report() -> None:
    handler = _Recording()

    heard = await _external_door(handler, CUT)

    assert (handler.refused, handler.results) == ([], [])
    [event] = heard.observed
    assert event.refused


async def test_an_acp_refusal_the_channel_imposed_is_the_channel_s_to_report() -> None:
    """The handler approved with an input ACP cannot apply: the channel
    refused, so the handler is not told it refused (RMK-432)."""
    handler = _Recording()

    heard = await _acp_permission(handler)

    assert handler.refused == []
    assert heard.sync == []
    [event] = heard.observed
    assert (event.is_error, event.refused) == (True, True)


async def test_the_tool_call_framework_event_carries_refused() -> None:
    kit = RoomKit()
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[BASH]),
            AIResponse(content="done"),
        ]
    )
    kit.register_channel(AIChannel("ai1", provider=provider, tool_handler=lambda *a: "ran"))
    framework: list[dict[str, Any]] = []

    @kit.on("tool_call")
    async def on_tool_call(event: Any) -> None:
        framework.append(dict(event.data))

    await _room(kit, "ai1")
    await _ask(kit)
    await kit.close()

    [data] = framework
    assert data.get("refused") is True


async def test_a_handler_s_tool_refused_error_is_marked_refused() -> None:
    async def refuses(name: str, arguments: dict[str, Any]) -> str:
        raise ToolRefusedError(json.dumps({"error": "not today"}))

    kit = RoomKit()
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[BASH]),
            AIResponse(content="done"),
        ]
    )
    kit.register_channel(
        AIChannel("ai1", provider=provider, tools=[_BASH_TOOL], tool_handler=refuses)
    )
    heard = _Heard(kit)
    await _room(kit, "ai1")
    await _ask(kit)
    await kit.close()

    [event] = heard.observed
    assert (event.is_error, event.refused) == (True, True)


async def test_an_acp_cancellation_reports_the_cancellation_envelope() -> None:
    heard = await _acp_cut(None)

    [event] = heard.observed
    assert json.loads(str(event.result)) == json.loads(
        cancelled_tool_error("Write", "The turn ended before its result.")
    )


async def test_a_handler_can_report_what_failed() -> None:
    class _Detailing(PolicyExternalToolHandler):
        async def on_tool_result(self, tool_name: str, tool_input: Any, result: str, **kw: Any):
            await self._fire_on_tool_hook(
                tool_name,
                tool_input,
                result,
                is_error=True,
                error_detail="disk full",
                tool_call_id=kw.get("tool_call_id", ""),
                room_id=kw.get("room_id"),
            )

    heard = await _external_door(_Detailing(), BASH)

    [event] = [*heard.sync, *heard.observed][-1:]
    assert event.error_detail == "disk full"


class _FailedClosed(PolicyExternalToolHandler):
    """A refusal that came from a failure: a gate that could not decide."""

    async def process_tool_call(self, tool_name: str, tool_input: Any, **kw: Any) -> ToolDecision:
        return ToolDecision(approved=False, reason="denied", detail="gate: approval db down")


async def _door(door: str, handler: Any) -> _Heard:
    if door == "ai":
        return await _external_door(handler, BASH)
    return await _acp_permission(handler)


@pytest.mark.parametrize("door", ["ai", "acp"])
async def test_a_refusal_from_a_failure_carries_what_failed_on_the_external_doors(
    door: str,
) -> None:
    """The observers read what failed (``error_detail``) as on the channel's
    own doors; the agent reads the refusal only (RMK-498, RFC §9.3)."""
    heard = await _door(door, _FailedClosed())

    [event] = heard.observed
    assert (event.is_error, event.refused) == (True, True)
    assert event.error_detail == "gate: approval db down"
    assert "approval db down" not in str(event.result)


async def test_the_policy_handler_hands_on_what_a_failed_closed_hook_said() -> None:
    handler = PolicyExternalToolHandler()

    async def failed_closed(event: ToolCallEvent) -> BeforeToolDecision:
        return BeforeToolDecision(allowed=False, detail="gate: approval db down")

    handler._before_tool_hook = failed_closed
    decision = await handler.process_tool_call("Bash", {"cmd": "ls"}, tool_call_id="p1")

    assert (decision.approved, decision.detail) == (False, "gate: approval db down")


async def _acp_ran_anyway(handler: Any) -> tuple[_Heard, list[Any]]:
    """An ACP agent whose permission RoomKit refused runs the call anyway and
    closes it completed."""
    with tempfile.TemporaryDirectory() as tmp:
        kit = RoomKit()
        channel, connection, _ = _channel(Path(tmp), handler=handler)
        connection.tool_status = "completed"
        connection.tool_raw_output = {"content": "written"}
        kit.register_channel(channel)
        heard = _Heard(kit)
        await _room(kit, "acp-agent")
        await _ask(kit)
        rows = [
            event.content
            for event in await kit.store.list_events("room-1")
            if event.type.value == "tool_call_end"
        ]
        await kit.close()
        return heard, rows


@pytest.mark.parametrize("handler", [None, _Denying()], ids=["no-handler", "denying-handler"])
async def test_an_acp_call_refused_then_run_is_reported_served_with_the_refusal_marked(
    handler: Any,
) -> None:
    """Reported as it ended, served, its flags unchanged; ``refused_but_ran``
    on its report and its END row tells an audit what it went past (RMK-498)."""
    heard, rows = await _acp_ran_anyway(handler)

    [event] = heard.observed
    assert (event.is_error, event.refused, event.refused_but_ran) == (False, False, True)
    [row] = rows
    assert (row.outcome, row.refused_but_ran) == ("served", True)

"""Every outcome of a tool call reaches ON_TOOL_CALL as the model read it, on
every door (RMK-395, RFC §9.3): a text call a stop or a cancellation cut, an
ACP agent's call with or without an external handler and its outcome, a
report's chain, a realtime skill activation, and a pipeline agent's handler.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import acp
import pytest
from acp import PromptResponse
from acp.schema import PermissionOption

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import RoomEvent, ToolCallContent
from roomkit.models.steering import Cancel
from roomkit.models.streaming import ToolCallStartMarker
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITool,
    AIToolCall,
    ServedCall,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.context import _ToolLoopContext
from roomkit.tools.external import PolicyExternalToolHandler
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_channels.test_acp import _channel, _RecordingToolHandler
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="Look up", parameters={"type": "object"})


async def until(predicate: Any, timeout: float = 5.0) -> None:
    """Wait until *predicate* holds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


def _observe(kit: RoomKit) -> list[ToolCallEvent]:
    seen: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        seen.append(event)

    return seen


def _call(name: str, served: str | None = None, **arguments: Any) -> AIResponse:
    """A response calling *name*; *served*, the result of a call the provider ran."""
    ran = ServedCall(result=served) if served is not None else None
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c1", name=name, arguments=arguments, served=ran)],
    )


async def _tool_rows(kit: RoomKit, room_id: str) -> list[tuple[str, str | None]]:
    return [
        (e.type.value, e.content.outcome)
        for e in await kit.store.list_events(room_id)
        if isinstance(e.content, ToolCallContent)
    ]


# -- A text call a stop or a cancellation cut is reported cancelled ----------


@pytest.mark.parametrize("streaming", [True, False])
async def test_a_stop_while_calls_are_announced_reports_them_cancelled(streaming: bool) -> None:
    ran: list[str] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(
            ai_responses=[_call("lookup"), AIResponse(content="done")], streaming=streaming
        ),
        tool_handler=lookup,
        tools=[LOOKUP],
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    seen = _observe(kit)
    loop = _ToolLoopContext(room_id="r1")
    loop.all_context_tools = [LOOKUP]
    context = AIContext(messages=[AIMessage(role="user", content="go")], tools=[LOOKUP])

    async for delta in channel._run_streaming_tool_loop(context, parent_loop_ctx=loop):
        if isinstance(delta, ToolCallStartMarker):
            channel.steer(Cancel(reason="user stopped"))
    await until(lambda: bool(seen))

    assert ran == []
    assert [(e.name, e.is_error, e.cancelled) for e in seen] == [("lookup", True, True)]
    await kit.close()


async def test_a_turn_cancelled_while_a_handler_runs_reports_the_call_cancelled() -> None:
    started = asyncio.Event()

    async def slow(name: str, arguments: dict[str, Any]) -> str:
        started.set()
        await asyncio.sleep(30)
        return "ok"

    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=MockAIProvider(ai_responses=[_call("lookup"), AIResponse(content="done")]),
            tool_handler=slow,
            tools=[LOOKUP],
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    seen = _observe(kit)
    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
    )
    await asyncio.wait_for(started.wait(), 5)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    await until(lambda: bool(seen))

    assert [(e.name, e.is_error, e.cancelled) for e in seen] == [("lookup", True, True)]
    assert ("tool_call_end", "cancelled") in await _tool_rows(kit, "r1")
    await kit.close()


# -- An ACP agent's call is reported with the outcome every channel names -----


async def _acp_room(
    tmp: Path, *, handler: Any = None, prompt: Any = None, **connection: Any
) -> tuple[RoomKit, list[ToolCallEvent]]:
    kit = RoomKit()
    channel, conn, _ = _channel(tmp, handler=handler, emit_updates=prompt is None)
    if prompt is not None:
        conn.prompt = lambda *a, **k: prompt(conn, *a, **k)  # type: ignore[method-assign]
    for name, value in connection.items():
        setattr(conn, name, value)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "sms")
    await kit.attach_channel("room-1", "acp-agent", category=ChannelCategory.INTELLIGENCE)
    seen = _observe(kit)
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.1)
    return kit, seen


async def test_an_acp_call_is_reported_without_an_external_handler(tmp_path: Path) -> None:
    kit, seen = await _acp_room(tmp_path)

    assert [(e.is_error, e.cancelled) for e in seen] == [(False, False)]
    assert ("tool_call_end", "served") in await _tool_rows(kit, "room-1")
    await kit.close()


async def _start_then_stop(connection: Any, session_id: str, prompt: list[Any], **kw: Any) -> Any:
    await connection.client.session_update(
        session_id,
        acp.start_tool_call(
            "tool-1", "Write", kind="edit", status="in_progress", raw_input={"path": "/tmp/n.md"}
        ),
    )
    return PromptResponse(stop_reason="cancelled")


async def test_an_acp_call_the_turn_ended_under_is_cancelled(tmp_path: Path) -> None:
    kit, seen = await _acp_room(tmp_path, prompt=_start_then_stop)

    assert ("tool_call_end", "cancelled") in await _tool_rows(kit, "room-1")
    assert [(e.is_error, e.cancelled) for e in seen] == [(True, True)]
    await kit.close()


async def test_an_acp_call_whose_permission_was_refused_is_refused(tmp_path: Path) -> None:
    kit, _ = await _acp_room(
        tmp_path,
        handler=_RecordingToolHandler(approved=False),
        tool_status="failed",
        tool_raw_output={"error": "permission denied"},
    )

    assert ("tool_call_end", "refused") in await _tool_rows(kit, "room-1")
    await kit.close()


# -- A report's chain folds each rewrite as the judged chain does ------------


@pytest.mark.parametrize("form", ["modify", "metadata"])
async def test_a_report_chain_shows_the_next_hook_each_rewrite(form: str) -> None:
    provider_run = _call("Read", served="secret")
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=MockAIProvider(
                ai_responses=[provider_run, AIResponse(content="done")], streaming=True
            ),
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    second: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="redact", priority=0)
    async def redact(event: ToolCallEvent, ctx: Any) -> HookResult:
        if form == "modify":
            return HookResult.modify(dataclasses.replace(event, result="[REDACTED]"))
        return HookResult(action="allow", metadata={"result": "[REDACTED]"})

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="cite", priority=1)
    async def cite(event: ToolCallEvent, ctx: Any) -> HookResult:
        second.append(event.result)
        return HookResult.allow()

    seen = _observe(kit)
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
    )
    await until(lambda: bool(seen))

    assert second == ["[REDACTED]"]
    # A report: the observers see what the agent read, not the rewrite.
    assert [e.result for e in seen] == ["secret"]
    await kit.close()


# -- A realtime skill activation is reported as the model reads it -----------


def _registry(tmp: Path) -> SkillRegistry:
    skill_dir = tmp / "s1"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: s1\ndescription: Test\nrequires: lookup\n---\nBody", encoding="utf-8"
    )
    registry = SkillRegistry()
    registry.discover(tmp)
    return registry


@pytest.mark.parametrize("missing", ["at-activation", "while-the-hooks-run"])
async def test_a_realtime_activation_missing_its_tools_is_refused_for_all(
    tmp_path: Path, missing: str
) -> None:
    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=lookup,
        tools=[] if missing == "at-activation" else [{"name": "lookup", "parameters": {}}],
        skills=_registry(tmp_path),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")
    seen = _observe(kit)

    if missing == "while-the-hooks-run":

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="handoff")
        async def handoff(event: ToolCallEvent, ctx: Any) -> HookResult:
            # Stands for a handoff reconfiguring the session meanwhile.
            channel._session_tools[session.id] = []
            return HookResult.allow()

    await provider.simulate_tool_call(session, "c1", "activate_skill", {"name": "s1"})
    await until(lambda: bool(provider.tool_results) and bool(seen))

    body = json.loads(provider.tool_results[0][2])
    assert body["error"] == "Required tools not available: lookup"
    assert [(e.name, e.is_error) for e in seen] == [("activate_skill", True)]
    assert seen[0].result == provider.tool_results[0][2]
    await kit.close()


# -- A pipeline agent's "not mine" answer is a call nothing served ------------


async def test_a_pipeline_agent_declining_a_call_is_unserved() -> None:
    async def declining(name: str, arguments: dict[str, Any]) -> str:
        return json.dumps({"error": f"Unknown tool: {name}"})

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel("rtv", provider=provider, transport=MockRealtimeTransport())
    kit = RoomKit()
    kit.register_channel(channel)
    agent = Agent(
        "agent-a", role="A", voice="v", system_prompt="x", tools=[LOOKUP], tool_handler=declining
    )
    pipeline = ConversationPipeline(stages=[PipelineStage(phase="a", agent_id="agent-a")])
    pipeline.install(kit, [agent], voice_channel_id="rtv")
    room = await kit.create_room()
    state = ConversationState(active_agent_id="agent-a", phase="a")
    await kit.store.update_room(set_conversation_state(room, state))
    await kit.attach_channel(room.id, "rtv")
    seen = _observe(kit)
    session = await channel.start_session(room.id, "u1", "ws")

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results) and bool(seen))

    assert json.loads(provider.tool_results[0][2]) == {"error": "No handler for tool lookup"}
    assert [e.is_error for e in seen] == [True]
    await kit.close()


# -- Every other way a text call is cut reports it once, cancelled ------------


class _StopAtToolStart(SimpleChannel):
    """A streaming transport that stops reading once a call starts (barge-in)."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self, stream: Any, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        async for item in stream:
            if isinstance(item, RoomEvent) and item.type == EventType.TOOL_CALL_START:
                return ChannelOutput.empty()
        return ChannelOutput.empty()


async def test_a_transport_that_stops_reading_reports_the_announced_call_cancelled() -> None:
    ran: list[str] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(_StopAtToolStart("screen"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=MockAIProvider(
                ai_responses=[_call("lookup"), AIResponse(content="done")], streaming=True
            ),
            tool_handler=lookup,
            tools=[LOOKUP],
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "screen")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    seen = _observe(kit)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await until(lambda: bool(seen))
    await asyncio.sleep(0.05)

    assert ran == []
    assert [(e.name, e.is_error, e.cancelled) for e in seen] == [("lookup", True, True)]
    assert ("tool_call_end", "cancelled") in await _tool_rows(kit, "r1")
    await kit.close()


async def _cancelled_inside(trigger: HookTrigger) -> list[ToolCallEvent]:
    """A turn cancelled while a SYNC *trigger* hook holds the call."""
    held = asyncio.Event()

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=MockAIProvider(ai_responses=[_call("lookup"), AIResponse(content="done")]),
            tool_handler=lookup,
            tools=[LOOKUP],
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(trigger, execution=HookExecution.SYNC, name="hold")
    async def hold(event: ToolCallEvent, ctx: Any) -> HookResult:
        held.set()
        await asyncio.sleep(30)
        return HookResult.allow()

    seen = _observe(kit)
    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
    )
    await asyncio.wait_for(held.wait(), 5)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    await until(lambda: bool(seen))
    await asyncio.sleep(0.05)
    await kit.close()
    return seen


@pytest.mark.parametrize("trigger", [HookTrigger.BEFORE_TOOL_USE, HookTrigger.ON_TOOL_CALL])
async def test_a_turn_cancelled_in_the_gate_or_the_judging_reports_the_call_once(
    trigger: HookTrigger,
) -> None:
    seen = await _cancelled_inside(trigger)

    assert [(e.name, e.is_error, e.cancelled) for e in seen] == [("lookup", True, True)]


async def test_an_acp_call_the_turn_ended_under_is_reported_once_with_a_handler(
    tmp_path: Path,
) -> None:
    kit, seen = await _acp_room(
        tmp_path, handler=PolicyExternalToolHandler(), prompt=_start_then_stop
    )

    assert ("tool_call_end", "cancelled") in await _tool_rows(kit, "room-1")
    assert [(e.is_error, e.cancelled) for e in seen] == [(True, True)]
    await kit.close()


async def _start_then_complete(
    connection: Any, session_id: str, prompt: list[Any], **kw: Any
) -> Any:
    await connection.client.session_update(
        session_id,
        acp.start_tool_call("tool-1", "Read", kind="read", status="in_progress", raw_input={}),
    )
    await connection.client.session_update(
        session_id, acp.update_tool_call("tool-1", status="completed", raw_output="ok")
    )
    return PromptResponse(stop_reason="end_turn")


@pytest.mark.parametrize("with_handler", [False, True], ids=["no-handler", "handler"])
async def test_an_acp_call_whose_report_is_cut_is_still_reported_once(
    tmp_path: Path, with_handler: bool
) -> None:
    kit = RoomKit()
    handler = PolicyExternalToolHandler() if with_handler else None
    channel, conn, _ = _channel(tmp_path, handler=handler, emit_updates=False)
    conn.prompt = lambda *a, **k: _start_then_complete(conn, *a, **k)  # type: ignore[method-assign]
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "sms")
    await kit.attach_channel("room-1", "acp-agent", category=ChannelCategory.INTELLIGENCE)
    held = asyncio.Event()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="hold")
    async def hold(event: ToolCallEvent, ctx: Any) -> HookResult:
        held.set()
        await asyncio.sleep(0.2)
        return HookResult.allow()

    seen = _observe(kit)
    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
        )
    )
    await asyncio.wait_for(held.wait(), 5)
    turn.cancel()
    await asyncio.gather(turn, return_exceptions=True)
    await until(lambda: bool(seen))
    await asyncio.sleep(0.3)

    assert [(e.tool_call_id, e.is_error, e.cancelled) for e in seen] == [("tool-1", False, False)]
    await kit.close()


async def test_an_acp_call_refused_then_approved_fails_on_its_own(tmp_path: Path) -> None:
    """The last permission decision stands: an approval clears a refusal."""

    class _ThenApproves(_RecordingToolHandler):
        async def process_tool_call(self, *args: Any, **kwargs: Any) -> Any:
            decision = await super().process_tool_call(*args, **kwargs)
            self.approved = True
            return decision

    async def refused_then_approved(
        connection: Any, session_id: str, prompt: list[Any], **kw: Any
    ) -> Any:
        await connection.client.session_update(
            session_id,
            acp.start_tool_call(
                "tool-1", "Write", kind="edit", status="pending", raw_input={"path": "/tmp/n.md"}
            ),
        )
        options = [
            PermissionOption(option_id="allow-once", name="Allow once", kind="allow_once"),
            PermissionOption(option_id="reject-once", name="Reject once", kind="reject_once"),
        ]
        for _ in range(2):
            await connection.client.request_permission(
                session_id, acp.update_tool_call("tool-1", title="Write"), options
            )
        await connection.client.session_update(
            session_id,
            acp.update_tool_call("tool-1", status="failed", raw_output={"error": "disk full"}),
        )
        return PromptResponse(stop_reason="end_turn")

    kit, _ = await _acp_room(
        tmp_path, handler=_ThenApproves(approved=False), prompt=refused_then_approved
    )

    assert ("tool_call_end", "failed") in await _tool_rows(kit, "room-1")
    await kit.close()


async def test_a_claimed_report_is_made_though_its_call_is_cut_meanwhile() -> None:
    """Once a call's one report is claimed, a cut of the call while its
    observers are told does not lose it (RFC §9.3)."""
    kit = RoomKit()
    await kit.create_room(room_id="r")
    release = asyncio.Event()
    observed: list[str] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        observed.append(event.tool_call_id)

    told = kit._tell_tool_call

    async def slow_tell(*args: Any) -> None:
        await release.wait()
        await told(*args)

    kit._tell_tool_call = slow_tell  # type: ignore[method-assign]
    event = ToolCallEvent(
        channel_id="c",
        channel_type=ChannelType.AI,
        tool_call_id="x1",
        name="lookup",
        arguments={},
        result="ok",
        room_id="r",
    )
    judging = asyncio.create_task(kit._judge_tool_call(event, "c", claim=lambda: True))
    await asyncio.sleep(0.05)
    judging.cancel()
    release.set()
    await asyncio.sleep(0.1)

    assert observed == ["x1"]
    await kit.close()

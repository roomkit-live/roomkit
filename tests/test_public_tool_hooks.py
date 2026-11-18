"""The executor's public tool wiring can run, decide, and report independently."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from roomkit import HookResult, RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, HookExecution, HookTrigger
from roomkit.models.task import Observation, Task
from roomkit.models.tool_call import ToolCallEvent


async def test_application_runs_the_chain_and_claims_its_own_reports() -> None:
    kit = RoomKit()
    room = await kit.create_room(room_id="r")
    executor = kit.turn_executor
    tools = executor.wiring.tools
    ctx = RoomContext(room=room)
    event = ToolCallEvent("tool", ChannelType.AI, "id", "lookup", {}, result="provider")
    raw = await tools.run_tool_call_chain(event, room.id, carrying=ctx)
    assert raw is not None and raw[0].event == event and raw[1] is None
    events: asyncio.Queue[ToolCallEvent] = asyncio.Queue()
    ran: list[str] = []

    @executor.hook(HookTrigger.ON_TOOL_CALL)
    async def rewrite(call: ToolCallEvent, room_ctx: RoomContext) -> HookResult:
        ran.append(room_ctx.room.id)
        return HookResult(
            action="allow",
            metadata={"result": "rewrite", "extra": True},
            tasks=[Task(id="t", room_id=room.id, title="check")],
            observations=[
                Observation(id="o", room_id=room.id, channel_id="tool", content="noted")
            ],
        )

    @executor.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def observe(call: ToolCallEvent, room_ctx: RoomContext) -> None:
        events.put_nowait(call)

    @executor.hook(HookTrigger.BEFORE_TOOL_USE)
    async def gate(call: ToolCallEvent, room_ctx: RoomContext) -> HookResult:
        return HookResult.block("approval")

    denied, selected = await tools.decide_before_tool_use(replace(event, room_id="r"), "bound")
    assert not denied.allowed and denied.reason == "approval"
    assert selected is not None and selected.room.id == room.id
    raw = await tools.run_tool_call_chain(event, room.id, carrying=ctx)
    assert raw is not None
    result, selected = raw
    assert selected is not None and selected.room.id == "r"
    assert result.event.result == "rewrite" and result.metadata["extra"]
    assert [t.title for t in result.tasks] == ["check"]
    assert [o.content for o in result.observations] == ["noted"]
    assert result.allowed and not result.failed_closed
    assert result.blocked_by is result.reason is None
    assert result.injected_events == result.hook_errors == []
    assert events.empty()

    await tools.tell_tool_call(event, selected, "bound")
    assert (await asyncio.wait_for(events.get(), 2)).room_id is None
    event = replace(event, room_id="r")
    verdict = await tools.judge_tool_call(event, "bound", carrying=ctx, claim=lambda: False)
    assert verdict is not None and verdict.result == "rewrite"
    await tools.report_tool_call(event, "bound", claim=lambda: False)
    await tools.report_tool_call(event, "bound")
    assert (await asyncio.wait_for(events.get(), 2)).result == "provider"
    refused = replace(event, refused=True)
    await tools.report_tool_call(refused, "bound")
    assert (await asyncio.wait_for(events.get(), 2)).refused
    await tools.observe_failed_tool_call(event, "bound")
    assert (await asyncio.wait_for(events.get(), 2)).result == "provider"
    assert events.empty() and ran == ["r"] * 4
    await kit.close()


async def test_raw_termination_and_the_history_a_kit_hook_can_reuse() -> None:
    from roomkit.models.event import EventSource, RoomEvent, TextContent

    kit = RoomKit()
    room = await kit.create_room(room_id="r")
    executor = kit.turn_executor
    tools = executor.wiring.tools
    carried = RoomContext(
        room=room.model_copy(update={"metadata": {"stale": True}}),
        history_start_index=3,
        recent_events=[
            RoomEvent(
                room_id=room.id,
                source=EventSource(channel_id="tool", channel_type=ChannelType.AI),
                content=TextContent(body="old version of the row"),
            )
        ],
    )
    event = ToolCallEvent("tool", ChannelType.AI, "id", "lookup", {}, room_id=room.id)

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="first")
    async def diagnostic(call: ToolCallEvent, ctx: RoomContext) -> HookResult:
        raise RuntimeError("failure")

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="second", priority=1)
    async def explicit(call: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("reason")

    raw = await tools.run_tool_call_chain(event, room.id, carrying=carried)
    assert raw is not None and raw[1] is not None
    result, fresh = raw
    assert "stale" not in fresh.room.metadata and fresh.recent_events == carried.recent_events
    assert fresh.history_start_index == 3
    assert not result.allowed and not result.failed_closed
    assert (result.blocked_by, result.reason) == ("second", "reason")
    assert result.hook_errors == [{"hook": "first", "error": "failure"}]

    for change in [{"id": "another-room"}, {"event_count": room.event_count + 1}]:
        stale = carried.model_copy(update={"room": room.model_copy(update=change)})
        raw = await tools.run_tool_call_chain(event, room.id, carrying=stale)
        assert raw is not None and raw[1] is not None and raw[1].recent_events == []

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="closed", priority=-1, fail_closed=True)
    async def closed(call: ToolCallEvent, ctx: RoomContext) -> HookResult:
        raise RuntimeError("failure")

    raw = await tools.run_tool_call_chain(event, room.id)
    assert raw is not None
    assert raw[0].failed_closed and not raw[0].allowed
    assert (raw[0].blocked_by, raw[0].reason) == ("closed", "hook_error:closed")
    await kit.close()

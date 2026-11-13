"""A registered handler's inherited hook helpers also work outside a turn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.telemetry import MockTelemetryProvider, SpanKind
from roomkit.tools import PolicyExternalToolHandler
from tests.test_channels.test_acp import _channel


async def test_off_turn_handler_hooks_use_the_registered_kits_telemetry(tmp_path: Path) -> None:
    telemetry = MockTelemetryProvider()
    kit = RoomKit(telemetry=telemetry)
    handler = PolicyExternalToolHandler()
    channel, _, _ = _channel(tmp_path, handler=handler)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")

    @kit.hook(HookTrigger.BEFORE_TOOL_USE)
    async def allow(event: Any, context: Any) -> HookResult:
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def report(event: Any, context: Any) -> None:
        pass

    assert (await handler.process_tool_call("Read", {}, room_id="r1")).approved
    await handler.on_tool_result("Read", {}, "read", room_id="r1")
    await kit.close()
    assert len(telemetry.get_spans(SpanKind.HOOK_SYNC)) == 1
    assert len(telemetry.get_spans(SpanKind.HOOK_ASYNC)) == 1
    assert telemetry.get_spans(SpanKind.HOOK_SYNC)[0].parent_id is None


async def test_inherited_approval_helper_preserves_the_raw_decision(tmp_path: Path) -> None:
    handler = PolicyExternalToolHandler()
    channel, _, _ = _channel(tmp_path, handler=handler)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")

    @kit.hook(HookTrigger.BEFORE_TOOL_USE)
    async def refuse(event: Any, context: Any) -> HookResult:
        return HookResult(action="block", reason="", metadata={"arguments": {"path": "other"}})

    raw = await handler._fire_before_hook("Read", {"path": "original"}, room_id="r1")
    assert not raw.allowed
    assert raw.reason == ""
    assert raw.arguments == {"path": "other"}
    assert raw.detail is None
    adapted = await handler.process_tool_call("Read", {"path": "original"}, room_id="r1")
    assert not adapted.approved
    assert adapted.reason == "Tool 'Read' denied by pre-execution hook."
    assert adapted.modified_input is None
    await kit.close()

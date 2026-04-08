"""Tests for ExternalToolHandler, PolicyExternalToolHandler, and BEFORE_TOOL_USE hook."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelDirection, ChannelType
from roomkit.models.event import EventSource, RoomEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import (
    PolicyExternalToolHandler,
    ToolDecision,
)
from roomkit.tools.policy import ToolPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_responses() -> list[AIResponse]:
    """AI response that calls a tool, then returns text."""
    return [
        AIResponse(
            content="Calling tool.",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[
                AIToolCall(id="tc-1", name="get_weather", arguments={"city": "Paris"}),
            ],
        ),
        AIResponse(
            content="It's sunny in Paris.",
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 10},
        ),
    ]


TOOLS = [
    AITool(
        name="get_weather",
        description="Get weather.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
]


async def _tool_handler(name: str, args: dict[str, Any]) -> str:
    return json.dumps({"temp": 22, "city": args.get("city", "unknown")})


# ---------------------------------------------------------------------------
# ToolDecision dataclass
# ---------------------------------------------------------------------------


class TestToolDecision:
    def test_approved(self) -> None:
        d = ToolDecision(approved=True)
        assert d.approved is True
        assert d.reason == ""
        assert d.modified_input is None
        assert d.result is None

    def test_denied_with_reason(self) -> None:
        d = ToolDecision(approved=False, reason="Not allowed")
        assert d.approved is False
        assert d.reason == "Not allowed"

    def test_approved_with_modified_input(self) -> None:
        d = ToolDecision(approved=True, modified_input={"city": "London"})
        assert d.modified_input == {"city": "London"}

    def test_approved_with_result_override(self) -> None:
        d = ToolDecision(approved=True, result="manual result")
        assert d.result == "manual result"


# ---------------------------------------------------------------------------
# PolicyExternalToolHandler — unit tests
# ---------------------------------------------------------------------------


class TestPolicyExternalToolHandler:
    async def test_approve_no_policy(self) -> None:
        handler = PolicyExternalToolHandler()
        decision = await handler.process_tool_call("Read", {"path": "/tmp"})
        assert decision.approved is True

    async def test_deny_by_policy(self) -> None:
        handler = PolicyExternalToolHandler(policy=ToolPolicy(deny=["Bash"]))
        decision = await handler.process_tool_call("Bash", {"command": "rm -rf /"})
        assert decision.approved is False
        assert "denied by policy" in decision.reason

    async def test_allow_by_policy(self) -> None:
        handler = PolicyExternalToolHandler(policy=ToolPolicy(deny=["Bash"]))
        decision = await handler.process_tool_call("Read", {"path": "/tmp"})
        assert decision.approved is True

    async def test_allow_list_policy(self) -> None:
        handler = PolicyExternalToolHandler(policy=ToolPolicy(allow=["Read", "Write"]))
        decision = await handler.process_tool_call("Read", {})
        assert decision.approved is True

        decision = await handler.process_tool_call("Bash", {})
        assert decision.approved is False

    async def test_on_tool_result_fires_hook(self) -> None:
        handler = PolicyExternalToolHandler()
        hook_called = False

        async def on_tool(event: ToolCallEvent) -> str | None:
            nonlocal hook_called
            hook_called = True
            assert event.name == "Read"
            assert event.result == "file contents"
            return None

        handler._on_tool_hook = on_tool
        handler._channel_id = "ai-1"

        await handler.on_tool_result("Read", {"path": "/tmp"}, "file contents")
        assert hook_called

    async def test_on_tool_result_no_hook(self) -> None:
        """on_tool_result does not error when no hook is injected."""
        handler = PolicyExternalToolHandler()
        await handler.on_tool_result("Read", {}, "ok")

    async def test_before_hook_denies(self) -> None:
        handler = PolicyExternalToolHandler()

        async def deny_all(event: ToolCallEvent) -> bool:
            return False

        handler._before_tool_hook = deny_all
        handler._channel_id = "ai-1"

        decision = await handler.process_tool_call("Read", {})
        assert decision.approved is False
        assert "BEFORE_TOOL_USE" in decision.reason

    async def test_before_hook_allows(self) -> None:
        handler = PolicyExternalToolHandler()

        async def allow_all(event: ToolCallEvent) -> bool:
            return True

        handler._before_tool_hook = allow_all
        handler._channel_id = "ai-1"

        decision = await handler.process_tool_call("Read", {})
        assert decision.approved is True

    async def test_before_hook_and_policy_combined(self) -> None:
        """Hook allows but policy denies — should be denied."""
        handler = PolicyExternalToolHandler(policy=ToolPolicy(deny=["Bash"]))

        async def allow_all(event: ToolCallEvent) -> bool:
            return True

        handler._before_tool_hook = allow_all
        handler._channel_id = "ai-1"

        decision = await handler.process_tool_call("Bash", {"cmd": "ls"})
        assert decision.approved is False
        assert "denied by policy" in decision.reason


# ---------------------------------------------------------------------------
# ExternalToolHandler — ABC contract
# ---------------------------------------------------------------------------


class TestExternalToolHandlerABC:
    async def test_start_stop_are_noops(self) -> None:
        handler = PolicyExternalToolHandler()
        await handler.start()
        await handler.stop()

    async def test_fire_before_hook_no_callback(self) -> None:
        handler = PolicyExternalToolHandler()
        handler._before_tool_hook = None
        result = await handler._fire_before_hook("Read", {})
        assert result is True  # fail-open

    async def test_fire_on_tool_hook_no_callback(self) -> None:
        handler = PolicyExternalToolHandler()
        handler._on_tool_hook = None
        # Should not raise
        await handler._fire_on_tool_hook("Read", {}, "result")

    async def test_fire_before_hook_with_callback(self) -> None:
        handler = PolicyExternalToolHandler()
        handler._channel_id = "ch-1"

        callback = AsyncMock(return_value=False)
        handler._before_tool_hook = callback

        result = await handler._fire_before_hook(
            "Bash", {"cmd": "ls"}, tool_call_id="tc-1", room_id="room-1"
        )
        assert result is False
        callback.assert_awaited_once()
        event = callback.call_args[0][0]
        assert isinstance(event, ToolCallEvent)
        assert event.name == "Bash"
        assert event.room_id == "room-1"

    async def test_fire_on_tool_hook_with_callback(self) -> None:
        handler = PolicyExternalToolHandler()
        handler._channel_id = "ch-1"

        callback = AsyncMock(return_value=None)
        handler._on_tool_hook = callback

        await handler._fire_on_tool_hook(
            "Read", {"path": "/tmp"}, "contents", tool_call_id="tc-2", room_id="room-1"
        )
        callback.assert_awaited_once()
        event = callback.call_args[0][0]
        assert event.name == "Read"
        assert event.result == "contents"


# ---------------------------------------------------------------------------
# BEFORE_TOOL_USE hook — framework integration
# ---------------------------------------------------------------------------


async def _trigger_ai(kit: RoomKit, ai: AIChannel, room_id: str) -> None:
    """Send an event to the AI channel via on_event (AI channels don't accept inbound)."""
    event = RoomEvent(
        room_id=room_id,
        source=EventSource(
            channel_id="sms-1",
            channel_type=ChannelType.SMS,
            direction=ChannelDirection.INBOUND,
        ),
        content=TextContent(body="What's the weather?"),
    )
    binding = ChannelBinding(
        channel_id=ai.channel_id,
        room_id=room_id,
        channel_type=ChannelType.AI,
    )
    context = await kit._build_context(room_id)
    await ai.on_event(event, binding, context)


class TestBeforeToolUseHook:
    async def test_hook_blocks_tool_execution(self) -> None:
        """A BEFORE_TOOL_USE hook that blocks prevents the tool from running."""
        provider = MockAIProvider(ai_responses=_make_tool_responses())
        ai = AIChannel(
            "ai-1",
            provider=provider,
            system_prompt="Test.",
            tool_handler=_tool_handler,
            tools=TOOLS,
        )

        kit = RoomKit()
        kit.register_channel(ai)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "ai-1", category=ChannelCategory.INTELLIGENCE)

        blocked_tools: list[str] = []

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny-weather")
        async def deny_weather(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            blocked_tools.append(event.name)
            return HookResult(action="block", reason="Weather tools disabled")

        await _trigger_ai(kit, ai, room.id)

        assert "get_weather" in blocked_tools

    async def test_hook_allows_tool_execution(self) -> None:
        """A BEFORE_TOOL_USE hook that allows lets the tool run normally."""
        provider = MockAIProvider(ai_responses=_make_tool_responses())
        ai = AIChannel(
            "ai-1",
            provider=provider,
            system_prompt="Test.",
            tool_handler=_tool_handler,
            tools=TOOLS,
        )

        kit = RoomKit()
        kit.register_channel(ai)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "ai-1", category=ChannelCategory.INTELLIGENCE)

        observed_tools: list[str] = []

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="allow-all")
        async def allow_all(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            observed_tools.append(event.name)
            return HookResult(action="allow")

        await _trigger_ai(kit, ai, room.id)

        assert "get_weather" in observed_tools

    async def test_hook_wired_to_external_handler(self) -> None:
        """BEFORE_TOOL_USE hooks are wired into ExternalToolHandler on register."""
        handler = PolicyExternalToolHandler()
        provider = MockAIProvider()
        ai = AIChannel(
            "ai-ext",
            provider=provider,
            system_prompt="Test.",
            external_tool_handler=handler,
        )

        kit = RoomKit()
        kit.register_channel(ai)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "ai-ext", category=ChannelCategory.INTELLIGENCE)

        # After register_channel + attach, the handler should have hooks injected
        assert handler._before_tool_hook is not None
        assert handler._on_tool_hook is not None
        assert handler._channel_id == "ai-ext"

    async def test_multiple_hooks_priority_order(self) -> None:
        """Multiple BEFORE_TOOL_USE hooks run in priority order; first block wins."""
        provider = MockAIProvider(ai_responses=_make_tool_responses())
        ai = AIChannel(
            "ai-1",
            provider=provider,
            system_prompt="Test.",
            tool_handler=_tool_handler,
            tools=TOOLS,
        )

        kit = RoomKit()
        kit.register_channel(ai)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "ai-1", category=ChannelCategory.INTELLIGENCE)

        call_order: list[str] = []

        @kit.hook(
            HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="audit", priority=10
        )
        async def audit(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            call_order.append("audit")
            return HookResult(action="allow")

        @kit.hook(
            HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="block", priority=20
        )
        async def block(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            call_order.append("block")
            return HookResult(action="block", reason="Denied")

        await _trigger_ai(kit, ai, room.id)

        # Lower priority number runs first; block stops further execution
        assert "audit" in call_order
        assert "block" in call_order
        assert call_order.index("audit") < call_order.index("block")

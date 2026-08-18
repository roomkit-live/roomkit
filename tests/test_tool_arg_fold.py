"""Folding a flattened hub-tool call back into ``params``, at the call sites.

``tests/test_tool_arg_validation.py`` covers the fold as a function. This file
covers what the two channels do with it: the model's own call is repaired
before the fail-closed gate, the repaired payload is what the hook and the
handler see, and a hook's rewritten arguments are still refused rather than
repaired.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomKit,
    TextContent,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelCategory, ChannelDirection, ChannelType
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

# A hub tool: one tool per domain, ``{action, params}``. FastMCP closes the
# schema for a typed function, which is what makes a hoisted key a hard error.
BOARDS_TOOL = {
    "name": "boards",
    "description": "Board operations.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string"},
            "params": {"type": "object"},
        },
        "required": ["action"],
    },
}

# A flat tool: no container to fold into, so an unknown argument stays an error.
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the weather.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

HOISTED_CALL = {"action": "list_columns", "board_id": "1a0a495f"}
FOLDED_CALL = {"action": "list_columns", "params": {"board_id": "1a0a495f"}}


def _recording_handler() -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        calls.append((name, arguments))
        return json.dumps({"ok": True})

    return handler, calls


# ---------------------------------------------------------------------------
# Classic AI channel
# ---------------------------------------------------------------------------


def _tool_call_provider(name: str, arguments: dict[str, Any]) -> MockAIProvider:
    return MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="t1", name=name, arguments=arguments)],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )


async def _ai_setup(
    provider: MockAIProvider, handler: Any, tools: list[dict[str, Any]]
) -> tuple[RoomKit, AIChannel, str]:
    ai = AIChannel("ai-1", provider=provider, tool_handler=handler, tools=tools)
    kit = RoomKit()
    kit.register_channel(ai)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "ai-1", category=ChannelCategory.INTELLIGENCE)
    return kit, ai, room.id


async def _trigger_ai(kit: RoomKit, ai: AIChannel, room_id: str) -> None:
    event = RoomEvent(
        room_id=room_id,
        source=EventSource(
            channel_id="sms-1",
            channel_type=ChannelType.SMS,
            direction=ChannelDirection.INBOUND,
        ),
        content=TextContent(body="list the columns"),
    )
    binding = ChannelBinding(
        channel_id=ai.channel_id, room_id=room_id, channel_type=ChannelType.AI
    )
    await ai.on_event(event, binding, await kit._build_context(room_id))


def _tool_error(provider: MockAIProvider) -> str:
    """The error the model was handed back, from the second round's messages."""
    tool_msgs = [m for m in provider.calls[1].messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    return json.loads(tool_msgs[0].content[0].result)["error"]


class TestClassicChannel:
    async def test_hoisted_call_executes_as_if_it_had_been_nested(self) -> None:
        handler, calls = _recording_handler()
        provider = _tool_call_provider("boards", HOISTED_CALL)
        kit, ai, room_id = await _ai_setup(provider, handler, [BOARDS_TOOL])

        await _trigger_ai(kit, ai, room_id)

        assert calls == [("boards", FOLDED_CALL)]

    async def test_hook_and_handler_see_the_folded_payload(self) -> None:
        """The repair is upstream of every gate, so nothing downstream sees the
        model's flat shape — including the post-hook validation, which would
        otherwise refuse what was just repaired."""
        handler, calls = _recording_handler()
        provider = _tool_call_provider("boards", HOISTED_CALL)
        kit, ai, room_id = await _ai_setup(provider, handler, [BOARDS_TOOL])

        seen_by_hook: list[dict[str, Any]] = []

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="observe")
        async def observe(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            seen_by_hook.append(event.arguments)
            return HookResult(action="allow")

        await _trigger_ai(kit, ai, room_id)

        assert seen_by_hook == [FOLDED_CALL]
        assert calls == [("boards", FOLDED_CALL)]

    async def test_both_forms_at_once_is_refused_with_a_message_that_decides(self) -> None:
        handler, calls = _recording_handler()
        provider = _tool_call_provider(
            "boards", {"action": "x", "params": {"board_id": "1"}, "column_id": "2"}
        )
        kit, ai, room_id = await _ai_setup(provider, handler, [BOARDS_TOOL])

        await _trigger_ai(kit, ai, room_id)

        assert calls == []
        error = _tool_error(provider)
        assert "'column_id'" in error
        assert "inside 'params'" in error

    async def test_unknown_argument_on_a_flat_tool_is_still_refused(self) -> None:
        handler, calls = _recording_handler()
        provider = _tool_call_provider("get_weather", {"city": "Laval", "units": "metric"})
        kit, ai, room_id = await _ai_setup(provider, handler, [WEATHER_TOOL])

        await _trigger_ai(kit, ai, room_id)

        assert calls == []
        assert "unknown argument 'units'" in _tool_error(provider)

    async def test_hook_rewritten_arguments_are_not_folded(self) -> None:
        """A hook is user code: a flat payload out of one is its bug, and the
        refusal names it instead of quietly reshaping what it returned."""
        handler, calls = _recording_handler()
        provider = _tool_call_provider("boards", FOLDED_CALL)
        kit, ai, room_id = await _ai_setup(provider, handler, [BOARDS_TOOL])

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="flatten")
        async def flatten(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult(action="allow", metadata={"arguments": dict(HOISTED_CALL)})

        await _trigger_ai(kit, ai, room_id)

        assert calls == []
        error = _tool_error(provider)
        assert "rewritten arguments" in error
        assert "unknown argument 'board_id'" in error


# ---------------------------------------------------------------------------
# Realtime voice channel
# ---------------------------------------------------------------------------


@pytest.fixture
def rt_provider() -> MockRealtimeProvider:
    return MockRealtimeProvider()


async def _rt_setup(
    provider: MockRealtimeProvider, handler: Any, tools: list[dict[str, Any]]
) -> tuple[RoomKit, RealtimeVoiceChannel, str]:
    channel = RealtimeVoiceChannel(
        "rt-1",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt-1")
    return kit, channel, room.id


def _rt_error(provider: MockRealtimeProvider) -> str:
    assert len(provider.tool_results) == 1
    return json.loads(provider.tool_results[0][2])["error"]


class TestRealtimeChannel:
    async def test_hoisted_call_executes_as_if_it_had_been_nested(
        self, rt_provider: MockRealtimeProvider
    ) -> None:
        handler, calls = _recording_handler()
        kit, channel, room_id = await _rt_setup(rt_provider, handler, [BOARDS_TOOL])
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await rt_provider.simulate_tool_call(session, "call-1", "boards", dict(HOISTED_CALL))
        await asyncio.sleep(0.05)

        assert calls == [("boards", FOLDED_CALL)]

    async def test_both_forms_at_once_is_refused_with_a_message_that_decides(
        self, rt_provider: MockRealtimeProvider
    ) -> None:
        handler, calls = _recording_handler()
        kit, channel, room_id = await _rt_setup(rt_provider, handler, [BOARDS_TOOL])
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await rt_provider.simulate_tool_call(
            session,
            "call-1",
            "boards",
            {"action": "x", "params": {"board_id": "1"}, "column_id": "2"},
        )
        await asyncio.sleep(0.05)

        assert calls == []
        error = _rt_error(rt_provider)
        assert "'column_id'" in error
        assert "inside 'params'" in error

    async def test_unknown_argument_on_a_flat_tool_is_still_refused(
        self, rt_provider: MockRealtimeProvider
    ) -> None:
        handler, calls = _recording_handler()
        kit, channel, room_id = await _rt_setup(rt_provider, handler, [WEATHER_TOOL])
        session = await channel.start_session(room_id, "user-1", "fake-ws")

        await rt_provider.simulate_tool_call(
            session, "call-1", "get_weather", {"city": "Laval", "units": "metric"}
        )
        await asyncio.sleep(0.05)

        assert calls == []
        assert "unknown argument 'units'" in _rt_error(rt_provider)

    async def test_hook_rewritten_arguments_are_not_folded(
        self, rt_provider: MockRealtimeProvider
    ) -> None:
        handler, calls = _recording_handler()
        kit, channel, room_id = await _rt_setup(rt_provider, handler, [BOARDS_TOOL])

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="flatten")
        async def flatten(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult(action="allow", metadata={"arguments": dict(HOISTED_CALL)})

        session = await channel.start_session(room_id, "user-1", "fake-ws")
        await rt_provider.simulate_tool_call(session, "call-1", "boards", dict(FOLDED_CALL))
        await asyncio.sleep(0.05)

        assert calls == []
        error = _rt_error(rt_provider)
        assert "rewritten arguments" in error
        assert "unknown argument 'board_id'" in error

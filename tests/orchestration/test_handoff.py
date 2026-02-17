"""Tests for HandoffHandler, HandoffMemoryProvider, and setup_handoff."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.room import Room
from roomkit.orchestration.handoff import (
    HANDOFF_TOOL,
    HandoffHandler,
    HandoffMemoryProvider,
    HandoffRequest,
    HandoffResult,
    _room_id_var,
    setup_handoff,
)
from roomkit.orchestration.state import (
    ConversationState,
    get_conversation_state,
    set_conversation_state,
)
from roomkit.providers.ai.base import AIMessage
from tests.conftest import make_event

# -- Helpers ------------------------------------------------------------------


def _make_mock_kit(room: Room, bindings: list[ChannelBinding]) -> MagicMock:
    """Create a mock RoomKit with store methods."""
    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.list_bindings = AsyncMock(return_value=bindings)
    kit.store.update_room = AsyncMock()
    kit.send_event = AsyncMock()
    return kit


def _ai_binding(channel_id: str, room_id: str = "r1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _transport_binding(channel_id: str, room_id: str = "r1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=ChannelType.SMS,
    )


# -- Models -------------------------------------------------------------------


class TestHandoffModels:
    def test_handoff_request_defaults(self):
        req = HandoffRequest(target_agent_id="agent-b")
        assert req.reason == ""
        assert req.summary == ""
        assert req.context == {}
        assert req.channel_escalation is None

    def test_handoff_result_defaults(self):
        result = HandoffResult()
        assert result.accepted is True
        assert result.new_agent_id is None
        assert result.message == ""

    def test_handoff_tool_definition(self):
        assert HANDOFF_TOOL.name == "handoff_conversation"
        assert "target" in HANDOFF_TOOL.parameters["properties"]
        assert "reason" in HANDOFF_TOOL.parameters["properties"]
        assert "summary" in HANDOFF_TOOL.parameters["properties"]
        assert HANDOFF_TOOL.parameters["required"] == ["target", "reason", "summary"]


# -- HandoffHandler -----------------------------------------------------------


class TestHandoffHandler:
    async def test_successful_handoff(self):
        room = Room(id="r1")
        bindings = [_transport_binding("sms1"), _ai_binding("agent-a"), _ai_binding("agent-b")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(kit=kit, router=router)
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={
                "target": "agent-b",
                "reason": "needs specialist",
                "summary": "user wants help with X",
            },
        )

        assert result.accepted is True
        assert result.new_agent_id == "agent-b"
        assert result.new_phase == "handling"

        # Verify state was persisted
        kit.store.update_room.assert_called_once()
        saved_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(saved_room)
        assert state.active_agent_id == "agent-b"
        assert state.previous_agent_id is None
        assert state.phase == "handling"
        assert state.context["handoff_summary"] == "user wants help with X"
        assert state.context["handoff_from"] == "agent-a"

        # Verify system event emitted
        kit.send_event.assert_called_once()
        call_kwargs = kit.send_event.call_args[1]
        assert call_kwargs["event_type"] == EventType.SYSTEM
        assert call_kwargs["metadata"]["handoff"] is True
        assert call_kwargs["metadata"]["from_agent"] == "agent-a"
        assert call_kwargs["metadata"]["to_agent"] == "agent-b"

    async def test_handoff_to_unknown_agent_rejected(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(kit=kit, router=router)
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-unknown", "reason": "test", "summary": "test"},
        )

        assert result.accepted is False
        assert "not found" in result.reason
        kit.store.update_room.assert_not_called()
        kit.send_event.assert_not_called()

    async def test_handoff_to_human(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(kit=kit, router=router)
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "human", "reason": "escalation", "summary": "needs human"},
        )

        assert result.accepted is True
        assert result.new_agent_id is None  # human = no active agent
        saved_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(saved_room)
        assert state.active_agent_id is None

    async def test_alias_resolution(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a"), _ai_binding("agent-advisor-v2")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(
            kit=kit,
            router=router,
            agent_aliases={"advisor": "agent-advisor-v2"},
        )
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "advisor", "reason": "test", "summary": "test"},
        )

        assert result.accepted is True
        assert result.new_agent_id == "agent-advisor-v2"

    async def test_phase_map(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a"), _ai_binding("agent-coder")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(
            kit=kit,
            router=router,
            phase_map={"agent-coder": "coding"},
        )
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-coder", "reason": "test", "summary": "test"},
        )

        assert result.new_phase == "coding"

    async def test_explicit_next_phase_overrides_phase_map(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a"), _ai_binding("agent-coder")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(
            kit=kit,
            router=router,
            phase_map={"agent-coder": "coding"},
        )
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={
                "target": "agent-coder",
                "reason": "test",
                "summary": "test",
                "next_phase": "review",
            },
        )

        assert result.new_phase == "review"

    async def test_handoff_increments_count(self):
        room = Room(id="r1")
        state = ConversationState(active_agent_id="agent-a")
        room = set_conversation_state(room, state)

        bindings = [_ai_binding("agent-a"), _ai_binding("agent-b")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(kit=kit, router=router)
        await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "test", "summary": "test"},
        )

        saved_room = kit.store.update_room.call_args[0][0]
        new_state = get_conversation_state(saved_room)
        assert new_state.handoff_count == 1
        assert len(new_state.phase_history) == 1


# -- HandoffMemoryProvider ----------------------------------------------------


class TestHandoffMemoryProvider:
    async def test_injects_handoff_context(self):
        inner = AsyncMock(spec=MemoryProvider)
        inner.retrieve = AsyncMock(
            return_value=MemoryResult(messages=[AIMessage(role="user", content="hello")])
        )

        provider = HandoffMemoryProvider(inner)

        room = Room(id="r1")
        state = ConversationState(
            context={
                "handoff_summary": "user needs portfolio help",
                "handoff_from": "agent-triage",
            }
        )
        room = set_conversation_state(room, state)

        event = make_event(room_id="r1")
        context = RoomContext(room=room)

        result = await provider.retrieve("r1", event, context, channel_id="agent-advisor")

        assert len(result.messages) == 2
        # Handoff context is prepended
        assert "agent-triage" in result.messages[0].content
        assert "user needs portfolio help" in result.messages[0].content
        # Original message follows
        assert result.messages[1].content == "hello"

    async def test_no_injection_without_handoff(self):
        inner = AsyncMock(spec=MemoryProvider)
        inner.retrieve = AsyncMock(
            return_value=MemoryResult(messages=[AIMessage(role="user", content="hello")])
        )

        provider = HandoffMemoryProvider(inner)
        room = Room(id="r1")
        event = make_event(room_id="r1")
        context = RoomContext(room=room)

        result = await provider.retrieve("r1", event, context)

        assert len(result.messages) == 1
        assert result.messages[0].content == "hello"

    async def test_name_wraps_inner(self):
        inner = AsyncMock(spec=MemoryProvider)
        inner.name = "SlidingWindow"
        provider = HandoffMemoryProvider(inner)
        assert provider.name == "Handoff(SlidingWindow)"

    async def test_delegates_ingest(self):
        inner = AsyncMock(spec=MemoryProvider)
        provider = HandoffMemoryProvider(inner)
        event = make_event(room_id="r1")
        await provider.ingest("r1", event, channel_id="ch1")
        inner.ingest.assert_called_once()

    async def test_delegates_clear(self):
        inner = AsyncMock(spec=MemoryProvider)
        provider = HandoffMemoryProvider(inner)
        await provider.clear("r1")
        inner.clear.assert_called_once_with("r1")

    async def test_delegates_close(self):
        inner = AsyncMock(spec=MemoryProvider)
        provider = HandoffMemoryProvider(inner)
        await provider.close()
        inner.close.assert_called_once()


# -- setup_handoff ------------------------------------------------------------


class TestSetupHandoff:
    def test_injects_handoff_tool(self):
        channel = MagicMock()
        channel._extra_tools = []
        channel._tool_handler = None

        handler = MagicMock(spec=HandoffHandler)
        setup_handoff(channel, handler)

        assert len(channel._extra_tools) == 1
        assert channel._extra_tools[0].name == "handoff_conversation"

    def test_wraps_tool_handler(self):
        channel = MagicMock()
        channel._extra_tools = []
        channel._tool_handler = AsyncMock(return_value='{"ok": true}')
        channel.channel_id = "agent-a"

        handler = MagicMock(spec=HandoffHandler)
        setup_handoff(channel, handler)

        # _tool_handler should be replaced with the wrapper
        assert channel._tool_handler is not None

    async def test_handoff_tool_call_dispatched(self):
        channel = MagicMock()
        channel._extra_tools = []
        channel._tool_handler = None
        channel.channel_id = "agent-a"

        handler = MagicMock(spec=HandoffHandler)
        handler.handle = AsyncMock(
            return_value=HandoffResult(accepted=True, new_agent_id="agent-b", new_phase="handling")
        )

        setup_handoff(channel, handler)
        wrapped = channel._tool_handler

        # Set the ContextVar as the routing hook would
        token = _room_id_var.set("r1")
        try:
            result_json = await wrapped(
                "handoff_conversation",
                {
                    "target": "agent-b",
                    "reason": "test",
                    "summary": "test",
                },
            )
        finally:
            _room_id_var.reset(token)

        result = json.loads(result_json)
        assert result["accepted"] is True
        assert result["new_agent_id"] == "agent-b"

        handler.handle.assert_called_once_with(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "test", "summary": "test"},
        )

    async def test_non_handoff_tool_delegates(self):
        original = AsyncMock(return_value='{"result": "ok"}')
        channel = MagicMock()
        channel._extra_tools = []
        channel._tool_handler = original
        channel.channel_id = "agent-a"

        handler = MagicMock(spec=HandoffHandler)
        setup_handoff(channel, handler)
        wrapped = channel._tool_handler

        result = await wrapped("some_other_tool", {"arg": "val"})
        assert json.loads(result) == {"result": "ok"}
        original.assert_called_once_with("some_other_tool", {"arg": "val"})

    async def test_handoff_without_room_id_returns_error(self):
        channel = MagicMock()
        channel._extra_tools = []
        channel._tool_handler = None
        channel.channel_id = "agent-a"

        handler = MagicMock(spec=HandoffHandler)
        setup_handoff(channel, handler)
        wrapped = channel._tool_handler

        # Don't set _room_id_var
        token = _room_id_var.set(None)
        try:
            result_json = await wrapped("handoff_conversation", {"target": "x"})
        finally:
            _room_id_var.reset(token)

        result = json.loads(result_json)
        assert "error" in result

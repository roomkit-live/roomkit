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
    build_handoff_tool,
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
    kit._hook_engine.run_async_hooks = AsyncMock()
    kit._lock_manager = MagicMock()
    kit._lock_manager.locked = MagicMock(return_value=_NoopLock())
    return kit


class _NoopLock:
    """Async context manager that does nothing (replaces real room lock in tests)."""

    async def __aenter__(self) -> None:
        pass

    async def __aexit__(self, *args: object) -> None:
        pass


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

    async def test_allowed_transition_accepted(self):
        room = Room(id="r1")
        state = ConversationState(phase="triage")
        room = set_conversation_state(room, state)

        bindings = [_ai_binding("agent-a"), _ai_binding("agent-b")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(
            kit=kit,
            router=router,
            phase_map={"agent-b": "handling"},
            allowed_transitions={"triage": {"handling"}, "handling": set()},
        )
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "ok", "summary": "s"},
        )
        assert result.accepted is True
        assert result.new_phase == "handling"

    async def test_disallowed_transition_rejected(self):
        room = Room(id="r1")
        state = ConversationState(phase="triage")
        room = set_conversation_state(room, state)

        bindings = [_ai_binding("agent-a"), _ai_binding("agent-b")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(
            kit=kit,
            router=router,
            phase_map={"agent-b": "resolution"},
            # triage can only go to handling, NOT resolution
            allowed_transitions={"triage": {"handling"}, "handling": {"resolution"}},
        )
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "skip", "summary": "s"},
        )
        assert result.accepted is False
        assert "not allowed" in result.reason

    async def test_no_transition_constraints_allows_anything(self):
        """Without allowed_transitions, any transition is accepted."""
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a"), _ai_binding("agent-b")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(kit=kit, router=router)
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "ok", "summary": "s"},
        )
        assert result.accepted is True


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

    def test_double_setup_raises(self):
        channel = MagicMock()
        channel._extra_tools = []
        channel._tool_handler = None

        handler = MagicMock(spec=HandoffHandler)
        setup_handoff(channel, handler)

        import pytest

        with pytest.raises(RuntimeError, match="already called"):
            setup_handoff(channel, handler)


# -- Hook firing --------------------------------------------------------------


class TestHandoffHookFiring:
    async def test_on_handoff_fired_on_success(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a"), _ai_binding("agent-b")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(kit=kit, router=router)
        await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "done", "summary": "s"},
        )

        # ON_HANDOFF and ON_PHASE_TRANSITION should both fire
        hook_calls = kit._hook_engine.run_async_hooks.call_args_list
        triggers = [c[0][1] for c in hook_calls]
        from roomkit.models.enums import HookTrigger

        assert HookTrigger.ON_HANDOFF in triggers
        assert HookTrigger.ON_PHASE_TRANSITION in triggers

    async def test_on_handoff_rejected_fired(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a")]  # agent-b NOT in room
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(kit=kit, router=router)
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "done", "summary": "s"},
        )

        assert result.accepted is False

        hook_calls = kit._hook_engine.run_async_hooks.call_args_list
        triggers = [c[0][1] for c in hook_calls]
        from roomkit.models.enums import HookTrigger

        assert HookTrigger.ON_HANDOFF_REJECTED in triggers


# -- Orchestration internal flag -----------------------------------------------


class TestOrchestrationInternalFlag:
    async def test_send_event_has_internal_flag(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a"), _ai_binding("agent-b")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(kit=kit, router=router)
        await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "test", "summary": "s"},
        )

        kit.send_event.assert_called_once()
        call_kwargs = kit.send_event.call_args[1]
        assert call_kwargs["metadata"]["_orchestration_internal"] is True


# -- build_handoff_tool -------------------------------------------------------


class TestBuildHandoffTool:
    def test_with_targets(self):
        tool = build_handoff_tool(
            [
                ("agent-advisor", "Gives financial advice"),
                ("agent-support", "Handles support tickets"),
            ]
        )
        assert tool.name == "handoff_conversation"
        target_prop = tool.parameters["properties"]["target"]
        assert target_prop["enum"] == ["agent-advisor", "agent-support"]

    def test_includes_descriptions(self):
        tool = build_handoff_tool(
            [
                ("agent-advisor", "Gives financial advice"),
            ]
        )
        assert "agent-advisor: Gives financial advice" in tool.description

    def test_empty_returns_generic(self):
        tool = build_handoff_tool([])
        assert tool is HANDOFF_TOOL

    def test_target_without_description(self):
        tool = build_handoff_tool([("agent-x", None)])
        assert "agent-x" in tool.description
        target_prop = tool.parameters["properties"]["target"]
        assert target_prop["enum"] == ["agent-x"]

    def test_preserves_other_parameters(self):
        tool = build_handoff_tool([("agent-a", "desc")])
        assert "reason" in tool.parameters["properties"]
        assert "summary" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["target", "reason", "summary"]


class TestSetupHandoffCustomTool:
    def test_custom_tool_injected(self):
        channel = MagicMock()
        channel._extra_tools = []
        channel._tool_handler = None

        custom_tool = build_handoff_tool([("agent-b", "specialist")])
        handler = MagicMock(spec=HandoffHandler)
        setup_handoff(channel, handler, tool=custom_tool)

        assert len(channel._extra_tools) == 1
        assert channel._extra_tools[0] is custom_tool

    def test_default_tool_when_none(self):
        channel = MagicMock()
        channel._extra_tools = []
        channel._tool_handler = None

        handler = MagicMock(spec=HandoffHandler)
        setup_handoff(channel, handler)

        assert channel._extra_tools[0] is HANDOFF_TOOL


# -- known_agents + on_handoff_complete ---------------------------------------


class TestKnownAgents:
    async def test_known_agent_bypasses_binding_check(self):
        """Handoff to a known agent succeeds even without a room binding."""
        room = Room(id="r1")
        # agent-b is NOT in bindings but IS in known_agents
        bindings = [_ai_binding("agent-a")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(
            kit=kit,
            router=router,
            known_agents={"agent-b"},
        )
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "test", "summary": "s"},
        )

        assert result.accepted is True
        assert result.new_agent_id == "agent-b"

    async def test_unknown_agent_still_rejected(self):
        """Agents not in known_agents nor bindings are still rejected."""
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        handler = HandoffHandler(
            kit=kit,
            router=router,
            known_agents={"agent-b"},
        )
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-c", "reason": "t", "summary": "s"},
        )

        assert result.accepted is False


class TestOnHandoffComplete:
    async def test_callback_called_on_success(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a"), _ai_binding("agent-b")]
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        callback = AsyncMock()
        handler = HandoffHandler(
            kit=kit,
            router=router,
            on_handoff_complete=callback,
        )
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "ok", "summary": "s"},
        )

        assert result.accepted is True
        callback.assert_called_once_with("r1", result)

    async def test_callback_not_called_on_rejection(self):
        room = Room(id="r1")
        bindings = [_ai_binding("agent-a")]  # agent-b NOT in room
        kit = _make_mock_kit(room, bindings)
        router = MagicMock()

        callback = AsyncMock()
        handler = HandoffHandler(
            kit=kit,
            router=router,
            on_handoff_complete=callback,
        )
        result = await handler.handle(
            room_id="r1",
            calling_agent_id="agent-a",
            arguments={"target": "agent-b", "reason": "t", "summary": "s"},
        )

        assert result.accepted is False
        callback.assert_not_called()


class TestSendGreeting:
    async def test_send_greeting_realtime(self):
        """send_greeting injects text into RealtimeVoiceChannel sessions."""
        room = Room(id="r1")
        room = set_conversation_state(
            room, ConversationState(phase="intake", active_agent_id="agent-triage")
        )
        kit = _make_mock_kit(room, [])

        # Mock a RealtimeVoiceChannel — use create_autospec=False
        # so we can set _provider without spec restrictions
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        mock_session = MagicMock()
        mock_provider = MagicMock()
        mock_provider.inject_text = AsyncMock()
        mock_rtv = MagicMock()
        mock_rtv.__class__ = RealtimeVoiceChannel
        mock_rtv._get_room_sessions.return_value = [mock_session]
        mock_rtv._provider = mock_provider
        kit._channels = {"voice": mock_rtv}

        handler = HandoffHandler(kit=kit, router=MagicMock())
        handler._greeting_map = {"agent-triage": "Welcome! How can I help?"}

        await handler.send_greeting("r1", channel_id="voice")

        mock_rtv._provider.inject_text.assert_called_once_with(
            mock_session, "Welcome! How can I help?", role="user"
        )

    async def test_send_greeting_traditional_voice(self):
        """send_greeting sends synthetic inbound for non-realtime channels."""
        room = Room(id="r1")
        room = set_conversation_state(
            room, ConversationState(phase="intake", active_agent_id="agent-triage")
        )
        kit = _make_mock_kit(room, [])
        kit._channels = {"voice": MagicMock()}  # not a RealtimeVoiceChannel
        kit.process_inbound = AsyncMock()

        handler = HandoffHandler(kit=kit, router=MagicMock())
        handler._greeting_map = {"agent-triage": "Welcome!"}

        await handler.send_greeting("r1", channel_id="voice")

        kit.process_inbound.assert_called_once()
        msg = kit.process_inbound.call_args[0][0]
        assert msg.channel_id == "voice"
        assert msg.content.body == "Welcome!"

    async def test_send_greeting_no_greeting_configured(self):
        """send_greeting does nothing when agent has no greeting."""
        room = Room(id="r1")
        room = set_conversation_state(
            room, ConversationState(phase="intake", active_agent_id="agent-triage")
        )
        kit = _make_mock_kit(room, [])
        kit._channels = {"voice": MagicMock()}
        kit.process_inbound = AsyncMock()

        handler = HandoffHandler(kit=kit, router=MagicMock())
        handler._greeting_map = {}  # no greetings

        await handler.send_greeting("r1", channel_id="voice")

        kit.process_inbound.assert_not_called()

    async def test_send_greeting_no_active_agent(self):
        """send_greeting does nothing when no agent is active."""
        room = Room(id="r1")
        kit = _make_mock_kit(room, [])

        handler = HandoffHandler(kit=kit, router=MagicMock())
        handler._greeting_map = {"agent-triage": "Hello"}

        await handler.send_greeting("r1", channel_id="voice")
        # No crash, no calls

    async def test_send_greeting_with_language(self):
        """send_greeting prepends language instruction when agent has language."""
        room = Room(id="r1")
        room = set_conversation_state(
            room, ConversationState(phase="intake", active_agent_id="agent-triage")
        )
        kit = _make_mock_kit(room, [])
        kit._channels = {"voice": MagicMock()}
        kit.process_inbound = AsyncMock()

        # Mock agent with language
        mock_agent = MagicMock()
        mock_agent.language = "French"

        handler = HandoffHandler(kit=kit, router=MagicMock())
        handler._greeting_map = {"agent-triage": "Welcome!"}
        handler._agents = {"agent-triage": mock_agent}

        await handler.send_greeting("r1", channel_id="voice")

        msg = kit.process_inbound.call_args[0][0]
        assert msg.content.body == "[Respond in French] Welcome!"


class TestSetLanguage:
    async def test_set_language_stores_in_state(self):
        """set_language persists language in conversation state."""
        room = Room(id="r1")
        room = set_conversation_state(
            room, ConversationState(phase="intake", active_agent_id="agent-triage")
        )
        kit = _make_mock_kit(room, [])
        kit._channels = {}

        handler = HandoffHandler(kit=kit, router=MagicMock())

        await handler.set_language("r1", "French")

        # Verify update_room was called with language in state
        updated_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(updated_room)
        assert state.context["language"] == "French"

    async def test_set_language_reconfigures_realtime(self):
        """set_language reconfigures realtime session with language in prompt."""
        room = Room(id="r1")
        room = set_conversation_state(
            room, ConversationState(phase="intake", active_agent_id="agent-triage")
        )
        kit = _make_mock_kit(room, [])

        from roomkit.channels.agent import Agent
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        mock_session = MagicMock()
        mock_rtv = MagicMock()
        mock_rtv.__class__ = RealtimeVoiceChannel
        mock_rtv._get_room_sessions.return_value = [mock_session]
        mock_rtv.reconfigure_session = AsyncMock()
        kit._channels = {"voice": mock_rtv}

        agent = Agent("agent-triage", role="Triage", language="English")
        handler = HandoffHandler(kit=kit, router=MagicMock(), event_channel_id="voice")
        handler._agents = {"agent-triage": agent}

        await handler.set_language("r1", "French", channel_id="voice")

        mock_rtv.reconfigure_session.assert_called_once()
        prompt = mock_rtv.reconfigure_session.call_args[1]["system_prompt"]
        assert "French" in prompt
        assert "English" not in prompt

"""Behavioral tests for orchestration strategies.

Tests end-to-end message flow, state transitions, and the loop cycle hook
that were not covered by the wiring-level tests.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelType,
    EventType,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.loop import Loop
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.mock import MockAIProvider

# -- Helpers ------------------------------------------------------------------


class _NoopLock:
    async def __aenter__(self) -> None:
        pass

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_agent(channel_id: str, description: str | None = None) -> Agent:
    return Agent(
        channel_id=channel_id,
        provider=MockAIProvider(responses=["ok"]),
        description=description,
    )


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
        channel_type=ChannelType.WEBSOCKET,
        category=ChannelCategory.TRANSPORT,
    )


def _make_event(
    room_id: str,
    channel_id: str,
    channel_type: ChannelType = ChannelType.AI,
) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(
            channel_id=channel_id,
            channel_type=channel_type,
            direction=ChannelDirection.INBOUND,
        ),
        type=EventType.MESSAGE,
        content=TextContent(body="test"),
    )


def _make_mock_kit(room: Room, bindings: list[ChannelBinding] | None = None) -> MagicMock:
    """Create a mock kit that tracks room state across updates."""
    kit = MagicMock()
    # Track room state so get_room returns the latest version
    _room_state: dict[str, Room] = {room.id: room}

    async def _get_room(rid: str) -> Room:
        return _room_state[rid]

    async def _update_room(r: Room) -> Room:
        _room_state[r.id] = r
        return r

    kit.get_room = AsyncMock(side_effect=_get_room)
    kit.store.update_room = AsyncMock(side_effect=_update_room)
    kit.store.list_bindings = AsyncMock(return_value=bindings or [])
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.hook_engine.run_async_hooks = AsyncMock()
    kit.lock_manager = MagicMock()
    kit.lock_manager.locked = MagicMock(return_value=_NoopLock())
    kit.channels = {}
    kit.register_channel = MagicMock()
    return kit


# -- Supervisor: handler wrapping idempotency ---------------------------------


class TestSupervisorHandlerIdempotency:
    async def test_double_install_does_not_stack_handlers(self):
        """Second install on a different room must not wrap the handler twice."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")

        mock_task = MagicMock()
        mock_task.id = "t1"

        kit1 = _make_mock_kit(Room(id="r1"))
        kit1.delegate = AsyncMock(return_value=mock_task)
        kit2 = _make_mock_kit(Room(id="r2"))
        kit2.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=False)
        await s.install(kit1, "r1")
        await s.install(kit2, "r2")

        # Only one delegation tool should be injected
        delegate_tools = [t for t in boss._injected_tools if t.name == "delegate_to_w1"]
        assert len(delegate_tools) == 1

        # Handler should still work — call delegation tool
        result = await boss.tool_handler("delegate_to_w1", {"task": "do it"})
        parsed = json.loads(result)
        assert parsed["status"] == "delegated"

        # Only the first kit's delegate should have been called
        assert kit1.delegate.call_count == 1
        assert kit2.delegate.call_count == 0

    async def test_delegation_error_returns_json_error(self):
        """Delegation failure should return JSON error, not raise."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(side_effect=RuntimeError("boom"))

        s = Supervisor(supervisor=boss, workers=[w1])
        await s.install(kit, "r1")

        result = await boss.tool_handler("delegate_to_w1", {"task": "fail"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "boom" in parsed["error"]

    async def test_original_handler_preserved(self):
        """User-defined tool_handler on supervisor should still be reachable."""
        boss = _make_agent("boss")
        original_called = False

        async def custom_handler(name: str, arguments: dict) -> str:
            nonlocal original_called
            original_called = True
            return json.dumps({"custom": True})

        boss.tool_handler = custom_handler

        kit = _make_mock_kit(Room(id="r1"))
        mock_task = MagicMock()
        mock_task.id = "t1"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=[_make_agent("w1")])
        await s.install(kit, "r1")

        # Unknown tool should fall through to original
        result = await boss.tool_handler("my_custom_tool", {"x": 1})
        parsed = json.loads(result)
        assert parsed["custom"] is True
        assert original_called


# -- Loop: cycle hook behavioral tests ----------------------------------------


class TestLoopCycleHook:
    """Tests for the AFTER_BROADCAST hook that drives producer↔reviewer cycling."""

    async def _setup_loop(
        self,
        max_iterations: int = 3,
    ) -> tuple[Loop, MagicMock, Agent, Agent]:
        writer = _make_agent("writer", "Writer agent")
        editor = _make_agent("editor", "Editor agent")
        bindings = [_ai_binding("writer"), _ai_binding("editor")]
        room = Room(id="r1")
        kit = _make_mock_kit(room, bindings)

        loop = Loop(agent=writer, reviewer=editor, max_iterations=max_iterations)
        await loop.install(kit, "r1")

        return loop, kit, writer, editor

    def _get_cycle_hook(self, kit: MagicMock):
        """Extract the AFTER_BROADCAST hook function from the mock kit."""
        for call in kit.hook_engine.add_room_hook.call_args_list:
            reg = call[0][1]
            if "loop_cycle" in reg.name:
                return reg.fn
        raise AssertionError("Loop cycle hook not found")

    def _make_context(self, bindings: list[ChannelBinding]) -> RoomContext:
        return RoomContext(room=Room(id="r1"), bindings=bindings)

    async def test_producer_event_transitions_to_reviewer(self):
        """After producer outputs, state should transition to reviewer."""
        _, kit, writer, editor = await self._setup_loop()
        hook = self._get_cycle_hook(kit)
        bindings = [_ai_binding("writer"), _ai_binding("editor")]
        context = self._make_context(bindings)

        event = _make_event("r1", "writer")
        await hook(event, context)

        room = await kit.get_room("r1")
        state = get_conversation_state(room)
        assert state.active_agent_id == "editor"
        assert state.phase == "editor"
        assert state.context["_loop_iteration"] == 1

    async def test_reviewer_event_transitions_back_to_producer(self):
        """After reviewer provides feedback, state should go back to producer."""
        _, kit, writer, editor = await self._setup_loop()
        hook = self._get_cycle_hook(kit)
        bindings = [_ai_binding("writer"), _ai_binding("editor")]
        context = self._make_context(bindings)

        # First: producer → reviewer
        event = _make_event("r1", "writer")
        await hook(event, context)

        # Now: reviewer → producer
        event = _make_event("r1", "editor")
        await hook(event, context)

        room = await kit.get_room("r1")
        state = get_conversation_state(room)
        assert state.active_agent_id == "writer"
        assert state.phase == "writer"

    async def test_iteration_counter_increments(self):
        """Each producer→reviewer transition should increment the counter."""
        _, kit, writer, editor = await self._setup_loop(max_iterations=5)
        hook = self._get_cycle_hook(kit)
        bindings = [_ai_binding("writer"), _ai_binding("editor")]
        context = self._make_context(bindings)

        for i in range(3):
            # Producer → reviewer
            await hook(_make_event("r1", "writer"), context)
            room = await kit.get_room("r1")
            state = get_conversation_state(room)
            assert state.context["_loop_iteration"] == i + 1

            # Reviewer → producer (no iteration increment)
            await hook(_make_event("r1", "editor"), context)

    async def test_max_iterations_stops_loop(self):
        """Once max_iterations is reached, hook should be a no-op."""
        _, kit, writer, editor = await self._setup_loop(max_iterations=2)
        hook = self._get_cycle_hook(kit)
        bindings = [_ai_binding("writer"), _ai_binding("editor")]
        context = self._make_context(bindings)

        # Cycle 1: producer → reviewer → producer
        await hook(_make_event("r1", "writer"), context)
        await hook(_make_event("r1", "editor"), context)

        # Cycle 2: producer → reviewer → producer
        await hook(_make_event("r1", "writer"), context)
        await hook(_make_event("r1", "editor"), context)

        # Cycle 3: producer output — but max reached, should not transition
        room = await kit.get_room("r1")
        state_before = get_conversation_state(room)
        await hook(_make_event("r1", "writer"), context)
        room = await kit.get_room("r1")
        state_after = get_conversation_state(room)

        # State should not have changed
        assert state_after.active_agent_id == state_before.active_agent_id
        assert state_after.context["_loop_iteration"] == 2

    async def test_approved_stops_loop(self):
        """After approve_output is called, hook should be a no-op."""
        _, kit, writer, editor = await self._setup_loop()
        hook = self._get_cycle_hook(kit)
        bindings = [_ai_binding("writer"), _ai_binding("editor")]
        context = self._make_context(bindings)

        # Producer → reviewer
        await hook(_make_event("r1", "writer"), context)

        # Reviewer calls approve_output
        await editor.tool_handler("approve_output", {"reason": "LGTM"})

        # Verify approved
        room = await kit.get_room("r1")
        state = get_conversation_state(room)
        assert state.context["_loop_approved"] is True

        # Another event should be a no-op
        await hook(_make_event("r1", "editor"), context)
        room = await kit.get_room("r1")
        state = get_conversation_state(room)
        # Should still be on editor (approve doesn't transition)
        assert state.context["_loop_approved"] is True

    async def test_transport_events_ignored(self):
        """Events from transport channels should not trigger loop cycling."""
        _, kit, writer, editor = await self._setup_loop()
        hook = self._get_cycle_hook(kit)
        bindings = [
            _ai_binding("writer"),
            _ai_binding("editor"),
            _transport_binding("ws"),
        ]
        context = self._make_context(bindings)

        # Event from transport channel
        event = _make_event("r1", "ws", ChannelType.WEBSOCKET)
        await hook(event, context)

        room = await kit.get_room("r1")
        state = get_conversation_state(room)
        # Should still be on initial state
        assert state.active_agent_id == "writer"
        assert state.context["_loop_iteration"] == 0

    async def test_unknown_channel_events_ignored(self):
        """Events from channels not in bindings should be ignored."""
        _, kit, writer, editor = await self._setup_loop()
        hook = self._get_cycle_hook(kit)
        # No binding for "unknown"
        bindings = [_ai_binding("writer"), _ai_binding("editor")]
        context = self._make_context(bindings)

        event = _make_event("r1", "unknown")
        await hook(event, context)

        room = await kit.get_room("r1")
        state = get_conversation_state(room)
        assert state.active_agent_id == "writer"
        assert state.context["_loop_iteration"] == 0


class TestLoopHandlerIdempotency:
    async def test_double_install_does_not_stack_approve_handler(self):
        """Second install must not wrap the reviewer's tool handler twice."""
        writer = _make_agent("writer")
        editor = _make_agent("editor")

        kit1 = _make_mock_kit(Room(id="r1"))
        kit2 = _make_mock_kit(Room(id="r2"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit1, "r1")
        await loop.install(kit2, "r2")

        # Only one approve_output tool
        approve_count = sum(1 for t in editor._injected_tools if t.name == "approve_output")
        assert approve_count == 1

        # Handler should still work — call approve on room r1
        result = await editor.tool_handler("approve_output", {"reason": "good"})
        parsed = json.loads(result)
        assert parsed["status"] == "approved"

    async def test_reviewer_non_approve_tool_falls_through(self):
        """Non-approve tools on reviewer should fall through to original handler."""
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        result = await editor.tool_handler("some_other_tool", {"x": 1})
        parsed = json.loads(result)
        assert "error" in parsed


# -- Loop: full cycle integration test ----------------------------------------


class TestLoopFullCycle:
    async def test_complete_produce_review_approve_cycle(self):
        """Test a full cycle: produce → review → produce → review → approve."""
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        bindings = [_ai_binding("writer"), _ai_binding("editor")]
        room = Room(id="r1")
        kit = _make_mock_kit(room, bindings)

        loop = Loop(agent=writer, reviewer=editor, max_iterations=5)
        await loop.install(kit, "r1")

        # Extract cycle hook
        cycle_hook = None
        for call in kit.hook_engine.add_room_hook.call_args_list:
            reg = call[0][1]
            if "loop_cycle" in reg.name:
                cycle_hook = reg.fn
                break
        assert cycle_hook is not None

        context = RoomContext(room=room, bindings=bindings)

        # Iteration 1: producer outputs → transitions to reviewer
        await cycle_hook(_make_event("r1", "writer"), context)
        r = await kit.get_room("r1")
        s = get_conversation_state(r)
        assert s.active_agent_id == "editor"
        assert s.context["_loop_iteration"] == 1

        # Reviewer gives feedback → transitions back to producer
        await cycle_hook(_make_event("r1", "editor"), context)
        r = await kit.get_room("r1")
        s = get_conversation_state(r)
        assert s.active_agent_id == "writer"

        # Iteration 2: producer outputs again
        await cycle_hook(_make_event("r1", "writer"), context)
        r = await kit.get_room("r1")
        s = get_conversation_state(r)
        assert s.active_agent_id == "editor"
        assert s.context["_loop_iteration"] == 2

        # Reviewer approves
        result = await editor.tool_handler("approve_output", {"reason": "Perfect"})
        parsed = json.loads(result)
        assert parsed["status"] == "approved"

        # Verify loop is done
        r = await kit.get_room("r1")
        s = get_conversation_state(r)
        assert s.context["_loop_approved"] is True

        # Further events should be no-ops
        await cycle_hook(_make_event("r1", "editor"), context)
        r = await kit.get_room("r1")
        s2 = get_conversation_state(r)
        assert s2.context["_loop_iteration"] == s.context["_loop_iteration"]

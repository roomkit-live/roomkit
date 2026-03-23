"""Behavioral tests for orchestration strategies.

Tests end-to-end message flow, state transitions, and the loop cycle hook
that were not covered by the wiring-level tests.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding
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


# -- Loop: framework-driven tests ---------------------------------------------


class TestLoopInstall:
    """Tests for Loop.install() with framework-driven auto_cycle."""

    async def test_agents_returns_only_producer(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        loop = Loop(agent=writer, reviewers=[editor])
        agents = loop.agents()
        assert len(agents) == 1
        assert agents[0].channel_id == "writer"

    async def test_initial_state_set(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewers=[editor], max_iterations=5)
        await loop.install(kit, "r1")

        room = await kit.get_room("r1")
        state = get_conversation_state(room)
        assert state.active_agent_id == "writer"
        assert state.context["_loop_iteration"] == 0
        assert state.context["_loop_approved"] is False
        assert state.context["_loop_max_iterations"] == 5

    async def test_reviewer_registered_not_attached(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))
        kit.channels = {}

        loop = Loop(agent=writer, reviewers=[editor])
        await loop.install(kit, "r1")

        kit.register_channel.assert_called_once_with(editor)

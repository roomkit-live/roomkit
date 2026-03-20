"""Tests for the Loop orchestration strategy."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding
from roomkit.models.room import Room
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.loop import Loop
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


def _make_mock_kit(room: Room, bindings: list[ChannelBinding] | None = None) -> MagicMock:
    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.list_bindings = AsyncMock(return_value=bindings or [])
    kit.store.update_room = AsyncMock()
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.hook_engine.run_async_hooks = AsyncMock()
    kit.lock_manager = MagicMock()
    kit.lock_manager.locked = MagicMock(return_value=_NoopLock())
    kit.channels = {}
    return kit


# -- Tests --------------------------------------------------------------------


class TestLoopAgents:
    def test_agents_returns_both(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        loop = Loop(agent=writer, reviewer=editor)

        result = loop.agents()
        assert len(result) == 2
        assert result[0].channel_id == "writer"
        assert result[1].channel_id == "editor"


class TestLoopInstall:
    async def test_installs_two_hooks(self):
        """Should install router (BEFORE_BROADCAST) + cycle (AFTER_BROADCAST)."""
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        assert kit.hook_engine.add_room_hook.call_count == 2

    async def test_sets_initial_state(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor, max_iterations=5)
        await loop.install(kit, "r1")

        updated_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(updated_room)
        assert state.active_agent_id == "writer"
        assert state.phase == "writer"
        assert state.context["_loop_iteration"] == 0
        assert state.context["_loop_approved"] is False
        assert state.context["_loop_max_iterations"] == 5

    async def test_wires_handoff_tools(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        # Both agents should have handoff_conversation
        assert any(t.name == "handoff_conversation" for t in writer._injected_tools)
        assert any(t.name == "handoff_conversation" for t in editor._injected_tools)

    async def test_injects_approve_tool(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        # Only reviewer gets approve_output
        assert any(t.name == "approve_output" for t in editor._injected_tools)
        assert not any(t.name == "approve_output" for t in writer._injected_tools)


class TestLoopApproval:
    async def test_approve_tool_sets_flag(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        room = Room(id="r1")
        kit = _make_mock_kit(room)

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        # Update the mock to return the room with state
        updated_room = kit.store.update_room.call_args[0][0]
        kit.get_room = AsyncMock(return_value=updated_room)

        # Call approve_output via tool handler
        result = await editor.tool_handler("approve_output", {"reason": "Looks good"})
        parsed = json.loads(result)

        assert parsed["status"] == "approved"

        # Verify the room was updated with approval flag
        last_update = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(last_update)
        assert state.context["_loop_approved"] is True


class TestLoopDoubleInstall:
    async def test_double_install_skips_tools(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")
        kit2 = _make_mock_kit(Room(id="r2"))
        await loop.install(kit2, "r2")

        for agent in [writer, editor]:
            handoff_count = sum(
                1 for t in agent._injected_tools if t.name == "handoff_conversation"
            )
            assert handoff_count == 1

        approve_count = sum(1 for t in editor._injected_tools if t.name == "approve_output")
        assert approve_count == 1

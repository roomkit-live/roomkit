"""Tests for the Loop orchestration strategy (framework-driven)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from roomkit.channels.agent import Agent
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


def _make_mock_kit(room: Room) -> MagicMock:
    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.update_room = AsyncMock()
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.channels = {}
    kit.register_channel = MagicMock()
    return kit


# -- Tests --------------------------------------------------------------------


class TestLoopAgents:
    def test_agents_returns_only_producer(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        loop = Loop(agent=writer, reviewer=editor)

        result = loop.agents()
        assert len(result) == 1
        assert result[0].channel_id == "writer"


class TestLoopInstall:
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

    async def test_registers_reviewer_on_kit(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        kit.register_channel.assert_called_once_with(editor)

    async def test_skips_already_registered_reviewer(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))
        kit.channels = {"editor": editor}

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        kit.register_channel.assert_not_called()

    async def test_wraps_producer_on_event(self):
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        original = writer.on_event
        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        # on_event should be wrapped
        assert writer.on_event is not original

    async def test_no_handoff_tools_injected(self):
        """Framework-driven loop should NOT inject handoff or approve tools."""
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        assert not any(t.name == "handoff_conversation" for t in writer._injected_tools)
        assert not any(t.name == "handoff_conversation" for t in editor._injected_tools)
        assert not any(t.name == "approve_output" for t in editor._injected_tools)

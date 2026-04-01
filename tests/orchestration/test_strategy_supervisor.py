"""Tests for the Supervisor orchestration strategy."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.orchestration.state import get_conversation_state
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


def _make_mock_kit(room: Room) -> MagicMock:
    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.update_room = AsyncMock()
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.lock_manager = MagicMock()
    kit.lock_manager.locked = MagicMock(return_value=_NoopLock())
    kit.channels = {}
    kit.register_channel = MagicMock()
    return kit


# -- Tests --------------------------------------------------------------------


class TestSupervisorAgents:
    def test_agents_returns_only_supervisor(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1"), _make_agent("w2")]
        s = Supervisor(supervisor=boss, workers=workers)

        result = s.agents()
        assert len(result) == 1
        assert result[0].channel_id == "boss"


class TestSupervisorInstall:
    async def test_installs_router_hook(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        kit.hook_engine.add_room_hook.assert_called_once()

    async def test_registers_workers_on_kit(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1"), _make_agent("w2")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        # Workers should be registered via register_channel
        assert kit.register_channel.call_count == 2

    async def test_skips_already_registered_workers(self):
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        kit.channels = {"w1": w1}  # Already registered

        s = Supervisor(supervisor=boss, workers=[w1])
        await s.install(kit, "r1")

        kit.register_channel.assert_not_called()

    async def test_injects_delegation_tools(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1", "Worker 1"), _make_agent("w2", "Worker 2")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        tool_names = [t.name for t in boss._injected_tools]
        assert "delegate_to_w1" in tool_names
        assert "delegate_to_w2" in tool_names

    async def test_sets_initial_state(self):
        boss = _make_agent("boss")
        workers = [_make_agent("w1")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        updated_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(updated_room)
        assert state.active_agent_id == "boss"
        assert state.phase == "supervisor"

    async def test_delegation_tool_handler(self):
        """Test that the delegation tool handler calls kit.delegate."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-123"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=False)
        await s.install(kit, "r1")

        # Call the delegation tool handler
        result = await boss.tool_handler("delegate_to_w1", {"task": "Do something"})
        parsed = json.loads(result)

        assert parsed["status"] == "delegated"
        assert parsed["worker"] == "w1"
        kit.delegate.assert_called_once()

    async def test_unknown_tool_falls_through(self):
        """Non-delegation tools should fall through to original handler."""
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=[_make_agent("w1")])
        await s.install(kit, "r1")

        result = await boss.tool_handler("unknown_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed

    async def test_double_install_skips_tools(self):
        """Second install should not duplicate delegation tools."""
        boss = _make_agent("boss")
        workers = [_make_agent("w1")]
        kit = _make_mock_kit(Room(id="r1"))
        mock_task = MagicMock()
        mock_task.task_id = "t1"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")
        kit2 = _make_mock_kit(Room(id="r2"))
        kit2.delegate = AsyncMock(return_value=mock_task)
        await s.install(kit2, "r2")

        tool_count = sum(1 for t in boss._injected_tools if t.name == "delegate_to_w1")
        assert tool_count == 1


class TestSupervisorShareChannels:
    """Tests for the share_channels parameter."""

    async def test_per_worker_tool_passes_share_channels(self):
        """Per-worker delegation tools pass share_channels to kit.delegate()."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-abc"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            wait_for_result=False,
            share_channels=["system", "ws-status"],
        )
        await s.install(kit, "r1")

        await boss.tool_handler("delegate_to_w1", {"task": "Do something"})

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["system", "ws-status"]

    async def test_per_worker_tool_inline_passes_share_channels(self):
        """Inline (wait=True) per-worker tools pass share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-abc"
        mock_task.result = MagicMock(status="completed", output="done", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            wait_for_result=True,
            share_channels=["email-out"],
        )
        await s.install(kit, "r1")

        await boss.tool_handler("delegate_to_w1", {"task": "Do something"})

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["email-out"]

    async def test_strategy_sequential_passes_share_channels(self):
        """Strategy-based sequential delegation passes share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.result = MagicMock(output="result", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            strategy="sequential",
            share_channels=["system"],
        )
        await s.install(kit, "r1")

        await boss.tool_handler("delegate_workers", {"task": "Analyze this"})

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["system"]

    async def test_strategy_parallel_passes_share_channels(self):
        """Strategy-based parallel delegation passes share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        w2 = _make_agent("w2")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.result = MagicMock(output="result", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1, w2],
            strategy="parallel",
            share_channels=["ws-status"],
        )
        await s.install(kit, "r1")

        await boss.tool_handler("delegate_workers", {"task": "Analyze this"})

        assert kit.delegate.call_count == 2
        for call in kit.delegate.call_args_list:
            _, kwargs = call
            assert kwargs["share_channels"] == ["ws-status"]

    async def test_default_share_channels_is_empty(self):
        """Without share_channels, kit.delegate() receives empty list."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-abc"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=False)
        await s.install(kit, "r1")

        await boss.tool_handler("delegate_to_w1", {"task": "Do something"})

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == []

    async def test_auto_delegate_one_pass_passes_share_channels(self):
        """auto_delegate with refine_task=False passes share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        room = Room(id="r1")
        kit = _make_mock_kit(room)

        mock_task = MagicMock()
        mock_task.result = MagicMock(output="worker result", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            strategy="sequential",
            auto_delegate=True,
            refine_task=False,
            share_channels=["system"],
        )
        await s.install(kit, "r1")

        # Build a user message event to trigger auto-delegate
        event = RoomEvent(
            room_id="r1",
            type=EventType.MESSAGE,
            source=EventSource(channel_id="user", channel_type=ChannelType.SMS),
            content=TextContent(body="Analyze this topic"),
        )
        binding = ChannelBinding(
            channel_id="user",
            room_id="r1",
            channel_type=ChannelType.SMS,
        )
        context = RoomContext(room=room, bindings=[], recent_events=[])

        # Invoke the wrapped on_event
        await boss.on_event(event, binding, context)

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["system"]

    async def test_auto_delegate_two_pass_passes_share_channels(self):
        """auto_delegate with refine_task=True passes share_channels."""
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        room = Room(id="r1")
        kit = _make_mock_kit(room)

        mock_task = MagicMock()
        mock_task.result = MagicMock(output="worker result", error=None)
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            strategy="parallel",
            auto_delegate=True,
            refine_task=True,
            share_channels=["ws-status", "email-out"],
        )
        await s.install(kit, "r1")

        event = RoomEvent(
            room_id="r1",
            type=EventType.MESSAGE,
            source=EventSource(channel_id="user", channel_type=ChannelType.SMS),
            content=TextContent(body="Analyze this topic"),
        )
        binding = ChannelBinding(
            channel_id="user",
            room_id="r1",
            channel_type=ChannelType.SMS,
        )
        context = RoomContext(room=room, bindings=[], recent_events=[])

        await boss.on_event(event, binding, context)

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["ws-status", "email-out"]

    async def test_share_channels_defensive_copy(self):
        """Mutating the original list after construction has no effect."""
        channels = ["system"]
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))

        mock_task = MagicMock()
        mock_task.id = "task-abc"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            wait_for_result=False,
            share_channels=channels,
        )

        # Mutate the original list after construction
        channels.append("hacked")

        await s.install(kit, "r1")
        await boss.tool_handler("delegate_to_w1", {"task": "Do something"})

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["system"]

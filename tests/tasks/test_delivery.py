"""Tests for background task delivery strategies."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from roomkit import (
    ChannelCategory,
    RoomKit,
)
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType, TaskStatus
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delivery import (
    BackgroundTaskDeliveryStrategy,
    ContextOnlyDelivery,
    ImmediateDelivery,
    TaskDeliveryContext,
    WaitForIdleDelivery,
)
from roomkit.tasks.models import DelegatedTaskResult

# -- Helpers ------------------------------------------------------------------


def _make_result(
    *,
    task_id: str = "task-abc",
    parent_room_id: str = "room-1",
    agent_id: str = "agent-bg",
) -> DelegatedTaskResult:
    return DelegatedTaskResult(
        task_id=task_id,
        child_room_id=f"{parent_room_id}::task-child",
        parent_room_id=parent_room_id,
        agent_id=agent_id,
        status=TaskStatus.COMPLETED,
        output="result text",
        duration_ms=42.0,
    )


def _make_agent(agent_id: str, response: str = "ok") -> Agent:
    return Agent(
        agent_id,
        provider=MockAIProvider(responses=[response]),
        role="Test Agent",
        description=f"Agent {agent_id}",
    )


# -- TaskDeliveryContext ------------------------------------------------------


class TestTaskDeliveryContext:
    async def test_room_id_property(self):
        kit = RoomKit()
        result = _make_result(parent_room_id="room-42")
        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")
        assert ctx.room_id == "room-42"
        await kit.close()

    async def test_find_transport_prefers_voice_over_text(self):
        """Voice channels should be preferred (most latency-sensitive)."""
        kit = RoomKit()
        result = _make_result()

        # Register channels so get_channel works
        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        kit.register_channel(sms)

        # Create a mock voice channel with async close
        voice_ch = MagicMock()
        voice_ch.channel_id = "voice-1"
        voice_ch.channel_type = ChannelType.VOICE
        voice_ch.category = ChannelCategory.TRANSPORT
        voice_ch.close = AsyncMock()
        kit._channels["voice-1"] = voice_ch

        # Create room with both bindings
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "sms-1")
        # Manually add voice binding
        voice_binding = ChannelBinding(
            channel_id="voice-1",
            room_id="room-1",
            channel_type=ChannelType.VOICE,
            category=ChannelCategory.TRANSPORT,
        )
        await kit.store.update_binding(voice_binding)

        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")
        transport_id = await ctx.find_transport_channel_id()
        assert transport_id == "voice-1"
        await kit.close()

    async def test_find_transport_falls_back_to_text(self):
        """When no voice channel, falls back to text transport."""
        kit = RoomKit()
        result = _make_result()

        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        kit.register_channel(sms)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "sms-1")

        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")
        transport_id = await ctx.find_transport_channel_id()
        assert transport_id == "sms-1"
        await kit.close()

    async def test_find_transport_returns_none_when_no_transport(self):
        """If only intelligence channels exist, returns None."""
        kit = RoomKit()
        result = _make_result()
        agent = _make_agent("agent-only")
        kit.register_channel(agent)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "agent-only", category=ChannelCategory.INTELLIGENCE)

        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")
        transport_id = await ctx.find_transport_channel_id()
        assert transport_id is None
        await kit.close()


# -- ContextOnlyDelivery -----------------------------------------------------


class TestContextOnlyDelivery:
    async def test_does_nothing(self):
        """ContextOnly should not call process_inbound."""
        kit = RoomKit()
        kit.process_inbound = AsyncMock()
        result = _make_result()
        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")

        strategy = ContextOnlyDelivery()
        await strategy.deliver(ctx)

        kit.process_inbound.assert_not_called()
        await kit.close()


# -- ImmediateDelivery --------------------------------------------------------


class TestImmediateDelivery:
    async def test_sends_process_inbound(self):
        """Should call process_inbound with a synthetic message."""
        kit = RoomKit()
        result = _make_result()

        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        kit.register_channel(sms)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "sms-1")

        kit.process_inbound = AsyncMock()
        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")

        strategy = ImmediateDelivery()
        await strategy.deliver(ctx)

        kit.process_inbound.assert_called_once()
        call_args = kit.process_inbound.call_args
        msg = call_args[0][0]
        assert msg.channel_id == "sms-1"
        assert msg.sender_id == "system"
        assert call_args[1]["room_id"] == "room-1"
        await kit.close()

    async def test_custom_prompt(self):
        """Custom prompt text should be passed in the inbound message."""
        kit = RoomKit()
        result = _make_result()

        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        kit.register_channel(sms)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "sms-1")

        kit.process_inbound = AsyncMock()
        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")

        strategy = ImmediateDelivery(prompt="Results are ready!")
        await strategy.deliver(ctx)

        msg = kit.process_inbound.call_args[0][0]
        assert msg.content.body == "Results are ready!"
        await kit.close()

    async def test_no_transport_skips(self):
        """If no transport channel exists, delivery is silently skipped."""
        kit = RoomKit()
        result = _make_result()
        agent = _make_agent("agent-only")
        kit.register_channel(agent)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "agent-only", category=ChannelCategory.INTELLIGENCE)

        kit.process_inbound = AsyncMock()
        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")

        strategy = ImmediateDelivery()
        await strategy.deliver(ctx)

        kit.process_inbound.assert_not_called()
        await kit.close()


# -- WaitForIdleDelivery ------------------------------------------------------


class TestWaitForIdleDelivery:
    async def test_waits_for_playback_on_voice(self):
        """For voice channels, should call wait_playback_done before process_inbound."""
        kit = RoomKit()
        result = _make_result()

        # Create a mock VoiceChannel
        from roomkit.channels.voice import VoiceChannel

        voice_ch = MagicMock(spec=VoiceChannel)
        voice_ch.channel_id = "voice-1"
        voice_ch.channel_type = ChannelType.VOICE
        voice_ch.category = ChannelCategory.TRANSPORT
        voice_ch.wait_playback_done = AsyncMock()
        kit._channels["voice-1"] = voice_ch

        await kit.create_room(room_id="room-1")
        voice_binding = ChannelBinding(
            channel_id="voice-1",
            room_id="room-1",
            channel_type=ChannelType.VOICE,
            category=ChannelCategory.TRANSPORT,
        )
        await kit.store.update_binding(voice_binding)

        kit.process_inbound = AsyncMock()
        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")

        strategy = WaitForIdleDelivery(playback_timeout=5.0)
        await strategy.deliver(ctx)

        voice_ch.wait_playback_done.assert_called_once_with("room-1", timeout=5.0)
        kit.process_inbound.assert_called_once()
        await kit.close()

    async def test_delivers_immediately_for_text(self):
        """For non-voice channels, should not wait — deliver immediately."""
        kit = RoomKit()
        result = _make_result()

        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        sms = SMSChannel("sms-1", provider=MockSMSProvider())
        kit.register_channel(sms)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "sms-1")

        kit.process_inbound = AsyncMock()
        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")

        strategy = WaitForIdleDelivery()
        await strategy.deliver(ctx)

        # process_inbound called without waiting for playback
        kit.process_inbound.assert_called_once()
        msg = kit.process_inbound.call_args[0][0]
        assert msg.channel_id == "sms-1"
        await kit.close()

    async def test_no_transport_skips(self):
        """If no transport channel exists, delivery is silently skipped."""
        kit = RoomKit()
        result = _make_result()
        agent = _make_agent("agent-only")
        kit.register_channel(agent)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "agent-only", category=ChannelCategory.INTELLIGENCE)

        kit.process_inbound = AsyncMock()
        ctx = TaskDeliveryContext(kit=kit, result=result, notify_channel_id="agent-x")

        strategy = WaitForIdleDelivery()
        await strategy.deliver(ctx)

        kit.process_inbound.assert_not_called()
        await kit.close()


# -- Integration with delegate() ---------------------------------------------


class TestDeliveryIntegration:
    async def test_global_strategy_invoked_on_delegate(self):
        """Framework-level strategy should be called after task completes."""
        calls: list[TaskDeliveryContext] = []

        class TrackingStrategy(BackgroundTaskDeliveryStrategy):
            async def deliver(self, ctx: TaskDeliveryContext) -> None:
                calls.append(ctx)

        kit = RoomKit(delivery_strategy=TrackingStrategy())

        agent = _make_agent("agent-bg", "result from bg")
        main_ai = AIChannel(
            "main-ai",
            provider=MockAIProvider(responses=["ok"]),
            system_prompt="main",
        )
        kit.register_channel(agent)
        kit.register_channel(main_ai)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "main-ai", category=ChannelCategory.INTELLIGENCE)

        task = await kit.delegate(
            room_id="room-1",
            agent_id="agent-bg",
            task="do background work",
            notify="main-ai",
        )
        await task.wait(timeout=5.0)

        assert len(calls) == 1
        assert calls[0].result.task_id == task.id
        assert calls[0].notify_channel_id == "main-ai"
        assert calls[0].room_id == "room-1"
        await kit.close()

    async def test_per_task_override(self):
        """Per-task strategy should override the global one."""
        global_calls: list[TaskDeliveryContext] = []
        override_calls: list[TaskDeliveryContext] = []

        class GlobalStrategy(BackgroundTaskDeliveryStrategy):
            async def deliver(self, ctx: TaskDeliveryContext) -> None:
                global_calls.append(ctx)

        class OverrideStrategy(BackgroundTaskDeliveryStrategy):
            async def deliver(self, ctx: TaskDeliveryContext) -> None:
                override_calls.append(ctx)

        kit = RoomKit(delivery_strategy=GlobalStrategy())

        agent = _make_agent("agent-bg", "result")
        main_ai = AIChannel(
            "main-ai",
            provider=MockAIProvider(responses=["ok"]),
            system_prompt="main",
        )
        kit.register_channel(agent)
        kit.register_channel(main_ai)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "main-ai", category=ChannelCategory.INTELLIGENCE)

        task = await kit.delegate(
            room_id="room-1",
            agent_id="agent-bg",
            task="do work",
            notify="main-ai",
            delivery_strategy=OverrideStrategy(),
        )
        await task.wait(timeout=5.0)

        assert len(global_calls) == 0
        assert len(override_calls) == 1
        await kit.close()

    async def test_per_task_none_disables_delivery(self):
        """Passing delivery_strategy=None should disable delivery even with global."""
        global_calls: list[TaskDeliveryContext] = []

        class GlobalStrategy(BackgroundTaskDeliveryStrategy):
            async def deliver(self, ctx: TaskDeliveryContext) -> None:
                global_calls.append(ctx)

        kit = RoomKit(delivery_strategy=GlobalStrategy())

        agent = _make_agent("agent-bg", "result")
        main_ai = AIChannel(
            "main-ai",
            provider=MockAIProvider(responses=["ok"]),
            system_prompt="main",
        )
        kit.register_channel(agent)
        kit.register_channel(main_ai)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "main-ai", category=ChannelCategory.INTELLIGENCE)

        task = await kit.delegate(
            room_id="room-1",
            agent_id="agent-bg",
            task="do work",
            notify="main-ai",
            delivery_strategy=None,
        )
        await task.wait(timeout=5.0)

        assert len(global_calls) == 0
        await kit.close()

    async def test_no_strategy_maintains_backward_compat(self):
        """Without any strategy, delegation works as before (no delivery)."""
        kit = RoomKit()

        agent = _make_agent("agent-bg", "result")
        main_ai = AIChannel(
            "main-ai",
            provider=MockAIProvider(responses=["ok"]),
            system_prompt="main",
        )
        kit.register_channel(agent)
        kit.register_channel(main_ai)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "main-ai", category=ChannelCategory.INTELLIGENCE)

        task = await kit.delegate(
            room_id="room-1",
            agent_id="agent-bg",
            task="do work",
            notify="main-ai",
        )
        result = await task.wait(timeout=5.0)

        assert result.status == TaskStatus.COMPLETED
        # System prompt was still injected
        binding = await kit.store.get_binding("room-1", "main-ai")
        assert binding is not None
        assert "BACKGROUND TASK COMPLETED" in binding.metadata.get("system_prompt", "")
        await kit.close()

    async def test_strategy_error_does_not_break_delegation(self):
        """If strategy.deliver() raises, the task still completes normally."""

        class BrokenStrategy(BackgroundTaskDeliveryStrategy):
            async def deliver(self, ctx: TaskDeliveryContext) -> None:
                raise RuntimeError("boom")

        kit = RoomKit(delivery_strategy=BrokenStrategy())

        agent = _make_agent("agent-bg", "result")
        main_ai = AIChannel(
            "main-ai",
            provider=MockAIProvider(responses=["ok"]),
            system_prompt="main",
        )
        kit.register_channel(agent)
        kit.register_channel(main_ai)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "main-ai", category=ChannelCategory.INTELLIGENCE)

        task = await kit.delegate(
            room_id="room-1",
            agent_id="agent-bg",
            task="do work",
            notify="main-ai",
        )
        result = await task.wait(timeout=5.0)

        # Task still completed despite strategy failure
        assert result.status == TaskStatus.COMPLETED
        await kit.close()


# -- DelegateHandler with delivery_strategy -----------------------------------


class TestDelegateHandlerDelivery:
    async def test_handler_passes_strategy_to_delegate(self):
        """DelegateHandler should forward delivery_strategy to kit.delegate()."""
        calls: list[TaskDeliveryContext] = []

        class TrackingStrategy(BackgroundTaskDeliveryStrategy):
            async def deliver(self, ctx: TaskDeliveryContext) -> None:
                calls.append(ctx)

        kit = RoomKit()

        agent = _make_agent("agent-bg", "result")
        main_ai = AIChannel(
            "main-ai",
            provider=MockAIProvider(responses=["ok"]),
            system_prompt="main",
        )
        kit.register_channel(agent)
        kit.register_channel(main_ai)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "main-ai", category=ChannelCategory.INTELLIGENCE)

        from roomkit.tasks.delegate import DelegateHandler

        handler = DelegateHandler(
            kit,
            delivery_strategy=TrackingStrategy(),
        )

        result = await handler.handle(
            room_id="room-1",
            calling_agent_id="main-ai",
            arguments={"agent": "agent-bg", "task": "do work"},
        )

        assert result["status"] == "delegated"

        # Give the async task runner time to complete
        await asyncio.sleep(0.5)

        assert len(calls) == 1
        assert calls[0].notify_channel_id == "main-ai"
        await kit.close()

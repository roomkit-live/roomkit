"""Tests for hook and broadcast side effect persistence (Area 3.4)."""

from __future__ import annotations

import pytest

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.task import Observation, Task
from tests.test_framework import SimpleChannel


class ObserverChannel(Channel):
    """Channel that produces observations on_event."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        obs = Observation(
            id=f"obs_{event.id}",
            room_id=event.room_id,
            channel_id=self.channel_id,
            content="sentiment positive",
            category="sentiment",
        )
        return ChannelOutput(observations=[obs])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class TaskCreatorChannel(Channel):
    """Channel that produces tasks on_event."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        task = Task(
            id=f"task_{event.id}",
            room_id=event.room_id,
            title="Follow up with client",
        )
        return ChannelOutput(tasks=[task])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestBroadcastObservationsPersisted:
    async def test_observations_stored_from_broadcast(self, kit: RoomKit) -> None:
        """Observations from channel on_event are persisted to the store."""
        ch = SimpleChannel("sms1")
        observer = ObserverChannel("observer1")
        kit.register_channel(ch)
        kit.register_channel(observer)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "observer1", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        observations = await kit.store.list_observations("r1")
        assert len(observations) == 1
        assert observations[0].category == "sentiment"


class TestBroadcastTasksPersisted:
    async def test_tasks_stored_from_broadcast(self, kit: RoomKit) -> None:
        """Tasks from channel on_event are persisted to the store."""
        ch = SimpleChannel("sms1")
        task_ch = TaskCreatorChannel("task_ai")
        kit.register_channel(ch)
        kit.register_channel(task_ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "task_ai", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        tasks = await kit.store.list_tasks("r1")
        assert len(tasks) == 1
        assert tasks[0].title == "Follow up with client"


class TestHookTasksPersisted:
    async def test_sync_hook_tasks_persisted(self, kit: RoomKit) -> None:
        """Tasks returned by sync hooks are persisted to the store."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="task_hook")
        async def task_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            task = Task(
                id="hook_task_1",
                room_id=event.room_id,
                title="Hook-created task",
            )
            return HookResult(action="allow", tasks=[task])

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        tasks = await kit.store.list_tasks("r1")
        assert len(tasks) == 1
        assert tasks[0].title == "Hook-created task"

    async def test_blocked_hook_tasks_still_persisted(self, kit: RoomKit) -> None:
        """Tasks from blocking hooks are still persisted (even when event blocked)."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="block_with_task")
        async def block_with_task(event: RoomEvent, ctx: RoomContext) -> HookResult:
            task = Task(
                id="blocked_task_1",
                room_id=event.room_id,
                title="Audit: blocked message",
            )
            return HookResult.block(reason="spam", tasks=[task])

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="spammer",
            content=TextContent(body="buy now"),
        )
        result = await kit.process_inbound(msg)
        assert result.blocked

        tasks = await kit.store.list_tasks("r1")
        assert len(tasks) == 1
        assert tasks[0].title == "Audit: blocked message"


class TestHookObservationsPersisted:
    async def test_sync_hook_observations_persisted(self, kit: RoomKit) -> None:
        """Observations returned by sync hooks are persisted."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="obs_hook")
        async def obs_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            obs = Observation(
                id="hook_obs_1",
                room_id=event.room_id,
                channel_id="system",
                content="Content scanned: clean",
                category="content_scan",
            )
            return HookResult(action="allow", observations=[obs])

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        observations = await kit.store.list_observations("r1")
        assert len(observations) == 1
        assert observations[0].category == "content_scan"


class TestOnTaskCreatedHookFires:
    async def test_on_task_hook_fires(self, kit: RoomKit) -> None:
        """ON_TASK_CREATED fires for tasks created during broadcast."""
        fired: list[RoomEvent] = []

        ch = SimpleChannel("sms1")
        task_ch = TaskCreatorChannel("task_ai")
        kit.register_channel(ch)
        kit.register_channel(task_ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "task_ai", category=ChannelCategory.INTELLIGENCE)

        async def on_task(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        from roomkit.core.hooks import HookRegistration

        kit.hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.ON_TASK_CREATED,
                execution=HookExecution.ASYNC,
                fn=on_task,
                name="test_on_task",
            )
        )

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert len(fired) == 1

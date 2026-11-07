"""Tests for hook and broadcast side effect persistence (Area 3.4)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit.channels.agent import Agent
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
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.task import Observation, Task
from roomkit.providers.ai.mock import MockAIProvider
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


class UpdatingChannel(Channel):
    """A channel whose output carries room metadata updates: an intelligence
    channel's from on_event, a transport's from deliver."""

    def __init__(self, channel_id: str, updates: dict[str, Any], *, agent: bool = False) -> None:
        super().__init__(channel_id)
        self.channel_type = ChannelType.AI if agent else ChannelType.WEBSOCKET
        self.category = ChannelCategory.INTELLIGENCE if agent else ChannelCategory.TRANSPORT
        self._updates = updates

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if self.category is ChannelCategory.INTELLIGENCE:
            return ChannelOutput(metadata_updates=dict(self._updates))
        return ChannelOutput.empty()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(metadata_updates=dict(self._updates))


class _NotesSources(UpdatingChannel):
    """A transport whose delivery notes, on the room, whose event it delivered."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id, {})

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(metadata_updates={f"seen_{event.source.channel_id}": True})


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


class TestBroadcastMetadataUpdatesApplied:
    async def test_the_targets_updates_are_written_on_the_room(
        self, kit: RoomKit, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RFC §10.1 step 15 (RMK-976): the room metadata updates of the
        targets' outputs, on_event's and deliver's alike, are merged into the
        room's metadata, which keeps its other keys; a key only the framework
        writes, of its namespace (§5.1) or of the room's private state
        (§14.1), is dropped with a warning, the rest applied."""
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(
            UpdatingChannel("ai1", {"stage": "qualified", "roomkit:forged": True}, agent=True)
        )
        kit.register_channel(
            UpdatingChannel("crm", {"crm_contact": "c_42", "author_register": {"version": 3}})
        )
        await kit.create_room(room_id="r1", metadata={"topic": "loan"})
        for channel_id in ("sms1", "ai1", "crm"):
            await kit.attach_channel("r1", channel_id)

        with caplog.at_level(logging.WARNING):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="sms1", sender_id="user1", content=TextContent(body="I want a loan")
                )
            )

        room = await kit.get_room("r1")
        assert room.metadata == {"topic": "loan", "stage": "qualified", "crm_contact": "c_42"}
        dropped = sorted(r.getMessage() for r in caplog.records if "drops" in r.getMessage())
        assert len(dropped) == 2
        assert "ai1" in dropped[0] and "'roomkit:forged'" in dropped[0]
        assert "crm" in dropped[1] and "'author_register'" in dropped[1]

    @pytest.mark.parametrize("delivery", ["a hook's injection", "a task's child room"])
    async def test_a_transport_s_updates_are_written_whatever_it_delivered(
        self, kit: RoomKit, delivery: str
    ) -> None:
        """RMK-976: a transport's updates are written on the room it delivered
        in, whatever it delivered there: an event a hook injected, or a
        delegated task's answer in its child room, which shares the
        transport."""
        kit.register_channel(_NotesSources("crm"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "crm")
        if delivery == "a task's child room":
            kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["done"])))
            task = await kit.delegate("r1", "worker", "go", share_channels=["crm"], wait=True)
            await task.wait()
            room = await kit.get_room(task.child_room_id)
            assert room.metadata.get("seen_worker") is True
            return

        kit.register_channel(SimpleChannel("sms1"))
        await kit.attach_channel("r1", "sms1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_id != "sms1":
                return HookResult.allow()
            note = RoomEvent(
                room_id="r1",
                source=EventSource(channel_id="hooknote", channel_type=ChannelType.SYSTEM),
                content=TextContent(body="note"),
            )
            return HookResult.allow([InjectedEvent(event=note, target_channel_ids=["crm"])])

        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="hi"))
        )
        room = await kit.get_room("r1")
        assert room.metadata.get("seen_hooknote") is True


class _AnsweringUpdater(UpdatingChannel):
    """An agent that answers ``done``, its output carrying room metadata
    updates."""

    def __init__(self, channel_id: str, updates: dict[str, Any]) -> None:
        super().__init__(channel_id, updates, agent=True)

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        answer = RoomEvent(
            room_id=event.room_id,
            source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
            content=TextContent(body="done"),
        )
        return ChannelOutput(
            responded=True, response_events=[answer], metadata_updates=dict(self._updates)
        )


def _failing_writes(monkeypatch: pytest.MonkeyPatch, kit: RoomKit, failing: str) -> None:
    """The kit's store fails the write *failing* names: the task ``t1``,
    the observation ``o1``, or a metadata update holding ``stage``."""
    store = kit.store
    add_task, add_observation = store.add_task, store.add_observation
    patch_room_metadata = store.patch_room_metadata

    async def write_task(task: Task) -> Task:
        if failing == "task" and task.id == "t1":
            raise RuntimeError("store down")
        return await add_task(task)

    async def write_observation(observation: Observation) -> Observation:
        if failing == "observation" and observation.id == "o1":
            raise RuntimeError("store down")
        return await add_observation(observation)

    async def write_metadata(room_id: str, updates: dict[str, Any], **kwargs: Any) -> Any:
        if failing == "metadata" and "stage" in updates:
            raise RuntimeError("store down")
        return await patch_room_metadata(room_id, updates, **kwargs)

    monkeypatch.setattr(store, "add_task", write_task)
    monkeypatch.setattr(store, "add_observation", write_observation)
    monkeypatch.setattr(store, "patch_room_metadata", write_metadata)


def _error_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name.startswith("roomkit") and r.levelno >= logging.ERROR
    ]


# The CRM's updates, in the order it writes them: the line names them sorted.
_CRM_UPDATES = {"stage": "qualified", "owner": "ana"}


class TestSideEffectWriteFailures:
    @pytest.mark.parametrize(
        ("failing", "blocks"),
        [("task", False), ("observation", False), ("metadata", False), ("task", True)],
        ids=["task", "observation", "metadata", "a-blocked-event-s-task"],
    )
    async def test_a_failed_side_effect_write_is_logged_and_the_call_goes_on(
        self,
        kit: RoomKit,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        failing: str,
        blocks: bool,
    ) -> None:
        """RFC §10.1 step 15, §15.2 (RMK-1498): a side effect the store fails
        to write is one ERROR line, the others still written, a blocked
        event's too; the call goes on, its error its turn's, and only a task
        stored fires ON_TASK_CREATED."""
        _failing_writes(monkeypatch, kit, failing)
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(UpdatingChannel("crm", _CRM_UPDATES))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "crm")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="triage")
        async def triage(event: RoomEvent, ctx: RoomContext) -> HookResult:
            tasks = [Task(id=f"t{i}", room_id="r1", title=f"Call back {i}") for i in (1, 2)]
            observations = [
                Observation(id=f"o{i}", room_id="r1", channel_id="sms1", content=f"note {i}")
                for i in (1, 2)
            ]
            if blocks:
                return HookResult.block(reason="spam", tasks=tasks, observations=observations)
            return HookResult(action="allow", tasks=tasks, observations=observations)

        created: list[str] = []

        @kit.hook(HookTrigger.ON_TASK_CREATED, execution=HookExecution.ASYNC, name="follow_up")
        async def follow_up(event: RoomEvent, ctx: RoomContext) -> None:
            created.append(event.metadata["task_id"])

        with caplog.at_level(logging.ERROR):
            message = InboundMessage(
                channel_id="sms1", sender_id="user1", content=TextContent(body="hi")
            )
            result = await kit.process_inbound(message)
            await kit.hook_engine.drain_observers()
        room = await kit.get_room("r1")
        tasks = sorted(task.id for task in await kit.store.list_tasks("r1"))
        observations = sorted(o.id for o in await kit.store.list_observations("r1"))
        await kit.close()

        expected = {
            "task": "Failed to persist task t1",
            "observation": "Failed to persist observation o1",
            "metadata": "Failed to write the room metadata updates ['owner', 'stage']",
        }
        assert (result.error, result.blocked, _error_lines(caplog)) == (
            None,
            blocks,
            [expected[failing]],
        )
        assert tasks == sorted(created) == (["t2"] if failing == "task" else ["t1", "t2"])
        assert observations == (["o2"] if failing == "observation" else ["o1", "o2"])
        stage = None if failing == "metadata" or blocks else "qualified"
        assert room.metadata.get("stage") == stage

    @pytest.mark.parametrize("writer", ["a shared transport", "the agent"])
    async def test_a_delegated_turn_s_failed_metadata_write_is_logged(
        self,
        kit: RoomKit,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        writer: str,
    ) -> None:
        """RFC §23.3 step 5, §10.1 step 15 (RMK-1498): the metadata updates a
        task's child room is given, by a transport shared into it or by the
        agent's own output, are written as a room's are, a failure one ERROR
        line, and the task still completes."""
        _failing_writes(monkeypatch, kit, "metadata")
        await kit.create_room(room_id="r1")
        share: list[str] | None = None
        if writer == "a shared transport":
            kit.register_channel(UpdatingChannel("crm", _CRM_UPDATES))
            kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["done"])))
            await kit.attach_channel("r1", "crm")
            share = ["crm"]
        else:
            kit.register_channel(_AnsweringUpdater("worker", _CRM_UPDATES))

        with caplog.at_level(logging.ERROR):
            task = await kit.delegate("r1", "worker", "go", share_channels=share, wait=True)
            result = await task.wait()
        child = await kit.get_room(task.child_room_id)
        await kit.close()

        assert (result.status, _error_lines(caplog), child.metadata.get("stage")) == (
            "completed",
            ["Failed to write the room metadata updates ['owner', 'stage']"],
            None,
        )


class TestDroppedSetSideEffects:
    async def test_a_set_the_lane_drops_as_stale_still_stores_its_hooks_side_effects(
        self, kit: RoomKit, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RFC §10.1 step 15, §13.5 (RMK-1499): a delivery set the room's lane
        drops before it ran, its index declared lost by another worker, still
        stores the task its event's hook gave before the call returns, and the
        task's ON_TASK_CREATED fires, as after a set that ran."""
        add_task = kit.store.add_task

        async def slow_add_task(task: Task) -> Task:
            # A store that takes a while: the call still returns after it.
            for _ in range(20):
                await asyncio.sleep(0)
            return await add_task(task)

        monkeypatch.setattr(kit.store, "add_task", slow_add_task)
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(SimpleChannel("ws1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="triage")
        async def triage(event: RoomEvent, ctx: RoomContext) -> HookResult:
            task = Task(id="t1", room_id="r1", title="Call the customer back")
            return HookResult(action="allow", tasks=[task])

        created: list[str] = []

        @kit.hook(HookTrigger.ON_TASK_CREATED, execution=HookExecution.ASYNC, name="follow_up")
        async def follow_up(event: RoomEvent, ctx: RoomContext) -> None:
            created.append(event.metadata["task_id"])

        # Another worker's lane declares the next index lost before this
        # process delivers it.
        room = await kit.get_room("r1")
        latest = -1 if room.latest_index is None else room.latest_index
        await kit.store.advance_delivered_index("r1", latest + 1, force=True)
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="user1", content=TextContent(body="Call me back")
            )
        )
        tasks = [task.id for task in await kit.store.list_tasks("r1")]
        await kit.hook_engine.drain_observers()
        await kit.close()

        assert (result.cancellation_reason, tasks, created) == ("delivery_stale", ["t1"], ["t1"])


class TestARoundThatFailsMidway:
    async def test_a_set_whose_cursor_write_fails_once_it_ran(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RMK-1515, today's behaviour: the store fails the room's delivery
        cursor write once, after the event's delivery set ran. The lane's
        round raises (lanes._claimed_round), and its finally drops the entry
        it ran as a cut one (_process_ready, ran=True): the transport was
        delivered to, but the set's aftermath never runs. The task the
        event's hook gave is not stored, ON_TASK_CREATED and AFTER_BROADCAST
        never fire, and the call reads its delivery cancelled. The cursor
        stays below the index: the room's next message waits for the gap
        timeout, which skips the index as lost."""
        kit = RoomKit(delivery_gap_timeout=0.01)
        # The lane's backoff between two looks at a hole, which no kit option
        # sets: short, so that the gap timeout is reached in milliseconds.
        kit._lanes._config.gap_backoff_initial = 0.002
        kit._lanes._config.gap_backoff_max = 0.01
        advance = kit.store.advance_delivered_index

        async def failing_once(room_id: str, index: int, **kwargs: Any) -> bool:
            monkeypatch.setattr(kit.store, "advance_delivered_index", advance)
            raise RuntimeError("store down")

        ws1 = SimpleChannel("ws1")
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(ws1)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="triage")
        async def triage(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if not isinstance(event.content, TextContent) or event.content.body != "Call me back":
                return HookResult(action="allow")
            task = Task(id="t1", room_id="r1", title="Call the customer back")
            return HookResult(action="allow", tasks=[task])

        later: list[str] = []

        @kit.hook(HookTrigger.ON_TASK_CREATED, execution=HookExecution.ASYNC, name="follow_up")
        async def follow_up(event: RoomEvent, ctx: RoomContext) -> None:
            later.append(f"task_created {event.metadata['task_id']}")

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: RoomEvent, ctx: RoomContext) -> None:
            assert isinstance(event.content, TextContent)
            later.append(f"after_broadcast {event.content.body}")

        skipped: list[tuple[int, int]] = []

        @kit.on("delivery_skipped")
        async def on_skipped(event: Any) -> None:
            skipped.append((event.data["from_index"], event.data["to_index"]))

        # The room's records are accounted for: the next cursor write is the
        # lane's, once the message's set ran.
        monkeypatch.setattr(kit.store, "advance_delivered_index", failing_once)
        first = await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="user1", content=TextContent(body="Call me back")
            )
        )
        tasks = [task.id for task in await kit.store.list_tasks("r1")]
        await kit.hook_engine.drain_observers()
        after_first = (len(ws1.delivered), list(later))
        second = await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="Hi"))
        )
        await kit.hook_engine.drain_observers()
        await kit.close()

        assert first.event is not None
        assert (first.cancellation_reason, tasks, after_first) == ("lane_cancelled", [], (1, []))
        index = first.event.index
        assert (second.cancellation_reason, len(ws1.delivered), skipped, later) == (
            None,
            2,
            [(index, index)],
            ["after_broadcast Hi"],
        )

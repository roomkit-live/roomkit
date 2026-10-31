"""ASYNC observers of a SYNC-run trigger run without holding up the operation (RFC §9, RMK-903)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.hooks import HookEngine, HookRegistration
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import EditContent, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.models.task import Task
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.test_event_ops import RecordingTransport
from tests.test_framework import SimpleChannel


def _ctx() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


def _register(engine: HookEngine, execution: HookExecution, fn, name: str) -> None:
    engine.register(
        HookRegistration(trigger=HookTrigger.BEFORE_TTS, execution=execution, fn=fn, name=name)
    )


class TestTheOperationGoesOn:
    async def test_a_sync_run_trigger_returns_before_a_slow_observer_ends(self) -> None:
        engine = HookEngine()
        seen: list[str] = []
        release = asyncio.Event()

        async def slow_observer(text: str, ctx: RoomContext) -> None:
            await release.wait()
            seen.append(text)

        _register(engine, HookExecution.ASYNC, slow_observer, "slow")
        result = await asyncio.wait_for(
            engine.run_sync_hooks(
                "r1", HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
            ),
            1,
        )
        assert result.allowed and seen == []  # returned while the observer still waits

        release.set()
        await engine.drain_observers()
        assert seen == ["hello"]  # it ran all the same

    async def test_run_observers_returns_at_once_too(self) -> None:
        engine = HookEngine()
        done, release = asyncio.Event(), asyncio.Event()

        async def slow_observer(text: str, ctx: RoomContext) -> None:
            await release.wait()
            done.set()

        _register(engine, HookExecution.ASYNC, slow_observer, "slow")
        await asyncio.wait_for(
            engine.run_observers(
                "r1", HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
            ),
            1,
        )
        assert not done.is_set()
        release.set()
        await engine.drain_observers()
        assert done.is_set()

    async def test_the_observer_sees_what_the_sync_chain_left(self) -> None:
        engine = HookEngine()
        seen: list[str] = []

        async def rewrite(text: str, ctx: RoomContext) -> HookResult:
            return HookResult.modify(text.upper())

        async def observer(text: str, ctx: RoomContext) -> None:
            seen.append(text)

        _register(engine, HookExecution.SYNC, rewrite, "rewrite")
        _register(engine, HookExecution.ASYNC, observer, "observer")
        await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
        )
        await engine.drain_observers()
        assert seen == ["HELLO"]

    async def test_an_observer_that_fails_is_logged_and_nothing_else(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = HookEngine()

        async def failing(text: str, ctx: RoomContext) -> None:
            raise RuntimeError("observer broke")

        _register(engine, HookExecution.ASYNC, failing, "failing")
        with caplog.at_level(logging.ERROR, logger="roomkit.hooks"):
            result = await engine.run_sync_hooks(
                "r1", HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
            )
            await engine.drain_observers()
        assert result.allowed
        assert any("Async hook failing failed" in r.getMessage() for r in caplog.records)


class _Tasker(SimpleChannel):
    """A transport whose reaction to each event files a task."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(tasks=[Task(id=f"t-{event.id}", room_id=event.room_id, title="t")])


class _FailingAgent(SimpleChannel):
    """An agent whose every turn fails."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        raise ValueError("the model is down")


def _said(text: str) -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=text))


def _bodies(channel: SimpleChannel) -> list[str]:
    return [getattr(event.content, "body", "") for event in channel.delivered]


async def _room_with(*channels: SimpleChannel) -> tuple[RoomKit, SimpleChannel]:
    """A room of ``sms``, whose messages keep their type (an edit), and a
    ``witness`` transport, with *channels*, an AI-typed one bound as an
    agent."""
    kit = RoomKit()
    witness = SimpleChannel("witness")
    await kit.create_room(room_id="r1")
    for channel in (RecordingTransport("sms"), witness, *channels):
        kit.register_channel(channel)
        agent = channel.channel_type is ChannelType.AI
        category = ChannelCategory.INTELLIGENCE if agent else ChannelCategory.TRANSPORT
        await kit.attach_channel("r1", channel.channel_id, category=category)
    return kit, witness


class TestTheRoomGoesOn:
    """An event's ASYNC hooks run once its delivery set has run: neither its
    call nor the room's next delivery waits for them (RFC §9.3, §10.1 step
    16, RMK-1020). They ran in the room's delivery lane, which held both."""

    @pytest.mark.parametrize(
        "trigger",
        [
            HookTrigger.AFTER_BROADCAST,
            HookTrigger.ON_EVENT_UPDATED,
            HookTrigger.ON_TASK_CREATED,
            HookTrigger.ON_ERROR,
        ],
    )
    async def test_neither_the_call_nor_the_next_delivery_waits_for_its_hooks(
        self, trigger: HookTrigger
    ) -> None:
        channels = {
            HookTrigger.ON_TASK_CREATED: [_Tasker("tasker")],
            HookTrigger.ON_ERROR: [_FailingAgent("ai", ChannelType.AI)],
        }.get(trigger, [])
        kit, witness = await _room_with(*channels)
        message = _said("One")
        if trigger is HookTrigger.ON_EVENT_UPDATED:
            said = await kit.process_inbound(message, room_id="r1")
            assert said.event is not None
            edit = EditContent(target_event_id=said.event.id, new_content=TextContent(body="1"))
            message = InboundMessage(
                channel_id="sms", sender_id="u", event_type=EventType.EDIT, content=edit
            )
        started, release = asyncio.Event(), asyncio.Event()

        @kit.hook(trigger, execution=HookExecution.ASYNC, name="slow")
        async def slow(event: RoomEvent, ctx: RoomContext) -> None:
            started.set()
            await release.wait()

        try:
            await asyncio.wait_for(kit.process_inbound(message, room_id="r1"), 1)
            await asyncio.wait_for(started.wait(), 1)
            await asyncio.wait_for(kit.process_inbound(_said("Two"), room_id="r1"), 1)
            assert "Two" in _bodies(witness)
        finally:
            release.set()
            await kit.close()

    async def test_an_answer_does_not_wait_for_the_hooks_of_what_it_answers(self) -> None:
        """A slow audit hook on a message no longer adds its time to the
        agent's answer: the answer is delivered while the hook runs."""
        agent = AIChannel("ai", provider=MockAIProvider(responses=["Answer."]))
        kit, witness = await _room_with()
        kit.register_channel(agent)
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        release = asyncio.Event()

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: RoomEvent, ctx: RoomContext) -> None:
            if getattr(event.content, "body", None) == "Hi.":
                await release.wait()

        try:
            await asyncio.wait_for(kit.process_inbound(_said("Hi."), room_id="r1"), 1)
            assert _bodies(witness) == ["Hi.", "Answer."]
        finally:
            release.set()
            await kit.close()

    async def test_event_processed_comes_once_its_after_broadcast_hooks_end(self) -> None:
        """``event_processed`` says the event finished its delivery set and
        its AFTER_BROADCAST hooks (RFC §8.2): it comes after them, the call
        before (RMK-1020)."""
        kit, _ = await _room_with()
        seen: list[str] = []
        release, processed = asyncio.Event(), asyncio.Event()

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="slow")
        async def slow(event: RoomEvent, ctx: RoomContext) -> None:
            await release.wait()
            seen.append("hook ended")

        @kit.on("event_processed")
        async def on_processed(event: Any) -> None:
            seen.append("event_processed")
            processed.set()

        try:
            await asyncio.wait_for(kit.process_inbound(_said("One"), room_id="r1"), 1)
            seen.append("call returned")
            release.set()
            await asyncio.wait_for(processed.wait(), 1)
            assert seen == ["call returned", "hook ended", "event_processed"]
        finally:
            release.set()
            await kit.close()


class TestClose:
    async def test_close_waits_for_the_observers_still_running(self) -> None:
        kit = RoomKit()
        room = await kit.create_room()
        finished, release = asyncio.Event(), asyncio.Event()

        @kit.hook(HookTrigger.BEFORE_TTS, execution=HookExecution.ASYNC)
        async def slow_observer(text: str, ctx: RoomContext) -> None:
            await release.wait()
            finished.set()

        await kit.hook_engine.run_sync_hooks(
            room.id, HookTrigger.BEFORE_TTS, "hello", _ctx(), skip_event_filter=True
        )
        assert not finished.is_set()
        closing = asyncio.ensure_future(kit.close())
        await asyncio.wait({closing}, timeout=0.05)
        assert not closing.done()  # it waits for the observer still running
        release.set()
        await asyncio.wait_for(closing, 5)
        assert finished.is_set()
        assert not kit.hook_engine._observers

    async def test_close_called_from_an_observer_returns(self) -> None:
        """An observer that ends the session (``kit.close()`` on "bye"): the
        close it runs does not wait for the observer it runs under."""
        kit = RoomKit()
        room = await kit.create_room()
        closed = asyncio.Event()

        @kit.hook(HookTrigger.BEFORE_TTS, execution=HookExecution.ASYNC, timeout=5.0)
        async def quit_on_bye(text: str, ctx: RoomContext) -> None:
            await kit.close()
            closed.set()

        await kit.hook_engine.run_sync_hooks(
            room.id, HookTrigger.BEFORE_TTS, "bye", _ctx(), skip_event_filter=True
        )
        await asyncio.wait_for(closed.wait(), 1)


class _SlowRelease(InMemoryLockManager):
    """A room lock whose release, once armed, takes a moment, as a networked
    lock manager's round trip does: an observer started as its section ends
    calls into the room meanwhile. ``released`` turns true once the armed
    section's lock is free."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.released = False
        self.observer_calls = asyncio.Event()

    @contextlib.asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        armed = False
        async with super().locked(room_id):
            yield
            if self.armed:
                self.armed, armed = False, True
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.observer_calls.wait(), 0.05)
        if armed:
            self.released = True


class _NotingStore(InMemoryStore):
    """Notes whether the armed section's lock was free when "reaction" committed."""

    def __init__(self, locks: _SlowRelease) -> None:
        super().__init__()
        self._locks = locks
        self.reaction_after_release: bool | None = None

    async def commit_event(self, room_id: str, event: RoomEvent, **kwargs: Any) -> RoomEvent:
        if isinstance(event.content, TextContent) and event.content.body == "reaction":
            self.reaction_after_release = self._locks.released
        return await super().commit_event(room_id, event, **kwargs)


class TestTheRoomLock:
    async def test_an_observer_s_call_into_the_room_waits_for_the_section(self) -> None:
        """An ASYNC BEFORE_BROADCAST observer starts as its event's pass ends,
        its lock still held while the release takes a round trip: its call
        into the room is a new arrival, which waits for the lock's release
        and commits after the event (RFC §9, §9.5.1, RMK-1039)."""
        locks = _SlowRelease()
        store = _NotingStore(locks)
        kit = RoomKit(store=store, lock_manager=locks)
        for channel_id in ("sms1", "sms2"):
            kit.register_channel(SimpleChannel(channel_id))
        await kit.create_room(room_id="r1")
        for channel_id in ("sms1", "sms2"):
            await kit.attach_channel("r1", channel_id)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="gate")
        async def gate(event: RoomEvent, ctx: RoomContext) -> HookResult:
            locks.armed = event.source.channel_id == "sms1"
            return HookResult.allow()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="react")
        async def react(event: RoomEvent, ctx: RoomContext) -> None:
            if event.source.channel_id == "sms1":
                locks.observer_calls.set()
                await kit.send_event("r1", "sms2", TextContent(body="reaction"))

        try:
            await kit.process_inbound(
                InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi")),
                room_id="r1",
            )
            await kit.hook_engine.drain_observers()
            events = await kit.store.list_events("r1")
            assert [e.content.body for e in events if isinstance(e.content, TextContent)][-2:] == [
                "hi",
                "reaction",
            ]
            assert locks.observer_calls.is_set()
            assert store.reaction_after_release is True
        finally:
            await kit.close()

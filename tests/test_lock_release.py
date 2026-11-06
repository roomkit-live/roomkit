"""A release of the room lock that fails changes nothing (RFC §13.5)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.locks import (
    InMemoryLockManager,
    ReleaseTolerantLockManager,
    RoomLockManager,
    _has_room_lock,
)
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel
from tests.test_streamed_response_failure import _FailsMidAnswer


class _LeaseLost(RoomLockManager):
    """A distributed lock whose lease is found expired at release while
    ``failing`` is set."""

    def __init__(self) -> None:
        self.inner = InMemoryLockManager()
        self.failing = False

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        async with self.inner.locked(room_id):
            yield
        if self.failing:
            raise ConnectionError("lease expired")


class _SlowRelease(RoomLockManager):
    """A lock whose release takes a round trip that never ends on its own."""

    def __init__(self) -> None:
        self.releasing = asyncio.Event()

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            self.releasing.set()
            await asyncio.Event().wait()


async def _room(locks: _LeaseLost) -> tuple[RoomKit, SimpleChannel]:
    kit = RoomKit(lock_manager=locks)
    sms, ws = SimpleChannel("sms"), SimpleChannel("ws")
    await kit.create_room(room_id="r1")
    for channel in (sms, ws):
        kit.register_channel(channel)
        await kit.attach_channel("r1", channel.channel_id)
    return kit, ws


def _message(body: str = "hello") -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=body))


class TestAnInboundMessage:
    async def test_committed_stays_committed_and_delivered(self) -> None:
        locks = _LeaseLost()
        kit, ws = await _room(locks)
        locks.failing = True
        try:
            result = await kit.process_inbound(_message())
            assert result.event is not None and not result.blocked
            assert [e.id for e in ws.delivered] == [result.event.id]
        finally:
            locks.failing = False
            await kit.close()

    async def test_reports_the_failed_release_as_its_error(self) -> None:
        locks = _LeaseLost()
        kit, _ = await _room(locks)
        locks.failing = True
        try:
            result = await kit.process_inbound(_message())
            assert isinstance(result.error, ConnectionError)
        finally:
            locks.failing = False
            await kit.close()

    async def test_reports_the_failed_release_before_a_later_stream_failure(self) -> None:
        locks = _LeaseLost()
        kit, _ = await _room(locks)
        kit.register_channel(AIChannel("ai", provider=_FailsMidAnswer()))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        locks.failing = True
        try:
            result = await kit.process_inbound(_message())
            assert isinstance(result.error, ConnectionError)
        finally:
            locks.failing = False
            await kit.close()

    async def test_refused_stays_refused(self) -> None:
        locks = _LeaseLost()
        kit, ws = await _room(locks)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="no_spam")
        async def no_spam(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("spam")

        locks.failing = True
        try:
            result = await kit.process_inbound(_message())
            assert (result.blocked, ws.delivered) == (True, [])
        finally:
            locks.failing = False
            await kit.close()

    async def test_the_next_message_takes_the_lock_again(self) -> None:
        locks = _LeaseLost()
        kit, ws = await _room(locks)
        locks.failing = True
        try:
            await kit.process_inbound(_message("first"))
            locks.failing = False
            result = await kit.process_inbound(_message("second"))
            assert (result.error, len(ws.delivered)) == (None, 2)
        finally:
            await kit.close()


class TestALifecycleChange:
    async def test_is_done_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        locks = _LeaseLost()
        kit, _ = await _room(locks)
        kit.register_channel(SimpleChannel("email"))
        locks.failing = True
        try:
            with caplog.at_level(logging.WARNING, logger="roomkit.locks"):
                await kit.attach_channel("r1", "email")
            assert await kit.store.get_binding("r1", "email") is not None
            assert any("lease expired" in r.getMessage() for r in caplog.records)
        finally:
            locks.failing = False
            await kit.close()


class TestARegenerate:
    async def test_reports_the_failed_release_as_its_error(self) -> None:
        locks = _LeaseLost()
        kit, _ = await _room(locks)
        kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["a", "b"])))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        try:
            await kit.process_inbound(_message())
            locks.failing = True
            result = await kit.regenerate_response("r1")
            assert result is not None and isinstance(result.error, ConnectionError)
        finally:
            locks.failing = False
            await kit.close()

    async def test_a_refusal_in_the_lane_reports_the_failed_release(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The room closed while the regeneration waited in its lane: the
        refusal carries the failed release of the lock its choice was made
        under (RMK-1443 G3)."""
        gate = asyncio.Event()

        class _Gated(MockAIProvider):
            async def generate(self, context: AIContext) -> AIResponse:
                if self.calls:
                    await gate.wait()
                return await super().generate(context)

        locks = _LeaseLost()
        kit, _ = await _room(locks)
        kit.register_channel(AIChannel("ai", provider=_Gated(responses=["a", "b", "c"])))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        queued = asyncio.Event()
        enqueue_regeneration = kit._enqueue_regeneration

        def enqueued(*args: Any, **kwargs: Any) -> Any:
            plan = enqueue_regeneration(*args, **kwargs)
            queued.set()
            return plan

        monkeypatch.setattr(kit, "_enqueue_regeneration", enqueued)
        try:
            await kit.process_inbound(_message())
            # A second message's turn keeps the lane busy at the gate.
            second = await kit.process_inbound(_message("again"), defer_delivery=True)
            locks.failing = True
            regeneration = asyncio.ensure_future(kit.regenerate_response("r1"))
            await asyncio.wait_for(queued.wait(), 1.0)
            await kit.close_room("r1")
            gate.set()
            result = await asyncio.wait_for(regeneration, 5.0)
            assert second.delivery is not None
            await second.delivery.wait()
            assert result is not None
            assert (result.blocked, result.reason) == (True, "room_closed")
            assert isinstance(result.error, ConnectionError)
        finally:
            locks.failing = False
            gate.set()
            await kit.close()


class TestTheKitsHold:
    async def test_an_error_inside_the_section_stays_the_one_raised(self) -> None:
        locks = _LeaseLost()
        locks.failing = True
        with pytest.raises(ValueError):
            async with ReleaseTolerantLockManager(locks).locked("r1"):
                raise ValueError("the section's own")

    async def test_a_cancellation_during_the_release_after_a_failure_is_delivered(self) -> None:
        """The section raised, and its caller is cancelled while the lock is
        released (a distributed lock's round trip): the cancellation reaches
        the caller rather than be logged as a failed release."""
        locks = _SlowRelease()

        async def section() -> str:
            try:
                async with ReleaseTolerantLockManager(locks).locked("r1"):
                    raise ValueError("the section's own")
            except ValueError:
                return "carried on"

        task = asyncio.create_task(section())
        await locks.releasing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_holds_the_lock_reentrantly(self) -> None:
        held = ReleaseTolerantLockManager(InMemoryLockManager())
        async with held.locked("r1"):
            assert _has_room_lock("r1", held)

    def test_the_kit_s_lock_manager_is_the_one_given(self) -> None:
        locks = _LeaseLost()
        assert RoomKit(lock_manager=locks).lock_manager is locks

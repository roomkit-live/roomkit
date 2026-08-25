"""Deferred delivery: ``process_inbound(..., defer_delivery=True)`` (RFC §10.1 step 18).

The call returns at the commit — the caller has its committed event (an HTTP
route can build its 200 from it) while the delivery set, the reentry passes
and any streamed responses follow in the room's lane. ``InboundResult.delivery``
is the grip on that tail: ``wait()`` resolves once the whole turn has run and
backfills ``delivery_results`` / ``error``, after which the result reads
exactly like a non-deferred call's. A hook refusal is decided under the room
lock, so ``blocked`` stays synchronous either way.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventStatus, EventType, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.hook import HookResult
from roomkit.providers.ai.base import AIContext, AIResponse, StreamEvent
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class _GatedAIProvider(MockAIProvider):
    """Mock provider whose generation blocks until the test opens the gate."""

    def __init__(self, *, streaming: bool = False) -> None:
        super().__init__(responses=["deferred reply"], streaming=streaming)
        self.gate = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        await self.gate.wait()
        return await super().generate(context)


class _StreamRaisingProvider(MockAIProvider):
    """Structured-streaming provider that raises before yielding anything."""

    def __init__(self, exc: Exception) -> None:
        super().__init__(streaming=True)
        self._exc = exc

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        raise self._exc
        yield  # pragma: no cover - keep this an async generator


class _SessionChannel(SimpleChannel):
    """Transport that records session connections."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.connected: list[Any] = []

    async def connect_session(self, session: Any, room_id: str, binding: ChannelBinding) -> None:
        self.connected.append(session)


async def _make_kit(ai: AIChannel, transport: SimpleChannel | None = None) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(transport or SimpleChannel("sms1"))
    # A second transport so the trigger has a delivery set of its own — the
    # source channel is not delivered back to, and delivery_results only
    # reports transports that were.
    kit.register_channel(SimpleChannel("sms2"))
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", transport.channel_id if transport else "sms1")
    await kit.attach_channel("r1", "sms2")
    await kit.attach_channel("r1", ai.channel_id, category=ChannelCategory.INTELLIGENCE)
    return kit


def _message(body: str = "hello") -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body=body))


async def _message_bodies(kit: RoomKit) -> list[str]:
    """DELIVERED message bodies — a refused event is stored BLOCKED (§10.1)."""
    events = await kit.store.list_events("r1")
    return [
        e.content.body
        for e in events
        if e.type == EventType.MESSAGE and e.status == EventStatus.DELIVERED
    ]


async def test_deferred_returns_committed_event_while_generation_in_flight() -> None:
    provider = _GatedAIProvider()
    kit = await _make_kit(AIChannel("ai1", provider=provider))

    result = await kit.process_inbound(_message(), defer_delivery=True)

    # The caller is back at the commit: its event is in the timeline, the
    # agent's turn has not produced anything yet.
    assert result.blocked is False
    assert result.event is not None
    assert result.delivery is not None
    assert result.delivery.done is False
    assert result.delivery_results == {}
    assert await _message_bodies(kit) == ["hello"]

    provider.gate.set()
    final = await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    assert final is result  # wait() reports the same result, backfilled
    assert result.delivery.done is True
    assert result.error is None
    assert "sms2" in result.delivery_results  # replaced-not-mutated dict was re-read
    assert await _message_bodies(kit) == ["hello", "deferred reply"]
    await kit.close()


async def test_blocked_stays_synchronous() -> None:
    kit = await _make_kit(AIChannel("ai1", provider=MockAIProvider()))

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def refuse(event: Any, ctx: RoomContext) -> HookResult:
        return HookResult.block("no spam")

    result = await kit.process_inbound(_message(), defer_delivery=True)

    # The refusal is decided under the room lock, before the deferred return.
    assert result.blocked is True
    assert result.reason == "no spam"
    # The contract does not fork on the outcome: the handle is there and its
    # near-empty cascade resolves at once.
    assert result.delivery is not None
    await asyncio.wait_for(result.delivery.wait(), timeout=2.0)
    assert await _message_bodies(kit) == []
    await kit.close()


async def test_streamed_reply_lands_after_wait() -> None:
    """Cascade completion is not turn completion: the streamed reply is
    generated while the detached consumer drains the stream, and wait()
    covers that too (parity with the reentrant-caller consumption path)."""
    provider = MockAIProvider(responses=["streamed reply"], streaming=True)
    kit = await _make_kit(AIChannel("ai1", provider=provider))

    result = await kit.process_inbound(_message(), defer_delivery=True)
    assert result.delivery is not None
    await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    assert result.error is None
    assert await _message_bodies(kit) == ["hello", "streamed reply"]
    await kit.close()


async def test_stream_error_surfaces_on_result_after_wait() -> None:
    exc = RuntimeError("stream exploded")
    kit = await _make_kit(AIChannel("ai1", provider=_StreamRaisingProvider(exc)))

    result = await kit.process_inbound(_message(), defer_delivery=True)
    assert result.delivery is not None
    await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    assert result.error is exc
    await kit.close()


async def test_waiting_path_has_no_handle() -> None:
    kit = await _make_kit(AIChannel("ai1", provider=MockAIProvider()))

    result = await kit.process_inbound(_message())

    assert result.delivery is None
    assert "sms2" in result.delivery_results  # the waiting path still reports step 18
    await kit.close()


async def test_deferred_connects_session_before_return() -> None:
    provider = _GatedAIProvider()
    transport = _SessionChannel("sms1")
    kit = await _make_kit(AIChannel("ai1", provider=provider), transport=transport)

    session = object()
    message = InboundMessage(
        channel_id="sms1", sender_id="u1", content=TextContent(body="hello"), session=session
    )
    result = await kit.process_inbound(message, defer_delivery=True)

    # Bound before the deferred return, while the agent's turn is still gated:
    # the session-connected invariant does not depend on waiting for delivery.
    assert transport.connected == [session]

    provider.gate.set()
    assert result.delivery is not None
    await asyncio.wait_for(result.delivery.wait(), timeout=5.0)
    await kit.close()


async def test_deferred_blocked_never_connects_session() -> None:
    transport = _SessionChannel("sms1")
    kit = await _make_kit(AIChannel("ai1", provider=MockAIProvider()), transport=transport)

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def refuse(event: Any, ctx: RoomContext) -> HookResult:
        return HookResult.block("no spam")

    message = InboundMessage(
        channel_id="sms1", sender_id="u1", content=TextContent(body="hello"), session=object()
    )
    result = await kit.process_inbound(message, defer_delivery=True)

    assert result.blocked is True
    assert transport.connected == []
    await kit.close()

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

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.lanes import DeliveryCascade, _active_lane_room
from roomkit.core.locks import _held_rooms
from roomkit.models.channel import ChannelBinding, RateLimit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventStatus, EventType, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.hook import HookResult
from roomkit.providers.ai.base import AIContext, AIResponse, StreamEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.context import get_current_span, reset_span, set_current_span
from roomkit.telemetry.mock import MockTelemetryProvider
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


async def _make_kit(
    ai: AIChannel,
    transport: SimpleChannel | None = None,
    *,
    telemetry: MockTelemetryProvider | None = None,
) -> RoomKit:
    kit = RoomKit(telemetry=telemetry)
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


async def test_deferred_refused_before_locked_region_has_no_handle() -> None:
    """A refusal shed before the locked region (here: rate limited) has no
    delivery to follow — the contract is `blocked is False implies delivery`,
    not `deferred implies delivery`."""
    kit = RoomKit(inbound_rate_limit=RateLimit(max_per_second=1))
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")

    first = await kit.process_inbound(_message(), defer_delivery=True)
    assert first.blocked is False
    assert first.delivery is not None
    await asyncio.wait_for(first.delivery.wait(), timeout=5.0)

    second = await kit.process_inbound(_message(), defer_delivery=True)
    assert second.blocked is True
    assert second.reason == "rate_limited"
    assert second.delivery is None
    await kit.close()


async def test_consumer_crash_surfaces_on_result() -> None:
    """An exception ESCAPING stream consumption (not one it returns) must
    reach the deferred caller — the waiting path propagates it, so wait()
    reporting success would be the silent fork."""
    provider = MockAIProvider(responses=["streamed reply"], streaming=True)
    kit = await _make_kit(AIChannel("ai1", provider=provider))

    boom = RuntimeError("consumption infrastructure exploded")

    async def raising(streams: Any, room_id: str) -> Exception | None:
        raise boom

    kit._process_streaming_responses = raising  # type: ignore[method-assign]

    result = await kit.process_inbound(_message(), defer_delivery=True)
    assert result.delivery is not None
    await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    assert result.error is boom
    await kit.close()


async def test_handle_wait_short_circuits_under_room_lock() -> None:
    """wait() from a context the lane cannot progress past (a sync hook under
    the room lock, a tool handler inside the lane) must return unwaited, the
    way cascade.wait() short-circuits — not hang on its own turn."""
    provider = _GatedAIProvider()
    kit = await _make_kit(AIChannel("ai1", provider=provider))

    result = await kit.process_inbound(_message(), defer_delivery=True)
    assert result.delivery is not None
    assert result.delivery.done is False

    async with kit._lock_manager.locked("r1"):
        # Generation still gated: a real wait would hang until timeout.
        same = await asyncio.wait_for(result.delivery.wait(), timeout=2.0)
    assert same is result

    provider.gate.set()
    final = await asyncio.wait_for(result.delivery.wait(), timeout=5.0)
    assert final.error is None
    await kit.close()


# -- Trace shape: the deferred tail stays in the inbound span's tree --


def _span_tree(telemetry: MockTelemetryProvider) -> list[tuple[str, str | None]]:
    """``(name, parent name)`` for every completed span, sorted."""
    by_id = {s.id: s for s in telemetry.spans}
    return sorted(
        (s.name, by_id[s.parent_id].name if s.parent_id in by_id else None)
        for s in telemetry.spans
    )


async def _traced_turn(*, streaming: bool, defer: bool) -> MockTelemetryProvider:
    """One full turn — trigger, AI reply, both hook passes — under a mock tracer."""
    telemetry = MockTelemetryProvider()
    provider = MockAIProvider(responses=["reply"], streaming=streaming)
    kit = await _make_kit(AIChannel("ai1", provider=provider), telemetry=telemetry)

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def before(event: Any, ctx: RoomContext) -> HookResult:
        return HookResult.allow()

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def after(event: Any, ctx: RoomContext) -> None:
        return None

    result = await kit.process_inbound(_message(), defer_delivery=defer)
    if result.delivery is not None:
        await asyncio.wait_for(result.delivery.wait(), timeout=5.0)
    await kit.close()
    return telemetry


@pytest.mark.parametrize("streaming", [False, True], ids=["reentry", "streamed"])
async def test_deferred_span_tree_matches_the_waiting_path(streaming: bool) -> None:
    """Every span of a deferred turn keeps the parent the waiting path gives
    it. The streamed segments and their hooks ride the detached consumer, the
    reentry pass and AFTER_BROADCAST ride the lane executor — both on fresh
    contexts — and none of them may surface as a trace root. The one
    deferred-only span is ``framework.detached``, a child of the inbound span."""
    waiting = _span_tree(await _traced_turn(streaming=streaming, defer=False))
    deferred = _span_tree(await _traced_turn(streaming=streaming, defer=True))

    assert ("framework.detached", "framework.inbound") in deferred
    deferred.remove(("framework.detached", "framework.inbound"))
    assert deferred == waiting
    assert [name for name, parent in deferred if parent is None] == ["framework.inbound"]
    # The whole turn hangs under the inbound span: the trigger's delivery
    # set, the reply's (reentry or streamed segment), and both hook passes.
    assert deferred.count(("framework.broadcast", "framework.inbound")) == 2
    assert ("hook.async.after", "framework.inbound") in deferred


async def test_detached_span_measures_the_deferred_tail() -> None:
    """``framework.inbound`` ends at the return — what the caller waited for —
    and a ``framework.detached`` child, opened at the deferral, covers the
    rest of the turn: still open when the call returns, ended with the
    consumer, so the turn's duration is readable from the trace."""
    telemetry = MockTelemetryProvider()
    provider = MockAIProvider(responses=["streamed reply"], streaming=True)
    kit = await _make_kit(AIChannel("ai1", provider=provider), telemetry=telemetry)

    result = await kit.process_inbound(_message(), defer_delivery=True)

    (inbound,) = [s for s in telemetry.spans if s.name == "framework.inbound"]
    assert inbound.attributes["deferred"] is True
    assert "framework.detached" in {s.name for s in telemetry.get_active_spans()}

    assert result.delivery is not None
    await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    (tail,) = [s for s in telemetry.spans if s.name == "framework.detached"]
    assert tail.parent_id == inbound.id
    assert tail.status == "ok"
    assert tail.attributes["streams"] == 1
    assert inbound.end_time is not None and tail.end_time is not None
    assert tail.end_time >= inbound.end_time
    await kit.close()


async def test_detached_span_reports_a_failed_tail() -> None:
    exc = RuntimeError("stream exploded")
    telemetry = MockTelemetryProvider()
    kit = await _make_kit(
        AIChannel("ai1", provider=_StreamRaisingProvider(exc)), telemetry=telemetry
    )

    result = await kit.process_inbound(_message(), defer_delivery=True)
    assert result.delivery is not None
    await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    (tail,) = [s for s in telemetry.spans if s.name == "framework.detached"]
    assert tail.status == "error"
    assert tail.error_message == str(exc)
    await kit.close()


async def test_close_in_flight_ends_the_detached_span() -> None:
    """close() cancels the consumer while the tail is still running: the
    detached span must not be left open — it ends as cancelled."""
    telemetry = MockTelemetryProvider()
    provider = _GatedAIProvider()
    kit = await _make_kit(AIChannel("ai1", provider=provider), telemetry=telemetry)

    result = await kit.process_inbound(_message(), defer_delivery=True)
    assert result.delivery is not None
    assert result.delivery.done is False

    # Open the gate only once close() has cancelled the consumer: the lane
    # then drains cleanly while the tail itself was abandoned mid-flight.
    asyncio.get_running_loop().call_later(0.05, provider.gate.set)
    await kit.close()

    assert result.delivery.done is True
    assert telemetry.get_active_spans() == []
    (tail,) = [s for s in telemetry.spans if s.name == "framework.detached"]
    assert tail.status == "error"
    assert tail.error_message == "cancelled"


async def test_detached_consumer_runs_lock_free_under_the_callers_span() -> None:
    """The consumer task starts on a fresh context — an inherited
    ``_held_rooms`` would fake lock reentrancy — and gets the caller's span
    back explicitly, the way the lane executor gets the planner's."""
    telemetry = MockTelemetryProvider()
    kit = RoomKit(telemetry=telemetry)
    seen: dict[str, Any] = {}

    async def observe(streams: Any, room_id: str) -> Exception | None:
        seen["held"] = _held_rooms.get()
        seen["lane"] = _active_lane_room.get()
        seen["span"] = get_current_span()
        return None

    kit._process_streaming_responses = observe  # type: ignore[method-assign]
    cascade = DeliveryCascade("r1", reentry_budget=1)
    cascade.add_streams([object()])

    caller_span = telemetry.start_span(SpanKind.CUSTOM, "caller")
    token = set_current_span(caller_span)
    try:
        async with kit._lock_manager.locked("r1"):
            task = kit._consume_streams_when_cascade_completes(cascade, "r1")
    finally:
        reset_span(token)
    await asyncio.wait_for(task, timeout=2.0)
    telemetry.end_span(caller_span)

    assert seen == {"held": frozenset(), "lane": None, "span": caller_span}
    (tail,) = [s for s in telemetry.spans if s.name == "framework.detached"]
    assert tail.parent_id == caller_span
    await kit.close()

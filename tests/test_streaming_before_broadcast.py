"""Streamed AI segments must pass through BEFORE_BROADCAST sync hooks.

The streaming path used to persist and re-broadcast segments without ever
running BEFORE_BROADCAST (only AFTER_BROADCAST fired). A pre-broadcast guard
on AI output — e.g. PII de-anonymisation — was therefore bypassed in
streaming: the stored row kept the raw text. These tests pin that a hook's
modification lands on the persisted segment and that a block drops it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.event_router import StreamingResponse
from roomkit.core.framework import RoomKit
from roomkit.core.lanes import DeliveryCascade
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.store_filter import EventFilter, PersistencePolicy
from roomkit.models.task import Observation, Task
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


async def _wire(kit: RoomKit, content: str) -> None:
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[AIResponse(content=content, finish_reason="stop")],
    )
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=provider))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)


def _ai_messages(events: list[RoomEvent]) -> list[RoomEvent]:
    return [e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == "ai1"]


_TOOL_EVENTS = (EventType.TOOL_CALL_START, EventType.TOOL_CALL_END)


def _tool_events(events: list[RoomEvent]) -> list[RoomEvent]:
    return [e for e in events if e.type in _TOOL_EVENTS]


async def _wire_tool_turn(kit: RoomKit) -> SimpleChannel:
    """A streamed turn that calls one tool, then answers; returns the transport."""
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="Looking.",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="lookup", arguments={"q": "x"})],
            ),
            AIResponse(content="Found.", finish_reason="stop"),
        ],
    )

    async def handler(name: str, args: dict[str, Any]) -> str:
        return "ok"

    transport = SimpleChannel("sms1")
    kit.register_channel(transport)
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            tools=[AITool(name="lookup", description="Look up")],
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return transport


async def test_streamed_segment_applies_before_broadcast_modification() -> None:
    kit = RoomKit()
    await _wire(kit, content="Hi [PERSON_1], welcome")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="deanon")
    async def deanon(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if not isinstance(event.content, TextContent) or "[PERSON_1]" not in event.content.body:
            return HookResult.allow()
        restored = event.content.body.replace("[PERSON_1]", "Alice")
        return HookResult.modify(event.model_copy(update={"content": TextContent(body=restored)}))

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    ai_msgs = _ai_messages(await kit.store.list_events("r1"))
    assert ai_msgs, "streamed AI segment was not persisted"
    assert ai_msgs[-1].content.body == "Hi Alice, welcome"
    assert "[PERSON_1]" not in ai_msgs[-1].content.body


async def test_streamed_segment_dropped_when_before_broadcast_blocks() -> None:
    kit = RoomKit()
    await _wire(kit, content="[SECRET] leaked")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="blocker")
    async def blocker(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id == "ai1":
            return HookResult.block("withheld")
        return HookResult.allow()

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    # The blocked segment never enters the timeline as a delivered message.
    assert _ai_messages(await kit.store.list_events("r1")) == []


async def test_streaming_transport_persists_modified_but_streams_raw() -> None:
    """With a real streaming transport attached: the persisted segment is
    de-anonymised by BEFORE_BROADCAST (C1 works on the streaming path), while
    the live chunks already carried the raw text — they precede the hook by
    construction. The raw-chunk exposure is what a host-side gate prevents by
    withholding the stream fn when PII is active."""
    kit = RoomKit()
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[AIResponse(content="Hi [PERSON_1]", finish_reason="stop")],
    )
    ws = WebSocketChannel("ws1")
    chunks: list[str] = []

    async def send_fn(conn_id: str, event: RoomEvent) -> None:
        return None

    async def stream_send_fn(conn_id: str, msg: object) -> None:
        delta = getattr(msg, "delta", None)
        if delta:
            chunks.append(delta)

    ws.register_connection("c1", send_fn, stream_send_fn=stream_send_fn, room_id="r1")
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(ws)
    kit.register_channel(AIChannel("ai1", provider=provider))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ws1")  # streaming-capable transport
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="deanon")
    async def deanon(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if isinstance(event.content, TextContent) and "[PERSON_1]" in event.content.body:
            restored = event.content.body.replace("[PERSON_1]", "Alice")
            return HookResult.modify(
                event.model_copy(update={"content": TextContent(body=restored)})
            )
        return HookResult.allow()

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    ai_msgs = _ai_messages(await kit.store.list_events("r1"))
    assert ai_msgs and ai_msgs[-1].content.body == "Hi Alice"  # persisted = de-anon
    assert "".join(chunks) == "Hi [PERSON_1]"  # live chunks = raw (by design)


def _body(event: RoomEvent) -> str | None:
    return event.content.body if isinstance(event.content, TextContent) else None


@pytest.mark.parametrize("action", ["allow", "modify", "block"])
@pytest.mark.parametrize("live_transport", [False, True])
async def test_streamed_hook_side_effects_survive_decision(
    action: Literal["allow", "modify", "block"], live_transport: bool
) -> None:
    """Side effects run once, even for a blocked segment, on either delivery path."""
    kit = RoomKit()
    await _wire(kit, content="original")
    audit = SimpleChannel("audit")
    kit.register_channel(audit)
    await kit.attach_channel("r1", "audit")
    chunks: list[str] = []
    if live_transport:
        ws = WebSocketChannel("ws1")

        async def send_fn(conn_id: str, event: RoomEvent) -> None:
            pass

        async def stream_send_fn(conn_id: str, msg: object) -> None:
            delta = getattr(msg, "delta", None)
            if delta:
                chunks.append(delta)

        ws.register_connection("c1", send_fn, stream_send_fn=stream_send_fn, room_id="r1")
        kit.register_channel(ws)
        await kit.attach_channel("r1", "ws1")

    hook_calls: list[str] = []
    created_tasks: list[str] = []
    after_tasks: list[list[str]] = []
    task = Task(id="follow-up", room_id="r1", title="Follow up")
    observation = Observation(id="seen", room_id="r1", channel_id="ai1", content="Observed")
    injected = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body="audit notice"),
    )

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def side_effects(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id != "ai1":
            return HookResult.allow()
        hook_calls.append(event.id)
        return HookResult(
            action=action,
            reason="withheld" if action == "block" else None,
            event=event.model_copy(update={"content": TextContent(body="modified")})
            if action == "modify"
            else None,
            tasks=[task],
            observations=[observation],
            injected_events=[InjectedEvent(event=injected, target_channel_ids=["audit"])],
        )

    @kit.hook(HookTrigger.ON_TASK_CREATED)
    async def task_created(event: RoomEvent, ctx: RoomContext) -> None:
        created_tasks.append(event.metadata["task_id"])

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def after_broadcast(event: RoomEvent, ctx: RoomContext) -> None:
        if event.source.channel_id == "ai1":
            after_tasks.append([t.id for t in await kit.store.list_tasks("r1")])

    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        assert len(hook_calls) == 1
        assert await kit.store.list_tasks("r1") == [task]
        assert await kit.store.list_observations("r1") == [observation]
        assert created_tasks == [task.id]
        events = await kit.store.list_events("r1")
        # The pipeline gives the injected event its id (RFC §8.1).
        injections = [e for e in events if _body(e) == "audit notice"]
        assert len(injections) == 1
        assert [_body(e) for e in audit.delivered].count("audit notice") == 1
        replies = _ai_messages(events)
        if action == "block":
            assert replies == []
            assert after_tasks == []
        else:
            assert len(replies) == 1
            assert replies[0].content.body == ("modified" if action == "modify" else "original")
            assert replies[0].index < injections[0].index
            assert after_tasks == [[task.id]]
        if live_transport:
            assert "".join(chunks) == "original"
    finally:
        await kit.close()


@pytest.mark.parametrize("detached", [False, True])
async def test_stream_hook_effects_without_persistence_or_source_binding(detached: bool) -> None:
    """Effects survive when the segment has no stored row or no delivery plan."""
    kit = RoomKit(
        persistence_policy=None
        if detached
        else PersistencePolicy(exclude_types={EventType.MESSAGE})
    )
    audit = SimpleChannel("audit")
    kit.register_channel(audit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "audit")
    if not detached:
        kit.register_channel(SimpleChannel("ai1", channel_type=ChannelType.AI))
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    task = Task(id="follow-up", room_id="r1", title="Follow up")
    observation = Observation(id="seen", room_id="r1", channel_id="ai1", content="Observed")
    notice = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body="notice"),
    )

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def side_effects(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id != "ai1":
            # The notice's own pass runs this hook too (RFC §9.4).
            return HookResult.allow()
        return HookResult(
            action="allow",
            tasks=[task],
            observations=[observation],
            injected_events=[InjectedEvent(event=notice, target_channel_ids=["audit"])],
        )

    async def stream() -> AsyncIterator[str]:
        yield "answer"

    try:
        response = StreamingResponse(
            stream=stream(),
            source_channel_id="ai1",
            source_channel_type=ChannelType.AI,
            trigger_event=notice,
        )
        await kit._handle_streaming_response(
            kit._get_router(),
            response,
            "r1",
            await kit._build_context("r1"),
            cascade=DeliveryCascade("r1", reentry_budget=1),
        )
        assert await kit.store.list_tasks("r1") == [task]
        assert await kit.store.list_observations("r1") == [observation]
        events = await kit.store.list_events("r1")
        # Not stored by the policy, or stored refused: a source with no
        # binding writes nothing into the room (RFC §7.5 rules 2 and 7).
        assert _ai_messages(events) == []
        if detached:
            refused = await kit.store.list_events(
                "r1", event_filter=EventFilter(include_blocked=True)
            )
            assert [e.blocked_by for e in _ai_messages(refused)] == ["source_unbound"]
        assert [_body(event) for event in events].count("notice") == 1
        assert [_body(event) for event in audit.delivered].count("notice") == 1
    finally:
        await kit.close()


async def test_stream_hook_tasks_wait_for_delivery_and_finish_before_return() -> None:
    """A slow transport keeps the turn pending through its post-delivery effects."""
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowTransport(SimpleChannel):
        async def deliver(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            if event.source.channel_id == "ai1":
                entered.set()
                await release.wait()
            return await super().deliver(event, binding, context)

    kit = RoomKit()
    await _wire(kit, content="answer")
    kit.register_channel(SlowTransport("slow"))
    await kit.attach_channel("r1", "slow")
    follow_up = Task(id="follow-up", room_id="r1", title="Follow up")

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def side_effects(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult(
            action="allow", tasks=[follow_up] if event.source.channel_id == "ai1" else []
        )

    processing = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not processing.done()
        assert await kit.store.list_tasks("r1") == []
        release.set()
        await asyncio.wait_for(processing, 2)
        assert await kit.store.list_tasks("r1") == [follow_up]
    finally:
        release.set()
        await asyncio.gather(processing, return_exceptions=True)
        await kit.close()


async def test_streamed_tool_call_events_carry_before_broadcast_modification() -> None:
    """The tool call's start and end cross the same gate as the text around them.

    A hook that labels a tool call for the reader must reach the stored row
    and the non-streaming channels, not the streamed text alone.
    """
    kit = RoomKit()
    transport = await _wire_tool_turn(kit)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="label")
    async def label(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.type not in _TOOL_EVENTS:
            return HookResult.allow()
        stamped = {**event.metadata, "label": f"Lookup · {event.type.value}"}
        return HookResult.modify(event.model_copy(update={"metadata": stamped}))

    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        expected = ["Lookup · tool_call_start", "Lookup · tool_call_end"]
        stored = _tool_events(await kit.store.list_events("r1"))
        assert [e.type for e in stored] == list(_TOOL_EVENTS)
        assert [e.metadata["label"] for e in stored] == expected
        # The delivery carries the modified event too, not the raw one.
        assert [e.metadata.get("label") for e in _tool_events(transport.delivered)] == expected
    finally:
        await kit.close()


async def test_streamed_tool_call_events_dropped_when_before_broadcast_blocks() -> None:
    """A blocked tool-call event is refused and recorded, and its effects stand.

    Refused means: out of the timeline the room received and undelivered, but
    committed with status BLOCKED as the audit record and announced as
    ``event_blocked``, like every other refusal (RFC §10.1 step 10).
    """
    kit = RoomKit()
    transport = await _wire_tool_turn(kit)
    announced: list[str] = []

    @kit.on("event_blocked")
    async def on_blocked(_evt: object) -> None:
        announced.append("blocked")

    task = Task(id="follow-up", room_id="r1", title="Follow up")
    observation = Observation(id="seen", room_id="r1", channel_id="ai1", content="Observed")
    notice = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body="tool call withheld"),
    )

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="blocker")
    async def blocker(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.type not in _TOOL_EVENTS:
            return HookResult.allow()
        if event.type == EventType.TOOL_CALL_END:
            return HookResult.block("withheld")
        # The effects ride the call's start alone, so each lands once.
        return HookResult(
            action="block",
            reason="withheld",
            tasks=[task],
            observations=[observation],
            injected_events=[InjectedEvent(event=notice, target_channel_ids=["sms1"])],
        )

    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        events = await kit.store.list_events("r1")
        assert _tool_events(events) == []
        assert _tool_events(transport.delivered) == []
        # The turn's text is untouched by a block aimed at its tool calls.
        assert [e.content.body for e in _ai_messages(events)] == ["Looking.", "Found."]
        assert await kit.store.list_tasks("r1") == [task]
        assert await kit.store.list_observations("r1") == [observation]
        assert [_body(e) for e in events].count("tool call withheld") == 1
        assert [_body(e) for e in transport.delivered].count("tool call withheld") == 1
        # The audit record: refused, kept, and announced once per event.
        audited = _tool_events(
            await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
        )
        assert [e.type for e in audited] == list(_TOOL_EVENTS)
        assert {e.status for e in audited} == {EventStatus.BLOCKED}
        assert {e.blocked_by for e in audited} == {"blocker"}
        assert len(announced) == 2
    finally:
        await kit.close()


async def test_streamed_tool_call_events_reach_the_streaming_transport_as_events() -> None:
    """With a live stream attached, the transport gets each event once, gated.

    The stream hands the channel consuming it every persisted event inline,
    so the lane leaves that channel out — of the tool-call events as of the
    text segments. Laning them there too delivered the same event id twice.
    """
    kit = RoomKit()
    await _wire_tool_turn(kit)
    ws = WebSocketChannel("ws1")
    chunks: list[str] = []
    sent: list[RoomEvent] = []

    async def send_fn(conn_id: str, event: RoomEvent) -> None:
        sent.append(event)

    async def stream_send_fn(conn_id: str, msg: object) -> None:
        delta = getattr(msg, "delta", None)
        if delta:
            chunks.append(delta)

    ws.register_connection("c1", send_fn, stream_send_fn=stream_send_fn, room_id="r1")
    kit.register_channel(ws)
    await kit.attach_channel("r1", "ws1")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="label")
    async def label(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.type not in _TOOL_EVENTS:
            return HookResult.allow()
        return HookResult.modify(
            event.model_copy(update={"metadata": {**event.metadata, "label": "Lookup"}})
        )

    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        assert "".join(chunks) == "Looking.Found."
        copies = _tool_events(sent)
        assert [e.type for e in copies] == list(_TOOL_EVENTS)
        assert len({e.id for e in copies}) == len(copies), "the same event was delivered twice"
        assert all(e.metadata.get("label") == "Lookup" for e in copies)
    finally:
        await kit.close()


@pytest.mark.parametrize("execution", [HookExecution.SYNC, HookExecution.ASYNC])
async def test_a_streamed_row_s_hook_reads_the_room_at_the_row(execution: HookExecution) -> None:
    """A row's hook, an observer alike, reads the room as it stands when the
    row comes, at the row's index: the record the room took while the tool
    ran is in what the answer's hook reads, not the stream's first context
    (RMK-1297)."""
    kit = RoomKit()
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="Looking.",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="lookup", arguments={"q": "x"})],
            ),
            AIResponse(content="Found.", finish_reason="stop"),
        ],
    )

    async def handler(name: str, args: dict[str, Any]) -> str:
        news = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="sms1", channel_type=ChannelType.SMS),
            content=TextContent(body="Any news?"),
        )
        await kit.commit_event("r1", news)
        return "ok"

    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=provider,
            tool_handler=handler,
            tools=[AITool(name="lookup", description="Look up")],
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    read: dict[str, tuple[int, list[str]]] = {}
    answered = asyncio.Event()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=execution, name="reads")
    async def reads(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if isinstance(event.content, TextContent) and event.source.channel_id == "ai1":
            read[event.content.body] = (event.index, [e.content.body for e in ctx.recent_events])
            if event.content.body == "Found.":
                answered.set()
        return HookResult.allow()

    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        await asyncio.wait_for(answered.wait(), 1.0)
        stored = {
            e.content.body: e.index
            for e in await kit.store.list_events("r1")
            if isinstance(e.content, TextContent)
        }
        assert read["Found."] == (stored["Found."], ["go", "Looking.", "Any news?"])
        assert read["Looking."] == (stored["Looking."], ["go"])
    finally:
        await kit.close()

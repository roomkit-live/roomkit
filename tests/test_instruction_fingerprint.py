"""The instruction fingerprint is the pipeline's (RFC §5.2, §9.4, §10.1.1 step 6).

The pipeline writes ``metadata.instruction`` on every response to an
instruction, over the text the instruction's content reads as (§6.6), and
removes the key from every other event it commits, whoever wrote it: a
channel cannot record the instruction's text in its place, and no event
claims an instruction that did not produce it.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from roomkit import AIChannel, HookResult, HookTrigger, RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType, RoomStatus
from roomkit.models.event import (
    CompositeContent,
    EventContent,
    EventSource,
    LocationContent,
    MediaContent,
    RichContent,
    RoomEvent,
    SystemContent,
    TemplateContent,
    TextContent,
    ToolCallContent,
)
from roomkit.models.hook import InjectedEvent
from roomkit.models.room import RoomTimers
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import AILikeChannel, SimpleChannel
from tests.test_responds_to import _ClockTool

ROOM = "r-fingerprint"
TEXT = "Handoff complete. Introduce yourself."


def _fingerprint(text: str) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "length": len(text)}


class _Parser(SimpleChannel):
    """A transport whose parser keeps the caller's metadata and adds *writes*."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.writes: dict[str, Any] = {}

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        event = await super().handle_inbound(message, context)
        return event.model_copy(update={"metadata": {**message.metadata, **self.writes}})


class _Writer(AILikeChannel):
    """An agent that writes *written* in its reply's metadata."""

    def __init__(self, channel_id: str, written: dict[str, Any]) -> None:
        super().__init__(channel_id, response="Hello.")
        self.written = written

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        output = await super().on_event(event, binding, context)
        output.response_events = [
            reply.model_copy(update={"metadata": dict(self.written)})
            for reply in output.response_events
        ]
        return output


async def _room(agent: Any, *, timers: RoomTimers | None = None) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(_Parser("sms"))
    kit.register_channel(SimpleChannel("speaker"))
    kit.register_channel(agent)
    await kit.create_room(room_id=ROOM, timers=timers)
    await kit.attach_channel(ROOM, "sms")
    await kit.attach_channel(ROOM, "speaker")
    await kit.attach_channel(ROOM, agent.channel_id, category=ChannelCategory.INTELLIGENCE)
    return kit


def _instruction(content: EventContent | None = None, **fields: Any) -> InboundMessage:
    return InboundMessage(
        channel_id="sms",
        sender_id="system",
        event_type=EventType.INSTRUCTION,
        content=content or TextContent(body=TEXT),
        addressed_to=["agent"],
        **fields,
    )


def _message(**fields: Any) -> InboundMessage:
    return InboundMessage(
        channel_id="sms", sender_id="caller", content=TextContent(body="hello"), **fields
    )


async def _stored(kit: RoomKit) -> list[RoomEvent]:
    events = await kit.store.list_events(ROOM, event_filter=EventFilter(include_blocked=True))
    return [e for e in events if e.type != EventType.CHANNEL_ATTACHED]


def _replies(events: list[RoomEvent]) -> list[RoomEvent]:
    """The agent's replies: its messages, not the copy of the instruction it
    keeps for its later turns."""
    return [e for e in events if e.source.channel_id == "agent" and e.type == EventType.MESSAGE]


# -- Written by the pipeline --


@pytest.mark.parametrize(
    "written",
    [{}, {"instruction": TEXT}, {"instruction": _fingerprint("another instruction")}],
    ids=["none", "its-text", "another-fingerprint"],
)
async def test_the_pipeline_records_the_instruction_whatever_the_channel_wrote(written):
    agent = _Writer("agent", written)
    kit = await _room(agent)

    await kit.process_inbound(_instruction(metadata={"instruction": "the caller's"}), room_id=ROOM)

    [reply] = _replies(await _stored(kit))
    assert reply.metadata["instruction"] == _fingerprint(TEXT)
    # The caller's key under that name is not the instruction's either.
    assert "instruction" not in agent.delivered[0].metadata
    await kit.close()


async def test_an_answer_to_a_message_records_no_instruction():
    kit = await _room(_Writer("agent", {"instruction": _fingerprint(TEXT)}))

    await kit.process_inbound(_message(), room_id=ROOM)

    [reply] = _replies(await _stored(kit))
    assert "instruction" not in reply.metadata
    await kit.close()


async def test_every_row_of_a_streamed_answer_records_the_instruction():
    call = AIToolCall(id="c1", name="clock", arguments={})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="Let me check.", tool_calls=[call]),
            AIResponse(content="It is noon."),
        ]
    )
    kit = await _room(AIChannel("agent", provider=provider, tools=[_ClockTool()]))

    await kit.process_inbound(_instruction(), room_id=ROOM)

    rows = [e for e in await _stored(kit) if e.source.channel_id == "agent"]
    kinds = {row.type for row in rows}
    rows_of_a_turn = {EventType.MESSAGE, EventType.TOOL_CALL_START, EventType.TOOL_CALL_END}
    assert rows_of_a_turn | {EventType.INSTRUCTION_READ} == kinds
    assert all(row.metadata.get("instruction") == _fingerprint(TEXT) for row in rows)
    await kit.close()


@pytest.mark.parametrize(
    ("content", "text"),
    [
        (RichContent(body="<b>Introduce yourself.</b>", plain_text="Introduce yourself."), None),
        (
            MediaContent(url="https://example.com/a.png", mime_type="image/png", caption="Look"),
            "[Media: Look]",
        ),
        (
            CompositeContent(parts=[TextContent(body="First."), TextContent(body="Then.")]),
            "First.\nThen.",
        ),
        # Read alike by every implementation of the RFC, so an instruction's
        # fingerprint is the same whichever computed it.
        (ToolCallContent(tool_name="lookup", tool_id="call_1"), "[Tool call: lookup]"),
        (SystemContent(body="Call transferred."), "Call transferred."),
        (
            CompositeContent(
                parts=[TextContent(body="First."), TextContent(body=""), TextContent(body="Then.")]
            ),
            "First.\nThen.",
        ),
        (
            CompositeContent(
                parts=[
                    SystemContent(body="Call transferred."),
                    ToolCallContent(tool_name="lookup", tool_id="call_1"),
                ]
            ),
            "Call transferred.\n[Tool call: lookup]",
        ),
        (
            CompositeContent(
                parts=[
                    LocationContent(latitude=45.5, longitude=-73.6, label="Office"),
                    TemplateContent(template_id="welcome"),
                ]
            ),
            "[Location: Office (45.5, -73.6)]\n[Template: welcome]",
        ),
    ],
    ids=[
        "rich",
        "media",
        "composite",
        "tool-call",
        "system",
        "composite-with-an-empty-part",
        "composite-of-a-notice-and-a-tool-call",
        "composite-of-a-location-and-a-template",
    ],
)
async def test_the_fingerprint_is_of_the_text_the_instruction_reads_as(content, text):
    """RFC §6.6: a rich text by its plain text, a media by its label, a tool
    call by its name, a system notice by its body, a composite by each
    part's own text, an empty one skipped; the model reads that same text."""
    text = text or "Introduce yourself."
    provider = MockAIProvider(["Hello."])
    kit = await _room(AIChannel("agent", provider=provider))

    await kit.process_inbound(_instruction(content), room_id=ROOM)

    [reply] = _replies(await _stored(kit))
    assert reply.metadata["instruction"] == _fingerprint(text)
    assert text in str(provider.calls[-1].messages[-1].content)
    await kit.close()


async def test_the_fingerprint_is_of_the_instruction_its_hooks_decided():
    """The result carries the instruction as its hooks left it; a modify
    keeps its visibility, its address and its standalone word (RFC §9.4)."""
    provider = MockAIProvider(["Hello."])
    kit = await _room(AIChannel("agent", provider=provider))

    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
    async def rewrite(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.modify(
            event.model_copy(
                update={
                    "content": TextContent(body="Greet the caller."),
                    "visibility": "all",
                    "addressed_to": ["speaker"],
                    "metadata": {**event.metadata, "standalone": True},
                }
            )
        )

    result = await kit.process_inbound(_instruction(), room_id=ROOM)

    decided = result.event
    assert decided is not None and decided.type == EventType.INSTRUCTION
    assert decided.content == TextContent(body="Greet the caller.")
    assert (decided.visibility, decided.addressed_to) == ("intelligence", ["agent"])
    assert "standalone" not in decided.metadata
    [reply] = _replies(await _stored(kit))
    assert reply.metadata["instruction"] == _fingerprint("Greet the caller.")
    await kit.close()


@pytest.mark.parametrize(
    "rewrite",
    [
        lambda metadata: {**metadata, "instruction": TEXT},
        lambda metadata: {k: v for k, v in metadata.items() if k != "instruction"},
    ],
    ids=["replaced-by-its-text", "removed"],
)
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streamed"])
async def test_a_hook_cannot_change_or_remove_a_response_s_fingerprint(rewrite, streaming):
    """RFC §9.4: a modify of a response to an instruction keeps the
    fingerprint the pipeline wrote; the hook's other changes stand."""
    agent = (
        AIChannel("agent", provider=MockAIProvider(["Hello."], streaming=True))
        if streaming
        else _Writer("agent", {})
    )
    kit = await _room(agent)

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def tamper(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id != "agent":
            return HookResult.allow()
        metadata = {**rewrite(event.metadata), "reviewed": True}
        return HookResult.modify(event.model_copy(update={"metadata": metadata}))

    await kit.process_inbound(_instruction(), room_id=ROOM)

    replies = _replies(await _stored(kit))
    assert replies
    assert all(r.metadata.get("instruction") == _fingerprint(TEXT) for r in replies)
    assert all(r.metadata.get("reviewed") is True for r in replies)
    await kit.close()


# -- Reserved: removed from every other event --


async def _caller(kit: RoomKit) -> None:
    await kit.process_inbound(_message(metadata={"instruction": "caller"}), room_id=ROOM)


async def _parser(kit: RoomKit) -> None:
    kit.get_channel("sms").writes = {"instruction": "parsed"}  # type: ignore[union-attr]
    await kit.process_inbound(_message(), room_id=ROOM)


async def _send_event(kit: RoomKit) -> None:
    await kit.send_event(
        ROOM, "sms", TextContent(body="hello"), metadata={"instruction": _fingerprint(TEXT)}
    )


async def _hook_modify(kit: RoomKit) -> None:
    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def claim(event: RoomEvent, ctx: RoomContext) -> HookResult:
        metadata = {**event.metadata, "instruction": _fingerprint(TEXT)}
        return HookResult.modify(event.model_copy(update={"metadata": metadata}))

    await kit.process_inbound(_message(), room_id=ROOM)


async def _hook_inject(kit: RoomKit) -> None:
    notice = RoomEvent(
        room_id=ROOM,
        source=EventSource(channel_id="system", channel_type=ChannelType.SMS),
        content=TextContent(body="notice"),
        metadata={"instruction": _fingerprint(TEXT)},
    )

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def notify(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id != "sms":
            return HookResult.allow()
        return HookResult.allow(injected=[InjectedEvent(event=notice)])

    await kit.process_inbound(_message(), room_id=ROOM)


@pytest.mark.parametrize(
    "write", [_caller, _parser, _send_event, _hook_modify, _hook_inject], ids=lambda f: f.__name__
)
async def test_no_other_event_keeps_the_reserved_key(write):
    kit = await _room(AILikeChannel("agent"))

    await write(kit)

    stored = await _stored(kit)
    assert len(stored) >= 2
    assert [e.id for e in stored if "instruction" in e.metadata] == []
    await kit.close()


# -- No activity --


async def test_an_instruction_is_no_activity():
    """RFC §10.1.1 step 5: neither ``last_activity_at`` nor a PAUSED room's
    status moves for an instruction its agent does not answer."""

    class _Silent(AILikeChannel):
        async def on_event(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            self.delivered.append(event)
            return ChannelOutput.empty()

    agent = _Silent("agent")
    kit = await _room(agent, timers=RoomTimers(inactive_after_seconds=3600))
    room = await kit.get_room(ROOM)
    await kit.store.update_room(room.model_copy(update={"status": RoomStatus.PAUSED}))

    result = await kit.process_inbound(_instruction(), room_id=ROOM)

    after = await kit.get_room(ROOM)
    assert not result.blocked and [e.type for e in agent.delivered] == [EventType.INSTRUCTION]
    assert after.status == RoomStatus.PAUSED
    assert after.timers.last_activity_at == room.timers.last_activity_at
    assert after.event_count == room.event_count
    await kit.close()

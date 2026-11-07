"""An INSTRUCTION event directs an agent and is never a participant's words (RFC §10.1.1).

The property under test: whatever path an instruction takes through the inbound
pipeline — delivered, refused, blocked by a hook — the room's timeline never
holds it, no transport receives it, and the model reads it as the application's
direction for one turn, not as something someone in the room said.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import pytest
from pydantic import ValidationError

from roomkit import AIChannel, HookResult, HookTrigger, RoomKit
from roomkit.channels.base import Channel
from roomkit.memory import BudgetAwareMemory, MemoryProvider, MemoryResult, SlidingWindowMemory
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventStatus, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

ROOM = "r-instruction"
INSTRUCTION = "Handoff complete. Introduce yourself to the caller."
FINGERPRINT = {"sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(), "length": 51}


class Speaker(Channel):
    """A transport that records what it was asked to deliver."""

    channel_type = ChannelType.VOICE

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[str] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(getattr(event.content, "body", ""))
        return ChannelOutput.empty()


async def _kit(
    *,
    streaming: bool = False,
    memory: MemoryProvider | None = None,
    provider: MockAIProvider | None = None,
    **agent: Any,
) -> tuple[RoomKit, Speaker, MockAIProvider]:
    """A room where the instruction enters on ``voice`` and ``speaker`` is another transport.

    The source channel never receives its own event, so what a transport is
    handed is read on a second one.
    """
    kit = RoomKit()
    speaker = Speaker("speaker")
    provider = provider or MockAIProvider(["Bonjour, je suis Paul."], streaming=streaming)
    kit.register_channel(Speaker("voice"))
    kit.register_channel(speaker)
    kit.register_channel(AIChannel("agent", provider=provider, memory=memory, **agent))
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "voice", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, "speaker", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, speaker, provider


def _instruction(**overrides: object) -> InboundMessage:
    fields: dict[str, object] = {
        "channel_id": "voice",
        "sender_id": "system",
        "event_type": EventType.INSTRUCTION,
        "content": TextContent(body=INSTRUCTION),
        "addressed_to": ["agent"],
    }
    fields.update(overrides)
    return InboundMessage(**fields)  # type: ignore[arg-type]


async def _messages(kit: RoomKit) -> list[RoomEvent]:
    """Every stored event but the bindings', BLOCKED audit records included."""
    events = await kit.store.list_events(ROOM, event_filter=EventFilter(include_blocked=True))
    return [e for e in events if e.type != EventType.CHANNEL_ATTACHED]


def _last_input(provider: MockAIProvider) -> str:
    content = provider.calls[-1].messages[-1].content
    return content if isinstance(content, str) else str(content)


@pytest.mark.parametrize("streaming", [False, True])
async def test_the_room_holds_the_agents_reply_and_its_own_copy_never_the_instruction(
    streaming: bool,
):
    """The instruction is never stored as the application's event; the agent
    keeps its own copy, visible to it alone, which no channel is delivered
    (RFC §10.1.1)."""
    kit, speaker, provider = await _kit(streaming=streaming)

    result = await kit.process_inbound(_instruction(), room_id=ROOM)

    assert not result.blocked
    stored = await _messages(kit)
    assert [(e.type, e.source.channel_id, e.content.body) for e in stored] == [
        (EventType.INSTRUCTION_READ, "agent", INSTRUCTION),
        (EventType.MESSAGE, "agent", "Bonjour, je suis Paul."),
    ]
    copy, reply = stored
    assert copy.visibility == "agent"
    # A fingerprint, never the text: the reply is stored and fanned out.
    assert reply.metadata["instruction"] == FINGERPRINT
    assert INSTRUCTION not in reply.model_dump_json()
    assert speaker.delivered == ["Bonjour, je suis Paul."]
    room = await kit.get_room(ROOM)
    assert room.event_count == len(await kit.store.list_events(ROOM))
    await kit.close()


async def test_the_model_reads_it_as_the_applications_direction_then_where_it_read_it():
    """A later turn replays it as the turn read it, before that turn's reply:
    the later request extends the instructed one (RFC §10.1.1)."""
    kit, _speaker, provider = await _kit()

    await kit.process_inbound(_instruction(), room_id=ROOM)
    directed = _last_input(provider)
    assert INSTRUCTION in directed
    assert directed != INSTRUCTION  # marked, never passed off as a participant's line

    await kit.process_inbound(
        InboundMessage(channel_id="voice", sender_id="caller", content=TextContent(body="Merci")),
        room_id=ROOM,
    )
    later = provider.calls[-1].messages
    assert [(m.role, m.content) for m in later] == [
        ("user", directed),
        ("assistant", "Bonjour, je suis Paul."),
        ("user", "Merci"),
    ]
    await kit.close()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"addressed_to": None}, "instruction_unaddressed"),
        ({"addressed_to": []}, "instruction_unaddressed"),
        # A name the kit does not register, or a transport's: no agent at all.
        ({"addressed_to": ["ghost"]}, "instruction_unaddressed"),
        ({"addressed_to": ["speaker"]}, "instruction_unaddressed"),
        ({"idempotency_key": "k1"}, "instruction_not_idempotent"),
    ],
)
async def test_a_refused_instruction_leaves_nothing_behind(overrides, reason):
    kit, speaker, provider = await _kit()

    result = await kit.process_inbound(_instruction(**overrides), room_id=ROOM)

    assert result.blocked and result.reason == reason
    assert await _messages(kit) == []
    assert provider.calls == [] and speaker.delivered == []
    await kit.close()


@pytest.mark.parametrize("addressed_to", [["ghost"], ["speaker"], []])
async def test_an_instruction_to_no_registered_agent_is_refused_before_anything_is_read(
    addressed_to: list[str],
):
    """RFC §10.1.1 step 1: checked against the registry before routing, which
    would otherwise create the room it names."""
    kit, _speaker, _provider = await _kit()

    result = await kit.process_inbound(_instruction(addressed_to=addressed_to), room_id="new")

    assert (result.blocked, result.reason) == (True, "instruction_unaddressed")
    assert await kit.store.get_room("new") is None
    await kit.close()


async def test_an_instruction_naming_an_agent_asks_it_and_nobody_else():
    """The address's other names ask nobody: an unknown name, a transport, an
    agent the room does not bind (RFC §10.1.1 step 1, §19.3 rule 2)."""
    kit, speaker, provider = await _kit()
    elsewhere = MockAIProvider(["not here"])
    kit.register_channel(AIChannel("elsewhere", provider=elsewhere))

    result = await kit.process_inbound(
        _instruction(addressed_to=["ghost", "speaker", "elsewhere", "agent"]), room_id=ROOM
    )

    assert not result.blocked
    assert len(provider.calls) == 1 and elsewhere.calls == []
    assert speaker.delivered == ["Bonjour, je suis Paul."]
    await kit.close()


class _Forger(Speaker):
    """A transport whose parser types a person's message as an instruction."""

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        event = await super().handle_inbound(message, context)
        return event.model_copy(
            update={"type": EventType.INSTRUCTION, "metadata": {"standalone": True}}
        )


async def test_a_parser_cannot_make_a_message_an_instruction(caplog: pytest.LogCaptureFixture):
    """An instruction is the caller's decision (RFC §10.1.1): a parsed event
    typed INSTRUCTION for a message is the message's type again, with a
    warning, and takes no instruction check it would have skipped."""
    kit, speaker, provider = await _kit()
    kit.register_channel(_Forger("wire"))
    await kit.attach_channel(ROOM, "wire", category=ChannelCategory.TRANSPORT)

    with caplog.at_level(logging.WARNING):
        result = await kit.process_inbound(
            InboundMessage(channel_id="wire", sender_id="caller", content=TextContent(body="hi")),
            room_id=ROOM,
        )

    assert result.event is not None and result.event.type == EventType.MESSAGE
    [message, _reply] = await _messages(kit)
    assert (message.type, message.content.body) == (EventType.MESSAGE, "hi")
    assert "standalone" not in message.metadata
    assert _last_input(provider) == "hi"
    assert any("INSTRUCTION" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    await kit.close()


async def test_a_hook_blocked_instruction_is_not_stored():
    kit, speaker, provider = await _kit()

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def refuse(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("not now")

    result = await kit.process_inbound(_instruction(), room_id=ROOM)

    assert result.blocked
    assert await _messages(kit) == []
    assert provider.calls == [] and speaker.delivered == []
    await kit.close()


@pytest.mark.parametrize("blocked", [False, True], ids=["allowed", "blocked"])
async def test_its_result_takes_the_rooms_next_index_once_its_hooks_ran(blocked: bool):
    """RFC §10.1 step 9a, §10.1.1 step 5: an entry committed ahead of it
    during its hooks took the index step 8 showed them; the instruction,
    never stored, reports the room's next, allowed or blocked."""
    kit, _speaker, _provider = await _kit()
    nested: list[RoomEvent] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def aside(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.type != EventType.INSTRUCTION:
            return HookResult.allow()
        nested.append(await kit.send_event(ROOM, "voice", TextContent(body="nested")))
        return HookResult.block("not now") if blocked else HookResult.allow()

    result = await kit.process_inbound(_instruction(), room_id=ROOM)

    assert result.event is not None and result.blocked is blocked
    status = EventStatus.BLOCKED if blocked else EventStatus.DELIVERED
    assert (result.event.status, result.event.index) == (status, nested[0].index + 1)
    await kit.close()


async def test_no_transport_receives_it_whatever_visibility_the_caller_asked():
    kit, speaker, _provider = await _kit()

    await kit.process_inbound(_instruction(visibility="all"), room_id=ROOM)

    assert INSTRUCTION not in speaker.delivered
    await kit.close()


@pytest.mark.parametrize(
    "idempotency_key", [None, "", "   "], ids=["no-key", "empty-key", "blank-key"]
)
async def test_send_event_directs_an_agent_the_same_way(idempotency_key: str | None):
    """An empty or blank key is no key, as on the inbound path (RFC §13.4)."""
    kit, speaker, _provider = await _kit()

    await kit.send_event(
        ROOM,
        "voice",
        TextContent(body=INSTRUCTION),
        event_type=EventType.INSTRUCTION,
        addressed_to=["agent"],
        idempotency_key=idempotency_key,
    )

    replies = [e for e in await _messages(kit) if e.type == EventType.MESSAGE]
    assert [(e.source.channel_id, e.metadata.get("instruction")) for e in replies] == [
        ("agent", FINGERPRINT)
    ]
    assert speaker.delivered == ["Bonjour, je suis Paul."]
    await kit.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"addressed_to": None},
        {"addressed_to": ["ghost", "speaker"]},
        {"addressed_to": ["agent"], "idempotency_key": "k"},
    ],
)
async def test_send_event_raises_for_an_instruction_it_would_refuse(overrides):
    """Its contract is the committed event: a refusal has nothing to return."""
    kit, speaker, provider = await _kit()

    with pytest.raises(ValueError):
        await kit.send_event(
            ROOM,
            "voice",
            TextContent(body=INSTRUCTION),
            event_type=EventType.INSTRUCTION,
            **overrides,
        )

    assert await _messages(kit) == []
    assert provider.calls == [] and speaker.delivered == []
    await kit.close()


class _CountingMemory(BudgetAwareMemory):
    """A budget with a floor that keeps events whatever the view: an empty view
    is not a blank page, only not calling it is."""

    def __init__(self) -> None:
        super().__init__(SlidingWindowMemory(), max_context_tokens=100_000, min_events=3)
        self.retrieved = 0

    async def retrieve(self, *args: object, **kwargs: object) -> MemoryResult:
        self.retrieved += 1
        return await super().retrieve(*args, **kwargs)  # type: ignore[arg-type]


async def _talking_room(
    *, streaming: bool = False
) -> tuple[RoomKit, MockAIProvider, _CountingMemory]:
    """A room where the caller and the agent already exchanged a line, the
    agent having used a tool to answer it."""
    memory = _CountingMemory()
    answer = AIResponse(content="Bonjour, je suis Paul.", finish_reason="stop")
    lookup = AIToolCall(id="t1", name="lookup_customer", arguments={"id": 7})
    provider = MockAIProvider(
        ai_responses=[AIResponse(content="", tool_calls=[lookup]), answer, answer],
        streaming=streaming,
    )

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "PREVIOUS-TOOL-RESULT"

    kit, _speaker, provider = await _kit(
        streaming=streaming,
        memory=memory,
        provider=provider,
        tools=[AITool(name="lookup_customer", description="Finds a customer", parameters={})],
        tool_handler=handler,
    )
    await kit.process_inbound(
        InboundMessage(channel_id="voice", sender_id="caller", content=TextContent(body="Allô")),
        room_id=ROOM,
    )
    assert memory.retrieved == 1
    return kit, provider, memory


async def _send_instruction(kit: RoomKit, via: str, **fields: object) -> None:
    if via == "process_inbound":
        await kit.process_inbound(_instruction(**fields), room_id=ROOM)
        return
    await kit.send_event(
        ROOM,
        "voice",
        TextContent(body=INSTRUCTION),
        event_type=EventType.INSTRUCTION,
        addressed_to=["agent"],
        **fields,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("via", ["process_inbound", "send_event"])
async def test_a_standalone_instruction_reads_nothing_of_the_room(via: str, streaming: bool):
    """RFC §10.1.1 step 7: no history, no memory provider call, and none of the
    room's working memories, nor its earlier tool calls, in the system prompt
    or with the input."""
    kit, provider, memory = await _talking_room(streaming=streaming)

    await _send_instruction(kit, via, standalone=True)

    assert memory.retrieved == 1
    [only] = provider.calls[-1].messages
    assert only.role == "user" and INSTRUCTION in str(only.content)
    assert "Allô" not in str(only.content)
    assert "PREVIOUS-TOOL-RESULT" not in str(only.content)
    assert "PREVIOUS-TOOL-RESULT" not in (provider.calls[-1].system_prompt or "")
    stored = await _messages(kit)
    assert stored[-1].metadata["instruction"] == FINGERPRINT
    assert all("standalone" not in e.metadata for e in stored)
    await kit.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("via", ["process_inbound", "send_event"])
async def test_an_instruction_without_standalone_reads_the_room(via: str, streaming: bool):
    """A metadata key of the same name is not the flag: only the typed field is."""
    kit, provider, memory = await _talking_room(streaming=streaming)

    await _send_instruction(kit, via, metadata={"standalone": True})

    assert memory.retrieved == 2
    messages = provider.calls[-1].messages
    contents = [str(m.content) for m in messages]
    assert any("Allô" in c for c in contents) and INSTRUCTION in contents[-1]
    # The agent's earlier tool call rides the history, at its place (RFC §6.4).
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant", "user"]
    assert "PREVIOUS-TOOL-RESULT" in contents[2]
    assert "PREVIOUS-TOOL-RESULT" not in (provider.calls[-1].system_prompt or "")
    await kit.close()


async def test_standalone_is_refused_on_anything_but_an_instruction():
    """Set on a message, it would land as a participant's line without the isolation asked."""
    kit, speaker, provider = await _kit()

    with pytest.raises(ValidationError):
        InboundMessage(
            channel_id="voice", sender_id="caller", content=TextContent(body="x"), standalone=True
        )
    with pytest.raises(ValueError):
        await kit.send_event(ROOM, "voice", TextContent(body="x"), standalone=True)

    assert await _messages(kit) == []
    assert provider.calls == [] and speaker.delivered == []
    await kit.close()

"""Tests for AIContext.response_metadata propagation to response events.

Turn-level metadata set by a BEFORE_AI_GENERATION hook (e.g. RAG source
attribution) must land in the metadata of every MESSAGE response event,
on both the non-streaming path (baked into response_events) and the
streaming path (merged when the core persists stream segments) — so the
stored row and the broadcast both carry it without post-hoc rewrites.
"""

from __future__ import annotations

from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import SyncPipelineResult
from roomkit.memory.base import MemoryResult
from roomkit.memory.mock import MockMemoryProvider
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventType,
    HookTrigger,
)
from roomkit.models.event import TextContent
from roomkit.models.hook import HookResult
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.room import Room
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import current_response_metadata
from tests.conftest import make_event

_SOURCES = [{"document_id": "doc-1", "name": "report.pdf", "relevance": 0.9}]


def _attribution_hook(gen_event: AIGenerationEvent) -> None:
    gen_event.ai_context.response_metadata["rag_sources"] = _SOURCES


def _binding(channel_id: str = "ai1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _ctx() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


# ---------------------------------------------------------------------------
# Channel-level (unit)
# ---------------------------------------------------------------------------


class TestResponseMetadataDirect:
    async def test_non_streaming_message_event_carries_metadata(self) -> None:
        provider = MockAIProvider(responses=["AI reply"])
        ch = AIChannel("ai1", provider=provider)

        async def _hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            _attribution_hook(gen_event)
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = _hook
        output = await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())

        assert len(output.response_events) == 1
        meta = output.response_events[0].metadata
        assert meta["rag_sources"] == _SOURCES
        # Existing usage stamp is preserved alongside.
        assert "ai_usage" in meta

    async def test_tool_round_message_events_carry_metadata(self) -> None:
        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return "result"

        responses = [
            AIResponse(
                content="Let me search.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "x"})],
            ),
            AIResponse(
                content="Here are the results.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]
        provider = MockAIProvider(ai_responses=responses)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        async def _hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            _attribution_hook(gen_event)
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = _hook
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"tools": [{"name": "search", "description": "Search"}]},
        )
        output = await ch.on_event(make_event(body="search", channel_id="sms1"), binding, _ctx())

        messages = [e for e in output.response_events if e.type == EventType.MESSAGE]
        tool_events = [e for e in output.response_events if e.type != EventType.MESSAGE]
        assert messages, "expected at least one MESSAGE event"
        for e in messages:
            assert e.metadata["rag_sources"] == _SOURCES
        for e in tool_events:
            assert "rag_sources" not in e.metadata

    async def test_streaming_output_carries_response_metadata(self) -> None:
        provider = MockAIProvider(responses=["AI reply"], streaming=True)
        ch = AIChannel("ai1", provider=provider)

        async def _hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            _attribution_hook(gen_event)
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = _hook
        output = await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())

        assert output.response_stream is not None
        assert output.response_metadata["rag_sources"] == _SOURCES

    async def test_streaming_tool_loop_output_carries_response_metadata(self) -> None:
        # Channels with tools take _start_streaming_tool_response — a
        # distinct entry point from the no-tools streaming path. Real
        # agents virtually always have tools, so this is the path that
        # matters in production.
        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            return "result"

        provider = MockAIProvider(responses=["AI reply"], streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        async def _hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            _attribution_hook(gen_event)
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = _hook
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"tools": [{"name": "search", "description": "Search"}]},
        )
        output = await ch.on_event(make_event(body="hi", channel_id="sms1"), binding, _ctx())

        assert output.response_stream is not None
        assert output.response_metadata["rag_sources"] == _SOURCES

    async def test_no_hook_means_empty_response_metadata(self) -> None:
        provider = MockAIProvider(responses=["AI reply"], streaming=True)
        ch = AIChannel("ai1", provider=provider)
        output = await ch.on_event(make_event(body="hi", channel_id="sms1"), _binding(), _ctx())
        assert output.response_metadata == {}


# ---------------------------------------------------------------------------
# Integration — stored events carry the metadata on both paths
# ---------------------------------------------------------------------------


async def _setup_kit(*, streaming: bool) -> RoomKit:
    from roomkit.channels import SMSChannel
    from roomkit.providers.sms.mock import MockSMSProvider

    kit = RoomKit()
    ai = AIChannel(
        "ai1",
        provider=MockAIProvider(responses=["AI says hi"], streaming=streaming),
        system_prompt="test",
    )
    sms = SMSChannel("sms1", provider=MockSMSProvider())
    kit.register_channel(ai)
    kit.register_channel(sms)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r1", "sms1")

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def attribute(event: AIGenerationEvent, ctx: Any) -> HookResult:
        _attribution_hook(event)
        return HookResult.allow()

    return kit


async def _ai_message_events(kit: RoomKit) -> list[Any]:
    events = await kit.get_timeline("r1")
    return [e for e in events if e.source.channel_id == "ai1" and e.type == EventType.MESSAGE]


class TestResponseMetadataStored:
    async def test_non_streaming_stored_event_has_metadata(self, advance) -> None:
        kit = await _setup_kit(streaming=False)
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="user1", content=TextContent(body="Hello AI")
            )
        )
        await advance(10)

        ai_events = await _ai_message_events(kit)
        assert len(ai_events) == 1
        assert ai_events[0].metadata["rag_sources"] == _SOURCES
        await kit.close()

    async def test_streaming_stored_event_has_metadata(self, advance) -> None:
        # The streaming path persists segments BEFORE any broadcast hook —
        # the regression this feature exists for: attribution must already
        # be on the event when the core persists it.
        kit = await _setup_kit(streaming=True)
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="user1", content=TextContent(body="Hello AI")
            )
        )
        await advance(20)

        ai_events = await _ai_message_events(kit)
        assert len(ai_events) == 1
        assert ai_events[0].metadata["rag_sources"] == _SOURCES
        await kit.close()


# ---------------------------------------------------------------------------
# One live record for the whole turn — every writer reaches the same object
# ---------------------------------------------------------------------------
#
# The record is created with the turn, before the context is built, and is
# what AIContext, the hooks, the tool handlers and the output all hold. What
# a tool handler learns mid-loop (a document it read) therefore lands on the
# MESSAGE events created after it, on both generation paths — which a plain
# dict could not do: Pydantic copied it into ChannelOutput at stream start.

_TOOL_ROUNDS = [
    AIResponse(
        content="Let me read.",
        finish_reason="tool_calls",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        tool_calls=[AIToolCall(id="tc1", name="read_page", arguments={"page": 3})],
    ),
    AIResponse(
        content="Page 3 says so.",
        finish_reason="stop",
        usage={"prompt_tokens": 20, "completion_tokens": 10},
    ),
]

_TOOLS_BINDING_META = {"tools": [{"name": "read_page", "description": "Read a page"}]}


async def _citing_handler(name: str, args: dict[str, Any]) -> str:
    """A host handler recording what the tool read, as a fact about the turn."""
    record = current_response_metadata()
    assert record is not None, "handler ran outside a turn"
    record.setdefault("cited", []).append({"tool": name, "page": args["page"]})
    return "page text"


class _CitingMemory(MockMemoryProvider):
    """A memory provider that attributes what it injected while building the context."""

    async def retrieve(self, room_id, current_event, context, *, channel_id=None) -> MemoryResult:
        record = current_response_metadata()
        assert record is not None, "retrieve ran outside a turn"
        record["rag_sources"] = _SOURCES
        return await super().retrieve(room_id, current_event, context, channel_id=channel_id)


class TestLiveRecord:
    def test_type_keeps_identity_where_a_dict_was_copied(self) -> None:
        record = ResponseMetadata()
        assert ChannelOutput(responded=True, response_metadata=record).response_metadata is record
        ai_context = AIContext(messages=[], response_metadata=record)
        assert ai_context.response_metadata is record
        # A bare dict is still accepted — wrapped, as a snapshot the caller chose.
        legacy = AIContext(messages=[], response_metadata={"rag_sources": _SOURCES})
        assert legacy.response_metadata == {"rag_sources": _SOURCES}
        assert ChannelOutput.empty().response_metadata == {}

    def test_accessor_is_none_outside_a_turn(self) -> None:
        assert current_response_metadata() is None

    async def test_tool_handler_write_lands_on_the_reply_non_streaming(self) -> None:
        provider = MockAIProvider(ai_responses=list(_TOOL_ROUNDS))
        ch = AIChannel("ai1", provider=provider, tool_handler=_citing_handler)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata=_TOOLS_BINDING_META,
        )
        output = await ch.on_event(make_event(body="read", channel_id="sms1"), binding, _ctx())

        messages = [e for e in output.response_events if e.type == EventType.MESSAGE]
        assert messages
        for e in messages:
            assert e.metadata["cited"] == [{"tool": "read_page", "page": 3}]

    async def test_tool_handler_write_lands_on_the_streamed_answer(self, advance) -> None:
        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        kit = RoomKit()
        ai = AIChannel(
            "ai1",
            provider=MockAIProvider(ai_responses=list(_TOOL_ROUNDS), streaming=True),
            tool_handler=_citing_handler,
            system_prompt="test",
        )
        kit.register_channel(ai)
        kit.register_channel(SMSChannel("sms1", provider=MockSMSProvider()))
        await kit.create_room(room_id="r1")
        await kit.attach_channel(
            "r1", "ai1", category=ChannelCategory.INTELLIGENCE, metadata=_TOOLS_BINDING_META
        )
        await kit.attach_channel("r1", "sms1")

        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="read"))
        )
        await advance(20)

        ai_events = await _ai_message_events(kit)
        assert ai_events, "expected the streamed answer to be persisted"
        # The segment persisted BEFORE the tool ran carries what was known then;
        # the answer, persisted after, carries what the handler wrote.
        assert ai_events[-1].metadata["cited"] == [{"tool": "read_page", "page": 3}]
        assert "cited" not in ai_events[0].metadata or ai_events[0] is ai_events[-1]
        await kit.close()

    async def test_memory_provider_write_lands_on_the_reply(self, advance) -> None:
        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        kit = RoomKit()
        ai = AIChannel(
            "ai1",
            provider=MockAIProvider(responses=["AI says hi"], streaming=True),
            memory=_CitingMemory(),
            system_prompt="test",
        )
        kit.register_channel(ai)
        kit.register_channel(SMSChannel("sms1", provider=MockSMSProvider()))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
        await kit.attach_channel("r1", "sms1")

        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="hi"))
        )
        await advance(20)

        ai_events = await _ai_message_events(kit)
        assert len(ai_events) == 1
        assert ai_events[0].metadata["rag_sources"] == _SOURCES
        await kit.close()

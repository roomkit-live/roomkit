"""Tests for AIContext.response_metadata propagation to response events.

Turn-level metadata set by a BEFORE_AI_GENERATION hook (e.g. RAG source
attribution) must land in the metadata of every MESSAGE response event,
on both the non-streaming path (baked into response_events) and the
streaming path (merged when the core persists stream segments) — so the
stored row and the broadcast both carry it without post-hoc rewrites.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.channels import SMSChannel
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
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.tools import current_response_metadata
from tests.conftest import make_event
from tests.tool_loop_modes import respond

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


class TestInboundResultResponseMetadata:
    """The caller receives the turn record on every completion path."""

    async def test_non_streaming_turn_returns_its_final_record(self) -> None:
        kit = await _setup_kit(streaming=False)

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="user1",
                content=TextContent(body="Hello AI"),
            )
        )

        assert result.response_metadata["rag_sources"] == _SOURCES
        await kit.close()

    async def test_deferred_wait_backfills_writes_made_during_the_stream(self) -> None:
        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        kit = RoomKit()
        ai = AIChannel(
            "ai1",
            provider=MockAIProvider(ai_responses=list(_TOOL_ROUNDS), streaming=True),
            tool_handler=_citing_handler,
        )
        kit.register_channel(ai)
        kit.register_channel(SMSChannel("sms1", provider=MockSMSProvider()))
        await kit.create_room(room_id="r1")
        await kit.attach_channel(
            "r1", "ai1", category=ChannelCategory.INTELLIGENCE, metadata=_TOOLS_BINDING_META
        )
        await kit.attach_channel("r1", "sms1")

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="user1",
                content=TextContent(body="Read page 3"),
            ),
            defer_delivery=True,
        )
        record = result.response_metadata
        assert result.delivery is not None

        waited = await result.delivery.wait()

        assert waited is result
        assert result.response_metadata is record
        assert record["cited"] == [{"tool": "read_page", "page": 3}]
        await kit.close()


class TestLiveRecord:
    def test_type_keeps_identity_where_a_dict_was_copied(self) -> None:
        record = ResponseMetadata()
        assert ChannelOutput(responded=True, response_metadata=record).response_metadata is record
        ai_context = AIContext(messages=[], response_metadata=record)
        assert ai_context.response_metadata is record
        # A bare dict is still accepted — wrapped, as a snapshot the caller chose.
        bare_dict = AIContext(messages=[], response_metadata={"rag_sources": _SOURCES})
        assert bare_dict.response_metadata == {"rag_sources": _SOURCES}
        assert ChannelOutput.empty().response_metadata == {}

    def test_a_record_counts_the_writes_to_its_keys(self) -> None:
        record = ResponseMetadata({"rag_sources": []})
        assert record.writes == 0
        record["cited"] = 1
        record.update(cited=2, credential="c")
        record.setdefault("cited", 3)  # present: nothing written
        record.setdefault("read", [])
        record.pop("credential")
        assert dict(record) and record.get("cited") == 2 and record != {}  # reads write nothing
        assert record.writes == 5
        record["read"].append("doc-1")  # inside a value: no key written
        assert record.writes == 5
        record.clear()
        assert record.writes == 8

    def test_accessor_is_none_outside_a_turn(self) -> None:
        assert current_response_metadata() is None

    @pytest.mark.parametrize("streaming", [True, False])
    # A turn that waited its place in the room hands its record on all the same.
    @pytest.mark.parametrize("serialized", [False, True], ids=["side by side", "one at a time"])
    async def test_tool_handler_write_lands_on_the_answer(
        self, advance, streaming: bool, serialized: bool
    ) -> None:
        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        kit = RoomKit()
        ai = AIChannel(
            "ai1",
            provider=MockAIProvider(ai_responses=list(_TOOL_ROUNDS), streaming=streaming),
            tool_handler=_citing_handler,
            system_prompt="test",
            serialize_room_turns=serialized,
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

    async def test_a_value_under_turns_lands_on_no_message(self, streaming: bool) -> None:
        """``turns`` is the framework's key: a value a tool writes there stays
        off every MESSAGE, its other keys ride them (RFC §6.7)."""

        async def writes_turns(name: str, args: dict[str, Any]) -> str:
            record = current_response_metadata()
            assert record is not None
            record.update(turns=3, cited=args["page"])
            return "page text"

        kit = RoomKit()
        ai = AIChannel(
            "ai1",
            provider=MockAIProvider(ai_responses=list(_TOOL_ROUNDS), streaming=streaming),
            tool_handler=writes_turns,
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

        answer = (await _ai_message_events(kit))[-1]
        assert answer.metadata["cited"] == 3
        assert "turns" not in answer.metadata
        await kit.close()

    async def test_a_cut_turn_gives_its_caller_what_its_tool_wrote(self) -> None:
        """A turn cut from outside tells its caller how it ended and what it
        wrote, as a turn that failed does (RFC §6.7, HITL 84991349)."""
        running = asyncio.Event()

        async def hangs(name: str, args: dict[str, Any]) -> str:
            record = current_response_metadata()
            assert record is not None
            record["cited"] = args["page"]
            running.set()
            await asyncio.Event().wait()
            return "never"

        kit = RoomKit()
        ai = AIChannel(
            "ai1",
            provider=MockAIProvider(ai_responses=list(_TOOL_ROUNDS), streaming=True),
            tool_handler=hangs,
        )
        kit.register_channel(ai)
        kit.register_channel(SMSChannel("sms1", provider=MockSMSProvider()))
        await kit.create_room(room_id="r1")
        await kit.attach_channel(
            "r1", "ai1", category=ChannelCategory.INTELLIGENCE, metadata=_TOOLS_BINDING_META
        )
        await kit.attach_channel("r1", "sms1")

        result = await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="read")),
            defer_delivery=True,
        )
        assert result.delivery is not None
        await asyncio.wait_for(running.wait(), 5)
        cut = await result.delivery.cancel(reason="stopped")

        assert cut.response_metadata["cited"] == 3
        assert cut.response_metadata["turns"]["ai1"]["loop_end_reason"] == "cancelled"
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

    async def test_a_hook_that_replaces_the_context_keeps_one_record(
        self, streaming: bool
    ) -> None:
        # A hook may return a NEW AIContext instead of mutating the one it got.
        # The turn still has one record: the tool handler's writes must land on
        # the record the reply is built from, whichever object the hook returned.
        provider = MockAIProvider(ai_responses=list(_TOOL_ROUNDS), streaming=streaming)
        ch = AIChannel("ai1", provider=provider, tool_handler=_citing_handler)

        async def _replacing_hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            fresh = gen_event.ai_context.model_copy(
                update={"response_metadata": ResponseMetadata({"rag_sources": _SOURCES})}
            )
            gen_event.ai_context = fresh
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = _replacing_hook
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata=_TOOLS_BINDING_META,
        )
        run = await respond(ch, make_event(body="read", channel_id="sms1"), binding, _ctx())

        assert run.metadata["rag_sources"] == _SOURCES
        assert run.metadata["cited"] == [{"tool": "read_page", "page": 3}]


class _FailsAfterTheCall(MockAIProvider):
    """The model calls ``read_page``, then fails on the round after it."""

    async def generate(self, context: AIContext) -> AIResponse:
        if self.calls:
            raise ProviderError("upstream down", provider="mock")
        return await super().generate(context)


@pytest.mark.parametrize("fails", [False, True])
async def test_a_before_tool_use_write_reaches_the_inbound_result(
    streaming: bool, fails: bool
) -> None:
    """A BEFORE_TOOL_USE hook runs under the call's turn: what it writes through
    ``current_response_metadata()`` is the record the caller reads, the turn
    answered or failed (RFC §6.7), so a host counts the calls a turn started."""
    provider_cls = _FailsAfterTheCall if fails else MockAIProvider
    kit = RoomKit()
    ai = AIChannel(
        "ai1",
        provider=provider_cls(ai_responses=list(_TOOL_ROUNDS), streaming=streaming),
        tool_handler=_citing_handler,
    )
    kit.register_channel(ai)
    kit.register_channel(SMSChannel("sms1", provider=MockSMSProvider()))
    await kit.create_room(room_id="r1")
    await kit.attach_channel(
        "r1", "ai1", category=ChannelCategory.INTELLIGENCE, metadata=_TOOLS_BINDING_META
    )
    await kit.attach_channel("r1", "sms1")

    @kit.hook(HookTrigger.BEFORE_TOOL_USE)
    async def count(event: Any, ctx: Any) -> HookResult:
        record = current_response_metadata()
        if record is not None:
            record["calls_started"] = record.get("calls_started", 0) + 1
        return HookResult.allow()

    result = await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="Read 3"))
    )

    assert (result.error is not None) is fails
    assert result.response_metadata["calls_started"] == 1
    await kit.close()

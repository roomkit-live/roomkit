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
from roomkit.models.channel import ChannelBinding
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
from roomkit.models.room import Room
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
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

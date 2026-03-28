"""Tests for the BEFORE_AI_GENERATION hook."""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import SyncPipelineResult
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

# ---------------------------------------------------------------------------
# Direct channel-level tests (unit)
# ---------------------------------------------------------------------------


def _make_channel_and_ctx(
    responses: list[str] | None = None,
    *,
    system_prompt: str = "Be helpful",
    streaming: bool = False,
) -> tuple[AIChannel, ChannelBinding, RoomContext]:
    provider = MockAIProvider(responses=responses or ["AI reply"], streaming=streaming)
    ch = AIChannel("ai1", provider=provider, system_prompt=system_prompt)
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    ctx = RoomContext(room=Room(id="r1"))
    return ch, binding, ctx


class TestBeforeGenerationHookDirect:
    """Direct channel-level tests — inject hook callback manually."""

    async def test_no_hook_proceeds_normally(self):
        ch, binding, ctx = _make_channel_and_ctx()
        event = make_event(body="hi", channel_id="sms1")
        output = await ch.on_event(event, binding, ctx)
        assert output.responded is True
        assert len(output.response_events) == 1

    async def test_hook_fires_with_correct_event(self):
        ch, binding, ctx = _make_channel_and_ctx()
        captured: list[AIGenerationEvent] = []

        async def _hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            captured.append(gen_event)
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = _hook
        event = make_event(body="question", channel_id="sms1", room_id="r1")
        await ch.on_event(event, binding, ctx)

        assert len(captured) == 1
        gen = captured[0]
        assert gen.channel_id == "ai1"
        assert gen.room_id == "r1"
        assert gen.provider_name is not None
        assert len(gen.ai_context.messages) >= 1

    async def test_hook_blocks_generation(self):
        ch, binding, ctx = _make_channel_and_ctx()
        provider = ch._provider

        async def _block(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            return SyncPipelineResult(
                allowed=False, reason="budget exceeded", blocked_by="budget_hook"
            )

        ch._before_generation_hook = _block
        event = make_event(body="hi", channel_id="sms1")
        output = await ch.on_event(event, binding, ctx)

        # Generation blocked — no response events, provider never called
        assert output.responded is False
        assert len(provider.calls) == 0

    async def test_hook_modifies_context(self):
        ch, binding, ctx = _make_channel_and_ctx()
        provider = ch._provider

        async def _modify(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            gen_event.ai_context.system_prompt = "You are a pirate"
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = _modify
        event = make_event(body="hi", channel_id="sms1")
        await ch.on_event(event, binding, ctx)

        # Provider should receive the modified context
        assert len(provider.calls) == 1
        assert provider.calls[0].system_prompt == "You are a pirate"

    async def test_hook_appends_message(self):
        from roomkit.providers.ai.base import AIMessage

        ch, binding, ctx = _make_channel_and_ctx()
        provider = ch._provider

        async def _inject(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            gen_event.ai_context.messages.insert(
                0, AIMessage(role="user", content="[INJECTED CONTEXT]")
            )
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = _inject
        event = make_event(body="hello", channel_id="sms1")
        await ch.on_event(event, binding, ctx)

        assert len(provider.calls) == 1
        messages = provider.calls[0].messages
        assert messages[0].content == "[INJECTED CONTEXT]"

    async def test_hook_fires_on_streaming_path(self):
        ch, binding, ctx = _make_channel_and_ctx(streaming=True)
        captured: list[AIGenerationEvent] = []

        async def _hook(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            captured.append(gen_event)
            return SyncPipelineResult(allowed=True)

        ch._before_generation_hook = _hook
        event = make_event(body="hi", channel_id="sms1", room_id="r1")
        output = await ch.on_event(event, binding, ctx)

        assert len(captured) == 1
        assert captured[0].channel_id == "ai1"
        # Streaming returns a response_stream, not response_events
        assert output.responded is True

    async def test_hook_blocks_streaming(self):
        ch, binding, ctx = _make_channel_and_ctx(streaming=True)

        async def _block(gen_event: AIGenerationEvent) -> SyncPipelineResult:
            return SyncPipelineResult(allowed=False, reason="policy violation")

        ch._before_generation_hook = _block
        event = make_event(body="hi", channel_id="sms1")
        output = await ch.on_event(event, binding, ctx)

        assert output.responded is False


# ---------------------------------------------------------------------------
# Integration tests (with RoomKit + hook engine)
# ---------------------------------------------------------------------------


class TestBeforeGenerationHookIntegration:
    """Integration tests using RoomKit.hook() decorator and process_inbound."""

    async def _setup_kit(
        self, responses: list[str] | None = None
    ) -> tuple[RoomKit, MockAIProvider]:
        from roomkit.channels import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        kit = RoomKit()
        ai_provider = MockAIProvider(responses=responses or ["AI says hi"])
        ai = AIChannel("ai1", provider=ai_provider, system_prompt="test")

        sms_provider = MockSMSProvider()
        sms = SMSChannel("sms1", provider=sms_provider)

        kit.register_channel(ai)
        kit.register_channel(sms)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
        await kit.attach_channel("r1", "sms1")

        return kit, ai_provider

    async def test_hook_fires_via_framework(self, advance):
        kit, ai_provider = await self._setup_kit()
        captured: list[AIGenerationEvent] = []

        @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
        async def observe(event, ctx):
            captured.append(event)
            return HookResult.allow()

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="Hello AI"),
        )
        await kit.process_inbound(msg)
        await advance()

        assert len(captured) == 1
        assert isinstance(captured[0], AIGenerationEvent)
        assert captured[0].channel_id == "ai1"
        assert captured[0].room_id == "r1"

    async def test_hook_blocks_via_framework(self, advance):
        kit, ai_provider = await self._setup_kit()

        @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
        async def block_gen(event, ctx):
            return HookResult.block(reason="over budget")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="Hello AI"),
        )
        await kit.process_inbound(msg)
        await advance()

        # Provider should NOT have been called
        assert len(ai_provider.calls) == 0

    async def test_hook_modifies_via_framework(self, advance):
        kit, ai_provider = await self._setup_kit()

        @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
        async def modify_prompt(event, ctx):
            event.ai_context.system_prompt = "You are a pirate"
            return HookResult.allow()

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="Hello AI"),
        )
        await kit.process_inbound(msg)
        await advance()

        assert len(ai_provider.calls) == 1
        assert ai_provider.calls[0].system_prompt == "You are a pirate"

    async def test_async_observer_fires(self, advance):
        kit, ai_provider = await self._setup_kit()
        observed: list[AIGenerationEvent] = []

        @kit.hook(HookTrigger.BEFORE_AI_GENERATION, execution=HookExecution.ASYNC)
        async def log_gen(event, ctx):
            observed.append(event)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="Hello AI"),
        )
        await kit.process_inbound(msg)
        await advance(20)

        assert len(observed) == 1

    async def test_multiple_hooks_priority_order(self, advance):
        kit, ai_provider = await self._setup_kit()
        order: list[str] = []

        @kit.hook(HookTrigger.BEFORE_AI_GENERATION, priority=10)
        async def second(event, ctx):
            order.append("second")
            # Should see modification from first hook
            assert event.ai_context.metadata.get("first") is True
            return HookResult.allow()

        @kit.hook(HookTrigger.BEFORE_AI_GENERATION, priority=0)
        async def first(event, ctx):
            order.append("first")
            event.ai_context.metadata["first"] = True
            return HookResult.allow()

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="Hello AI"),
        )
        await kit.process_inbound(msg)
        await advance()

        assert order == ["first", "second"]

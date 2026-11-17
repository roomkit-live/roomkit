"""The context retained by a generation hook is the one the thinker receives."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest

from roomkit import Agent, AIChannel, HookResult, HookTrigger, MockSpeakPolicy, RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.models.tool_call import AIGenerationEvent
from roomkit.providers.ai import AIContext, AIMessage
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking import MockThinker, Thought
from tests.test_framework import SimpleChannel


class EditingThinker(MockThinker):
    """Record the real context, then change that object through its public fields."""

    def __init__(self) -> None:
        super().__init__([Thought("A quiet impression.")])
        self.changed = asyncio.Event()

    async def think(self, previous: Thought, context: AIContext) -> Thought:
        thought = await super().think(previous, context)
        context.system_prompt = "Changed by the thinker."
        context.messages.append(AIMessage(role="user", content="A thinker annotation."))
        self.changed.set()
        return thought


@pytest.mark.parametrize("channel_kind", [AIChannel, Agent], ids=["channel", "agent"])
@pytest.mark.parametrize("replace_context", [False, True], ids=["allow", "replacement"])
async def test_a_hook_and_the_thinker_keep_the_same_selected_context(
    channel_kind: type[AIChannel], replace_context: bool
) -> None:
    thinker, provider = EditingThinker(), MockAIProvider(responses=["Unexpected answer"])
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(
        channel_kind(
            "ai",
            provider=provider,
            speak_policy=MockSpeakPolicy(["silent"]),
            thinker=thinker,
            think_wait=1,
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    originals: list[AIContext] = []
    selected: list[AIGenerationEvent] = []

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def keep(event: AIGenerationEvent, context: Any) -> HookResult:
        assert event.purpose == "thought"
        originals.append(event.ai_context)
        if replace_context:
            replacement = event.ai_context.model_copy(
                update={
                    "system_prompt": "Replacement.",
                    "messages": list(event.ai_context.messages),
                }
            )
            event = dataclasses.replace(event, ai_context=replacement)
            selected.append(event)
            return HookResult.modify(event)
        selected.append(event)
        return HookResult.allow()

    try:
        async with asyncio.timeout(5):
            result = await kit.process_inbound(
                InboundMessage(
                    channel_id="sms", sender_id="alice", content=TextContent(body="Listen.")
                ),
                room_id="r1",
            )
            await thinker.changed.wait()
        assert result.error is None and provider.calls == []
        assert len(selected) == len(originals) == len(thinker.calls) == 1
        kept = selected[0].ai_context
        assert kept is thinker.calls[0][1]
        assert kept.system_prompt == "Changed by the thinker."
        assert kept.messages[-1].content == "A thinker annotation."
        kept.messages.append(AIMessage(role="user", content="A later hook annotation."))
        assert thinker.calls[0][1].messages[-1].content == "A later hook annotation."
        if replace_context:
            assert originals[0] is not kept
            assert originals[0].system_prompt != "Changed by the thinker."
            assert all(
                message.content != "A thinker annotation." for message in originals[0].messages
            )
        else:
            assert originals[0] is kept
    finally:
        await asyncio.wait_for(kit.close(), 3)

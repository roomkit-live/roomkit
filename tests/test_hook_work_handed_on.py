"""A hook's work handed to another context continues its turn's chain (RFC §8.3).

The body of an ASYNC hook may hand its work to another context: a task outside
the turn, another process. It reads the turn's chain depth and actor there
(``current_tool_chain_depth``, ``current_tool_actor_id``) and the work runs
under them (``tool_turn_context``), so what it sends continues the turn's
chain, as it would have sent from the hook's body.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import current_tool_actor_id, current_tool_chain_depth, tool_turn_context
from tests.test_framework import SimpleChannel


async def _delivered_depth(kit: RoomKit, body: str) -> int:
    for event in await kit.store.list_events("r1"):
        if getattr(event.content, "body", None) == body:
            return event.chain_depth
    raise AssertionError(f"{body!r} was not delivered")


async def test_a_hooks_body_hands_its_turns_depth_and_actor_on() -> None:
    kit = RoomKit()
    handed: asyncio.Future[tuple[int, str | None]] = asyncio.get_running_loop().create_future()

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def observe(event: Any, ctx: Any) -> None:
        if not handed.done():
            handed.set_result((current_tool_chain_depth(), current_tool_actor_id()))

    ai = AIChannel(
        "ai1",
        provider=MockAIProvider(
            ai_responses=[AIResponse(content=text) for text in ("One.", "Two.", "Three.")]
        ),
    )
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        depth, actor = await asyncio.wait_for(handed, 5)
        asker = next(
            event.source.participant_id
            for event in await kit.store.list_events("r1")
            if getattr(event.content, "body", None) == "go"
        )
        # The response to a person's message is one deep, and its author asked.
        assert depth == 1
        assert actor == asker

        async def elsewhere(body: str, *, under_the_turn: bool) -> None:
            if not under_the_turn:
                await kit.deliver("r1", body)
                return
            with tool_turn_context(room_id="r1", actor_id=actor, chain_depth=depth):
                await kit.deliver("r1", body)

        # A task that holds nothing of the turn: only what was handed on.
        fresh = contextvars.Context()
        await asyncio.create_task(elsewhere("handed", under_the_turn=True), context=fresh)
        await asyncio.create_task(elsewhere("bare", under_the_turn=False), context=fresh)

        assert await _delivered_depth(kit, "handed") == depth
        assert await _delivered_depth(kit, "bare") == 0
    finally:
        await kit.close()

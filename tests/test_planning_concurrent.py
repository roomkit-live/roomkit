"""Concurrent room turns retain only their own channel-managed plan."""

from __future__ import annotations

import asyncio

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.planning_support import answer, message, plan
from tests.test_framework import SimpleChannel


class RoomPlans(MockAIProvider):
    def __init__(self, streaming: bool) -> None:
        super().__init__(streaming=streaming)
        self.by_room: dict[str, list[AIContext]] = {}
        self.both = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        assert context.room is not None
        room_id = context.room.room.id
        calls = self.by_room.setdefault(room_id, [])
        calls.append(context.model_copy(deep=True))
        if len(calls) == 1:
            if len(self.by_room) == 2:
                self.both.set()
            await asyncio.wait_for(self.both.wait(), 3)
            return plan([{"title": f"SECRET_{room_id}", "status": "pending"}])
        return answer()


async def test_two_live_rooms_keep_separate_plans_in_their_real_messages(streaming: bool) -> None:
    model = RoomPlans(streaming)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ws"))
    kit.register_channel(AIChannel("agent", provider=model, enable_planning=True))
    for name in ("a", "b"):
        await kit.create_room(room_id=name)
        await kit.attach_channel(name, "ws")
        await kit.attach_channel(name, "agent")
    try:
        for body in ("Plan.", "Continue."):
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(kit.process_inbound(message(body), room_id=name) for name in ("a", "b"))
                ),
                5,
            )
            assert all(result.error is None for result in results)
        for own, other in (("a", "b"), ("b", "a")):
            calls = model.by_room[own]
            assert len(calls) == 3
            assert f"SECRET_{own}" in str(calls[-1].messages[-1].content)
            assert all(
                f"SECRET_{other}" not in str(msg.content)
                for call in calls
                for msg in call.messages
            )
    finally:
        await kit.close()

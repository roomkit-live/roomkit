"""A strategy's passes bypass speaking and thinking on the actual executor."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import Agent, MockSpeakPolicy, RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking import MockThinker
from tests.test_framework import SimpleChannel


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("refine", [False, True])
async def test_strategy_passes_bypass_both_the_silent_policy_and_its_thinker(
    streaming: bool, refine: bool
) -> None:
    policy, thinker = MockSpeakPolicy(["silent"]), MockThinker()
    provider = MockAIProvider(
        responses=["Task.", "Presented."] if refine else ["Presented."], streaming=streaming
    )
    boss = Agent("boss", provider=provider, speak_policy=policy, thinker=thinker)
    worker = Agent("worker", provider=MockAIProvider(responses=["Found."]))
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(boss)
    kit.register_channel(worker)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.install_strategy(
        "r1",
        Supervisor(
            supervisor=boss,
            workers=[worker],
            strategy="parallel",
            auto_delegate=True,
            refine_task=refine,
        ),
    )
    try:
        answer = await asyncio.wait_for(
            kit.process_inbound(
                InboundMessage(
                    channel_id="sms", sender_id="alice", content=TextContent(body="Go.")
                ),
                room_id="r1",
            ),
            5,
        )
        assert [e.content.body for e in answer.response_events] == ["Presented."]
        assert len(provider.calls) == (2 if refine else 1)
        assert policy.turns == [] and thinker.calls == []
    finally:
        await asyncio.wait_for(kit.close(), 3)

"""A turn reads host definitions after its configuration callback."""

from __future__ import annotations

import pytest

from roomkit import Agent, AIChannel, AIChannelTurnConfig, ChannelBinding, Room, RoomContext
from roomkit.models import ChannelCategory, ChannelType
from roomkit.providers.ai import AITool, MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond


@pytest.mark.parametrize("channel_type", [AIChannel, Agent], ids=["channel", "agent"])
async def test_a_configuration_callback_can_edit_the_host_definition(
    channel_type: type[AIChannel], streaming: bool
) -> None:
    async def configured(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        channel.extra_tools[0].description = "edited while configuring"
        return AIChannelTurnConfig()

    provider = MockAIProvider(responses=["done"], streaming=streaming)
    channel = channel_type(
        "ai",
        provider=provider,
        config_provider=configured,
        tools=[AITool(name="read", description="before")],
        tool_search=False,
    )
    binding = ChannelBinding(
        room_id="r1",
        channel_id="ai",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    try:
        await respond(channel, make_event(body="hello"), binding, RoomContext(room=Room(id="r1")))
    finally:
        await channel.close()
    assert provider.calls[0].tools[0].description == "edited while configuring"

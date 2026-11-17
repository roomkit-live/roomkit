"""A turn's skill registry keeps planning after its skill declarations."""

from __future__ import annotations

from pathlib import Path

from roomkit import RoomKit
from roomkit.channels._turn_config import AIChannelTurnConfig
from roomkit.channels.ai import AIChannel
from roomkit.models import ChannelBinding, RoomContext
from roomkit.providers.ai import MockAIProvider
from roomkit.skills import Skill, SkillMetadata, SkillRegistry
from tests.planning_support import answer, message
from tests.test_framework import SimpleChannel


async def test_a_turn_registry_keeps_skill_declarations_before_planning(streaming: bool) -> None:
    skills = SkillRegistry()
    skills.add(
        Skill(
            metadata=SkillMetadata(name="helper", description="Help"),
            instructions="Help.",
            path=Path("."),
        )
    )

    async def configure(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(skills=skills)

    model = MockAIProvider(ai_responses=[answer()], streaming=streaming)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ws"))
    kit.register_channel(
        AIChannel(
            "agent",
            provider=model,
            skills=SkillRegistry(),
            enable_planning=True,
            config_provider=configure,
        )
    )
    try:
        await kit.create_room(room_id="r")
        for channel in ("ws", "agent"):
            await kit.attach_channel("r", channel)
        await kit.process_inbound(message())
        names = [tool.name for tool in model.calls[0].tools]
        assert (
            names.index("activate_skill")
            < names.index("read_skill_reference")
            < names.index("plan_tasks")
        )
    finally:
        await kit.close()

"""Per-turn config provider — dynamic system prompt / tools resolution.

Channel config snapshotted at attach time goes stale (admin edits never
reach existing rooms). ``AIChannel(config_provider=...)`` resolves config
fresh on every turn; binding-metadata overrides stay authoritative for
sampling/prompt, while the provider's toolset REPLACES the deprecated
``binding.metadata["tools"]`` snapshot.
"""

from __future__ import annotations

from roomkit.channels._turn_config import AIChannelTurnConfig
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

TOOL_V1 = {"name": "outlook", "description": "mail", "parameters": {}}
TOOL_V2 = AITool(name="gmail", description="mail", parameters={})


def _binding(metadata: dict | None = None) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata=metadata or {},
    )


def _ctx(binding: ChannelBinding) -> RoomContext:
    return RoomContext(room=Room(id="r1"), bindings=[binding])


def _channel(**kwargs) -> AIChannel:
    defaults = {"provider": MockAIProvider(responses=["ok"]), "system_prompt": "static prompt"}
    defaults.update(kwargs)
    return AIChannel("ai1", **defaults)


class TestConfigProvider:
    async def test_provider_overrides_prompt_and_tools(self) -> None:
        calls: list[tuple] = []

        async def provider(binding, context):
            calls.append((binding.room_id, context.room.id))
            return AIChannelTurnConfig(system_prompt="fresh prompt", tools=[TOOL_V2])

        ch = _channel(config_provider=provider)
        # Even a stale binding snapshot must not win over the resolver.
        binding = _binding(metadata={"tools": [TOOL_V1]})
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert calls == [("r1", "r1")]
        assert ai_ctx.system_prompt.startswith("fresh prompt")
        tool_names = {t.name for t in ai_ctx.tools}
        assert "gmail" in tool_names
        assert "outlook" not in tool_names  # the snapshot is dead data

    async def test_binding_metadata_still_wins_for_sampling_and_prompt(self) -> None:
        async def provider(binding, context):
            return AIChannelTurnConfig(system_prompt="fresh prompt", temperature=0.9)

        ch = _channel(config_provider=provider)
        binding = _binding(metadata={"system_prompt": "explicit override", "temperature": 0.1})
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.system_prompt.startswith("explicit override")
        assert ai_ctx.temperature == 0.1

    async def test_partial_config_falls_back_to_defaults(self) -> None:
        async def provider(binding, context):
            return AIChannelTurnConfig(tools=[TOOL_V2])  # no prompt override

        ch = _channel(config_provider=provider)
        binding = _binding()
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.system_prompt.startswith("static prompt")
        assert {t.name for t in ai_ctx.tools} == {"gmail"}

    async def test_provider_returning_none_keeps_static_path(self) -> None:
        async def provider(binding, context):
            return None

        ch = _channel(config_provider=provider)
        binding = _binding(metadata={"tools": [TOOL_V1]})
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.system_prompt.startswith("static prompt")
        assert {t.name for t in ai_ctx.tools} == {"outlook"}

    async def test_no_provider_is_byte_identical_to_static_path(self) -> None:
        binding = _binding(metadata={"tools": [TOOL_V1]})
        static_ctx = await _channel()._build_context(make_event(), binding, _ctx(binding))

        assert static_ctx.system_prompt.startswith("static prompt")
        assert {t.name for t in static_ctx.tools} == {"outlook"}

    async def test_provider_tools_flow_through_skill_injection(self) -> None:
        # The resolver result feeds the SAME downstream pipeline (skills,
        # filters) as static tools — it replaces only the base list.
        from pathlib import Path

        from roomkit.skills import SkillRegistry
        from roomkit.skills.models import Skill, SkillMetadata

        async def provider(binding, context):
            return AIChannelTurnConfig(tools=[TOOL_V2])

        registry = SkillRegistry()
        meta = SkillMetadata(name="s1", description="a skill")
        registry._metadata["s1"] = meta
        registry._skills["s1"] = Skill(metadata=meta, instructions="body", path=Path("/tmp"))
        registry._paths["s1"] = Path("/tmp")
        ch = _channel(config_provider=provider, skills=registry)
        binding = _binding()
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        names = {t.name for t in ai_ctx.tools}
        assert "gmail" in names
        assert "activate_skill" in names  # skill infra injected on top
        assert "<available_skills>" in ai_ctx.system_prompt


class TestReasoningOverrides:
    """Reasoning settings ride the same three-tier chain as sampling.

    A thinking model costs 2-3x the tokens and latency of a direct answer, and
    that trade is not the same in an agent's tool loop as in a chat turn — so
    the switch has to be steerable per room and per turn, not only per
    provider instance.
    """

    async def test_channel_default_reaches_context(self) -> None:
        ch = _channel(enable_thinking=False, reasoning_effort="low")
        binding = _binding()
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.enable_thinking is False
        assert ai_ctx.reasoning_effort == "low"

    async def test_unset_stays_none(self) -> None:
        # Nothing set anywhere: the provider is left to defer to the model.
        ch = _channel()
        binding = _binding()
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.enable_thinking is None
        assert ai_ctx.reasoning_effort is None

    async def test_turn_config_overrides_channel_default(self) -> None:
        async def provider(binding, context):
            return AIChannelTurnConfig(enable_thinking=True, reasoning_effort="xhigh")

        ch = _channel(config_provider=provider, enable_thinking=False, reasoning_effort="low")
        binding = _binding()
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.enable_thinking is True
        assert ai_ctx.reasoning_effort == "xhigh"

    async def test_binding_metadata_wins_over_turn_config(self) -> None:
        async def provider(binding, context):
            return AIChannelTurnConfig(enable_thinking=True, reasoning_effort="xhigh")

        ch = _channel(config_provider=provider, enable_thinking=True)
        binding = _binding(metadata={"enable_thinking": False, "reasoning_effort": "low"})
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.enable_thinking is False
        assert ai_ctx.reasoning_effort == "low"

    async def test_metadata_false_is_a_value_not_an_absence(self) -> None:
        # False is the whole point of the override — it must not read as unset
        # and fall through to a channel default that enables thinking.
        ch = _channel(enable_thinking=True)
        binding = _binding(metadata={"enable_thinking": False})
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.enable_thinking is False

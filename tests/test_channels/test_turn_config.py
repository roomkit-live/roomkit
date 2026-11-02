"""Per-turn config provider — dynamic system prompt / tools resolution.

Channel config snapshotted at attach time goes stale (admin edits never
reach existing rooms). ``AIChannel(config_provider=...)`` resolves config
fresh on every turn; binding-metadata overrides stay authoritative for
sampling/prompt, while the provider's toolset REPLACES the deprecated
``binding.metadata["tools"]`` snapshot.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

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


def _schema(field: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {field: {"type": "string"}},
        "required": [field],
        "additionalProperties": False,
    }


# Each per-turn setting the context carries: (key, channel default, turn value).
_SETTINGS = [
    ("system_prompt", "channel prompt", "turn prompt"),
    ("temperature", 0.3, 0.9),
    ("max_tokens", 321, 654),
    ("thinking_budget", 2048, 4096),
    ("enable_thinking", True, False),
    ("reasoning_effort", "low", "high"),
    ("response_schema", _schema("channel"), _schema("turn")),
]


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


def _resolved(ai_ctx: Any, key: str, expected: Any) -> bool:
    """Whether the context carries *expected* for *key*; the prompt grows
    past the resolved one (skills, sandbox, notes), so it is a prefix."""
    value = getattr(ai_ctx, key)
    return value.startswith(expected) if key == "system_prompt" else value == expected


class TestANullDefers:
    """``None`` means "not set here" at every level (RFC Appendix A.9, RMK-346).

    A host that serializes an empty form field as ``null`` into the binding
    metadata must not clear the channel's prompt or settings for that room:
    a binding ``null`` reads like the config provider's ``None``.
    """

    @pytest.mark.parametrize(("key", "channel_value", "turn_value"), _SETTINGS)
    async def test_a_binding_null_keeps_the_channel_default(
        self, key: str, channel_value: Any, turn_value: Any
    ) -> None:
        ch = _channel(**{key: channel_value})
        binding = _binding(metadata={key: None})
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert _resolved(ai_ctx, key, channel_value)

    @pytest.mark.parametrize(("key", "channel_value", "turn_value"), _SETTINGS)
    async def test_a_binding_null_keeps_the_turn_value(
        self, key: str, channel_value: Any, turn_value: Any
    ) -> None:
        async def provider(binding, context):
            return AIChannelTurnConfig(**{key: turn_value})

        ch = _channel(config_provider=provider, **{key: channel_value})
        binding = _binding(metadata={key: None})
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert _resolved(ai_ctx, key, turn_value)

    @pytest.mark.parametrize(("key", "channel_value", "turn_value"), _SETTINGS)
    async def test_a_turn_none_keeps_the_channel_default(
        self, key: str, channel_value: Any, turn_value: Any
    ) -> None:
        async def provider(binding, context):
            return AIChannelTurnConfig(**{key: None})

        ch = _channel(config_provider=provider, **{key: channel_value})
        binding = _binding()
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert _resolved(ai_ctx, key, channel_value)

    async def test_a_zero_budget_in_the_binding_lifts_the_channel_budget(self) -> None:
        ch = _channel(thinking_budget=2048)
        binding = _binding(metadata={"thinking_budget": 0})
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.thinking_budget == 0

    async def test_a_null_tools_declares_no_toolset(self) -> None:
        ch = _channel()
        binding = _binding(metadata={"tools": None})
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.tools == []


class TestABindingPromptThatIsNoText:
    """A binding's system prompt is read as AIContext reads a text (RFC §6.4,
    Appendix A.9, RMK-1237): any other value fails the turn with pydantic's
    validation error, every setting it refuses named, before the turn reads
    the prompt as a text, where 5 raised a TypeError and 0 lifted the
    channel's prompt unseen."""

    @pytest.mark.parametrize("prompt", [5, 0, False, [], {"a": 1}])
    async def test_a_prompt_that_is_no_text_fails_the_turn(self, prompt: Any) -> None:
        ch = _channel(system_prompt="The channel's prompt.")
        binding = _binding(metadata={"system_prompt": prompt, "temperature": "hot"})

        with pytest.raises(ValidationError) as raised:
            await ch._build_context(make_event(), binding, _ctx(binding))

        refused = [error["loc"] for error in raised.value.errors()]
        assert refused == [("system_prompt",), ("temperature",)]


async def _off_turn(binding, context):
    return AIChannelTurnConfig(enable_thinking=False)


async def _budget_turn(binding, context):
    return AIChannelTurnConfig(thinking_budget=4096)


class TestThinkingOffAcrossLevels:
    """An ``enable_thinking: false`` turns off a budget a less specific level
    set (RFC §6.7, Appendix A.9, RMK-346): the budget states the switch first,
    so without this a room that says off on a channel built with a budget,
    the form a host sends with an empty budget field, would think anyway."""

    @pytest.mark.parametrize(
        ("metadata", "config_provider", "budget"),
        [
            ({"enable_thinking": False}, None, None),
            ({"enable_thinking": False, "thinking_budget": None}, None, None),
            ({}, _off_turn, None),
            ({"enable_thinking": False}, _budget_turn, None),
            ({"enable_thinking": False, "thinking_budget": 1024}, None, 1024),
            ({"thinking_budget": 1024}, _off_turn, 1024),
        ],
        ids=[
            "room-off-over-channel-budget",
            "room-off-with-null-budget",
            "turn-off-over-channel-budget",
            "room-off-over-turn-budget",
            "same-level-budget-decides",
            "more-specific-budget-decides",
        ],
    )
    async def test_off_turns_off_a_less_specific_budget(
        self, metadata: dict[str, Any], config_provider: Any, budget: int | None
    ) -> None:
        ch = _channel(thinking_budget=2048, config_provider=config_provider)
        binding = _binding(metadata=metadata)
        ai_ctx = await ch._build_context(make_event(), binding, _ctx(binding))

        assert ai_ctx.thinking_budget == budget
        assert ai_ctx.enable_thinking is False

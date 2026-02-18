"""Tests for Agent (AIChannel subclass with structured identity)."""

from __future__ import annotations

from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event


def _make_agent(**kwargs) -> Agent:
    """Create an Agent with a MockAIProvider."""
    defaults = {
        "provider": MockAIProvider(responses=["ok"]),
    }
    defaults.update(kwargs)
    return Agent("agent-test", **defaults)


def _binding(room_id: str = "r1") -> ChannelBinding:
    return ChannelBinding(
        channel_id="agent-test",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


class TestAgentBasics:
    def test_agent_is_aichannel(self):
        agent = _make_agent()
        assert isinstance(agent, AIChannel)

    def test_defaults_all_none(self):
        agent = _make_agent()
        assert agent.role is None
        assert agent.description is None
        assert agent.scope is None
        assert agent.voice is None
        assert agent.greeting is None
        assert agent.language is None

    def test_fields_stored(self):
        agent = _make_agent(
            role="Triage",
            description="Routes callers",
            scope="Financial only",
            voice="voice-id-123",
            greeting="Welcome! How can I help?",
        )
        assert agent.role == "Triage"
        assert agent.description == "Routes callers"
        assert agent.scope == "Financial only"
        assert agent.voice == "voice-id-123"
        assert agent.greeting == "Welcome! How can I help?"

    def test_greeting_not_in_identity_block(self):
        agent = _make_agent(greeting="Hello there!")
        block = agent._build_identity_block()
        assert block is None  # greeting is not an identity field


class TestLanguage:
    def test_language_in_identity_block(self):
        agent = _make_agent(role="Triage", language="French")
        block = agent._build_identity_block()
        assert "Language: Always respond in French" in block

    def test_language_only_identity_block(self):
        agent = _make_agent(language="Spanish")
        block = agent._build_identity_block()
        assert block is not None
        assert "Language: Always respond in Spanish" in block

    def test_language_override_in_identity_block(self):
        agent = _make_agent(role="Triage", language="English")
        block = agent._build_identity_block(language="French")
        assert "Language: Always respond in French" in block
        assert "English" not in block

    def test_language_stored(self):
        agent = _make_agent(language="fr")
        assert agent.language == "fr"


class TestIdentityBlock:
    def test_identity_block_role_only(self):
        agent = _make_agent(role="Triage receptionist")
        block = agent._build_identity_block()
        assert block is not None
        assert "Role: Triage receptionist" in block
        assert "Description:" not in block
        assert "Scope:" not in block

    def test_identity_block_all_fields(self):
        agent = _make_agent(
            role="Triage",
            description="Routes callers",
            scope="Financial only",
        )
        block = agent._build_identity_block()
        assert block is not None
        assert "Role: Triage" in block
        assert "Description: Routes callers" in block
        assert "Scope: Financial only" in block
        assert "--- Agent Identity ---" in block

    def test_voice_not_in_identity_block(self):
        agent = _make_agent(voice="voice-id-123")
        block = agent._build_identity_block()
        # voice is not an identity field — block should be None
        assert block is None

    def test_no_metadata_returns_none(self):
        agent = _make_agent()
        assert agent._build_identity_block() is None


class TestBuildContext:
    async def test_build_context_appends_identity(self):
        agent = _make_agent(
            role="Advisor",
            description="Gives financial advice",
            system_prompt="Be helpful.",
        )
        event = make_event(room_id="r1")
        binding = _binding()
        ctx = RoomContext(room=Room(id="r1"), bindings=[binding])

        ai_ctx = await agent._build_context(event, binding, ctx)

        assert isinstance(ai_ctx, AIContext)
        assert ai_ctx.system_prompt is not None
        assert ai_ctx.system_prompt.startswith("Be helpful.")
        assert "Role: Advisor" in ai_ctx.system_prompt
        assert "Description: Gives financial advice" in ai_ctx.system_prompt

    async def test_build_context_no_metadata_unchanged(self):
        agent = _make_agent(system_prompt="Be helpful.")
        event = make_event(room_id="r1")
        binding = _binding()
        ctx = RoomContext(room=Room(id="r1"), bindings=[binding])

        ai_ctx = await agent._build_context(event, binding, ctx)

        assert ai_ctx.system_prompt == "Be helpful."

    async def test_build_context_respects_binding_override(self):
        agent = _make_agent(
            role="Advisor",
            system_prompt="Default prompt.",
        )
        event = make_event(room_id="r1")
        binding = _binding()
        binding.metadata["system_prompt"] = "Override prompt."
        ctx = RoomContext(room=Room(id="r1"), bindings=[binding])

        ai_ctx = await agent._build_context(event, binding, ctx)

        assert ai_ctx.system_prompt is not None
        assert ai_ctx.system_prompt.startswith("Override prompt.")
        assert "Role: Advisor" in ai_ctx.system_prompt

    async def test_build_context_no_system_prompt_with_identity(self):
        agent = _make_agent(role="Advisor")
        event = make_event(room_id="r1")
        binding = _binding()
        ctx = RoomContext(room=Room(id="r1"), bindings=[binding])

        ai_ctx = await agent._build_context(event, binding, ctx)

        assert ai_ctx.system_prompt is not None
        assert "Role: Advisor" in ai_ctx.system_prompt


class TestConfigOnly:
    def test_config_only_no_provider(self):
        agent = Agent("agent-test", role="Triage", system_prompt="Hello.")
        assert agent.is_config_only is True
        assert isinstance(agent, AIChannel)

    def test_config_only_with_provider_is_false(self):
        agent = _make_agent(role="Triage")
        assert agent.is_config_only is False

    async def test_config_only_generate_raises(self):
        import pytest

        agent = Agent("agent-test", role="Triage", system_prompt="Hello.")
        ai_ctx = AIContext(messages=[], system_prompt="test")
        with pytest.raises(RuntimeError, match="no AI provider"):
            await agent._provider.generate(ai_ctx)

    def test_config_only_identity_block_works(self):
        agent = Agent("agent-test", role="Triage", description="Routes callers")
        block = agent._build_identity_block()
        assert block is not None
        assert "Role: Triage" in block
        assert "Description: Routes callers" in block

    def test_config_only_fields(self):
        agent = Agent(
            "agent-test",
            role="Triage",
            description="Routes callers",
            scope="Financial only",
            voice="Aoede",
            system_prompt="Greet callers.",
        )
        assert agent.role == "Triage"
        assert agent.description == "Routes callers"
        assert agent.scope == "Financial only"
        assert agent.voice == "Aoede"
        assert agent._system_prompt == "Greet callers."

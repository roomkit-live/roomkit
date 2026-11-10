"""Agent — AIChannel subclass with structured identity metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_router import AIRouter
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import AgentIdentity, ChannelDeclaration
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse

if TYPE_CHECKING:
    from roomkit.models.context import RoomContext


class _NullAIProvider(AIProvider):
    """Placeholder provider for config-only agents (speech-to-speech mode).

    Config-only agents are used with :class:`RealtimeVoiceChannel` where
    the AI reasoning happens inside the realtime provider (Gemini Live,
    OpenAI Realtime), not in a separate AIChannel.
    """

    @property
    def model_name(self) -> str:
        return "_null"

    async def generate(self, context: AIContext) -> AIResponse:
        msg = (
            "This Agent has no AI provider. "
            "Config-only agents are used with RealtimeVoiceChannel "
            "for speech-to-speech orchestration."
        )
        raise RuntimeError(msg)


class Agent(AIChannel):
    """AI agent with structured identity metadata.

    Extends :class:`AIChannel` with ``role``, ``description``, ``scope``,
    ``voice``, and ``greeting`` fields.  The first three are auto-injected
    into the system prompt as an identity block, unless
    ``identity_in_prompt=False`` (a host that renders identity in its own
    prompt keeps the fields readable without the block); ``voice`` is read by
    :meth:`ConversationPipeline.install` to auto-wire the voice map;
    ``greeting`` is spoken directly via TTS when a new voice session
    becomes ready (controlled by ``auto_greet``).

    When ``provider`` is omitted the agent is **config-only** — it holds
    identity and prompt data for speech-to-speech orchestration via
    :class:`RealtimeVoiceChannel` but cannot generate responses itself.

    Example::

        # Full agent (STT → LLM → TTS)
        triage = Agent(
            "agent-triage",
            provider=GeminiAIProvider(config),
            role="Triage receptionist",
            description="Routes callers to the right specialist",
            voice="21m00Tcm4TlvDq8ikWAM",
            system_prompt="Greet callers warmly.",
        )

        # Config-only agent (speech-to-speech)
        triage = Agent(
            "agent-triage",
            role="Triage receptionist",
            description="Routes callers to the right specialist",
            voice="Aoede",
            system_prompt="Greet callers warmly.",
        )
    """

    def __init__(
        self,
        channel_id: str,
        *,
        provider: AIProvider | None = None,
        router: AIRouter | None = None,
        name: str | None = None,
        role: str | None = None,
        description: str | None = None,
        purpose: str | None = None,
        scope: str | None = None,
        voice: str | None = None,
        greeting: str | None = None,
        language: str | None = None,
        auto_greet: bool = True,
        identity_in_prompt: bool = True,
        **kwargs: Any,
    ) -> None:
        if provider is None and router is None:
            provider = _NullAIProvider()
        super().__init__(channel_id, provider=provider, router=router, **kwargs)
        # Human-readable display name (e.g. "Research Assistant"), distinct from the
        # channel_id and from ``role`` (the agent's instructions). Used to label
        # the agent in orchestration timelines instead of a truncated role.
        self.name = name
        self.role = role
        self.description = description
        # The agent's own configured instructions, in concise form — surfaced to
        # an orchestrator (e.g. a Supervisor) so it can frame tasks knowing what
        # this agent does and which tools it uses, without injecting the full
        # runtime system prompt. Distinct from ``description`` (a short label).
        self.purpose = purpose
        self.scope = scope
        self.voice = voice
        self.greeting = greeting
        self.language = language
        self.auto_greet = auto_greet
        # False: a host that renders the agent's identity in its own prompt
        # turns the block off everywhere RoomKit writes it, the fields kept.
        self.identity_in_prompt = identity_in_prompt

    def declaration(self) -> ChannelDeclaration:
        identity = AgentIdentity(
            name=self.name,
            role=self.role,
            description=self.description,
            scope=self.scope,
            greeting=self.greeting,
            language=self.language,
            purpose=self.purpose,
            voice=self.voice,
            auto_greet=self.auto_greet,
        )
        return super().declaration().model_copy(update={"identity": identity})

    @property
    def is_config_only(self) -> bool:
        """Whether this agent has no AI provider (config-only mode)."""
        return self._router is None and isinstance(self._provider, _NullAIProvider)

    @property
    def system_prompt(self) -> str | None:
        """The base system prompt for this agent."""
        return self._system_prompt

    def build_identity_block(self, language: str | None = None) -> str | None:
        """Public access to :meth:`_build_identity_block` for orchestration callers."""
        return self._build_identity_block(language=language)

    def _build_identity_block(self, language: str | None = None) -> str | None:
        """Build the identity block appended to the system prompt.

        Args:
            language: Override language (e.g. from conversation state).
                Falls back to ``self.language`` when ``None``.

        Returns ``None`` when all identity fields are ``None``, or when
        ``identity_in_prompt`` is off: the turn's prompt, a handoff's and a
        realtime pipeline's all read the block here.
        """
        if not self.identity_in_prompt:
            return None
        lines: list[str] = []
        if self.role is not None:
            lines.append(f"Role: {self.role}")
        if self.description is not None:
            lines.append(f"Description: {self.description}")
        if self.scope is not None:
            lines.append(f"Scope: {self.scope}")
        lang = language or self.language
        if lang is not None:
            lines.append(f"Language: Always respond in {lang}")
        if not lines:
            return None
        return "\n--- Agent Identity ---\n" + "\n".join(lines)

    def _prompt_identity(self, context: RoomContext) -> str | None:
        """The identity block, in the room's language when the conversation
        state sets one (``handler.set_language()``), else ``self.language``;
        written into the prompt before the turn is measured (RFC §20)."""
        from roomkit.orchestration.state import get_conversation_state

        state = get_conversation_state(context.room)
        room_language = state.context.get("language") if state.context else None
        return self._build_identity_block(language=room_language)

"""Agent — AIChannel subclass with structured identity metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse

if TYPE_CHECKING:
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent


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
    into the system prompt as an identity block; ``voice`` is read by
    :meth:`ConversationPipeline.install` to auto-wire the voice map;
    ``greeting`` is played once when a new session starts via
    :meth:`HandoffHandler.send_greeting`.

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
        role: str | None = None,
        description: str | None = None,
        scope: str | None = None,
        voice: str | None = None,
        greeting: str | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(channel_id, provider=provider or _NullAIProvider(), **kwargs)
        self.role = role
        self.description = description
        self.scope = scope
        self.voice = voice
        self.greeting = greeting
        self.language = language

    @property
    def is_config_only(self) -> bool:
        """Whether this agent has no AI provider (config-only mode)."""
        return isinstance(self._provider, _NullAIProvider)

    @property
    def system_prompt(self) -> str | None:
        """The base system prompt for this agent."""
        return self._system_prompt

    def build_identity_block(self, language: str | None = None) -> str | None:
        """Build the identity block appended to the system prompt.

        Args:
            language: Override language (e.g. from conversation state).
                Falls back to ``self.language`` when ``None``.

        Returns ``None`` when all identity fields are ``None``.
        """
        return self._build_identity_block(language=language)

    def _build_identity_block(self, language: str | None = None) -> str | None:
        """Build the identity block appended to the system prompt.

        Args:
            language: Override language (e.g. from conversation state).
                Falls back to ``self.language`` when ``None``.

        Returns ``None`` when all identity fields are ``None``.
        """
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

    async def _build_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext:
        """Build AI context, appending the identity block to the system prompt.

        Checks conversation state for a per-room language override
        (set via ``handler.set_language()``), falling back to ``self.language``.
        """
        from roomkit.orchestration.state import get_conversation_state

        ai_context = await super()._build_context(event, binding, context)
        # Per-room language override from conversation state
        state = get_conversation_state(context.room)
        room_language = state.context.get("language") if state.context else None
        identity = self._build_identity_block(language=room_language)
        if identity is not None:
            prompt = (ai_context.system_prompt or "") + identity
            ai_context = ai_context.model_copy(update={"system_prompt": prompt})
        return ai_context

"""Anam AI provider configuration."""

from __future__ import annotations

from pydantic import BaseModel


class AnamConfig(BaseModel):
    """Configuration for the Anam AI Realtime provider.

    Anam AI renders photorealistic talking-head avatars via a cloud
    pipeline (STT → LLM → TTS → face animation) and delivers
    synchronized audio+video over WebRTC.

    Either ``persona_id`` (pre-configured on Anam dashboard) or inline
    persona fields (``avatar_id``, ``voice_id``, ``llm_id``) can be
    used — but not both.

    Attributes:
        api_key: Anam API key.
        persona_id: Pre-configured persona ID on Anam dashboard.
        avatar_id: Inline persona: avatar model ID.
        voice_id: Inline persona: voice ID.
        llm_id: Inline persona: LLM model ID.
        system_prompt: Default system prompt (overridable per-session).
        language_code: BCP-47 language code for STT/TTS.
        enable_audio_passthrough: If True, bypass Anam's built-in STT
            and send raw PCM audio for transcription externally.
        timeout: Connection timeout in seconds.
    """

    api_key: str
    persona_id: str | None = None
    avatar_id: str | None = None
    avatar_model: str | None = None
    voice_id: str | None = None
    llm_id: str | None = None
    system_prompt: str | None = None
    language_code: str = "en"
    enable_audio_passthrough: bool = False
    timeout: float = 30.0

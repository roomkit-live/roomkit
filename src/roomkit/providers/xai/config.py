"""xAI Realtime provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class XAIRealtimeConfig(BaseModel):
    """Configuration for the xAI Grok Realtime provider.

    Attributes:
        api_key: xAI API key (or set ``XAI_API_KEY`` env var).
        model: Model identifier for the realtime session.
        base_url: WebSocket base URL for the xAI Realtime API.
        voice: Default voice — ``eve``, ``ara``, ``rex``, ``sal``, ``leo``.
        transcription_model: Model used for input audio transcription.
    """

    api_key: SecretStr
    model: str = "grok-2-audio"
    base_url: str = "wss://api.x.ai/v1/realtime"
    voice: str = "eve"
    transcription_model: str = "grok-2-audio"

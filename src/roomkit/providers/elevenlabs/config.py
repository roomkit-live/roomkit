"""ElevenLabs Conversational AI configuration."""

from __future__ import annotations

from pydantic import BaseModel


class ElevenLabsRealtimeConfig(BaseModel):
    """Configuration for the ElevenLabs Conversational AI realtime provider.

    Attributes:
        api_key: ElevenLabs API key.
        agent_id: Pre-configured agent ID from the ElevenLabs dashboard.
        requires_auth: When True, use a signed URL for authentication
            (recommended for client-facing deployments).  When False,
            the ``api_key`` is sent as a header on the WebSocket.
        base_url: WebSocket base URL.  Override for regional endpoints
            (e.g. ``"wss://api.eu.residency.elevenlabs.io"`` for EU).
    """

    api_key: str
    agent_id: str
    requires_auth: bool = False
    base_url: str = "wss://api.elevenlabs.io"

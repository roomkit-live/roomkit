"""ElevenLabs Conversational AI configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


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
        tool_timeout_s: How long a client tool call may stay pending before
            the provider answers the agent with an error.  The ElevenLabs
            agent applies its own per-tool timeout server-side; keep this
            one above it so the agent's own timeout is what the user hears.
        response_idle_ms: Quiet period after the last audio chunk of a turn
            before the provider declares the response finished.  ElevenLabs
            sends no end-of-audio marker, so the end of a turn is inferred
            from silence on the audio stream.  A tool call in flight holds
            the turn open — the agent keeps speaking once the result lands.
    """

    api_key: SecretStr
    agent_id: str
    requires_auth: bool = False
    base_url: str = "wss://api.elevenlabs.io"
    tool_timeout_s: float = 30.0
    response_idle_ms: int = 800

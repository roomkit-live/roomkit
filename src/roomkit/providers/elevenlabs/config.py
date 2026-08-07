"""ElevenLabs Conversational AI configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr, field_validator


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

    api_key: SecretStr = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    requires_auth: bool = False
    base_url: str = Field(default="wss://api.elevenlabs.io", min_length=1)
    tool_timeout_s: float = Field(default=30.0, gt=0)
    response_idle_ms: int = Field(default=800, gt=0)

    @field_validator("base_url")
    @classmethod
    def _validate_websocket_url(cls, value: str) -> str:
        if not value.startswith(("ws://", "wss://")):
            raise ValueError("base_url must use ws:// or wss://")
        return value

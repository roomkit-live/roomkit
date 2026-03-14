"""NVIDIA PersonaPlex provider configuration."""

from __future__ import annotations

from pydantic import BaseModel


class PersonaPlexConfig(BaseModel):
    """Configuration for the PersonaPlex Realtime provider.

    PersonaPlex is a self-hosted speech-to-speech model from NVIDIA.
    It requires a running PersonaPlex server (GPU recommended: A100/H100).

    Attributes:
        server_url: WebSocket URL of the PersonaPlex server.
        ssl_verify: Whether to verify SSL certificates. Set to False for
            self-signed certs (default for PersonaPlex dev servers).
        default_voice_prompt: Default voice prompt file (e.g. ``NATF2.pt``).
            Can be overridden per-session via the ``voice`` parameter.
        response_end_timeout: Seconds of silence after the last audio/text
            token before firing the ``response_end`` callback.
    """

    server_url: str = "wss://localhost:8998/api/chat"
    ssl_verify: bool = False
    default_voice_prompt: str = "NATF2.pt"
    response_end_timeout: float = 1.0
    seed: int = -1

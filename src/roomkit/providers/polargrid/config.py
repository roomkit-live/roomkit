"""PolarGrid provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class PolarGridConfig(BaseModel):
    """PolarGrid AI provider configuration.

    Wraps the PolarGrid chat-completions endpoint via the official
    ``polargrid-sdk`` async client. PolarGrid is a Canadian-hosted
    inference network with regional edges in Toronto, Vancouver, and
    Montreal — useful when data residency on Canadian soil matters.

    Attributes:
        api_key: PolarGrid API key (``pg_...``), sent as a Bearer token.
        model: Model identifier (e.g. ``"qwen-3.5-9b"``, ``"qwen-3.5-27b"``).
        region: Region to pin. One of ``"toronto"``/``"vancouver"``/
            ``"montreal"`` (or the IDs ``"yto-01"``/``"yvr-02"``/
            ``"yul-01"``). ``None`` lets the SDK auto-route to the
            fastest edge — convenient for dev, but pin a region in
            production when residency matters.
        max_tokens: Maximum tokens in the response. ``None`` lets the
            server pick its default (the API caps at 4096).
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling probability (0.0-1.0).
        timeout: HTTP request timeout in seconds.
        max_retries: SDK-level retry count. Default 0 because RoomKit's
            RetryPolicy handles retries at the right layer with proper
            backoff and fallback.
        debug: Enable verbose logging in the underlying SDK.
    """

    api_key: SecretStr
    model: str = "qwen-3.5-9b"
    region: str | None = None
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: float = 30.0
    max_retries: int = 0
    debug: bool = False

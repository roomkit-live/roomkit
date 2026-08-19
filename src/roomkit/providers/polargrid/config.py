"""PolarGrid provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.polargrid.models import region_choices, resolve_region_id


class PolarGridConfig(BaseModel):
    """PolarGrid AI provider configuration.

    Wraps the PolarGrid chat-completions endpoint via the official
    ``polargrid-sdk`` async client. PolarGrid is a Canadian-hosted
    inference network with regional edges in Toronto, Vancouver, and
    Montreal — useful when data residency on Canadian soil matters.

    Attributes:
        api_key: PolarGrid API key (``pg_...``), sent as a Bearer token.
        model: Model identifier. Defaults to ``"qwen-3.8-27b"``, the LLM
            the Toronto edge (the SDK's default region) serves — verified
            live 2026-08-19. The fleet is mid-rollout from
            ``"qwen-3.5-27b"``, which only ``yul-01`` still serves; pin
            that model explicitly if you are pinned to ``yul-01``. The
            catalog varies per edge — call ``list_models()`` on the raw
            SDK client to enumerate what's loaded.
        region: Region to pin. One of ``"toronto"``/``"vancouver"``/
            ``"montreal"`` (or the IDs ``"yto-01"``/``"yvr-02"``/
            ``"yul-01"``). ``None`` lets the SDK auto-route to the
            nearest edge that already serves the configured model
            (``routing_model``, polargrid-sdk 0.10.0) — convenient for
            dev, but pin a region in production when residency matters.
        max_tokens: Maximum tokens in the response. ``None`` lets the
            server pick its default (the API caps at 4096).
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling probability (0.0-1.0).
        thinking: Toggle qwen's reasoning via the ``enable_thinking``
            request flag (polargrid-sdk 0.8.5+). ``True`` turns reasoning
            on — qwen emits it inline as ``<think>...</think>``, which the
            provider surfaces as ``AIResponse.thinking`` /
            ``StreamThinkingDelta``. ``False`` turns it off. ``None``
            (default) leaves the flag unset (model/edge default). Thinking
            responses are larger and slower, so raise ``timeout`` and
            ``max_tokens`` when enabling it.
        timeout: HTTP request timeout in seconds.
        max_retries: SDK-level retry count. Default 0 because RoomKit's
            RetryPolicy handles retries at the right layer with proper
            backoff and fallback.
        debug: Enable verbose logging in the underlying SDK.
    """

    api_key: SecretStr
    model: str = "qwen-3.8-27b"
    region: str | None = None
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    thinking: bool | None = None
    timeout: float = 30.0
    max_retries: int = 0
    debug: bool = False

    @field_validator("region")
    @classmethod
    def _validate_region(cls, value: str | None) -> str | None:
        """Reject an unknown pinned region at construction.

        The SDK does not validate ``region``: an unknown value is interpolated
        straight into ``https://api.<region>.edge.polargrid.ai``, so a typo
        like ``"yul-2"`` silently yields an unroutable host that only fails
        much later with an opaque DNS error. Fail here instead, loudly.
        """
        if value is None or resolve_region_id(value) is not None:
            return value
        raise ValueError(f"unknown PolarGrid region {value!r}. Valid: {region_choices()}")

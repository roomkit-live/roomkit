"""Anthropic provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class AnthropicConfig(BaseModel):
    """Anthropic AI provider configuration."""

    api_key: SecretStr
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 60.0
    """Request timeout in seconds (default 60s)."""
    base_url: str | None = None
    """Override the base URL (e.g., for Claude Code sandbox proxy)."""
    extra_headers: dict[str, str] | None = None
    """Extra headers sent with every request (e.g., X-Tenant-ID)."""
    use_adaptive_thinking: bool = False
    """Send extended thinking as ``{"type": "adaptive"}`` instead of the
    deprecated ``{"type": "enabled", "budget_tokens": N}``. Anthropic's newer
    models (Opus 4.7/4.8, Fable 5) reject ``budget_tokens`` with HTTP 400;
    adaptive is the modern, recommended shape on Opus 4.6+ and Sonnet 4.6.
    Leave False for older models (Sonnet 4.5 and earlier) that only accept
    ``budget_tokens``."""
    supports_custom_temperature: bool = True
    """When False, ``temperature`` is omitted from requests. Anthropic's
    reasoning models (Opus 4.7/4.8, Fable 5) removed the sampling parameters
    and reject ``temperature`` with HTTP 400."""

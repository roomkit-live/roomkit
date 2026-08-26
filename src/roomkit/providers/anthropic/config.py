"""Anthropic provider configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, SecretStr, field_validator

_ADAPTIVE_THINKING_MODEL_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


class AnthropicConfig(BaseModel):
    """Anthropic AI provider configuration."""

    api_key: SecretStr
    model: str
    """Model identifier. Required so upgrading RoomKit cannot silently change
    a caller's model, cost, latency, or behavior."""
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 60.0
    """Request timeout in seconds (default 60s)."""
    base_url: str | None = None
    """Override the base URL (e.g., for Claude Code sandbox proxy).

    The SDK appends ``/v1/messages`` to this, so a value that already ends in
    that path is dropped down to its parent: Microsoft Foundry documents the
    Claude surface as the whole ``<resource>/anthropic/v1/messages`` URL, and
    pasting it verbatim would otherwise post to ``/v1/messages/v1/messages``.
    A bare trailing ``/v1`` is deliberately left alone — unlike the full path
    it is not unambiguously wrong, and a gateway may route on it."""
    extra_headers: dict[str, str] | None = None
    """Extra headers sent with every request (e.g., X-Tenant-ID)."""
    enable_prompt_caching: bool = True
    """Apply Anthropic prompt caching (explicit ``cache_control`` markers) to
    the stable request prefix — tools, system prompt, and the conversation
    suffix. Every tool-loop round re-sends the full context; without markers
    it is billed at the full input rate on every round, with them the prefix
    re-reads at the cached rate (10%). Disable for proxies that reject
    ``cache_control`` blocks."""
    use_adaptive_thinking: bool = False
    """Send extended thinking as ``{"type": "adaptive"}`` instead of the
    deprecated ``{"type": "enabled", "budget_tokens": N}``. Anthropic's newer
    models reject ``budget_tokens`` with HTTP 400. Official modern models are
    profiled automatically; an explicit value or custom ``base_url`` is left
    untouched for compatibility with proxies and older deployments."""
    supports_custom_temperature: bool = True
    """When False, ``temperature`` is omitted from requests. Anthropic's
    modern reasoning models removed the sampling parameters and reject
    ``temperature`` with HTTP 400. Official modern models are profiled
    automatically unless this field is explicitly set."""

    @field_validator("base_url")
    @classmethod
    def _drop_the_path_the_sdk_appends(cls, value: str | None) -> str | None:
        """Reduce a pasted endpoint to the base the SDK expects."""
        if value is None:
            return value
        trimmed = value.strip().rstrip("/")
        suffix = "/v1/messages"
        if trimmed.lower().endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
        return trimmed or value.strip()

    def model_post_init(self, __context: Any) -> None:
        """Apply safe defaults for Anthropic's modern first-party models."""
        if self.base_url is not None or not self.model.startswith(
            _ADAPTIVE_THINKING_MODEL_PREFIXES
        ):
            return
        if "use_adaptive_thinking" not in self.model_fields_set:
            self.use_adaptive_thinking = True
        if "supports_custom_temperature" not in self.model_fields_set:
            self.supports_custom_temperature = False

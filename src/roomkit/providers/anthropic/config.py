"""Anthropic provider configuration."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.vendor_endpoint import ANTHROPIC_BASE_URL, is_vendor_endpoint

# Claude models that still take ``temperature`` and the ``budget_tokens``
# thinking shape: the 4.6 generation and everything before it. Every other
# ``claude-`` id gets the modern contract (adaptive thinking, no sampling
# parameters), which Anthropic has applied to each release since Opus 4.7.
# Listing the old models rather than the new ones is what makes a release
# nobody catalogued yet — Opus 5.5 shipped one — work on the day it ships:
# the legacy set is closed, the modern one keeps growing. (4.6 accepts both
# thinking shapes; ``budget_tokens`` is deprecated there, not refused.)
_LEGACY_MODEL = re.compile(r"^claude-(?:2|3|instant|(?:opus|sonnet|haiku)-4-(?:[0-6]\b|\d{8}))")


def _is_modern_claude(model: str) -> bool:
    return model.startswith("claude-") and not _LEGACY_MODEL.match(model)


class AnthropicConfig(BaseModel):
    """Anthropic AI provider configuration."""

    api_key: SecretStr
    model: str
    """Model identifier. Required so upgrading RoomKit cannot silently change
    a caller's model, cost, latency, or behavior."""
    max_tokens: int = 1024
    timeout: float = 60.0
    """Request timeout in seconds (default 60s)."""
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, kept apart from ``timeout`` so a host
    that no longer accepts connections is given up on in seconds rather
    than after the read budget. The SDK's own default."""
    max_retries: int = 0
    """SDK-level retry count. Default 0 because RoomKit's RetryPolicy
    handles retries at the right layer with proper backoff and fallback; the
    SDK's own retries would multiply its attempts (RMK-509)."""
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
        if not is_vendor_endpoint(self.base_url, ANTHROPIC_BASE_URL):
            return
        if not _is_modern_claude(self.model):
            return
        if "use_adaptive_thinking" not in self.model_fields_set:
            self.use_adaptive_thinking = True
        if "supports_custom_temperature" not in self.model_fields_set:
            self.supports_custom_temperature = False

"""DeepSeek provider configuration."""

from __future__ import annotations

from roomkit.providers.openai.config import OpenAIConfig


class DeepSeekConfig(OpenAIConfig):
    """DeepSeek AI provider configuration.

    DeepSeek serves an OpenAI-compatible Chat Completions API at
    ``https://api.deepseek.com``, so this **subclasses** :class:`OpenAIConfig`
    and inherits every request field (``temperature``, ``include_stream_usage``,
    ``use_max_completion_tokens``, ``supports_custom_temperature``,
    ``extra_body`` …). Inheriting — rather than re-declaring them — keeps the
    two configs from drifting apart: any field the inherited
    :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` reads is guaranteed
    to exist here.

    DeepSeek also fronts an Anthropic-compatible endpoint at
    ``/anthropic``. RoomKit talks to the OpenAI-shaped one because it is the
    richer of the two: the Anthropic path drops ``cache_control``, images and
    the models listing, and ignores ``budget_tokens``.

    Only the endpoint and DeepSeek's own thinking switch are added on top.
    """

    base_url: str = "https://api.deepseek.com/v1"
    """DeepSeek's OpenAI-compatible endpoint. Override only to point at a
    proxy."""

    model: str
    """DeepSeek model id — ``"deepseek-v4-pro"`` or ``"deepseek-v4-flash"``.
    Required, so upgrading RoomKit cannot silently change a caller's model,
    cost, latency, or behavior. See
    :mod:`roomkit.providers.deepseek.models` for the curated catalog."""

    enable_thinking: bool | None = None
    """Force DeepSeek's thinking mode on (``True``) or off (``False``) for
    every turn. ``None`` sends nothing and leaves the model's own default,
    which is thinking **on** for both V4 models. A per-turn
    ``AIContext.thinking_budget`` overrides this."""

    reasoning_effort: str | None = None
    """Reasoning depth — ``"low"`` | ``"high"`` | ``"max"``. DeepSeek takes it
    nested inside its ``thinking`` object, not as OpenAI's top-level field, and
    it is the *only* lever over reasoning length: token budgets are ignored by
    this API. ``None`` leaves the model's default. Redeclared here for the
    narrower value set — the inherited field documents OpenAI's, which includes
    values DeepSeek rejects."""

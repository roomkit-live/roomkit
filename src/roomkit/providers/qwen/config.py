"""Qwen (Alibaba Cloud Model Studio) provider configuration."""

from __future__ import annotations

from roomkit.providers.openai.config import OpenAIConfig


class QwenConfig(OpenAIConfig):
    """Qwen AI provider configuration.

    Alibaba Cloud Model Studio serves the Qwen lineup behind an
    OpenAI-compatible Chat Completions API, so this **subclasses**
    :class:`OpenAIConfig` and inherits every request field (``temperature``,
    ``include_stream_usage``, ``use_max_completion_tokens``,
    ``supports_custom_temperature``, ``extra_body`` …). Inheriting — rather
    than re-declaring them — keeps the two configs from drifting apart: any
    field the inherited
    :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` reads is guaranteed
    to exist here.

    Model Studio also hosts third-party models (DeepSeek, Kimi, GLM, MiniMax)
    on the same endpoint. This provider is scoped to the Qwen family it is
    named for — its catalog, its prices and its thinking switch are Qwen's.
    Reaching the others through it would work on the wire and report the wrong
    metadata for every one of them.

    Only the endpoint and Qwen's own thinking switch are added on top.
    """

    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    """Model Studio's OpenAI-compatible endpoint, international deployment.

    Alibaba serves several, and the account decides which one holds your key:

    * international / Singapore — the default above
    * China (Beijing) — ``https://dashscope.aliyuncs.com/compatible-mode/v1``
    * US (Virginia) — ``https://dashscope-us.aliyuncs.com/compatible-mode/v1``
    * workspace-scoped, any region — ``https://{WorkspaceId}.{region}.maas
      .aliyuncs.com/compatible-mode/v1``

    The workspace form takes the id from the Model Studio console, which is why
    it cannot be a default: there is no correct value to guess. Prices differ by
    deployment too — :mod:`roomkit.providers.qwen.models` carries the
    international list.
    """

    model: str
    """Qwen model id — e.g. ``"qwen3.7-max"``, ``"qwen3.7-plus"``,
    ``"qwen3-coder-plus"``. Required, so upgrading RoomKit cannot silently
    change a caller's model, cost, latency, or behavior. See
    :mod:`roomkit.providers.qwen.models` for the curated catalog."""

    enable_thinking: bool | None = None
    """Force Qwen's thinking mode on (``True``) or off (``False``) for every
    turn. ``None`` sends nothing and leaves the model's own default, which
    differs across the lineup. A per-turn ``AIContext.thinking_budget``
    overrides this."""

    reasoning_effort: str | None = None
    """Not sent to Model Studio, and kept only because the inherited config
    declares it. Qwen sizes thinking with ``enable_thinking`` and a token
    ``thinking_budget``, not with an effort tier — Model Studio accepts
    ``reasoning_effort`` only for the third-party DeepSeek models it also
    hosts, which this provider does not catalog. Use ``enable_thinking`` here
    or ``AIContext.thinking_budget`` per turn."""

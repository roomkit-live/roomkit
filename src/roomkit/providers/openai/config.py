"""OpenAI provider configuration."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.vendor_endpoint import OPENAI_BASE_URL, is_vendor_endpoint

_MAX_COMPLETION_TOKEN_MODEL_PREFIXES = ("gpt-4.1", "gpt-5", "gpt-6", "o1", "o3", "o4")
_FIXED_TEMPERATURE_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


class OpenAIConfig(BaseModel):
    """OpenAI AI provider configuration.

    Attributes:
        api_key: API key for authentication.
        base_url: Custom base URL for OpenAI-compatible APIs (e.g., Ollama, LM Studio,
            Azure OpenAI, or other providers). If None, uses the default OpenAI API.
        model: Model identifier to use.
        max_tokens: Maximum tokens in the response.
    """

    _vendor_rule_on_tool_turns: ClassVar[bool] = False
    """Whether this config's provider knows what its vendor takes of reasoning
    on a turn with tools, and applies that rule: a derivative that does sets it
    true and refuses ``supports_reasoning_effort_with_tools`` rather than
    ignore it (RFC §6.7)."""

    api_key: SecretStr
    base_url: str | None = None
    model: str
    """Model identifier. Required so upgrading RoomKit cannot silently change
    a caller's model, cost, latency, or behavior."""
    max_tokens: int = 1024
    timeout: float = 30.0
    """HTTP request timeout in seconds. Override for servers that need
    longer (e.g. Ollama cold-starting a model on first request)."""
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, kept apart from ``timeout`` so a host
    that no longer accepts connections is given up on in seconds rather
    than after the read budget. The SDK's own default."""
    max_retries: int = 0
    """SDK-level retry count. Default 0 because RoomKit's RetryPolicy
    handles retries at the right layer with proper backoff and fallback."""
    include_stream_usage: bool = True
    """Request token usage in streaming responses via
    ``stream_options.include_usage``; it is included in the final
    :class:`StreamDone` event. Every tool round streams, so without it a
    turn reports no usage and prices at zero. Set False for an
    OpenAI-compatible server that rejects ``stream_options``."""
    use_max_completion_tokens: bool = False
    """Send the output cap as ``max_completion_tokens`` instead of the
    deprecated ``max_tokens``. OpenAI's newer models (o-series, gpt-5,
    gpt-4.1) reject ``max_tokens`` outright. Leave False for
    OpenAI-compatible servers (vLLM, LM Studio, older Azure deployments)
    that only understand ``max_tokens``. Official modern models are profiled
    automatically unless this field is explicitly set."""
    supports_custom_temperature: bool = True
    """When False, ``temperature`` is omitted from requests. OpenAI's
    reasoning models (o-series, gpt-5) accept only the default
    ``temperature=1`` and reject any other value with HTTP 400."""
    reasoning_effort: str | None = None
    """Reasoning depth for OpenAI reasoning models (o-series, gpt-5):
    ``"none"`` | ``"low"`` | ``"medium"`` | ``"high"`` | ``"xhigh"`` |
    ``"max"`` (availability varies by model). Controls how long the model
    reasons (quality vs latency/cost); the reasoning trace itself stays hidden
    in the Chat Completions API. ``None`` = the model's default. The turn's
    own effort outranks this one. On a turn with tools the model catalogue
    decides what is sent (RFC §6.7): the effort up to GPT-5.2, ``"none"``
    instead from GPT-5.4 on, and nothing at all for a model it does not tag or
    one behind ``base_url``, unless ``supports_reasoning_effort_with_tools``
    says otherwise. Only configure this for reasoning models — others reject
    the parameter."""
    default_headers: dict[str, str] | None = None
    """Extra HTTP headers sent on every request, passed to the SDK's
    ``default_headers``. Use for an OpenAI-compatible endpoint behind a
    reverse proxy that needs custom headers, or a non-Bearer
    ``Authorization`` scheme (e.g. Basic). ``None`` sends only the SDK's
    own headers; the ``api_key`` Bearer token is unaffected."""
    extra_body: dict[str, Any] | None = None
    """Extra JSON fields merged into every Chat Completions request body
    via the SDK's ``extra_body``. The route for server-specific params the
    OpenAI schema omits — e.g. vLLM guided decoding
    (``guided_json``/``guided_choice``) and extra sampling (``top_k``,
    ``repetition_penalty``, ``min_p``). ``None`` sends a vanilla body."""
    supports_response_schema: bool | None = None
    """Whether the server honours a ``json_schema`` response format (RFC §6.7).
    ``None`` keeps the provider's default; set it for a server behind
    ``base_url`` that differs, so a turn carrying a response schema is refused
    up front instead of answered in prose."""
    supports_response_schema_with_tools: bool | None = None
    """Whether the server takes a ``json_schema`` response format beside function
    tools and still lets the model call them. ``None`` keeps the provider's
    default (on for OpenAI's own endpoint, off behind a ``base_url``)."""
    supports_reasoning_effort_with_tools: bool | None = None
    """Whether the server takes ``reasoning_effort`` beside function tools (RFC
    §6.7), for a model the provider cannot know: one behind a ``base_url``, or
    on OpenAI's own endpoint one the catalogue does not tag. ``True`` sends the
    turn's effort on a turn with tools as on any other; ``False`` leaves it
    out. ``None`` leaves it out too, with a warning logged once when an effort
    is left out. A model the catalogue tags follows its tag whatever this says.
    Read by the OpenAI and LiteLLM providers; the configs of the derivatives
    that know their vendor's rule (Meta, Cerebras, xAI, OpenRouter, DeepSeek,
    Qwen) refuse it."""

    @field_validator("supports_reasoning_effort_with_tools")
    @classmethod
    def _refused_where_the_vendor_rule_applies(cls, value: bool | None) -> bool | None:
        """Refuse a statement this config's provider would ignore: it applies
        its vendor's own rule on a turn with tools (RFC §6.7)."""
        if value is not None and cls._vendor_rule_on_tool_turns:
            raise ValueError(
                f"{cls.__name__} takes no supports_reasoning_effort_with_tools: its provider "
                "applies its vendor's own rule on a turn with tools"
            )
        return value

    def model_post_init(self, __context: Any) -> None:
        """Apply safe defaults for modern models on OpenAI's own endpoint."""
        if not is_vendor_endpoint(self.base_url, OPENAI_BASE_URL):
            return
        if (
            self.model.startswith(_MAX_COMPLETION_TOKEN_MODEL_PREFIXES)
            and "use_max_completion_tokens" not in self.model_fields_set
        ):
            self.use_max_completion_tokens = True
        if (
            self.model.startswith(_FIXED_TEMPERATURE_MODEL_PREFIXES)
            and "supports_custom_temperature" not in self.model_fields_set
        ):
            self.supports_custom_temperature = False


class OpenAIImageConfig(BaseModel):
    """OpenAI image-generation provider configuration (RFC §25).

    Separate from :class:`OpenAIConfig` because it configures a different
    endpoint with a disjoint model lineup — sampling temperature, reasoning
    effort and completion caps mean nothing to ``/v1/images``, and an image
    model means nothing to Chat Completions.

    Attributes:
        api_key: API key for authentication.
        base_url: Custom base URL for an OpenAI-compatible images endpoint.
            ``None`` uses the default OpenAI API.
        model: Image model identifier (e.g. ``"gpt-image-2"``). Required, for
            the same reason the chat config requires one: upgrading RoomKit
            must not silently change a caller's cost or output.
        quality: ``"low"`` | ``"medium"`` | ``"high"`` | ``"auto"``, or ``None``
            for the model's default. Multiplies both the token count and the
            latency, so it is a deployment decision rather than a per-call one.
        background: ``"transparent"`` | ``"opaque"`` | ``"auto"``. Transparent
            requires a ``png`` or ``webp`` output format.
        output_format: ``"png"`` | ``"jpeg"`` | ``"webp"``. ``None`` leaves the
            vendor default, which the response reports back and this provider
            reads rather than assuming.
        timeout: HTTP request timeout in seconds. Higher than the chat default
            because a high-quality image routinely takes more than 30s.
        connect_timeout: TCP connect timeout in seconds, kept apart from
            ``timeout`` so a host that no longer accepts connections is given
            up on in seconds rather than after the read budget.
        max_retries: SDK-level retry count. 0 because RoomKit's RetryPolicy
            handles retries at the right layer.
    """

    api_key: SecretStr
    base_url: str | None = None
    model: str
    quality: str | None = None
    background: str | None = None
    output_format: str | None = None
    timeout: float = 120.0
    connect_timeout: float = 5.0
    max_retries: int = 0
    default_headers: dict[str, str] | None = None
    """Extra HTTP headers sent on every request, passed to the SDK's
    ``default_headers`` — same role as on :class:`OpenAIConfig`."""

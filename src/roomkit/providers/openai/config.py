"""OpenAI provider configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, SecretStr

_MAX_COMPLETION_TOKEN_MODEL_PREFIXES = ("gpt-4.1", "gpt-5", "o1", "o3", "o4")
_FIXED_TEMPERATURE_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAIConfig(BaseModel):
    """OpenAI AI provider configuration.

    Attributes:
        api_key: API key for authentication.
        base_url: Custom base URL for OpenAI-compatible APIs (e.g., Ollama, LM Studio,
            Azure OpenAI, or other providers). If None, uses the default OpenAI API.
        model: Model identifier to use.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
    """

    api_key: SecretStr
    base_url: str | None = None
    model: str = "gpt-5.6-sol"
    """Model identifier. The default tracks OpenAI's current frontier model —
    a default that names a retired snapshot is a 404 at the first request,
    so it moves when the lineup does. Pin an id explicitly to opt out."""
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 30.0
    """HTTP request timeout in seconds. Override for servers that need
    longer (e.g. Ollama cold-starting a model on first request)."""
    max_retries: int = 0
    """SDK-level retry count. Default 0 because RoomKit's RetryPolicy
    handles retries at the right layer with proper backoff and fallback."""
    include_stream_usage: bool = False
    """When True, request token usage in streaming responses via
    ``stream_options.include_usage``. The usage is included in the
    final :class:`StreamDone` event."""
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
    in the Chat Completions API. ``None`` = the model's default. GPT-5.6 tool
    turns on OpenAI's endpoint use ``"none"`` because Chat Completions function
    tools reject that family at higher effective efforts. Only configure this
    for reasoning models — others reject the parameter."""
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

    def model_post_init(self, __context: Any) -> None:
        """Apply safe defaults for modern models on OpenAI's own endpoint."""
        if self.base_url is not None:
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

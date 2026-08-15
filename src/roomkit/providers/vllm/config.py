"""vLLM provider configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, SecretStr


class VLLMConfig(BaseModel):
    """Configuration for a local vLLM server.

    vLLM exposes an OpenAI-compatible API, so this config is translated
    into an ``OpenAIConfig`` by :func:`create_vllm_provider`.

    Attributes:
        model: Model name loaded by the vLLM server (required).
        base_url: Base URL of the vLLM OpenAI-compatible endpoint.
        api_key: Bearer token sent as ``Authorization: Bearer <key>``.
            Matches ``vllm serve --api-key``; default ``"none"`` for the
            common no-auth local server.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        timeout: HTTP request timeout in seconds. Increase for vLLM servers that
            load models lazily on first request.
        headers: Extra HTTP headers on every request — for a reverse proxy
            that needs custom headers, or a non-Bearer ``Authorization``
            scheme. Maps to ``OpenAIConfig.default_headers``.
        extra_body: Extra JSON fields merged into every request body — the
            route for vLLM-specific params (``guided_json``/``guided_choice``
            guided decoding, ``top_k``/``repetition_penalty`` sampling).
            Maps to ``OpenAIConfig.extra_body``.
        enable_thinking: Turn the model's reasoning block on or off. ``None``
            leaves the model's own default, which for current Qwen builds is
            *on* at the most verbose effort — reasoning then competes with the
            answer for ``max_tokens`` and can consume the whole budget, leaving
            an empty ``content``. Set ``False`` for tool loops that only need
            the final answer.
        reasoning_effort: Reasoning verbosity when thinking is on —
            ``"low"``, ``"medium"`` or ``"xhigh"``. Accepted values depend on
            the served model's chat template; vLLM raises a template error on
            an unknown one.
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: SecretStr = SecretStr("none")
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 0
    """SDK-level retry count. Default 0 because RoomKit's RetryPolicy
    handles retries at the right layer with proper backoff and fallback."""
    include_stream_usage: bool = False
    """When True, request token usage in streaming responses."""
    headers: dict[str, str] | None = None
    """Extra HTTP headers sent on every request (proxy headers, non-Bearer
    auth). ``None`` sends only the SDK defaults."""
    extra_body: dict[str, Any] | None = None
    """Extra request-body fields for vLLM-specific params (guided decoding,
    extra sampling). ``None`` sends a vanilla body."""
    enable_thinking: bool | None = None
    """Reasoning block on/off. ``None`` leaves the model's own default."""
    reasoning_effort: str | None = None
    """Reasoning verbosity when thinking is on (``"low"``/``"medium"``/``"xhigh"``)."""

    def chat_template_kwargs(self) -> dict[str, Any]:
        """The ``chat_template_kwargs`` implied by the reasoning settings.

        vLLM renders the model's own chat template server-side, so reasoning is
        steered through template kwargs rather than a sampling parameter. Empty
        when neither knob is set, so a vanilla body stays vanilla.
        """
        kwargs: dict[str, Any] = {}
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

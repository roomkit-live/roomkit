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
        top_p: Nucleus sampling cutoff. ``None`` leaves the server's default.
        top_k: Top-k sampling cutoff — a vLLM extension, not an OpenAI field.
            ``None`` leaves the server's default.
        min_p: Minimum-probability sampling cutoff — a vLLM extension.
            ``None`` leaves the server's default.
        presence_penalty: Penalty on tokens already present in the output.
            The knob Qwen's own guidance raises (to ``1.5``) for non-thinking
            mode, where the failure it addresses is degenerate repetition.
            ``None`` leaves the server's default.
        repetition_penalty: Multiplicative repetition penalty — a vLLM
            extension, distinct from ``presence_penalty``. ``None`` leaves the
            server's default.
        extra_body: Extra JSON fields merged into every request body — the
            route for vLLM params this config does not model
            (``guided_json``/``guided_choice`` guided decoding, and any
            sampler added by a newer server). Maps to
            ``OpenAIConfig.extra_body``, and an entry here wins over the
            typed fields above.
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
    top_p: float | None = None
    """Nucleus sampling cutoff. ``None`` leaves the server's default."""
    top_k: int | None = None
    """Top-k sampling cutoff (vLLM extension). ``None`` leaves the default."""
    min_p: float | None = None
    """Minimum-probability cutoff (vLLM extension). ``None`` leaves the default."""
    presence_penalty: float | None = None
    """Penalty on already-present tokens. ``None`` leaves the default."""
    repetition_penalty: float | None = None
    """Multiplicative repetition penalty (vLLM extension). ``None`` leaves the default."""
    extra_body: dict[str, Any] | None = None
    """Extra request-body fields for vLLM params this config does not model
    (guided decoding, a newer server's sampler). ``None`` sends a vanilla
    body; an entry here wins over the typed sampling fields."""
    enable_thinking: bool | None = None
    """Reasoning block on/off. ``None`` leaves the model's own default."""
    reasoning_effort: str | None = None
    """Reasoning verbosity when thinking is on (``"low"``/``"medium"``/``"xhigh"``)."""

    def sampling_body(self) -> dict[str, Any]:
        """The request-body fields implied by the sampling settings.

        All of them ride the body rather than the SDK's named parameters:
        ``top_k``, ``min_p`` and ``repetition_penalty`` are vLLM extensions the
        OpenAI SDK has no argument for, and ``top_p``/``presence_penalty`` are
        read from the same body by the server, so routing the five through one
        place keeps the split invisible to the caller.

        Only the knobs actually set are emitted — ``None`` means "the server
        decides", which is not the same as sending its documented default and
        is the only honest answer for a server whose model we cannot see.
        Tested with ``is not None`` so an explicit ``0`` survives: ``min_p=0``
        and ``presence_penalty=0`` are meaningful values, not absences.
        """
        fields = ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
        return {name: value for name in fields if (value := getattr(self, name)) is not None}

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

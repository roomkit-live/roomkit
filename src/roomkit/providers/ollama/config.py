"""Ollama provider configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, SecretStr

#: Effort levels accepted by Ollama's ``think`` parameter for reasoning models.
ThinkEffort = Literal["low", "medium", "high"]


class OllamaConfig(BaseModel):
    """Ollama AI provider configuration.

    Wraps the native Ollama ``/api/chat`` endpoint via the
    ``ollama-python`` SDK. Prefer this over the OpenAI-compatible
    shim when the model is a reasoning model (DeepSeek-R1, Qwen 3
    thinking variants, etc.) because Ollama exposes the ``think``
    parameter and streams the reasoning content as a separate
    ``thinking`` field — both ignored by the OpenAI-compat endpoint.

    Attributes:
        host: Base URL of the Ollama server. Default points at the
            local daemon. The native API lives under ``/api`` on the
            same host; the SDK appends the path.
        model: Model identifier to use (e.g. ``"qwen3:8b"``,
            ``"llama3.2"``, ``"deepseek-r1:7b"``).
        max_tokens: Maximum tokens to generate in the response. Maps
            to Ollama's ``options.num_predict``. ``None`` lets the
            server pick its default.
        temperature: Sampling temperature. Maps to
            ``options.temperature``.
        timeout: HTTP request timeout in seconds. Long default because
            local models cold-start on first request and reasoning
            models can take 30-60s before the first token.
        max_retries: SDK-level retry count. Default 0 because
            RoomKit's RetryPolicy handles retries at the right layer
            with proper backoff and fallback.
        think: Whether and how hard the model should reason before
            answering. ``None`` (default) means "use the model's
            default" — reasoning models think, others don't.
            ``True``/``False`` force thinking on or off as a boolean.
            One of ``"low"``, ``"medium"``, ``"high"`` selects an
            effort level for models that support it (Ollama 0.7+ on
            reasoning-capable models like gpt-oss and deepseek-r1).
            Effort strings pass straight through to the Ollama API;
            unsupported models silently downgrade to boolean
            behavior. ``AIContext.thinking_budget`` overrides this at
            request time: ``None``/``0`` → ``think=False``, ``>0`` →
            uses this config value if it's a string, otherwise
            ``think=True``.
        keep_alive: How long the model stays loaded in memory after
            the request. Maps to Ollama's ``keep_alive`` parameter.
            Strings like ``"5m"`` or integer seconds. ``None`` uses
            the server default (5 minutes).
        num_ctx: Context window size. Maps to ``options.num_ctx``.
            ``None`` uses the model's default (typically 2048 — bump
            for long contexts).
        api_key: Bearer token for a protected Ollama endpoint — Ollama
            Cloud/Turbo, or a self-hosted server behind a reverse proxy
            that checks ``Authorization: Bearer``. Sent as the
            ``Authorization`` header. ``None`` (default) leaves auth to
            the SDK, which still falls back to the ``OLLAMA_API_KEY``
            environment variable when it is set. Prefer this field when
            the key comes from a secret manager rather than the process
            environment.
        headers: Extra HTTP headers attached to every request — for a
            reverse proxy that needs custom headers, or a non-Bearer
            ``Authorization`` scheme (e.g. Basic). ``api_key`` takes
            precedence over an ``Authorization`` entry supplied here.
            ``None`` (default) sends only the SDK's own headers.
    """

    host: str = "http://localhost:11434"
    model: str = "llama3.2"
    max_tokens: int | None = None
    temperature: float = 0.7
    timeout: float = 120.0
    max_retries: int = 0
    think: bool | ThinkEffort | None = None
    keep_alive: str | int | None = None
    num_ctx: int | None = None
    api_key: SecretStr | None = None
    headers: dict[str, str] | None = None

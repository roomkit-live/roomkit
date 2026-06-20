"""OpenRouter AI provider — generates responses via OpenRouter's OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig
from roomkit.providers.openrouter.models import MODELS


class OpenRouterAIProvider(OpenAIAIProvider):
    """AI provider using OpenRouter's OpenAI-compatible Chat Completions API.

    Subclasses :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` — only
    client initialisation, provider name, and model discovery differ. All
    message building, tool handling, response parsing, and streaming are
    inherited unchanged, since OpenRouter speaks the OpenAI Chat Completions
    API verbatim.
    """

    _config: OpenRouterConfig

    def __init__(self, config: OpenRouterConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenRouterAIProvider. "
                "Install it with: pip install roomkit[openrouter]"
            ) from exc
        self._config = config
        self._api_status_error = _openai.APIStatusError
        self._api_connection_error = _openai.APIConnectionError
        self._client = _openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            default_headers=self._merged_headers(config),
        )

    @staticmethod
    def _attribution_headers(config: OpenRouterConfig) -> dict[str, str]:
        """Build OpenRouter's optional app-attribution headers.

        ``HTTP-Referer`` (site URL) and ``X-Title`` (app name) let an app show
        up on OpenRouter's leaderboards and analytics. Omitted when unset.
        """
        headers: dict[str, str] = {}
        if config.site_url:
            headers["HTTP-Referer"] = config.site_url
        if config.app_name:
            headers["X-Title"] = config.app_name
        return headers

    def _merged_headers(self, config: OpenRouterConfig) -> dict[str, str] | None:
        """Attribution headers plus any inherited ``default_headers``.

        ``default_headers`` comes from :class:`OpenAIConfig` (proxy or
        non-Bearer headers); it layers on top of attribution and wins on key
        collisions. ``None`` when nothing is set.
        """
        headers = self._attribution_headers(config)
        headers.update(config.default_headers or {})
        return headers or None

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "openrouter"

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Add temperature and OpenRouter's unified ``reasoning`` parameter.

        Where the OpenAI parent sends a top-level ``reasoning_effort`` (only
        OpenAI models honour it), OpenRouter normalises thinking across every
        upstream provider through a single ``reasoning`` object — so Claude,
        Gemini, and DeepSeek all surface a reasoning trace. It is sent via the
        OpenAI SDK's ``extra_body`` passthrough. Reasoning is omitted on tool
        turns, matching the parent (some models reject it alongside tools); the
        streamed trace is surfaced by the inherited ``delta.reasoning`` reader.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        reasoning = self._resolve_reasoning(context)
        if reasoning is not None and not context.tools:
            kwargs.setdefault("extra_body", {})["reasoning"] = reasoning

    def _resolve_reasoning(self, context: AIContext) -> dict[str, Any] | None:
        """Build OpenRouter's ``reasoning`` object for this turn, or ``None`` to omit it.

        ``thinking_budget`` gates per-turn (mirrors the Mistral provider):
        ``None`` passes the configured ``reasoning_effort`` through (omitted when
        unset, so the model decides); ``0`` disables reasoning explicitly; ``>0``
        maps the budget straight to OpenRouter's Anthropic-style ``max_tokens``
        reasoning cap.
        """
        budget = context.thinking_budget
        if budget is None:
            effort = self._config.reasoning_effort
            return {"effort": effort} if effort else None
        if budget <= 0:
            return {"enabled": False}
        return {"max_tokens": budget}

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline snapshot of popular OpenRouter model slugs.

        A small representative slice — see :meth:`list_models` for the full,
        always-current catalog OpenRouter exposes live.
        """
        return list(MODELS)

    async def list_models(self) -> list[ModelInfo]:
        """List every model OpenRouter currently exposes, with live metadata.

        OpenRouter's ``/models`` items omit the ``object``/``owned_by`` fields
        the OpenAI SDK's ``Model`` type requires, so the inherited
        ``models.list()`` cannot parse them — this reads the raw JSON instead
        and maps id, display name, context window, and vision support directly.
        Curated metadata backfills anything the endpoint leaves blank.
        """
        data = await self._fetch_models_json()
        live = [self._parse_model(item) for item in data]
        return self._merge_curated(live)

    async def _fetch_models_json(self) -> list[dict[str, Any]]:
        """GET the raw ``/models`` payload and return its ``data`` array."""
        # httpx ships with the openai SDK; imported lazily so the curated
        # catalog (available_models) stays usable without the HTTP stack.
        import httpx

        url = f"{self._config.base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {self._config.api_key.get_secret_value()}"}
        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data", [])
        return data if isinstance(data, list) else []

    @staticmethod
    def _parse_model(item: dict[str, Any]) -> ModelInfo:
        """Map one OpenRouter ``/models`` entry to a :class:`ModelInfo`.

        Vision support is read from ``architecture.input_modalities``; it stays
        ``None`` ("unknown") when the endpoint reports no modalities.
        """
        architecture = item.get("architecture") or {}
        modalities = architecture.get("input_modalities") or []
        return ModelInfo(
            id=item["id"],
            display_name=item.get("name"),
            context_window=item.get("context_length"),
            supports_vision=("image" in modalities) if modalities else None,
        )

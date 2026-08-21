"""LiteLLM AI provider — generates responses via a LiteLLM proxy (AI gateway)."""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo, ModelPricing
from roomkit.providers.litellm.config import LiteLLMConfig
from roomkit.providers.openai.ai import OpenAIAIProvider


def _rate_per_million(value: object) -> float | None:
    """Convert a LiteLLM per-token cost to a per-million rate, or ``None``.

    ``bool`` is excluded explicitly — it passes an ``isinstance`` check against
    ``int`` and would price a model at one dollar per million tokens.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value * 1_000_000


class LiteLLMAIProvider(OpenAIAIProvider):
    """AI provider using a LiteLLM proxy's OpenAI-compatible API.

    Subclasses :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` — the
    proxy speaks the OpenAI Chat Completions API verbatim, so all message
    building, tool handling, response parsing, and streaming are inherited
    unchanged. The streamed reasoning trace is surfaced by the inherited
    ``reasoning_content`` reader, which is exactly the field LiteLLM
    normalises reasoning into. What differs: the provider name, reasoning
    parameters (LiteLLM's cross-provider ``reasoning_effort`` / ``thinking``),
    and model metadata — the deployment's own, via ``/model/info``.
    """

    _config: LiteLLMConfig
    _install_extra: ClassVar[str] = "litellm"

    def __init__(self, config: LiteLLMConfig) -> None:
        super().__init__(config)

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "litellm"

    @property
    def supports_vision(self) -> bool:
        """True: whether a routed model reads images is the gateway's call.

        The configured model is a deployment-chosen alias, so the parent's
        fallback prefixes — OpenAI's own model names — say nothing about it,
        and inheriting them would report vision-capable routes as text-only
        and drop images before they reached the wire. Passing them through
        lets a multimodal route work and a text-only one answer with an
        error, the same bargain the vLLM and Ollama providers make.
        """
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Empty: a gateway's models are its operator's config, not knowable offline.

        Every deployment names its own aliases and routes, so no curated list
        can describe them, and inheriting OpenAI's would hand out one of their
        context windows for any alias that happened to collide. Empty makes
        :attr:`context_window` ``None``, the honest answer for an alias roomkit
        has never seen. Call :meth:`list_models` for the real set — it asks the
        proxy, which does know.
        """
        return []

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Add temperature and LiteLLM's normalised reasoning parameters.

        LiteLLM translates OpenAI's top-level ``reasoning_effort`` for every
        upstream it fronts (Anthropic thinking budgets, Gemini, DeepSeek, …),
        and accepts Anthropic's ``thinking`` object where an explicit token
        budget is wanted. ``thinking_budget`` gates per-turn (mirrors the
        OpenRouter provider): ``None`` passes the effort through, the turn's
        own effort outranking the configured one; ``0`` disables reasoning
        explicitly via LiteLLM's ``reasoning_effort="none"``; ``>0`` maps to a
        ``thinking`` budget, sent via the SDK's ``extra_body`` passthrough.
        Reasoning is omitted on tool turns, matching the parent — the gateway
        fronts the same upstreams that reject it alongside tools.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        if context.tools:
            return
        budget = context.thinking_budget
        if budget is None:
            effort = context.reasoning_effort or self._config.reasoning_effort
            if effort is not None:
                kwargs["reasoning_effort"] = effort
        elif budget <= 0:
            kwargs["reasoning_effort"] = "none"
        else:
            kwargs.setdefault("extra_body", {})["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }

    async def list_models(self) -> list[ModelInfo]:
        """List the models this proxy deployment exposes, with live metadata.

        Reads LiteLLM's ``/model/info`` rather than the barer ``/v1/models``:
        alongside each public model name it reports the context window, vision
        support, and per-token costs from the proxy's own cost map — fed
        straight into :class:`ModelInfo`, so history trimming and budget
        dashboards work against the deployment's real numbers. A load-balanced
        model group lists one entry per deployment under the same public name;
        they are collapsed to the first, which carries the group's metadata.
        """
        data = await self._fetch_model_info()
        models: dict[str, ModelInfo] = {}
        for item in data:
            name = item.get("model_name")
            if not isinstance(name, str) or name in models:
                continue
            info = item.get("model_info")
            models[name] = self._parse_model(name, info if isinstance(info, dict) else {})
        return list(models.values())

    async def _fetch_model_info(self) -> list[dict[str, Any]]:
        """GET the raw ``/model/info`` payload and return its ``data`` array."""
        # httpx ships with the openai SDK; imported lazily so the class stays
        # importable (available_models, catalogs) without the HTTP stack.
        import httpx

        url = f"{self._config.base_url.rstrip('/')}/model/info"
        headers = {"Authorization": f"Bearer {self._config.api_key.get_secret_value()}"}
        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data", [])
        return data if isinstance(data, list) else []

    @staticmethod
    def _parse_model(name: str, info: dict[str, Any]) -> ModelInfo:
        """Map one ``/model/info`` entry to a :class:`ModelInfo`.

        Fields absent from the proxy's cost map stay ``None`` ("unknown") —
        an operator-defined alias the map has never heard of reports nothing,
        and inventing a window or a price for it would be worse.
        """
        window = info.get("max_input_tokens")
        return ModelInfo(
            id=name,
            context_window=window if isinstance(window, int) else None,
            supports_vision=(
                info["supports_vision"] if isinstance(info.get("supports_vision"), bool) else None
            ),
            pricing=LiteLLMAIProvider._parse_pricing(info),
        )

    @staticmethod
    def _parse_pricing(info: dict[str, Any]) -> ModelPricing | None:
        """Build :class:`ModelPricing` from a ``/model/info`` entry, or ``None``.

        LiteLLM quotes per-token; roomkit's catalog rates are per-million.
        Only built when both base rates are present — a partial price would
        bill half a conversation. A ``0``/``0`` pair is also ``None``: LiteLLM
        defaults *unknown* costs to zero rather than null (verified live
        against 1.79.0), so free-of-charge is indistinguishable from unmapped
        here, and a $0 price would tell a budget dashboard the route is free
        while the gateway may well be billing it. ``verified`` is today's
        date: these rates were read live from the deployment's own cost map,
        not copied from a vendor page at some past release.
        """
        input_rate = _rate_per_million(info.get("input_cost_per_token"))
        output_rate = _rate_per_million(info.get("output_cost_per_token"))
        if input_rate is None or output_rate is None:
            return None
        if input_rate == 0 and output_rate == 0:
            return None
        return ModelPricing(
            input_per_million=input_rate,
            output_per_million=output_rate,
            cache_read_per_million=_rate_per_million(info.get("cache_read_input_token_cost")),
            cache_write_per_million=_rate_per_million(info.get("cache_creation_input_token_cost")),
            verified=date.today(),
        )

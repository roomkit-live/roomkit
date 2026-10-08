"""LiteLLM AI provider — generates responses via a LiteLLM proxy (AI gateway)."""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo, ModelPricing
from roomkit.providers.ai.model_tags import SPEECH_CAPABILITY, TRANSCRIPTION_CAPABILITY
from roomkit.providers.ai.reasoning import thinking_switch, turn_setting
from roomkit.providers.litellm.config import LiteLLMConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.utils import http_timeout


def _rate_per_million(value: object) -> float | None:
    """Convert a LiteLLM per-token cost to a per-million rate, or ``None``.

    ``bool`` is excluded explicitly — it passes an ``isinstance`` check against
    ``int`` and would price a model at one dollar per million tokens.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value * 1_000_000


# LiteLLM's cost-map ``mode`` of a speech model (``whisper-1`` is
# ``audio_transcription``, ``tts-1`` ``audio_speech``).
_MODE_TAGS: dict[Any, list[str]] = {
    "audio_transcription": [TRANSCRIPTION_CAPABILITY],
    "audio_speech": [SPEECH_CAPABILITY],
}


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
        own effort outranking the configured one; ``>0`` maps to a
        ``thinking`` budget, sent via the SDK's ``extra_body`` passthrough.
        Reasoning is omitted on tool turns unless the config's
        ``supports_reasoning_effort_with_tools`` says the proxy takes it there:
        the model behind the alias is not known here, and some upstreams reject
        it alongside tools (RFC §6.7 allows the omission where the model is
        unknown). The flag covers the budget too, LiteLLM's other spelling of
        the effort.

        ``0`` sends no reasoning parameters at all. LiteLLM has no disable
        token that survives every translator (verified live on 1.79.0: the
        Gemini mapper rejects ``"none"`` with a 500, the Anthropic mapper
        rejects both ``"none"`` and ``"disable"``), so an explicit off would
        break on some routes while working on others. Omission is the one
        portable spelling — the routed model's own default applies. A
        deployment that must force thinking off for an alias states it where
        the upstream is known: per-model in the proxy's ``config.yaml``, or
        via ``extra_body`` here with the token its translator accepts.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        # Off, as the turn states it (RFC §6.7), sends nothing: see above.
        if thinking_switch(context) is False:
            return
        budget = context.thinking_budget
        effort = turn_setting(context.reasoning_effort, self._config.reasoning_effort)
        if context.tools and self._tool_turn_setting(budget or effort) is None:
            return
        if budget:
            kwargs.setdefault("extra_body", {})["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
            return
        if effort is not None:
            kwargs["reasoning_effort"] = effort

    async def list_models(self) -> list[ModelInfo]:
        """List the models this proxy deployment exposes, with live metadata.

        Reads LiteLLM's ``/model/info`` rather than the barer ``/v1/models``:
        alongside each public model name it reports the context window, vision
        support, and per-token costs from the proxy's own cost map — fed
        straight into :class:`ModelInfo`, so history trimming and budget
        dashboards work against the deployment's real numbers. A load-balanced
        model group lists one entry per deployment under the same public name;
        they are folded into one :class:`ModelInfo` that promises only what
        every deployment delivers (see :meth:`_merge_deployments`).
        """
        data = await self._fetch_model_info()
        models: dict[str, ModelInfo] = {}
        for item in data:
            name = item.get("model_name")
            if not isinstance(name, str):
                continue
            info = item.get("model_info")
            parsed = self._parse_model(name, info if isinstance(info, dict) else {})
            previous = models.get(name)
            models[name] = (
                parsed if previous is None else self._merge_deployments(previous, parsed)
            )
        return self._listing(list(models.values()))

    @staticmethod
    def _merge_deployments(a: ModelInfo, b: ModelInfo) -> ModelInfo:
        """Fold two deployments of one load-balanced alias into the group's promise.

        The router may hand any request to any deployment, and ``/model/info``
        does not say which — so the group's metadata is what every member can
        honour. Payload order is routing config, not truth: keeping the first
        entry made the window, vision flag and price flip with a reshuffle of
        the proxy's ``config.yaml``.

        The window is the smallest (the larger one overflows on the smaller
        route; one unknown makes the group's unknown). Vision is ``False`` as
        soon as one deployment says so — that route refuses images — ``True``
        only when unanimous, otherwise unknown. Pricing survives only when the
        deployments quote the same rates: a group billing differently per
        route has no single honest price, and picking one would misprice every
        request the router sends the other way.
        """
        if a.context_window is None or b.context_window is None:
            window = None
        else:
            window = min(a.context_window, b.context_window)
        if a.supports_vision is False or b.supports_vision is False:
            vision: bool | None = False
        elif a.supports_vision is None or b.supports_vision is None:
            vision = None
        else:
            vision = True
        return ModelInfo(
            id=a.id,
            context_window=window,
            supports_vision=vision,
            capabilities=a.capabilities if a.capabilities == b.capabilities else [],
            pricing=a.pricing if a.pricing == b.pricing else None,
        )

    async def _fetch_model_info(self) -> list[dict[str, Any]]:
        """GET the raw ``/model/info`` payload and return its ``data`` array."""
        # httpx ships with the openai SDK; imported lazily so the class stays
        # importable (available_models, catalogs) without the HTTP stack.
        import httpx

        url = f"{self._config.base_url.rstrip('/')}/model/info"
        headers = {"Authorization": f"Bearer {self._config.api_key.get_secret_value()}"}
        async with httpx.AsyncClient(timeout=http_timeout(self._config)) as client:
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
        and inventing a window or a price for it would be worse. Its ``mode``
        tags a speech model, whatever alias the operator gave it.
        """
        window = info.get("max_input_tokens")
        return ModelInfo(
            id=name,
            capabilities=list(_MODE_TAGS.get(info.get("mode"), [])),
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

"""Tests for the per-provider model catalog (available_models / list_models).

Curated catalogs are classmethods — they need neither an API key nor the
provider SDK, so they are exercised directly on the class. Live ``list_models``
overrides are tested by building the provider via ``__new__`` (skipping the
SDK-importing ``__init__``) and injecting a fake client, which keeps the tests
offline while still covering the response→ModelInfo mapping and curated merge.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr, ValidationError

from roomkit.providers.ai import ModelInfo, ModelPricing
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.xai.ai import XAIAIProvider

# (provider class, config class) for the providers that ship a curated catalog.
CURATED = [
    (AnthropicAIProvider, AnthropicConfig),
    (OpenAIAIProvider, OpenAIConfig),
    (GeminiAIProvider, GeminiConfig),
    (MistralAIProvider, MistralConfig),
    (OllamaAIProvider, OllamaConfig),
]

# Curated providers that deliberately retain a model default. OpenAI and
# Anthropic require callers to choose explicitly because their model selection
# materially changes cost, latency, and behavior.
DEFAULTED_CURATED = [
    (GeminiAIProvider, GeminiConfig),
    (MistralAIProvider, MistralConfig),
    (OllamaAIProvider, OllamaConfig),
]

# Providers whose vendor publishes a per-token list price, so every model in
# the catalog must carry one. Ollama (open weights pulled onto your own
# hardware) publishes none — demanding a price there would only invent one.
# PolarGrid quotes its public model on its models page; its customer-pilot
# model is not advertised, so the guard never reaches it.
PRICED = [
    AnthropicAIProvider,
    OpenAIAIProvider,
    GeminiAIProvider,
    MistralAIProvider,
    XAIAIProvider,
    OpenRouterAIProvider,
    PolarGridAIProvider,
]


def _bare(cls: type[AIProvider]) -> Any:
    """Instantiate a provider without running its SDK-importing ``__init__``."""
    return cls.__new__(cls)


# --- ModelInfo + base ABC ------------------------------------------------------


def test_modelinfo_defaults() -> None:
    m = ModelInfo(id="x")
    assert m.id == "x"
    assert m.display_name is None
    assert m.context_window is None
    assert m.supports_vision is None
    assert m.deprecated is False
    assert m.pricing is None


# --- ModelPricing --------------------------------------------------------------


def test_pricing_defaults_to_usd_without_cache_rates() -> None:
    p = ModelPricing(input_per_million=3.0, output_per_million=15.0, verified=date(2026, 8, 5))
    assert p.currency == "USD"
    assert p.cache_read_per_million is None
    assert p.cache_write_per_million is None
    assert p.long_context_threshold_tokens is None
    assert p.long_context_input_multiplier == 1.0
    assert p.long_context_output_multiplier == 1.0


def test_cost_for_prices_input_and_output() -> None:
    p = ModelPricing(input_per_million=1.5, output_per_million=7.5, verified=date(2026, 8, 5))
    # The conversation that opened RMK-116: 12,693 in / 20 out on a Gemini Flash.
    cost = p.cost_for({"input_tokens": 12_693, "output_tokens": 20})
    assert cost == pytest.approx(12_693 * 1.5e-6 + 20 * 7.5e-6)


def test_cost_for_uses_the_cache_rates_when_the_model_has_them() -> None:
    p = ModelPricing(
        input_per_million=5.0,
        output_per_million=25.0,
        cache_read_per_million=0.5,
        cache_write_per_million=6.25,
        verified=date(2026, 8, 5),
    )
    cost = p.cost_for(
        {
            "input_tokens": 1_000,
            "output_tokens": 100,
            "cache_read_input_tokens": 100_000,
            "cache_creation_input_tokens": 10_000,
        }
    )
    # The cached prefix costs a tenth of the input rate — the whole point of
    # carrying a cache rate rather than billing every token at the input price.
    assert cost == pytest.approx(0.005 + 0.0025 + 0.05 + 0.0625)


def test_cost_for_omits_cache_counter_without_a_per_token_rate() -> None:
    p = ModelPricing(input_per_million=2.0, output_per_million=6.0, verified=date(2026, 8, 5))
    assert p.cost_for({"cache_read_input_tokens": 1_000_000}) == 0.0


def test_cost_for_applies_long_context_multipliers_to_all_token_classes() -> None:
    p = ModelPricing(
        input_per_million=2.0,
        output_per_million=10.0,
        cache_read_per_million=0.2,
        cache_write_per_million=2.5,
        long_context_threshold_tokens=200_000,
        long_context_input_multiplier=2.0,
        long_context_output_multiplier=1.5,
        verified=date(2026, 8, 5),
    )
    usage = {
        "input_tokens": 100_000,
        "cache_read_input_tokens": 100_000,
        "cache_creation_input_tokens": 1,
        "output_tokens": 10_000,
    }

    assert p.cost_for(usage) == pytest.approx(((0.2 + 0.02 + 0.0000025) * 2) + 0.15)


def test_cost_for_ignores_unknown_counters_and_missing_keys() -> None:
    p = ModelPricing(input_per_million=1.0, output_per_million=1.0, verified=date(2026, 8, 5))
    assert p.cost_for({}) == 0.0
    assert p.cost_for({"reasoning_tokens": 5_000, "input_tokens": 1_000}) == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_per_million", -1),
        ("output_per_million", float("inf")),
        ("cache_read_per_million", -1),
        ("cache_write_per_million", float("nan")),
        ("long_context_threshold_tokens", 0),
        ("long_context_input_multiplier", 0),
        ("long_context_output_multiplier", -1),
        ("currency", ""),
    ],
)
def test_pricing_rejects_invalid_financial_fields(field: str, value: Any) -> None:
    fields: dict[str, Any] = {
        "input_per_million": 1.0,
        "output_per_million": 1.0,
        "verified": date(2026, 8, 5),
        field: value,
    }

    with pytest.raises(ValidationError):
        ModelPricing(**fields)


@pytest.mark.parametrize("value", [-1, True, 1.5, "100"])
def test_cost_for_rejects_invalid_usage_counters(value: Any) -> None:
    p = ModelPricing(input_per_million=1.0, output_per_million=1.0, verified=date(2026, 8, 5))
    usage: Any = {"input_tokens": value}

    with pytest.raises(ValueError, match="non-negative integer"):
        p.cost_for(usage)


def test_merge_curated_backfills_pricing() -> None:
    class _Cat(AIProvider):
        @property
        def model_name(self) -> str:
            return "c"

        async def generate(self, context: AIContext) -> AIResponse:
            return AIResponse(content="")

        @classmethod
        def available_models(cls) -> list[ModelInfo]:
            return [
                ModelInfo(
                    id="a",
                    pricing=ModelPricing(
                        input_per_million=1.0, output_per_million=2.0, verified=date(2026, 8, 5)
                    ),
                )
            ]

    merged = {m.id: m for m in _Cat._merge_curated([ModelInfo(id="a"), ModelInfo(id="b")])}
    # A live endpoint reports ids, rarely rates: the curated price survives.
    assert merged["a"].pricing is not None
    assert merged["a"].pricing.input_per_million == 1.0
    assert merged["b"].pricing is None


def test_base_available_models_is_empty() -> None:
    class _Bare(AIProvider):
        @property
        def model_name(self) -> str:
            return "bare"

        async def generate(self, context: AIContext) -> AIResponse:
            return AIResponse(content="")

    assert _Bare.available_models() == []


async def test_base_list_models_falls_back_to_curated() -> None:
    # MockAIProvider does not override list_models → it returns the curated set.
    provider = MockAIProvider()
    live = await provider.list_models()
    assert [m.id for m in live] == [m.id for m in MockAIProvider.available_models()]


def test_context_window_resolves_from_catalog() -> None:
    # "mock" is in MockAIProvider's catalog with context_window=8192.
    assert MockAIProvider().context_window == 8192


def test_openai_gpt_5_6_luna_has_its_smaller_untiered_context() -> None:
    luna = next(
        model for model in OpenAIAIProvider.available_models() if model.id == "gpt-5.6-luna"
    )

    assert luna.context_window == 400_000
    assert luna.pricing is not None
    assert luna.pricing.long_context_threshold_tokens is None


def test_context_window_none_when_model_absent() -> None:
    class _Bare(AIProvider):
        @property
        def model_name(self) -> str:
            return "some-custom-local-model"  # not in any catalog

        async def generate(self, context: AIContext) -> AIResponse:
            return AIResponse(content="")

    assert _Bare().context_window is None


def test_merge_curated_backfills_missing_metadata() -> None:
    class _Cat(AIProvider):
        @property
        def model_name(self) -> str:
            return "c"

        async def generate(self, context: AIContext) -> AIResponse:
            return AIResponse(content="")

        @classmethod
        def available_models(cls) -> list[ModelInfo]:
            return [ModelInfo(id="a", display_name="A", context_window=100, supports_vision=True)]

    merged = {m.id: m for m in _Cat._merge_curated([ModelInfo(id="a"), ModelInfo(id="b")])}
    # Known id: metadata filled in from the catalog.
    assert merged["a"].display_name == "A"
    assert merged["a"].context_window == 100
    assert merged["a"].supports_vision is True
    # Unknown id: passes through untouched.
    assert merged["b"].display_name is None
    assert merged["b"].context_window is None


def test_merge_curated_prefers_live_values() -> None:
    class _Cat(AIProvider):
        @property
        def model_name(self) -> str:
            return "c"

        async def generate(self, context: AIContext) -> AIResponse:
            return AIResponse(content="")

        @classmethod
        def available_models(cls) -> list[ModelInfo]:
            return [ModelInfo(id="a", display_name="Curated", context_window=100)]

    merged = _Cat._merge_curated([ModelInfo(id="a", display_name="Live", context_window=200)])[0]
    assert merged.display_name == "Live"
    assert merged.context_window == 200


# --- Curated catalogs (offline, no SDK, no key) --------------------------------


@pytest.mark.parametrize(("provider_cls", "_config_cls"), CURATED)
def test_curated_catalog_is_nonempty_and_unique(
    provider_cls: type[AIProvider], _config_cls: type
) -> None:
    models = provider_cls.available_models()
    assert models, f"{provider_cls.__name__} has an empty curated catalog"
    assert all(isinstance(m, ModelInfo) for m in models)
    ids = [m.id for m in models]
    assert len(ids) == len(set(ids)), f"{provider_cls.__name__} has duplicate model ids"


@pytest.mark.parametrize(("provider_cls", "config_cls"), DEFAULTED_CURATED)
def test_default_config_model_is_in_catalog(
    provider_cls: type[AIProvider], config_cls: type
) -> None:
    # The model a provider defaults to should be discoverable in its catalog.
    default = config_cls.model_fields["model"].default
    ids = {m.id for m in provider_cls.available_models()}
    assert default in ids, f"{provider_cls.__name__} default {default!r} missing from catalog"


# --- Every model a vendor prices carries its price ----------------------------
#
# The gap this closes: `gemini-3.6-flash` shipped in the catalog while the
# consumer's separate rate sheet stopped at 3.5, so a whole conversation billed
# zero. Nothing noticed, because a catalog with no prices in it is internally
# consistent. This runs offline on every commit — unlike `make check-models`,
# which needs a network and only runs at release.


@pytest.mark.parametrize("provider_cls", PRICED)
def test_every_model_in_a_priced_catalog_carries_a_price(provider_cls: type[AIProvider]) -> None:
    for model in provider_cls.available_models():
        if model.deprecated:
            # A retired id is one its vendor may have stopped quoting.
            continue
        assert model.pricing is not None, (
            f"{provider_cls.__name__}: {model.id} has no price — add it to the catalog "
            "from the vendor's own price list, or the consumer bills this model at zero"
        )


@pytest.mark.parametrize("provider_cls", PRICED)
def test_catalog_prices_are_well_formed(provider_cls: type[AIProvider]) -> None:
    for model in provider_cls.available_models():
        pricing = model.pricing
        if pricing is None:
            continue
        assert pricing.input_per_million > 0, f"{model.id} input rate"
        assert pricing.output_per_million > 0, f"{model.id} output rate"
        assert pricing.currency, f"{model.id} currency"
        assert pricing.verified <= date.today(), f"{model.id} verified in the future"
        for rate in (pricing.cache_read_per_million, pricing.cache_write_per_million):
            assert rate is None or rate >= 0, f"{model.id} negative cache rate"


def test_catalog_entry_returns_the_active_models_metadata() -> None:
    entry = MockAIProvider().catalog_entry()
    assert entry is not None
    assert entry.id == "mock"
    assert entry.context_window == 8192


def test_catalog_entry_is_none_for_an_unknown_model() -> None:
    class _Bare(AIProvider):
        @property
        def model_name(self) -> str:
            return "some-custom-local-model"

        async def generate(self, context: AIContext) -> AIResponse:
            return AIResponse(content="")

    assert _Bare().catalog_entry() is None


# --- supports_vision follows the catalog, not a parallel prefix table ----------
#
# Both providers used to answer this from a hardcoded tuple of prefixes that
# nothing updated with the lineup: Anthropic's stopped at claude-opus-4 and
# OpenAI's predated GPT-5 entirely, so every current model reported text-only
# and images were dropped before reaching the wire. These pin the fix.


@pytest.mark.parametrize(
    ("provider_cls", "config_cls"),
    [(AnthropicAIProvider, AnthropicConfig), (OpenAIAIProvider, OpenAIConfig)],
)
def test_vision_is_reported_for_every_vision_model_in_the_catalog(
    provider_cls: type[AIProvider], config_cls: type
) -> None:
    for model in provider_cls.available_models():
        if model.supports_vision is not True:
            continue
        provider = _bare(provider_cls)
        provider._config = config_cls(api_key=SecretStr("k"), model=model.id)
        assert provider.supports_vision, f"{model.id} reported text-only"


@pytest.mark.parametrize(
    ("provider_cls", "config_cls", "model", "expected"),
    [
        # Unknown id, right family → keep the family's capability.
        (AnthropicAIProvider, AnthropicConfig, "claude-something-newer", True),
        # Unknown id, no family → no claim to vision.
        (AnthropicAIProvider, AnthropicConfig, "some-local-model", False),
        (OpenAIAIProvider, OpenAIConfig, "gpt-5.9-unreleased", True),
        (OpenAIAIProvider, OpenAIConfig, "Qwen/Qwen3-VL-8B-Instruct", False),
    ],
)
def test_vision_falls_back_to_family_prefix_for_unknown_ids(
    provider_cls: type[AIProvider], config_cls: type, model: str, expected: bool
) -> None:
    provider = _bare(provider_cls)
    provider._config = config_cls(api_key=SecretStr("k"), model=model)
    assert provider.supports_vision is expected


def test_mock_catalog() -> None:
    ids = {m.id for m in MockAIProvider.available_models()}
    assert ids == {"mock", "mock-vision"}


def test_azure_has_no_offline_catalog() -> None:
    # Azure deployments are user-named → no meaningful curated list.
    assert AzureAIProvider.available_models() == []


# --- Live list_models (fake client injected, offline) --------------------------


async def test_openai_list_models_maps_and_merges() -> None:
    provider = _bare(OpenAIAIProvider)
    provider._client = SimpleNamespace(
        models=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(id="gpt-4o"), SimpleNamespace(id="text-embedding-3")]
                )
            )
        )
    )
    models = {m.id: m for m in await provider.list_models()}
    # Known chat model: backfilled from the curated catalog.
    assert models["gpt-4o"].display_name == "GPT-4o"
    assert models["gpt-4o"].supports_vision is True
    # Unknown id from the raw endpoint: passes through with id only.
    assert models["text-embedding-3"].display_name is None


async def test_anthropic_list_models_maps_and_merges() -> None:
    provider = _bare(AnthropicAIProvider)
    provider._client = SimpleNamespace(
        models=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(id="claude-opus-4-8", display_name="Claude Opus 4.8")]
                )
            )
        )
    )
    models = {m.id: m for m in await provider.list_models()}
    assert models["claude-opus-4-8"].display_name == "Claude Opus 4.8"
    # context_window comes from the curated catalog (the API list omits it).
    assert models["claude-opus-4-8"].context_window == 1_000_000


async def test_gemini_list_models_strips_prefix_and_filters() -> None:
    async def _pager() -> Any:
        yield SimpleNamespace(
            name="models/gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            input_token_limit=1_048_576,
            supported_actions=["generateContent"],
        )
        yield SimpleNamespace(
            name="models/text-embedding-004",
            display_name="Embedding",
            input_token_limit=2048,
            supported_actions=["embedContent"],
        )

    provider = _bare(GeminiAIProvider)
    provider._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(list=AsyncMock(return_value=_pager())))
    )
    ids = [m.id for m in await provider.list_models()]
    # "models/" prefix stripped; embedding model filtered out.
    assert ids == ["gemini-2.5-flash"]


async def test_ollama_list_models_reads_installed() -> None:
    provider = _bare(OllamaAIProvider)
    provider._client = SimpleNamespace(
        list=AsyncMock(
            return_value=SimpleNamespace(
                models=[
                    SimpleNamespace(model="llama3.2:latest"),
                    SimpleNamespace(model="custom-local-model"),
                ]
            )
        )
    )
    ids = [m.id for m in await provider.list_models()]
    assert ids == ["llama3.2:latest", "custom-local-model"]


async def test_mistral_list_models_maps_and_merges() -> None:
    provider = _bare(MistralAIProvider)
    provider._client = SimpleNamespace(
        models=SimpleNamespace(
            list_async=AsyncMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(id="mistral-large-latest"), SimpleNamespace(id="ft:xyz")]
                )
            )
        )
    )
    models = {m.id: m for m in await provider.list_models()}
    assert models["mistral-large-latest"].display_name == "Mistral Large 3"
    assert models["ft:xyz"].display_name is None

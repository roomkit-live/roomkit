"""Tests for the per-provider model catalog (available_models / list_models).

Curated catalogs are classmethods — they need neither an API key nor the
provider SDK, so they are exercised directly on the class. Live ``list_models``
overrides are tested by building the provider via ``__new__`` (skipping the
SDK-importing ``__init__``) and injecting a fake client, which keeps the tests
offline while still covering the response→ModelInfo mapping and curated merge.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.providers.ai import ModelInfo
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

# (provider class, config class) for the providers that ship a curated catalog.
CURATED = [
    (AnthropicAIProvider, AnthropicConfig),
    (OpenAIAIProvider, OpenAIConfig),
    (GeminiAIProvider, GeminiConfig),
    (MistralAIProvider, MistralConfig),
    (OllamaAIProvider, OllamaConfig),
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


@pytest.mark.parametrize(("provider_cls", "config_cls"), CURATED)
def test_default_config_model_is_in_catalog(
    provider_cls: type[AIProvider], config_cls: type
) -> None:
    # The model a provider defaults to should be discoverable in its catalog.
    default = config_cls.model_fields["model"].default
    ids = {m.id for m in provider_cls.available_models()}
    assert default in ids, f"{provider_cls.__name__} default {default!r} missing from catalog"


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

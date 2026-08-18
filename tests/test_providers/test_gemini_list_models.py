"""Live model listing for the Gemini family, across its two naming surfaces.

The Developer API and Vertex answer the same call with differently shaped
entries, and only the Vertex shape exercises the id and filtering rules: it
prefixes ids with ``publishers/google/models/`` and declares no
``supported_actions`` at all, so a listing there also carries embedding and
image models.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from roomkit.providers.gemini import GeminiAIProvider, GeminiConfig


class _Pager:
    """Async-iterable stand-in for the SDK's model pager."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for item in self._items:
                yield item

        return gen()


def _provider(items: list[Any]) -> GeminiAIProvider:
    provider = GeminiAIProvider(GeminiConfig(api_key="test-key"))
    provider._client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(list=AsyncMock(return_value=_Pager(items))))
    )
    return provider


def _entry(name: str, actions: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        supported_actions=actions,
        display_name=None,
        input_token_limit=None,
    )


async def test_vertex_ids_lose_their_publisher_prefix() -> None:
    """A Vertex id is the model name, not the resource path.

    The prefixed form matches nothing in the curated catalog (so metadata comes
    back empty) and a caller storing it as a model name gets a rejected request.
    """
    provider = _provider([_entry("publishers/google/models/gemini-2.5-flash")])

    models = await provider.list_models()

    assert [m.id for m in models] == ["gemini-2.5-flash"]
    # Proof the id now matches the curated entry: the backfill filled a window
    # the Vertex listing itself never reports.
    assert models[0].context_window == 1_048_576


async def test_developer_api_ids_lose_their_models_prefix() -> None:
    provider = _provider([_entry("models/gemini-3.5-flash", ["generateContent"])])

    models = await provider.list_models()

    assert [m.id for m in models] == ["gemini-3.5-flash"]


async def test_declared_actions_still_exclude_non_generative_models() -> None:
    provider = _provider(
        [
            _entry("models/gemini-3.5-flash", ["generateContent"]),
            _entry("models/text-embedding-004", ["embedContent"]),
        ]
    )

    models = await provider.list_models()

    assert [m.id for m in models] == ["gemini-3.5-flash"]


async def test_undeclared_actions_exclude_embedding_and_image_models() -> None:
    """Vertex declares no actions, so the family name is what separates them."""
    provider = _provider(
        [
            _entry("publishers/google/models/gemini-2.5-pro"),
            _entry("publishers/google/models/text-embedding-005"),
            _entry("publishers/google/models/multimodalembedding"),
            _entry("publishers/google/models/image-segmentation-001"),
        ]
    )

    models = await provider.list_models()

    assert [m.id for m in models] == ["gemini-2.5-pro"]


async def test_the_gemini_embedding_line_is_excluded_too() -> None:
    """``gemini-embedding-*`` carries the family name but answers embedContent."""
    provider = _provider(
        [
            _entry("publishers/google/models/gemini-3.5-flash"),
            _entry("publishers/google/models/gemini-embedding-2"),
        ]
    )

    models = await provider.list_models()

    assert [m.id for m in models] == ["gemini-3.5-flash"]


async def test_an_uncurated_gemini_model_is_kept() -> None:
    """A model newer than the curated catalog must not be filtered away."""
    provider = _provider([_entry("publishers/google/models/gemini-9-flash")])

    models = await provider.list_models()

    assert [m.id for m in models] == ["gemini-9-flash"]

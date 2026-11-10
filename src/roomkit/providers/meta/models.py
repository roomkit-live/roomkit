"""Offline metadata for Meta's Muse Spark, Muse Wonder and Muse Ember chat models.

Hand-maintained list returned by ``MetaAIProvider.available_models`` — the
context windows and rates roomkit needs before it can make a network call.
:meth:`~roomkit.providers.meta.ai.MetaAIProvider.list_models` reads the
account's ``/v1/models`` for what is live.

Sourced from the Meta Model API docs (dev.meta.ai/docs/models and
/docs/pricing-rate-limits), verified 2026-09-27 (Muse Wonder 2026-10-12), and
the ids against a live ``GET /v1/models`` the same days. Muse Ember's id,
window and rates were read from its model and pricing pages on 2026-11-09.

Every model reads images (and video, PDF and audio, which roomkit's AI parts do
not carry), calls tools and always reasons — ``"thinking"`` is on every entry,
and ``reasoning_effort="none"`` is refused by the service (checked live on
Muse Wonder too, which takes ``"minimal"`` to ``"max"``). The reasoning itself
is not returned by Chat Completions, only counted. Muse Wonder is a preview,
Standard mode only: it has no contributor id.

The ``-contributor`` ids are the same models at a fraction of the price,
because Meta trains its models on their traffic. Spark and Wonder have no
long-context premium. Ember is Standard-only and triples all token rates
above 200,000 input tokens, cached tokens included.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 9, 27)
_WONDER_VERIFIED = date(2026, 10, 12)
_EMBER_VERIFIED = date(2026, 11, 9)
_WINDOW = 1_048_576
_CAPS = ["tools", "thinking"]

_STANDARD = ModelPricing(
    input_per_million=1.25,
    output_per_million=4.25,
    cache_read_per_million=0.15,
    verified=_VERIFIED,
)
_WONDER = ModelPricing(
    input_per_million=4.00,
    output_per_million=20.00,
    cache_read_per_million=0.20,
    verified=_WONDER_VERIFIED,
)
_EMBER = ModelPricing(
    input_per_million=0.10,
    output_per_million=0.50,
    cache_read_per_million=0.004,
    long_context_threshold_tokens=200_000,
    long_context_input_multiplier=3.0,
    long_context_output_multiplier=3.0,
    verified=_EMBER_VERIFIED,
)
_CONTRIBUTOR = ModelPricing(
    input_per_million=0.10,
    output_per_million=0.20,
    cache_read_per_million=0.002,
    verified=_VERIFIED,
)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="muse-ember-1.0",
        display_name="Muse Ember 1.0",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_EMBER,
    ),
    ModelInfo(
        id="muse-wonder-1.0",
        display_name="Muse Wonder 1.0 (preview)",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_WONDER,
    ),
    ModelInfo(
        id="muse-spark-1.3",
        display_name="Muse Spark 1.3",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_STANDARD,
    ),
    ModelInfo(
        id="muse-spark-1.2",
        display_name="Muse Spark 1.2",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_STANDARD,
    ),
    ModelInfo(
        id="muse-spark-1.1",
        display_name="Muse Spark 1.1",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_STANDARD,
    ),
    ModelInfo(
        id="muse-spark-1.3-contributor",
        display_name="Muse Spark 1.3 (Contributor: Meta trains on the traffic)",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_CONTRIBUTOR,
    ),
    ModelInfo(
        id="muse-spark-1.2-contributor",
        display_name="Muse Spark 1.2 (Contributor: Meta trains on the traffic)",
        context_window=_WINDOW,
        supports_vision=True,
        capabilities=_CAPS,
        pricing=_CONTRIBUTOR,
    ),
]

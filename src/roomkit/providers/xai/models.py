"""Offline metadata for xAI Grok text/multimodal models.

Hand-maintained list returned by ``XAIAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what xAI currently offers. Call
:meth:`~roomkit.providers.xai.ai.XAIAIProvider.list_models` for that; it reads
the account's ``/v1/models``.

Sourced from the xAI model docs (docs.x.ai/developers/models), verified
2026-08-13.

Scope is the chat-capable text + multimodal models. The realtime speech-to-speech
models (``grok-2-audio``) belong to
:class:`~roomkit.providers.xai.realtime.XAIRealtimeProvider`, not here.

Every current Grok text model accepts image input and supports function calling
and structured outputs, so ``supports_vision=True`` throughout is documented
fact, not a guess. ``capabilities`` states reasoning support *positively*:
``"thinking"`` is present on the reasoning models and deliberately absent from
``grok-4.20-0309-non-reasoning``, which is the one model that rejects a
reasoning request. An id missing from this catalog (an alias such as
``grok-latest``, or a model newer than this snapshot) reports empty
capabilities — "unknown", which consumers read as "allow", matching the
family's behaviour.

Aliases are not modelled here (``ModelInfo`` has no alias field), but the API
accepts them. xAI publishes the rule rather than a table: ``<model>`` names the
latest stable release of that line and ``<model>-latest`` its newest, so
``grok-4.6-latest`` and ``grok-4.5-latest`` each resolve inside their own line.
A bare ``grok-latest`` tracks whichever line xAI currently calls latest and is
deliberately not restated here — it moved off ``grok-4.3`` when 4.6 shipped, and
would go stale again on the next release. Dated ids stay put:
``grok-4.20``/``grok-4.20-reasoning-latest`` → ``grok-4.20-0309-reasoning``;
``grok-code-fast``/``grok-code-fast-1`` → ``grok-build-0.1``.

Prices come from the same model docs, read 2026-08-13. Every entry represents
xAI's 200k-token prompt threshold, beyond which all token rates double.
``cache_write`` is unset: xAI publishes a cached-input rate but bills nothing
for populating the cache.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 8, 13)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="grok-4.6",
        display_name="Grok 4.6",
        context_window=500_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=6.0,
            # 4.6 charges more for a cache hit than 4.5 does at the same
            # input and output rates — the one place the two rate cards differ.
            cache_read_per_million=0.5,
            long_context_threshold_tokens=200_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=2.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="grok-4.5",
        display_name="Grok 4.5",
        context_window=500_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=6.0,
            cache_read_per_million=0.3,
            long_context_threshold_tokens=200_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=2.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="grok-4.3",
        display_name="Grok 4.3",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=2.5,
            cache_read_per_million=0.2,
            long_context_threshold_tokens=200_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=2.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="grok-4.20-0309-reasoning",
        display_name="Grok 4.20 (reasoning)",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=2.5,
            cache_read_per_million=0.2,
            long_context_threshold_tokens=200_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=2.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="grok-4.20-0309-non-reasoning",
        display_name="Grok 4.20 (non-reasoning)",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools"],
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=2.5,
            cache_read_per_million=0.2,
            long_context_threshold_tokens=200_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=2.0,
            verified=_VERIFIED,
        ),
    ),
    # Effort on this one sizes the agent swarm rather than the reasoning depth,
    # and it additionally accepts ``"xhigh"``.
    ModelInfo(
        id="grok-4.20-multi-agent-0309",
        display_name="Grok 4.20 Multi-Agent",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=2.5,
            cache_read_per_million=0.2,
            long_context_threshold_tokens=200_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=2.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="grok-build-0.1",
        display_name="Grok Build 0.1",
        context_window=256_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=1.0,
            output_per_million=2.0,
            cache_read_per_million=0.2,
            long_context_threshold_tokens=200_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=2.0,
            verified=_VERIFIED,
        ),
    ),
]

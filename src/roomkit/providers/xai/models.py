"""Curated catalog of xAI Grok text/multimodal models.

Hand-maintained, offline list returned by ``XAIAIProvider.available_models``.
Sourced from the xAI model docs (docs.x.ai/developers/models, verified
2026-07-30). Call :meth:`~roomkit.providers.xai.ai.XAIAIProvider.list_models`
for what the account's ``/v1/models`` endpoint reports right now.

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
accepts them: ``grok-4.5-latest``/``grok-build-latest`` → ``grok-4.5``;
``grok-4.3-latest``/``grok-latest`` → ``grok-4.3``;
``grok-4.20``/``grok-4.20-reasoning-latest`` → ``grok-4.20-0309-reasoning``;
``grok-code-fast``/``grok-code-fast-1`` → ``grok-build-0.1``.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="grok-4.5",
        display_name="Grok 4.5",
        context_window=500_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
    ),
    ModelInfo(
        id="grok-4.3",
        display_name="Grok 4.3",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
    ),
    ModelInfo(
        id="grok-4.20-0309-reasoning",
        display_name="Grok 4.20 (reasoning)",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
    ),
    ModelInfo(
        id="grok-4.20-0309-non-reasoning",
        display_name="Grok 4.20 (non-reasoning)",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools"],
    ),
    # Effort on this one sizes the agent swarm rather than the reasoning depth,
    # and it additionally accepts ``"xhigh"``.
    ModelInfo(
        id="grok-4.20-multi-agent-0309",
        display_name="Grok 4.20 Multi-Agent",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
    ),
    ModelInfo(
        id="grok-build-0.1",
        display_name="Grok Build 0.1",
        context_window=256_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
    ),
]

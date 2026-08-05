"""Offline metadata for OpenAI chat/multimodal models.

Hand-maintained list returned by ``OpenAIAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what OpenAI currently offers. Call ``OpenAIAIProvider.list_models()``
for that; it queries the account's ``/v1/models``.

Sourced from the OpenAI models and deprecations docs
(developers.openai.com/api/docs/models, .../deprecations), verified 2026-08-05.

Scope is the chat/responses-capable text + multimodal models; embeddings,
audio (whisper/tts), and image-generation models are intentionally omitted.

``deprecated=True`` marks a model OpenAI has given a shutdown date. Models
already past their shutdown are removed outright rather than flagged — a dead
id is a 404, and keeping it here only invites one.

The GPT-5.6 tier reaches its "pro" depth through ``reasoning.mode: "pro"`` on
the base model rather than through a separate id, which is why — unlike 5.4
and 5.5 — it has no ``-pro`` entries.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

_CTX_1M = 1_050_000

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-5.6-sol",
        display_name="GPT-5.6 Sol",
        context_window=_CTX_1M,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-5.6-terra",
        display_name="GPT-5.6 Terra",
        context_window=_CTX_1M,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-5.6-luna",
        display_name="GPT-5.6 Luna",
        context_window=_CTX_1M,
        supports_vision=True,
    ),
    ModelInfo(id="gpt-5.5", display_name="GPT-5.5", context_window=_CTX_1M, supports_vision=True),
    ModelInfo(
        id="gpt-5.5-pro",
        display_name="GPT-5.5 Pro",
        context_window=_CTX_1M,
        supports_vision=True,
    ),
    ModelInfo(id="gpt-5.4", display_name="GPT-5.4", context_window=_CTX_1M, supports_vision=True),
    ModelInfo(
        id="gpt-5.4-pro",
        display_name="GPT-5.4 Pro",
        context_window=_CTX_1M,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-5.4-mini",
        display_name="GPT-5.4 mini",
        context_window=400_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-5.4-nano",
        display_name="GPT-5.4 nano",
        context_window=400_000,
        supports_vision=True,
    ),
    ModelInfo(id="gpt-5.1", display_name="GPT-5.1", context_window=400_000, supports_vision=True),
    ModelInfo(
        id="gpt-4.1", display_name="GPT-4.1", context_window=1_047_576, supports_vision=True
    ),
    ModelInfo(
        id="gpt-4.1-mini",
        display_name="GPT-4.1 mini",
        context_window=1_047_576,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-4.1-nano",
        display_name="GPT-4.1 nano",
        context_window=1_047_576,
        supports_vision=True,
    ),
    ModelInfo(id="gpt-4o", display_name="GPT-4o", context_window=128_000, supports_vision=True),
    # --- Deprecated: OpenAI has announced a shutdown date -----------------
    ModelInfo(
        id="gpt-5",
        display_name="GPT-5",
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="gpt-5-mini",
        display_name="GPT-5 mini",
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="gpt-5-nano",
        display_name="GPT-5 nano",
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="gpt-5.2",
        display_name="GPT-5.2",
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="o3",
        display_name="o3",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="o3-pro",
        display_name="o3-pro",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="o4-mini",
        display_name="o4-mini",
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
    ),
]

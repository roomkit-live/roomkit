"""Offline metadata for Mistral chat/multimodal models.

Hand-maintained list returned by ``MistralAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what Mistral currently offers. Call ``MistralAIProvider.list_models()``
for that; it queries the account's models endpoint.

Sourced from the Mistral model cards (docs.mistral.ai/models), verified
2026-08-05.

Scope is general chat/multimodal models; embeddings, moderation, OCR, audio
(Voxtral), and code-completion (Codestral/Devstral) models are omitted. The whole
current Mistral 3 family is multimodal, and every member carries a 256k window
except Ministral 3 3B, which is half that. The deprecated 128k-tier models leave
``context_window`` as ``None`` — Mistral documents them only as "128k" without a
firm token integer.

Both the dated ids and their ``-latest`` aliases are listed so either form
resolves here.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

_CTX = 262_144

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="mistral-large-latest",
        display_name="Mistral Large 3",
        context_window=_CTX,
        supports_vision=True,
    ),
    ModelInfo(
        id="mistral-large-2512",
        display_name="Mistral Large 3",
        context_window=_CTX,
        supports_vision=True,
    ),
    ModelInfo(
        id="mistral-medium-latest",
        display_name="Mistral Medium 3.5",
        context_window=_CTX,
        supports_vision=True,
    ),
    ModelInfo(
        id="mistral-medium-3-5",
        display_name="Mistral Medium 3.5",
        context_window=_CTX,
        supports_vision=True,
    ),
    ModelInfo(
        id="mistral-small-latest",
        display_name="Mistral Small 4",
        context_window=_CTX,
        supports_vision=True,
    ),
    ModelInfo(
        id="mistral-small-2603",
        display_name="Mistral Small 4",
        context_window=_CTX,
        supports_vision=True,
    ),
    ModelInfo(
        id="ministral-14b-latest",
        display_name="Ministral 3 14B",
        context_window=_CTX,
        supports_vision=True,
    ),
    ModelInfo(
        id="ministral-8b-latest",
        display_name="Ministral 3 8B",
        context_window=_CTX,
        supports_vision=True,
    ),
    ModelInfo(
        id="ministral-3b-latest",
        display_name="Ministral 3 3B",
        context_window=131_072,
        supports_vision=True,
    ),
    ModelInfo(
        id="magistral-medium-latest",
        display_name="Magistral Medium 1.2",
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="magistral-small-latest",
        display_name="Magistral Small 1.2",
        supports_vision=True,
        deprecated=True,
    ),
    ModelInfo(
        id="pixtral-large-latest",
        display_name="Pixtral Large",
        supports_vision=True,
        deprecated=True,
    ),
]

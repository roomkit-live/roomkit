"""Curated catalog of OpenAI chat/multimodal models.

Hand-maintained, offline list returned by ``OpenAIAIProvider.available_models``.
Sourced from the OpenAI models docs (developers.openai.com/api/docs/models).
The lineup changes fast — refresh this against the live docs, or call
``OpenAIAIProvider.list_models()`` for what the account's API reports right now.

Scope is the chat/responses-capable text + multimodal models; embeddings,
audio (whisper/tts), and image-generation models are intentionally omitted.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-5.5", display_name="GPT-5.5", context_window=1_050_000, supports_vision=True
    ),
    ModelInfo(
        id="gpt-5.5-pro",
        display_name="GPT-5.5 Pro",
        context_window=1_050_000,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-5.4", display_name="GPT-5.4", context_window=1_050_000, supports_vision=True
    ),
    ModelInfo(
        id="gpt-5.4-pro",
        display_name="GPT-5.4 Pro",
        context_window=1_050_000,
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
    ModelInfo(id="o3", display_name="o3", context_window=200_000, supports_vision=True),
    ModelInfo(id="o3-pro", display_name="o3-pro", context_window=200_000, supports_vision=True),
    ModelInfo(id="gpt-5", display_name="GPT-5", context_window=400_000, supports_vision=True),
    ModelInfo(
        id="gpt-5-mini", display_name="GPT-5 mini", context_window=400_000, supports_vision=True
    ),
    ModelInfo(
        id="gpt-5-nano", display_name="GPT-5 nano", context_window=400_000, supports_vision=True
    ),
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
    ModelInfo(
        id="gpt-5.2",
        display_name="GPT-5.2",
        context_window=400_000,
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
    ModelInfo(
        id="gpt-5-codex",
        display_name="GPT-5-Codex",
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
    ),
]

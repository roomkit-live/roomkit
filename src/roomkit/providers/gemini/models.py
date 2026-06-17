"""Curated catalog of Google Gemini models.

Hand-maintained, offline list returned by ``GeminiAIProvider.available_models``.
Sourced from the Gemini API models docs (ai.google.dev/gemini-api/docs/models).
The lineup changes fast — refresh this against the live docs, or call
``GeminiAIProvider.list_models()`` for what the Gemini API reports right now.

Ids carry no ``models/`` prefix, matching the form ``GeminiConfig.model`` and the
generate-content calls use. Current text/multimodal models all report a
1,048,576-token input window and accept image input.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

_CTX = 1_048_576

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
    ),
    ModelInfo(
        id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro (Preview)",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
    ),
    ModelInfo(
        id="gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash-Lite",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
    ),
    ModelInfo(
        id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
    ),
    ModelInfo(
        id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
    ),
    ModelInfo(
        id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
    ),
    ModelInfo(
        id="gemini-3-flash-preview",
        display_name="Gemini 3 Flash (Preview)",
        context_window=_CTX,
        supports_vision=True,
        deprecated=True,
    ),
]

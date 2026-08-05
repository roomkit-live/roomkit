"""Offline metadata for Google Gemini models.

Hand-maintained list returned by ``GeminiAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what the Gemini API currently offers. Call
``GeminiAIProvider.list_models()`` for that; it reports ``inputTokenLimit``
per model straight from the API.

Sourced from the Gemini API models docs (ai.google.dev/gemini-api/docs/models),
verified 2026-08-05.

Ids carry no ``models/`` prefix, matching the form ``GeminiConfig.model`` and the
generate-content calls use. Current text/multimodal models all report a
1,048,576-token input window and accept image input.

Scope is the text/multimodal generate-content models. The image-generation
(``*-image``, Nano Banana), TTS, Live, embedding, Veo, Lyria, robotics and
Deep Research variants are out of scope for this provider. Google lists
``gemini-3-pro-preview`` and ``gemini-3.1-flash-lite-preview`` as shut down;
they are absent rather than flagged, since a retired id is a 404.
"""

from __future__ import annotations

from roomkit.providers.ai.base import ModelInfo

_CTX = 1_048_576

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-3.6-flash",
        display_name="Gemini 3.6 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
    ),
    ModelInfo(
        id="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        context_window=_CTX,
        supports_vision=True,
        capabilities=["thinking", "audio", "video"],
    ),
    ModelInfo(
        id="gemini-3.5-flash-lite",
        display_name="Gemini 3.5 Flash-Lite",
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

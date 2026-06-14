"""Gemini on Vertex AI — same models, regional endpoint, no data retention.

Vertex AI serves the very same Gemini models as the public Gemini Developer
API, but through a Google Cloud project with a **pinned region**. That is what
makes it the right backend when data residency matters (e.g. Québec Law 25 /
PIPEDA): prompts and responses are processed in the chosen region and are not
retained to train Google's models. The only differences from
:class:`~roomkit.providers.gemini.ai.GeminiAIProvider` are how the client is
built (Vertex mode + ADC auth instead of an API key) — everything else
(generation, streaming, thinking, model catalog) is inherited unchanged.
"""

from __future__ import annotations

from pydantic import SecretStr

from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig


class GeminiVertexConfig(GeminiConfig):
    """Gemini-on-Vertex configuration.

    Subclasses :class:`GeminiConfig`, inheriting every generation field
    (``model``, ``max_tokens``, ``temperature``, ``thinking_level``) so the two
    cannot drift. Authentication uses Application Default Credentials (ADC) — the
    standard Google Cloud chain (``gcloud auth application-default login``,
    ``GOOGLE_APPLICATION_CREDENTIALS``, or workload identity) — so ``api_key`` is
    not required.
    """

    api_key: SecretStr | None = None
    """Optional and unused on Vertex — authentication is via ADC, not an API key."""

    project: str
    """Google Cloud project id that hosts the Vertex AI API."""

    location: str
    """Vertex region — **required, no default**. Pin it to keep data in-region
    for residency (e.g. ``"northamerica-northeast1"`` for Montréal,
    ``"europe-west1"``). A default like ``"global"`` could route out of region
    and defeat the whole point, so the choice is made explicit."""


class GeminiVertexProvider(GeminiAIProvider):
    """Gemini provider backed by Vertex AI in a specific Google Cloud region.

    Subclasses :class:`GeminiAIProvider` — only client construction differs
    (Vertex mode + ADC instead of an API key). All generation, streaming,
    thinking, and model discovery are inherited.
    """

    _config: GeminiVertexConfig

    def __init__(self, config: GeminiVertexConfig) -> None:
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiVertexProvider. "
                "Install it with: pip install roomkit[gemini]"
            ) from exc

        self._config = config
        self._genai = _genai
        self._types = _types
        self._client = _genai.Client(
            vertexai=True,
            project=config.project,
            location=config.location,
        )

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

import json
from typing import Any

from pydantic import SecretStr

from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig


class GeminiVertexConfig(GeminiConfig):
    """Gemini-on-Vertex configuration.

    Subclasses :class:`GeminiConfig`, inheriting every generation field
    (``model``, ``max_tokens``, ``temperature``, ``thinking_level``) so the two
    cannot drift. There is no API key on Vertex: the caller is authenticated
    either by an explicit service-account key or, failing that, by Application
    Default Credentials — the standard Google chain
    (``gcloud auth application-default login``, ``GOOGLE_APPLICATION_CREDENTIALS``,
    workload identity).
    """

    api_key: SecretStr | None = None
    """Optional and unused on Vertex — the identity comes from
    ``service_account_json`` or from ADC, never from a key on the request."""

    project: str
    """Google Cloud project id that hosts the Vertex AI API."""

    location: str
    """Vertex region — **required, no default**. Pin it to keep data in-region
    for residency (e.g. ``"northamerica-northeast1"`` for Montréal,
    ``"europe-west1"``). A default like ``"global"`` could route out of region
    and defeat the whole point, so the choice is made explicit."""

    service_account_json: SecretStr | None = None
    """A service-account key file's contents, as JSON, authenticating **as**
    that account instead of as the process.

    ADC answers "who is this machine", which is the wrong question wherever one
    deployment serves several projects: the ambient identity belongs to whoever
    runs the server, so a caller naming someone else's project gets
    ``PERMISSION_DENIED`` no matter what it puts in ``project``. Passing a key
    here makes the identity travel with the configuration, which is what lets
    one process serve one project per tenant.

    ``None`` keeps the ADC chain, which stays right for a single-project
    deployment and for local development."""


class GeminiVertexProvider(GeminiAIProvider):
    """Gemini provider backed by Vertex AI in a specific Google Cloud region.

    Subclasses :class:`GeminiAIProvider` — only client construction differs
    (Vertex mode, and an identity that is not an API key). All generation,
    streaming, thinking, and model discovery are inherited.
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
            credentials=self._credentials(config),
        )

    @staticmethod
    def _credentials(config: GeminiVertexConfig) -> Any | None:
        """Service-account credentials from the configured key, else ``None``.

        ``None`` is not a failure: it hands the client back to the ADC chain,
        which is the right answer for a deployment that owns its project.

        The scope is stated explicitly because a key parsed without one yields
        credentials the transport refuses, and ``cloud-platform`` is the only
        scope Vertex accepts — there is no narrower one to ask for.
        """
        if config.service_account_json is None:
            return None
        raw = config.service_account_json.get_secret_value().strip()
        if not raw:
            return None
        try:
            from google.oauth2 import service_account
        except ImportError as exc:  # pragma: no cover - google-auth ships with google-genai
            raise ImportError(
                "google-auth is required to authenticate with a service-account key."
            ) from exc
        try:
            info = json.loads(raw)
        except ValueError as exc:
            raise ValueError(
                "service_account_json is not valid JSON. Paste the key file's "
                "contents, not its path."
            ) from exc
        try:
            return service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except ValueError as exc:
            # The other JSON Google hands out is an authorized_user file, which
            # names no service account and fails here with a terser message.
            raise ValueError(
                f"service_account_json is not a service-account key "
                f"(type={info.get('type', 'missing')!r}): {exc}"
            ) from exc

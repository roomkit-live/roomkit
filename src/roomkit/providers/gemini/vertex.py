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
from roomkit.providers.gemini.sdk import build_vertex_genai_client


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

    impersonate_service_account: str | None = None
    """Email of a service account to **borrow** instead of holding its key.

    The identity a caller needs and the secret it holds are separate problems,
    and organizations increasingly forbid the second: Google enforces
    ``constraints/iam.disableServiceAccountKeyCreation`` by default on recent
    organizations, so a project owner cannot hand out a key even when willing.
    Here the owner instead grants this deployment's own identity
    ``roles/iam.serviceAccountTokenCreator`` on one of their service accounts,
    and Vertex is called as that account with short-lived tokens nobody ever
    downloads — revocable from their side, in one command, without telling us.

    Combines with the fields above rather than replacing them: the borrowing
    identity is ``service_account_json`` when set, otherwise ADC."""


class GeminiVertexProvider(GeminiAIProvider):
    """Gemini provider backed by Vertex AI in a specific Google Cloud region.

    Subclasses :class:`GeminiAIProvider` — only client construction differs
    (Vertex mode, and an identity that is not an API key). All generation,
    streaming, thinking, and model discovery are inherited.
    """

    _config: GeminiVertexConfig

    def __init__(self, config: GeminiVertexConfig) -> None:
        self._config = config
        # The client carries the connect/read split; see ``build_genai_client``
        # for why it cannot go on the request.
        self._client, self._http, self._types = build_vertex_genai_client(
            config,
            provider="GeminiVertexProvider",
            project=config.project,
            location=config.location,
            credentials=self._credentials(config),
        )

    @staticmethod
    def _credentials(config: GeminiVertexConfig) -> Any | None:
        """The identity this client calls as, or ``None`` for the ADC chain.

        Three answers, in the order they are read: borrow a named service
        account, authenticate as a configured key, or stay ambient. ``None`` is
        not a failure — it is the right answer for a deployment that owns the
        project it calls.

        The scope is stated explicitly throughout because credentials parsed
        without one are refused by the transport, and ``cloud-platform`` is the
        only scope Vertex accepts.
        """
        source = GeminiVertexProvider._key_credentials(config)
        target = (config.impersonate_service_account or "").strip()
        if not target:
            return source
        return GeminiVertexProvider._borrowed_credentials(source, target)

    @staticmethod
    def _borrowed_credentials(source: Any | None, target: str) -> Any:
        """Short-lived credentials for ``target``, minted by whoever we are.

        The source is this deployment's own identity — its key when configured,
        else ADC, resolved explicitly here because impersonation needs something
        to sign the request with and the client cannot fall back on its own.
        """
        try:
            import google.auth
            from google.auth import impersonated_credentials
        except ImportError as exc:  # pragma: no cover - ships with google-genai
            raise ImportError("google-auth is required to impersonate a service account.") from exc

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        if source is None:
            try:
                source, _ = google.auth.default(scopes=scopes)
            except Exception as exc:
                raise ValueError(
                    f"Cannot borrow {target}: this deployment has no Google credentials "
                    "of its own to borrow it with."
                ) from exc
        return impersonated_credentials.Credentials(
            source_credentials=source,
            target_principal=target,
            target_scopes=scopes,
        )

    @staticmethod
    def _key_credentials(config: GeminiVertexConfig) -> Any | None:
        """Credentials from the configured service-account key, else ``None``."""
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

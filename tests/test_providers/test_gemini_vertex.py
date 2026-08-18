"""Tests for the Gemini-on-Vertex provider."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr, ValidationError

from roomkit.providers.gemini.vertex import GeminiVertexConfig, GeminiVertexProvider


def _mock_genai_module() -> MagicMock:
    """Return a MagicMock that behaves like the google.genai module."""
    mod = MagicMock()
    mod.types = MagicMock()
    return mod


def _genai_modules(mock_genai: MagicMock) -> dict[str, Any]:
    """Build the sys.modules patch dict so the google.genai imports resolve."""
    return {
        "google": MagicMock(genai=mock_genai),
        "google.genai": mock_genai,
    }


def _vconfig(**overrides: Any) -> GeminiVertexConfig:
    defaults: dict[str, Any] = {
        "project": "my-proj",
        "location": "northamerica-northeast1",
    }
    defaults.update(overrides)
    return GeminiVertexConfig(**defaults)


class TestGeminiVertexConfig:
    def test_project_required(self) -> None:
        with pytest.raises(ValidationError):
            GeminiVertexConfig(location="us-central1")  # type: ignore[call-arg]

    def test_location_required(self) -> None:
        # Location is intentionally required (no default) to keep data in-region.
        with pytest.raises(ValidationError):
            GeminiVertexConfig(project="p")  # type: ignore[call-arg]

    def test_api_key_optional(self) -> None:
        # Vertex authenticates via ADC, not an API key.
        assert _vconfig().api_key is None

    def test_inherits_gemini_defaults(self) -> None:
        cfg = _vconfig()
        assert cfg.model == "gemini-3.1-flash-lite"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 1.0

    def test_custom_values(self) -> None:
        cfg = _vconfig(project="p2", location="europe-west1", model="gemini-3.5-flash")
        assert cfg.project == "p2"
        assert cfg.location == "europe-west1"
        assert cfg.model == "gemini-3.5-flash"


class TestGeminiVertexProvider:
    def test_client_built_in_vertex_mode(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.vertex import GeminiVertexProvider

            GeminiVertexProvider(_vconfig())

            mock_genai.Client.assert_called_once_with(
                vertexai=True,
                project="my-proj",
                location="northamerica-northeast1",
                # No key configured: the client falls back to the ADC chain,
                # which is what a single-project deployment runs on.
                credentials=None,
            )

    def test_no_api_key_passed_to_client(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.vertex import GeminiVertexProvider

            GeminiVertexProvider(_vconfig(api_key="should-be-ignored"))

            _, kwargs = mock_genai.Client.call_args
            assert "api_key" not in kwargs

    def test_inherits_gemini_provider(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.vertex import GeminiVertexProvider

            provider = GeminiVertexProvider(_vconfig(model="gemini-3.5-flash"))
            # Subclasses GeminiAIProvider — name-based so it survives the
            # importlib.reload other Gemini tests do to the parent module.
            assert "GeminiAIProvider" in {c.__name__ for c in type(provider).__mro__}
            # Inherited behaviour works unchanged.
            assert provider.model_name == "gemini-3.5-flash"
            assert provider.supports_vision is True


_EMAIL = "luge-agent@example-project.iam.gserviceaccount.com"


def _service_account_key() -> str:
    """A structurally real key file — generated, never checked in."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "example-project",
            "private_key_id": "0" * 40,
            "private_key": pem,
            "client_email": _EMAIL,
            "client_id": "1234567890",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


class TestGeminiVertexIdentity:
    """Who the provider authenticates as.

    Vertex takes no API key, so the identity is either a service account named
    by a key in the configuration or whatever ADC the process happens to hold.
    The distinction decides whether one deployment can serve several projects:
    ADC answers "who is this machine", so a caller naming another tenant's
    project is refused however correct that project id is.
    """

    def test_no_key_leaves_the_adc_chain_in_place(self) -> None:
        assert GeminiVertexProvider._credentials(_vconfig()) is None
        assert (
            GeminiVertexProvider._credentials(_vconfig(service_account_json=SecretStr("  ")))
            is None
        )

    def test_a_key_authenticates_as_that_service_account(self) -> None:
        creds = GeminiVertexProvider._credentials(
            _vconfig(service_account_json=SecretStr(_service_account_key()))
        )

        assert creds is not None
        assert creds.service_account_email == _EMAIL
        # Stated explicitly: credentials parsed without a scope are refused by
        # the transport, and cloud-platform is the only scope Vertex accepts.
        assert list(creds.scopes) == ["https://www.googleapis.com/auth/cloud-platform"]

    def test_the_key_reaches_the_client(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.vertex import GeminiVertexProvider as Provider

            Provider(_vconfig(service_account_json=SecretStr(_service_account_key())))

            _, kwargs = mock_genai.Client.call_args
            assert kwargs["credentials"] is not None
            assert kwargs["credentials"].service_account_email == _EMAIL

    def test_borrowing_an_account_mints_tokens_for_it(self) -> None:
        """The identity a caller needs and the secret it holds are separate.

        Organizations increasingly forbid the secret — Google enforces
        ``disableServiceAccountKeyCreation`` by default on recent ones — so a
        project owner grants us the right to borrow an account instead, and no
        key is ever downloaded.
        """
        creds = GeminiVertexProvider._credentials(
            _vconfig(
                service_account_json=SecretStr(_service_account_key()),
                impersonate_service_account="theirs@their-project.iam.gserviceaccount.com",
            )
        )

        assert creds is not None
        assert creds.service_account_email == "theirs@their-project.iam.gserviceaccount.com"
        # The key is the borrower, not the borrowed: it signs the request that
        # asks Google for the other account's token.
        assert creds._source_credentials.service_account_email == _EMAIL

    def test_borrowing_without_an_identity_of_our_own_says_which_half_is_missing(self) -> None:
        """Impersonation needs something to sign with, and "no credentials at
        all" must not read as "that account refused us"."""
        import google.auth

        def _no_adc(**_kwargs: Any) -> Any:
            raise google.auth.exceptions.DefaultCredentialsError("none found")

        with patch.object(google.auth, "default", _no_adc):
            with pytest.raises(ValueError, match="no Google credentials of its own"):
                GeminiVertexProvider._credentials(
                    _vconfig(
                        impersonate_service_account="theirs@their-project.iam.gserviceaccount.com"
                    )
                )

    def test_a_path_instead_of_the_key_says_so(self) -> None:
        """The likeliest mistake is pasting the filename, and it must not read
        as a credentials failure hours later."""
        with pytest.raises(ValueError, match="not valid JSON"):
            GeminiVertexProvider._credentials(
                _vconfig(service_account_json=SecretStr("/etc/secrets/key.json"))
            )

    def test_the_other_google_json_is_named_for_what_it_is(self) -> None:
        """An ``authorized_user`` file is what ``gcloud auth`` writes, so it is
        the other thing someone will paste here."""
        adc = json.dumps({"type": "authorized_user", "client_id": "x", "refresh_token": "y"})

        with pytest.raises(ValueError, match="authorized_user"):
            GeminiVertexProvider._credentials(_vconfig(service_account_json=SecretStr(adc)))

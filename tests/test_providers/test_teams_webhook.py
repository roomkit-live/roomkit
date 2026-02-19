"""Tests for BotFrameworkTeamsProvider.verify_signature()."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from roomkit.providers.teams.config import TeamsConfig


def _config(**overrides: Any) -> TeamsConfig:
    defaults: dict[str, Any] = {
        "app_id": "test-app-id",
        "app_password": "test-app-password",
    }
    defaults.update(overrides)
    return TeamsConfig(**defaults)


def _make_provider(config: TeamsConfig | None = None) -> Any:
    """Create a BotFrameworkTeamsProvider with mocked botbuilder dependency.

    The BotFrameworkTeamsProvider constructor imports from ``botbuilder.core``
    at runtime, so we patch those modules in ``sys.modules`` before importing
    the class.
    """
    cfg = config or _config()

    mock_core = MagicMock()
    mock_schema = MagicMock()

    with patch.dict(sys.modules, {
        "botbuilder": MagicMock(),
        "botbuilder.core": mock_core,
        "botbuilder.schema": mock_schema,
    }):
        from roomkit.providers.teams.bot_framework import BotFrameworkTeamsProvider

        provider = BotFrameworkTeamsProvider(cfg)

    return provider


def _mock_jwt_module(
    *,
    decode_return: dict[str, Any] | None = None,
    get_key_side_effect: Exception | None = None,
) -> MagicMock:
    """Build a mock ``jwt`` module suitable for patching into ``sys.modules``.

    The ``verify_signature`` method does::

        import jwt
        from jwt import PyJWKClient

    So the mock module needs both ``decode`` and ``PyJWKClient`` attributes.
    """
    mock = MagicMock()

    mock_signing_key = MagicMock()
    mock_jwk_client_instance = MagicMock()

    if get_key_side_effect:
        mock_jwk_client_instance.get_signing_key_from_jwt.side_effect = get_key_side_effect
    else:
        mock_jwk_client_instance.get_signing_key_from_jwt.return_value = mock_signing_key

    mock.PyJWKClient.return_value = mock_jwk_client_instance

    if decode_return is not None:
        mock.decode.return_value = decode_return

    return mock


class TestTeamsSignatureVerification:
    """Tests for BotFrameworkTeamsProvider.verify_signature()."""

    def test_verify_strips_bearer_prefix(self) -> None:
        provider = _make_provider()

        mock_jwt = _mock_jwt_module(decode_return={
            "iss": "https://api.botframework.com",
            "aud": "test-app-id",
        })

        with patch.dict(sys.modules, {"jwt": mock_jwt}):
            result = provider.verify_signature(b"ignored", "Bearer some.jwt.token")

        assert result is True
        # Verify the Bearer prefix was stripped — jwt.decode received the raw token
        mock_jwt.decode.assert_called()
        call_args = mock_jwt.decode.call_args
        assert call_args[0][0] == "some.jwt.token"

    def test_verify_empty_token(self) -> None:
        provider = _make_provider()

        mock_jwt = _mock_jwt_module()

        with patch.dict(sys.modules, {"jwt": mock_jwt}):
            result = provider.verify_signature(b"ignored", "")

        assert result is False

    def test_verify_invalid_jwt(self) -> None:
        provider = _make_provider()

        mock_jwt = _mock_jwt_module(
            get_key_side_effect=Exception("Invalid token"),
        )

        with patch.dict(sys.modules, {"jwt": mock_jwt}):
            result = provider.verify_signature(b"ignored", "Bearer invalid.jwt.token")

        assert result is False

    def test_verify_no_pyjwt(self) -> None:
        provider = _make_provider()

        # Setting a sys.modules entry to None causes ``import jwt`` to raise
        # ImportError, which verify_signature converts to ValueError.
        with patch.dict(sys.modules, {"jwt": None}), pytest.raises(ValueError, match="PyJWT"):
            provider.verify_signature(b"ignored", "Bearer some.jwt.token")

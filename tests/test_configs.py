"""Tests for provider configurations."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.sendgrid.config import SendGridConfig
from roomkit.providers.twilio.config import TwilioConfig


class TestTwilioConfig:
    def test_create(self) -> None:
        cfg = TwilioConfig(
            account_sid="AC123",
            auth_token=SecretStr("secret"),
            from_number="+15551234567",
        )
        assert cfg.account_sid == "AC123"
        assert cfg.auth_token.get_secret_value() == "secret"

    def test_secret_str_masked(self) -> None:
        cfg = TwilioConfig(
            account_sid="AC123",
            auth_token=SecretStr("secret"),
            from_number="+15551234567",
        )
        assert "secret" not in str(cfg.auth_token)

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            TwilioConfig()  # type: ignore[call-arg]


class TestAnthropicConfig:
    def test_defaults(self) -> None:
        cfg = AnthropicConfig(api_key=SecretStr("sk-test"))
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.max_tokens == 1024


class TestSendGridConfig:
    def test_create(self) -> None:
        cfg = SendGridConfig(api_key=SecretStr("SG.test"), from_email="test@example.com")
        assert cfg.from_name is None

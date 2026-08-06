"""Tests for provider configurations."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.openai.config import OpenAIConfig
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
    def test_model_is_required(self) -> None:
        with pytest.raises(ValidationError):
            AnthropicConfig(api_key=SecretStr("sk-test"))  # type: ignore[call-arg]

    def test_request_defaults_profile_selected_model(self) -> None:
        cfg = AnthropicConfig(api_key=SecretStr("sk-test"), model="claude-opus-5")
        assert cfg.model == "claude-opus-5"
        assert cfg.max_tokens == 1024
        assert cfg.use_adaptive_thinking is True
        assert cfg.supports_custom_temperature is False


class TestOpenAIConfig:
    def test_model_is_required(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIConfig(api_key=SecretStr("sk-test"))  # type: ignore[call-arg]

    def test_request_defaults_profile_selected_model(self) -> None:
        cfg = OpenAIConfig(api_key=SecretStr("sk-test"), model="gpt-5.6-sol")
        assert cfg.model == "gpt-5.6-sol"
        assert cfg.use_max_completion_tokens is True
        assert cfg.supports_custom_temperature is False


class TestSendGridConfig:
    def test_create(self) -> None:
        cfg = SendGridConfig(api_key=SecretStr("SG.test"), from_email="test@example.com")
        assert cfg.from_name is None

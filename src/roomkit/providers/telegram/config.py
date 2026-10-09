"""Telegram Bot API provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr, field_validator

from roomkit.providers.url_safety import validate_api_base_url


class TelegramConfig(BaseModel):
    """Telegram Bot API provider configuration."""

    bot_token: SecretStr
    webhook_secret: SecretStr | None = None
    timeout: float = 30.0
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, separate from the request ``timeout``."""
    # Opt in to Bot API 10.1 Rich Messages (native tables/headings) for text
    # sends, falling back to entity formatting on any failure. Off by default:
    # the format is new and older Telegram clients may not render it.
    rich_messages: bool = False
    api_base_url: str = "https://api.telegram.org"
    """The Bot API server; another one (a self-hosted Bot API server, a local
    fake) must be HTTPS, or HTTP on this machine only."""

    @field_validator("api_base_url")
    @classmethod
    def _secure_api_base_url(cls, value: str) -> str:
        return validate_api_base_url(value)

    @property
    def base_url(self) -> str:
        return f"{self.api_base_url}/bot{self.bot_token.get_secret_value()}"

    @property
    def file_base_url(self) -> str:
        """Base URL for downloading a file resolved by ``getFile``.

        Telegram serves file content from a different path than the one that
        answers Bot API methods, so this is not a suffix of :attr:`base_url`.
        Like it, it embeds the bot token and must never reach a log.
        """
        return f"{self.api_base_url}/file/bot{self.bot_token.get_secret_value()}"

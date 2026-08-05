"""Telegram Bot API provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class TelegramConfig(BaseModel):
    """Telegram Bot API provider configuration."""

    bot_token: SecretStr
    webhook_secret: SecretStr | None = None
    timeout: float = 30.0
    # Opt in to Bot API 10.1 Rich Messages (native tables/headings) for text
    # sends, falling back to entity formatting on any failure. Off by default:
    # the format is new and older Telegram clients may not render it.
    rich_messages: bool = False

    @property
    def base_url(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token.get_secret_value()}"

    @property
    def file_base_url(self) -> str:
        """Base URL for downloading a file resolved by ``getFile``.

        Telegram serves file content from a different path than the one that
        answers Bot API methods, so this is not a suffix of :attr:`base_url`.
        Like it, it embeds the bot token and must never reach a log.
        """
        return f"https://api.telegram.org/file/bot{self.bot_token.get_secret_value()}"

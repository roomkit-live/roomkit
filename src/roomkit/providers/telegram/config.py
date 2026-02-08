"""Telegram Bot API provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class TelegramConfig(BaseModel):
    """Telegram Bot API provider configuration."""

    bot_token: SecretStr
    timeout: float = 30.0

    @property
    def base_url(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token.get_secret_value()}"

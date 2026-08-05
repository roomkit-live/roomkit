"""Telegram Bot API provider."""

from roomkit.providers.telegram.api import TelegramBotAPI
from roomkit.providers.telegram.base import TelegramProvider
from roomkit.providers.telegram.bot import TelegramBotProvider
from roomkit.providers.telegram.config import TelegramConfig
from roomkit.providers.telegram.entities import entity_text, mentions_bot
from roomkit.providers.telegram.mock import MockTelegramProvider
from roomkit.providers.telegram.update import (
    TelegramCallback,
    TelegramUpdate,
    parse_telegram_callback,
    parse_telegram_update,
)
from roomkit.providers.telegram.webhook import (
    TelegramMessageParts,
    parse_telegram_message,
    parse_telegram_webhook,
)

__all__ = [
    "MockTelegramProvider",
    "TelegramBotAPI",
    "TelegramBotProvider",
    "TelegramCallback",
    "TelegramConfig",
    "TelegramMessageParts",
    "TelegramProvider",
    "TelegramUpdate",
    "entity_text",
    "mentions_bot",
    "parse_telegram_callback",
    "parse_telegram_message",
    "parse_telegram_update",
    "parse_telegram_webhook",
]

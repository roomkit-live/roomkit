"""Tests for Telegram webhook parsing."""

from __future__ import annotations

from roomkit.models.event import LocationContent, TextContent
from roomkit.providers.telegram import parse_telegram_webhook


class TestParseTelegramWebhook:
    def test_parse_text_message(self) -> None:
        payload = {
            "update_id": 100,
            "message": {
                "message_id": 1,
                "from": {"id": 999, "first_name": "Alice"},
                "chat": {"id": 555, "type": "private"},
                "date": 1700000000,
                "text": "Hello from Telegram",
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")

        assert len(messages) == 1
        msg = messages[0]
        assert msg.channel_id == "tg-main"
        assert msg.sender_id == "999"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello from Telegram"
        assert msg.external_id == "1"
        assert msg.idempotency_key == "1"
        assert msg.metadata["chat_id"] == "555"
        assert msg.metadata["date"] == 1700000000

    def test_parse_photo_message(self) -> None:
        payload = {
            "update_id": 101,
            "message": {
                "message_id": 2,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000001,
                "photo": [
                    {"file_id": "small_id", "width": 90, "height": 90},
                    {"file_id": "large_id", "width": 800, "height": 600},
                ],
                "caption": "Nice pic",
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")

        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Nice pic"
        assert msg.metadata["file_id"] == "large_id"
        assert msg.metadata["media_type"] == "photo"

    def test_parse_location_message(self) -> None:
        payload = {
            "update_id": 102,
            "message": {
                "message_id": 3,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000002,
                "location": {"latitude": 48.8566, "longitude": 2.3522},
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")

        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg.content, LocationContent)
        assert msg.content.latitude == 48.8566
        assert msg.content.longitude == 2.3522

    def test_parse_empty_payload(self) -> None:
        messages = parse_telegram_webhook({}, channel_id="tg-main")
        assert messages == []

    def test_parse_non_message_update_skipped(self) -> None:
        payload = {
            "update_id": 103,
            "callback_query": {
                "id": "abc",
                "from": {"id": 999},
                "data": "button_1",
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")
        assert messages == []

    def test_parse_unsupported_message_type_skipped(self) -> None:
        payload = {
            "update_id": 104,
            "message": {
                "message_id": 4,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000003,
                "sticker": {"file_id": "sticker_id"},
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")
        assert messages == []

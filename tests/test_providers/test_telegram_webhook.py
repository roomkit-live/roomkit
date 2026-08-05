"""Tests for Telegram webhook parsing."""

from __future__ import annotations

from roomkit.models.event import LocationContent, TextContent
from roomkit.providers.telegram import parse_telegram_message, parse_telegram_webhook


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

    def test_parse_voice_message(self) -> None:
        """A voice note has no caption — the file_id carries the message."""
        payload = {
            "update_id": 105,
            "message": {
                "message_id": 5,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000004,
                "voice": {
                    "duration": 3,
                    "mime_type": "audio/ogg",
                    "file_id": "AwACAgIAAxkBAAIC",
                    "file_unique_id": "AgADbQ4AAlOZAUo",
                    "file_size": 8342,
                },
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")

        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == ""
        assert msg.metadata["file_id"] == "AwACAgIAAxkBAAIC"
        assert msg.metadata["media_type"] == "voice"
        assert msg.metadata["duration"] == 3
        assert msg.metadata["mime_type"] == "audio/ogg"
        assert msg.metadata["file_size"] == 8342

    def test_parse_audio_message(self) -> None:
        payload = {
            "update_id": 106,
            "message": {
                "message_id": 6,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000005,
                "audio": {
                    "duration": 215,
                    "file_name": "song.mp3",
                    "mime_type": "audio/mpeg",
                    "performer": "Artist",
                    "title": "Song",
                    "file_id": "CQACAgIAAxkBAAID",
                    "file_unique_id": "AgADXQ8AAg",
                    "file_size": 3452160,
                },
                "caption": "listen to this",
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")

        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "listen to this"
        assert msg.metadata["file_id"] == "CQACAgIAAxkBAAID"
        assert msg.metadata["media_type"] == "audio"
        assert msg.metadata["duration"] == 215
        assert msg.metadata["file_name"] == "song.mp3"
        assert msg.metadata["mime_type"] == "audio/mpeg"

    def test_parse_video_note_message(self) -> None:
        """Video notes carry a duration but no mime_type — absent keys stay absent."""
        payload = {
            "update_id": 107,
            "message": {
                "message_id": 7,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000006,
                "video_note": {
                    "duration": 8,
                    "length": 384,
                    "file_id": "DQACAgIAAxkBAAIE",
                    "file_unique_id": "AgADhw4AAg",
                    "file_size": 512000,
                },
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")

        assert len(messages) == 1
        msg = messages[0]
        assert msg.metadata["file_id"] == "DQACAgIAAxkBAAIE"
        assert msg.metadata["media_type"] == "video_note"
        assert msg.metadata["duration"] == 8
        assert "mime_type" not in msg.metadata
        assert "file_name" not in msg.metadata

    def test_parse_video_message(self) -> None:
        payload = {
            "update_id": 108,
            "message": {
                "message_id": 8,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000007,
                "video": {
                    "duration": 12,
                    "width": 640,
                    "height": 480,
                    "file_name": "IMG_0042.MOV",
                    "mime_type": "video/quicktime",
                    "file_id": "BAACAgIAAxkBAAIF",
                    "file_unique_id": "AgADrg4AAg",
                    "file_size": 1048576,
                },
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")

        assert len(messages) == 1
        msg = messages[0]
        assert msg.metadata["file_id"] == "BAACAgIAAxkBAAIF"
        assert msg.metadata["media_type"] == "video"
        assert msg.metadata["duration"] == 12
        assert msg.metadata["file_name"] == "IMG_0042.MOV"
        assert msg.metadata["mime_type"] == "video/quicktime"

    def test_parse_document_message(self) -> None:
        payload = {
            "update_id": 109,
            "message": {
                "message_id": 9,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000008,
                "document": {
                    "file_name": "rapport.pdf",
                    "mime_type": "application/pdf",
                    "file_id": "BQACAgIAAxkBAAIG",
                    "file_unique_id": "AgADkg4AAg",
                    "file_size": 240128,
                },
            },
        }
        messages = parse_telegram_webhook(payload, channel_id="tg-main")

        assert len(messages) == 1
        msg = messages[0]
        assert msg.metadata["file_id"] == "BQACAgIAAxkBAAIG"
        assert msg.metadata["media_type"] == "document"
        assert msg.metadata["file_name"] == "rapport.pdf"
        assert msg.metadata["mime_type"] == "application/pdf"
        assert msg.metadata["file_size"] == 240128
        assert "duration" not in msg.metadata

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


class TestParseTelegramMessage:
    """The layer below attribution: a consumer whose sender is not message.from.id."""

    def test_reads_a_voice_note_without_building_an_inbound_message(self) -> None:
        parts = parse_telegram_message(
            {
                "message_id": 5,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000004,
                "voice": {
                    "duration": 3,
                    "mime_type": "audio/ogg",
                    "file_id": "AwACAgIAAxkBAAIC",
                    "file_size": 8342,
                },
            }
        )

        assert parts is not None
        assert isinstance(parts.content, TextContent)
        assert parts.content.body == ""
        assert parts.metadata["file_id"] == "AwACAgIAAxkBAAIC"
        assert parts.metadata["media_type"] == "voice"
        assert parts.metadata["duration"] == 3
        assert parts.metadata["chat_id"] == "555"
        assert parts.message_id == "5"

    def test_sender_is_offered_not_imposed(self) -> None:
        """The Telegram sender is available, and a caller is free to ignore it."""
        parts = parse_telegram_message(
            {
                "message_id": 6,
                "from": {"id": 999},
                "chat": {"id": 555},
                "date": 1700000005,
                "text": "hi",
            }
        )

        assert parts is not None
        assert parts.sender_id == "999"

    def test_takes_the_message_not_the_update(self) -> None:
        """A whole Update has no content of its own — the message is nested in it."""
        assert parse_telegram_message({"update_id": 1, "message": {"text": "hi"}}) is None

    def test_unsupported_message_returns_none(self) -> None:
        assert parse_telegram_message({"message_id": 7, "sticker": {"file_id": "x"}}) is None

    def test_webhook_is_this_function_plus_attribution(self) -> None:
        """parse_telegram_webhook must stay a thin layer over this one."""
        msg = {
            "message_id": 8,
            "from": {"id": 999},
            "chat": {"id": 555},
            "date": 1700000006,
            "document": {
                "file_name": "rapport.pdf",
                "mime_type": "application/pdf",
                "file_id": "BQACAgIAAxkBAAIG",
                "file_size": 240128,
            },
        }
        parts = parse_telegram_message(msg)
        inbound = parse_telegram_webhook({"update_id": 9, "message": msg}, channel_id="tg-main")

        assert parts is not None
        assert len(inbound) == 1
        assert inbound[0].sender_id == parts.sender_id
        assert inbound[0].content == parts.content
        assert inbound[0].metadata == parts.metadata
        assert inbound[0].external_id == parts.message_id


class TestMessageProtocolFacts:
    """Entities, the reply and the album — read, and kept out of the attribution."""

    def test_entities_are_handed_back_unsliced(self) -> None:
        """Offsets count UTF-16 units; slicing by them is entity_text's job, not this one's."""
        entities = [{"type": "mention", "offset": 4, "length": 9}]
        parts = parse_telegram_message(
            {"message_id": 1, "chat": {"id": 555}, "text": "hey @luge_bot", "entities": entities}
        )

        assert parts is not None
        assert parts.entities == entities

    def test_a_captions_entities_come_from_their_own_key(self) -> None:
        parts = parse_telegram_message(
            {
                "message_id": 2,
                "chat": {"id": 555},
                "caption": "look @luge_bot",
                "caption_entities": [{"type": "mention", "offset": 5, "length": 9}],
                "photo": [{"file_id": "abc"}],
            }
        )

        assert parts is not None
        assert parts.entities == [{"type": "mention", "offset": 5, "length": 9}]

    def test_a_message_with_no_markup_has_no_entities(self) -> None:
        parts = parse_telegram_message({"message_id": 3, "chat": {"id": 555}, "text": "plain"})

        assert parts is not None
        assert parts.entities == []

    def test_the_message_a_reply_answers(self) -> None:
        """What ties an answer back to the force_reply prompt that asked for it."""
        parts = parse_telegram_message(
            {
                "message_id": 4,
                "chat": {"id": 555},
                "text": "because it was wrong",
                "reply_to_message": {"message_id": 88, "from": {"id": 42}},
            }
        )

        assert parts is not None
        assert parts.reply_to_message_id == "88"

    def test_a_message_that_answers_nothing(self) -> None:
        parts = parse_telegram_message({"message_id": 5, "chat": {"id": 555}, "text": "hi"})

        assert parts is not None
        assert parts.reply_to_message_id is None

    def test_an_album_is_one_post_delivered_as_several(self) -> None:
        parts = parse_telegram_message(
            {
                "message_id": 6,
                "chat": {"id": 555},
                "media_group_id": "13285469384512",
                "photo": [{"file_id": "abc"}],
            }
        )

        assert parts is not None
        assert parts.media_group_id == "13285469384512"

    def test_a_lone_photo_belongs_to_no_album(self) -> None:
        parts = parse_telegram_message(
            {"message_id": 7, "chat": {"id": 555}, "photo": [{"file_id": "abc"}]}
        )

        assert parts is not None
        assert parts.media_group_id is None

    def test_the_new_facts_stay_out_of_the_inbound_metadata(self) -> None:
        """The InboundMessage a text, photo or location update builds is unchanged.

        These are protocol facts for a consumer reading the lower layer, not
        payload for every delivery record downstream of parse_telegram_webhook.
        """
        text = {
            "message_id": 1,
            "from": {"id": 999},
            "chat": {"id": 555},
            "date": 1700000000,
            "text": "hey @luge_bot",
            "entities": [{"type": "mention", "offset": 4, "length": 9}],
            "reply_to_message": {"message_id": 88},
        }
        photo = {
            "message_id": 2,
            "from": {"id": 999},
            "chat": {"id": 555},
            "date": 1700000001,
            "media_group_id": "132854",
            "photo": [{"file_id": "abc", "file_size": 100}],
        }
        location = {
            "message_id": 3,
            "from": {"id": 999},
            "chat": {"id": 555},
            "date": 1700000002,
            "location": {"latitude": 45.5, "longitude": -73.5},
        }

        for msg in (text, photo, location):
            inbound = parse_telegram_webhook({"update_id": 1, "message": msg}, channel_id="tg")
            assert len(inbound) == 1
            assert set(inbound[0].metadata) <= {
                "chat_id",
                "date",
                "file_id",
                "media_type",
                "duration",
                "mime_type",
                "file_name",
                "file_size",
            }

    def test_metadata_is_exactly_what_it_was_for_a_text_message(self) -> None:
        inbound = parse_telegram_webhook(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "from": {"id": 999},
                    "chat": {"id": 555},
                    "date": 1700000000,
                    "text": "hey @luge_bot",
                    "entities": [{"type": "mention", "offset": 4, "length": 9}],
                },
            },
            channel_id="tg",
        )

        assert inbound[0].metadata == {"chat_id": "555", "date": 1700000000}

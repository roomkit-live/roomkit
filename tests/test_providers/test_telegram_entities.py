"""Tests for the entity primitives — UTF-16 slicing, and the five ways a
Telegram message reaches a bot."""

from __future__ import annotations

from typing import Any

from roomkit.providers.telegram import entity_text, mentions_bot

BOT = "luge_bot"
BOT_ID = 42


def _mention(text: str, offset: int, length: int) -> dict[str, Any]:
    return {"text": text, "entities": [{"type": "mention", "offset": offset, "length": length}]}


class TestEntityText:
    def test_slices_a_plain_ascii_mention(self) -> None:
        text = "hey @luge_bot what's up"

        assert entity_text(text, {"offset": 4, "length": 9}) == "@luge_bot"

    def test_counts_an_astral_character_as_two_units(self) -> None:
        """Telegram measures in UTF-16 code units; Python indexes code points.

        An emoji is one code point and two code units, so every offset after it
        is off by one per emoji when sliced the Python way.
        """
        text = "🎉 @luge_bot"
        # Telegram: the emoji occupies units 0-1, the space unit 2, the mention 3.
        entity = {"offset": 3, "length": 9}

        assert entity_text(text, entity) == "@luge_bot"
        # The control: what a code-point slice would have returned instead.
        assert text[3 : 3 + 9] != "@luge_bot"

    def test_counts_several_astral_characters(self) -> None:
        text = "🎉🎊🎈 @luge_bot!"

        assert entity_text(text, {"offset": 7, "length": 9}) == "@luge_bot"

    def test_an_entity_pointing_past_the_text_is_empty(self) -> None:
        assert entity_text("short", {"offset": 100, "length": 5}) == ""

    def test_a_negative_bound_is_empty_rather_than_a_slice_from_the_end(self) -> None:
        """Entities are composed by someone else's client, so the bound is theirs."""
        assert entity_text("hey @luge_bot", {"offset": -9, "length": 9}) == ""
        assert entity_text("hey @luge_bot", {"offset": 4, "length": -1}) == ""


class TestMentionsBot:
    def test_a_mention_entity(self) -> None:
        assert mentions_bot(_mention("hey @luge_bot", 4, 9), bot_username=BOT) is True

    def test_a_mention_after_an_emoji(self) -> None:
        """The case a code-point slice gets wrong: same message, emoji in front."""
        msg = _mention("🎉 @luge_bot", 3, 9)

        assert mentions_bot(msg, bot_username=BOT) is True

    def test_a_mention_of_another_bot(self) -> None:
        assert mentions_bot(_mention("hey @other_bot", 4, 10), bot_username=BOT) is False

    def test_a_mention_is_matched_case_insensitively(self) -> None:
        assert mentions_bot(_mention("hey @Luge_Bot", 4, 9), bot_username=BOT) is True

    def test_a_text_mention_names_the_bots_id(self) -> None:
        msg = {
            "text": "hey Luge",
            "entities": [
                {"type": "text_mention", "offset": 4, "length": 4, "user": {"id": BOT_ID}}
            ],
        }

        assert mentions_bot(msg, bot_id=BOT_ID) is True

    def test_a_text_mention_of_someone_else(self) -> None:
        msg = {
            "text": "hey Bob",
            "entities": [{"type": "text_mention", "offset": 4, "length": 3, "user": {"id": 99}}],
        }

        assert mentions_bot(msg, bot_id=BOT_ID) is False

    def test_a_bot_command(self) -> None:
        """Telegram routes commands itself — a delivered one was meant for us."""
        msg = {"text": "/status", "entities": [{"type": "bot_command", "offset": 0, "length": 7}]}

        assert mentions_bot(msg, bot_username=BOT, bot_id=BOT_ID) is True

    def test_a_reply_to_the_bots_own_message(self) -> None:
        msg = {"text": "yes please", "reply_to_message": {"from": {"id": BOT_ID}}}

        assert mentions_bot(msg, bot_id=BOT_ID) is True

    def test_a_reply_to_someone_elses_message(self) -> None:
        msg = {"text": "yes please", "reply_to_message": {"from": {"id": 99}}}

        assert mentions_bot(msg, bot_username=BOT, bot_id=BOT_ID) is False

    def test_a_handle_posted_as_plain_text_with_no_entity(self) -> None:
        """Some clients post an @-mention without marking it up."""
        assert mentions_bot({"text": "hey @luge_bot"}, bot_username=BOT) is True

    def test_ordinary_group_chatter(self) -> None:
        assert mentions_bot({"text": "lunch at noon?"}, bot_username=BOT, bot_id=BOT_ID) is False

    def test_a_caption_and_its_own_entities(self) -> None:
        """A caption's markup is carried under ``caption_entities``, not ``entities``."""
        msg = {
            "caption": "look 🎉 @luge_bot",
            "caption_entities": [{"type": "mention", "offset": 8, "length": 9}],
            "photo": [{"file_id": "abc"}],
        }

        assert mentions_bot(msg, bot_username=BOT) is True

    def test_a_bot_that_has_not_said_who_it_is(self) -> None:
        assert mentions_bot(_mention("hey @luge_bot", 4, 9)) is False

    def test_a_username_alone_is_enough_for_the_mention_paths(self) -> None:
        assert mentions_bot(_mention("hey @luge_bot", 4, 9), bot_username=BOT) is True

    def test_an_id_alone_is_enough_for_the_reply_path(self) -> None:
        msg = {"text": "ok", "reply_to_message": {"from": {"id": BOT_ID}}}

        assert mentions_bot(msg, bot_id=BOT_ID) is True

    def test_a_message_with_no_text_at_all(self) -> None:
        msg = {"sticker": {"file_id": "x"}}

        assert mentions_bot(msg, bot_username=BOT, bot_id=BOT_ID) is False

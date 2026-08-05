"""Tests for reading a Telegram Update envelope — which form it took, and what
an inline-button press carries."""

from __future__ import annotations

from roomkit.providers.telegram import parse_telegram_callback, parse_telegram_update


class TestUpdateForms:
    def test_a_new_message(self) -> None:
        payload = {"update_id": 1, "message": {"message_id": 9, "text": "hi"}}

        update = parse_telegram_update(payload)

        assert update is not None
        assert update.edited is False
        assert update.callback is None
        assert update.message == {"message_id": 9, "text": "hi"}

    def test_an_edited_message_hands_back_the_same_shape(self) -> None:
        """What differs is the flag — the message is read by the same function."""
        payload = {"update_id": 2, "edited_message": {"message_id": 9, "text": "hi (fixed)"}}

        update = parse_telegram_update(payload)

        assert update is not None
        assert update.edited is True
        assert update.message == {"message_id": 9, "text": "hi (fixed)"}

    def test_a_callback_query(self) -> None:
        payload = {"update_id": 3, "callback_query": {"id": "cq-1", "data": "a:xyz"}}

        update = parse_telegram_update(payload)

        assert update is not None
        assert update.message is None
        assert update.callback is not None
        assert update.callback.data == "a:xyz"

    def test_a_form_this_does_not_cover(self) -> None:
        assert parse_telegram_update({"update_id": 4, "poll_answer": {"poll_id": "p"}}) is None

    def test_an_empty_payload(self) -> None:
        assert parse_telegram_update({}) is None


class TestCallback:
    def test_reads_the_press_and_the_message_it_hangs_off(self) -> None:
        cq = {
            "id": "cq-1",
            "data": "a:0f2b",
            "from": {"id": 777, "username": "someone"},
            "message": {
                "message_id": 55,
                "chat": {"id": -100999, "type": "supergroup"},
                "text": "Approve the draft?",
            },
        }

        callback = parse_telegram_callback(cq)

        assert callback.id == "cq-1"
        assert callback.data == "a:0f2b"
        assert callback.sender_id == "777"
        assert callback.chat_id == "-100999"
        assert callback.message_id == 55
        assert callback.message_text == "Approve the draft?"

    def test_a_press_with_no_message_attached(self) -> None:
        """An inline-mode result has no message behind it — nothing to edit."""
        callback = parse_telegram_callback({"id": "cq-2", "data": "d:1", "from": {"id": 777}})

        assert callback.chat_id == ""
        assert callback.message_id is None
        assert callback.message_text == ""

    def test_a_press_carrying_no_data(self) -> None:
        callback = parse_telegram_callback({"id": "cq-3", "from": {"id": 777}})

        assert callback.data == ""

    def test_the_sender_is_read_not_judged(self) -> None:
        """Whoever pressed it is a fact; whether they were allowed to is not."""
        callback = parse_telegram_callback({"id": "cq-4", "data": "a:x", "from": {"id": 999}})

        assert callback.sender_id == "999"

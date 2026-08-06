"""What form an Update took, before anything is decided about it.

Telegram wraps everything a bot receives in one ``Update`` object and names the
kind by which key is present. Three matter to a bot with a webhook: a new
``message``, an ``edited_message``, and a ``callback_query`` — someone pressing
an inline button. Reading them apart is protocol, and neither of the decisions
that follow is: who the sender is, and whether they were allowed to press that
button.

The message forms hand back the raw ``message`` object, which is what
:func:`~roomkit.providers.telegram.webhook.parse_telegram_message` and
:func:`~roomkit.providers.telegram.entities.mentions_bot` both read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TelegramCallback:
    """An inline-button press, read but not judged.

    Attributes:
        id: The callback query's id. Answering it with
            :meth:`~roomkit.providers.telegram.api.TelegramBotAPI.answer_callback_query`
            is what stops the spinner on the button.
        data: The button's ``callback_data``, verbatim. Chosen by whoever built
            the keyboard, but posted by whoever pressed it — any client can
            send arbitrary bytes to its own bot's webhook, so what it points at
            is a claim to check, never a fact.
        sender_id: The Telegram account that pressed it.
        chat_id: The chat holding the message the button hangs off, empty when
            Telegram sent no message with the query (an inline-mode result).
        message_id: That message's id, or None in the same case.
        message_text: That message's text, so an outcome can be appended to
            what was already said rather than replacing it.
    """

    id: str
    data: str
    sender_id: str
    chat_id: str
    message_id: int | None
    message_text: str


@dataclass(frozen=True)
class TelegramUpdate:
    """One Telegram Update, told apart by the form it took.

    Exactly one of :attr:`message` and :attr:`callback` is set.

    Attributes:
        message: The raw ``message`` object, for a new or edited message.
        edited: True when that message is an edit of one already delivered.
            The distinction is the application's to act on — some treat an edit
            as a new turn, others ignore it.
        callback: The button press, for a ``callback_query`` update.
    """

    message: dict[str, Any] | None
    edited: bool
    callback: TelegramCallback | None


def parse_telegram_callback(cq: dict[str, Any]) -> TelegramCallback:
    """Read a Telegram ``callback_query`` into its parts.

    Args:
        cq: The ``callback_query`` object of an Update.

    Returns:
        The press, with nothing decided about who was allowed to make it.
    """
    message = cq.get("message")
    sender = cq.get("from")
    message = message if isinstance(message, dict) else {}
    sender = sender if isinstance(sender, dict) else {}
    chat = message.get("chat")
    chat = chat if isinstance(chat, dict) else {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    message_text = message.get("text")
    return TelegramCallback(
        id=str(cq.get("id") or ""),
        data=str(cq.get("data") or ""),
        sender_id=str(sender.get("id") or ""),
        chat_id=str(chat_id) if chat_id is not None else "",
        message_id=(
            message_id
            if isinstance(message_id, int) and not isinstance(message_id, bool)
            else None
        ),
        message_text=message_text if isinstance(message_text, str) else "",
    )


def parse_telegram_update(payload: dict[str, Any]) -> TelegramUpdate | None:
    """Read a Telegram Update envelope and say which form it took.

    Args:
        payload: The Update object, as Telegram POSTs it to a webhook.

    Returns:
        The update, or None for a form this does not cover — a channel post, a
        poll answer, an inline query. Those are kinds a webhook only receives
        when its ``allowed_updates`` asked for them.
    """
    if not isinstance(payload, dict):
        return None
    if isinstance(message := payload.get("message"), dict):
        return TelegramUpdate(message=message, edited=False, callback=None)
    if isinstance(edited := payload.get("edited_message"), dict):
        return TelegramUpdate(message=edited, edited=True, callback=None)
    if isinstance(cq := payload.get("callback_query"), dict):
        return TelegramUpdate(message=None, edited=False, callback=parse_telegram_callback(cq))
    return None

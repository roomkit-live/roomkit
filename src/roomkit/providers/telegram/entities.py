"""Message entities, and the one question every bot in a group has to answer.

Telegram marks up a message out of band: the text arrives plain, and a parallel
list of entities says which stretch of it is a mention, a command, a link. The
offsets in that list count **UTF-16 code units**, which is what Python does not
index by — so slicing a mention out with ``text[offset:offset + length]`` is
right until the message contains an emoji, and quietly wrong after.

That, and the five ways a Telegram message can be addressed to a bot, is
protocol knowledge with no product decision in it. Whether to *answer* is the
application's call; whether it was *asked* is this module's.
"""

from __future__ import annotations

import re
from typing import Any


def entity_text(text: str, entity: dict[str, Any]) -> str:
    """Return the stretch of ``text`` an entity covers.

    Telegram's ``offset`` and ``length`` count UTF-16 code units; Python indexes
    strings by code point. The two agree until a character outside the Basic
    Multilingual Plane — an emoji, some scripts, a musical symbol — appears
    earlier in the message, at which point every offset after it is off by one
    per such character. Encoding to UTF-16-LE puts the slice on the same basis
    Telegram measured it in.

    Args:
        text: The message text (or caption) the entity indexes into.
        entity: A Telegram ``MessageEntity``, read for ``offset`` and ``length``.

    Returns:
        The substring the entity covers, empty when the entity points outside
        the text.
    """
    offset = entity.get("offset", 0)
    length = entity.get("length", 0)
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or not isinstance(length, int)
        or isinstance(length, bool)
    ):
        return ""
    if offset < 0 or length < 0:
        # Telegram never sends these, but the entities reaching mentions_bot are
        # composed by someone else's client. A negative bound would slice from
        # the end of the string and return a stretch nobody pointed at.
        return ""
    encoded = text.encode("utf-16-le")
    return encoded[offset * 2 : (offset + length) * 2].decode("utf-16-le", errors="replace")


def mentions_bot(
    msg: dict[str, Any],
    *,
    bot_username: str | None = None,
    bot_id: int | None = None,
) -> bool:
    """Say whether a message addresses the bot. The fact, not the policy.

    True on any of five things, which are the five ways Telegram lets someone
    reach a bot in a room full of people:

    1. A reply to a message the bot itself sent.
    2. An unqualified ``bot_command`` entity, or one whose ``@username`` suffix
       names this bot. Administrators and bots without privacy mode can receive
       commands addressed to other bots, so delivery alone is not attribution.
    3. A ``mention`` entity whose text is ``@bot_username``.
    4. A ``text_mention`` entity naming the bot's user id — how a mention of an
       account with no username is carried.
    5. ``@bot_username`` present as plain text with no entity at all, which is
       what some clients post.

    Whether to *answer* is not decided here. A group that only responds when
    addressed asks this; a group that responds to everything never needs to.

    Args:
        msg: A Telegram ``message`` object.
        bot_username: The bot's username, without the ``@``. Without it, cases
            3 and 5 cannot be checked.
        bot_id: The bot's numeric user id. Without it, cases 1 and 4 cannot be
            checked.

    Returns:
        True if the message addresses the bot. False when neither identifier
        was supplied — nothing can be attributed to a bot that has not said
        who it is.
    """
    if not bot_username and bot_id is None:
        return False

    reply = msg.get("reply_to_message")
    reply = reply if isinstance(reply, dict) else {}
    reply_from = reply.get("from")
    reply_from = reply_from if isinstance(reply_from, dict) else {}
    if bot_id is not None and reply_from.get("id") == bot_id:
        return True

    text = msg.get("text")
    if not isinstance(text, str):
        text = msg.get("caption")
    text = text if isinstance(text, str) else ""
    username = bot_username.removeprefix("@") if bot_username else None
    handle = f"@{username}" if username else None

    entities = msg.get("entities") or msg.get("caption_entities") or []
    entities = entities if isinstance(entities, list) else []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        etype = entity.get("type")
        if etype == "bot_command":
            command = entity_text(text, entity)
            _, separator, target = command.partition("@")
            if not separator:
                if command:
                    return True
                continue
            if username and target.casefold() == username.casefold():
                return True
            continue
        if (
            etype == "mention"
            and handle
            and entity_text(text, entity).casefold() == handle.casefold()
        ):
            return True
        user = entity.get("user")
        if (
            etype == "text_mention"
            and bot_id is not None
            and isinstance(user, dict)
            and user.get("id") == bot_id
        ):
            return True

    if handle is None:
        return False
    # Telegram usernames contain ASCII letters, digits, and underscores. The
    # boundaries prevent @demo_bot from matching @demo_bot_extra or an email.
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(handle)}(?![A-Za-z0-9_])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

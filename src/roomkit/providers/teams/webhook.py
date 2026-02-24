"""Microsoft Teams webhook parsing helpers."""

from __future__ import annotations

import re
from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent

_AT_MENTION_RE = re.compile(r"<at>[^<]*</at>\s*")


def parse_teams_activity(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract common fields from a Bot Framework Activity payload.

    Returns a dict with ``activity_type``, ``conversation_id``,
    ``conversation_type``, ``conversation_name``, ``is_group``,
    ``service_url``, ``tenant_id``, ``sender_id``, ``sender_name``,
    ``bot_id``, ``members_added``, and ``members_removed``.  Useful for
    handling lifecycle events (``conversationUpdate``) alongside message
    parsing.
    """
    conversation = payload.get("conversation", {})
    is_group = conversation.get("isGroup", False) or conversation.get("conversationType") in (
        "groupChat",
        "channel",
    )
    sender = payload.get("from", {})
    recipient = payload.get("recipient", {})

    return {
        "activity_type": payload.get("type", ""),
        "conversation_id": conversation.get("id", ""),
        "conversation_type": conversation.get("conversationType", "personal"),
        "conversation_name": conversation.get("name", ""),
        "is_group": is_group,
        "service_url": payload.get("serviceUrl", ""),
        "tenant_id": payload.get("channelData", {}).get("tenant", {}).get("id", ""),
        "sender_id": sender.get("id", ""),
        "sender_name": sender.get("name", ""),
        "bot_id": recipient.get("id", ""),
        "reply_to_id": payload.get("replyToId", ""),
        "members_added": [m.get("id", "") for m in payload.get("membersAdded", [])],
        "members_removed": [m.get("id", "") for m in payload.get("membersRemoved", [])],
    }


def is_bot_added(payload: dict[str, Any], bot_id: str | None = None) -> bool:
    """Check if a ``conversationUpdate`` Activity indicates the bot was added.

    Args:
        payload: Raw Bot Framework Activity dict.
        bot_id: The bot's AAD ID.  If *None*, falls back to
            ``payload["recipient"]["id"]`` (the bot in most Activities).

    Returns:
        ``True`` if the Activity is a ``conversationUpdate`` with the bot
        in ``membersAdded``.
    """
    if payload.get("type") != "conversationUpdate":
        return False
    bid = bot_id or payload.get("recipient", {}).get("id", "")
    if not bid:
        return False
    return any(m.get("id") == bid for m in payload.get("membersAdded", []))


def parse_teams_webhook(
    payload: dict[str, Any],
    channel_id: str,
) -> list[InboundMessage]:
    """Convert a Bot Framework Activity payload into InboundMessages.

    Bot Framework sends one Activity per HTTP POST.  Only ``type="message"``
    activities with non-empty text are converted.  ``<at>BotName</at>``
    mention tags are stripped from group chat messages.
    """
    if payload.get("type") != "message":
        return []

    text = payload.get("text", "").strip()
    if not text:
        return []

    # Strip <at>BotName</at> mentions from group chats
    conversation = payload.get("conversation", {})
    is_group = conversation.get("isGroup", False) or conversation.get("conversationType") in (
        "groupChat",
        "channel",
    )
    if is_group:
        text = _AT_MENTION_RE.sub("", text).strip()
        if not text:
            return []

    sender = payload.get("from", {})
    sender_id = sender.get("id", "")
    conversation_id = conversation.get("id", "")

    # Check if the bot was @mentioned
    bot_id = payload.get("recipient", {}).get("id", "")
    bot_mentioned = (
        any(
            e.get("type") == "mention" and e.get("mentioned", {}).get("id") == bot_id
            for e in payload.get("entities", [])
        )
        if bot_id
        else False
    )
    # In personal chats the bot is always implicitly addressed
    if not is_group:
        bot_mentioned = True

    reply_to_id = payload.get("replyToId") or None

    return [
        InboundMessage(
            channel_id=channel_id,
            sender_id=sender_id,
            content=TextContent(body=text),
            external_id=payload.get("id"),
            thread_id=reply_to_id,
            idempotency_key=payload.get("id"),
            metadata={
                "sender_name": sender.get("name", ""),
                "conversation_id": conversation_id,
                "conversation_type": conversation.get("conversationType", "personal"),
                "is_group": is_group,
                "bot_mentioned": bot_mentioned,
                "service_url": payload.get("serviceUrl", ""),
                "tenant_id": payload.get("channelData", {}).get("tenant", {}).get("id", ""),
                "reply_to_id": reply_to_id or "",
            },
        )
    ]


def parse_teams_reactions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse reaction events from a ``messageReaction`` Activity.

    Teams sends ``messageReaction`` activities when a user adds or removes
    a reaction (like, heart, laugh, etc.) on a message.  This helper
    normalises both ``reactionsAdded`` and ``reactionsRemoved`` into a flat
    list of dicts.

    Args:
        payload: Raw Bot Framework Activity dict.

    Returns:
        A list of dicts, each with keys ``action`` (``"add"`` or
        ``"remove"``), ``emoji``, ``sender_id``, ``sender_name``, and
        ``target_activity_id``.  Returns an empty list if the Activity
        is not a ``messageReaction`` or has no reactions.
    """
    if payload.get("type") != "messageReaction":
        return []

    sender = payload.get("from", {})
    sender_id = sender.get("id", "")
    sender_name = sender.get("name", "")
    target_activity_id = payload.get("replyToId", "")

    results: list[dict[str, Any]] = []
    for reaction in payload.get("reactionsAdded", []):
        results.append(
            {
                "action": "add",
                "emoji": reaction.get("type", ""),
                "sender_id": sender_id,
                "sender_name": sender_name,
                "target_activity_id": target_activity_id,
            }
        )
    for reaction in payload.get("reactionsRemoved", []):
        results.append(
            {
                "action": "remove",
                "emoji": reaction.get("type", ""),
                "sender_id": sender_id,
                "sender_name": sender_name,
                "target_activity_id": target_activity_id,
            }
        )
    return results

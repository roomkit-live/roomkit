"""Channel implementations and transport-channel factories."""

from __future__ import annotations

from typing import Any

from roomkit.channels._ai_router import AIRouter as AIRouter
from roomkit.channels._ai_router import FallbackRouter as FallbackRouter
from roomkit.channels._ai_router import RouteReason as RouteReason
from roomkit.channels._ai_router import RouteRequest as RouteRequest
from roomkit.channels._note_blocks import add_turn_note as add_turn_note
from roomkit.channels._tool_search_constants import TOOL_FIND_TOOLS as TOOL_FIND_TOOLS
from roomkit.channels._tool_search_constants import TOOL_LIST_TOOLS as TOOL_LIST_TOOLS
from roomkit.channels._tool_search_constants import (
    TOOL_SEARCH_INFRA_TOOL_NAMES as TOOL_SEARCH_INFRA_TOOL_NAMES,
)
from roomkit.channels._turn_notes import TURN_NOTES_HEADER as TURN_NOTES_HEADER
from roomkit.channels._turn_notes import split_turn_notes as split_turn_notes
from roomkit.channels.acp import ACPChannel as ACPChannel
from roomkit.channels.addressing import Addressing as Addressing
from roomkit.channels.addressing import ChannelAddressing as ChannelAddressing
from roomkit.channels.addressing import addressing_of as addressing_of
from roomkit.channels.ai import AIChannel as AIChannel
from roomkit.channels.cli import CLIChannel as CLIChannel
from roomkit.channels.declared import DeclaredChannel as DeclaredChannel
from roomkit.channels.host import HostChannel as HostChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel as RealtimeVoiceChannel
from roomkit.channels.skill_script_tool import RunSkillScriptTool as RunSkillScriptTool
from roomkit.channels.transport import TransportChannel
from roomkit.channels.voice import VoiceChannel as VoiceChannel
from roomkit.channels.websocket import (
    SendFn,
    StreamChunk,
    StreamEnd,
    StreamError,
    StreamMessage,
    StreamSendFn,
    StreamStart,
    WebSocketChannel,
)
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType

__all__ = [
    "ACPChannel",
    "AIChannel",
    "AIRouter",
    "Addressing",
    "BuzzChannel",
    "CLIChannel",
    "ChannelAddressing",
    "DeclaredChannel",
    "DiscordChannel",
    "EmailChannel",
    "FallbackRouter",
    "HTTPChannel",
    "HostChannel",
    "MessengerChannel",
    "RCSChannel",
    "RealtimeVoiceChannel",
    "RouteReason",
    "RouteRequest",
    "RunSkillScriptTool",
    "SMSChannel",
    "SendFn",
    "StreamChunk",
    "StreamEnd",
    "StreamError",
    "StreamMessage",
    "StreamSendFn",
    "StreamStart",
    "TOOL_FIND_TOOLS",
    "TOOL_LIST_TOOLS",
    "TOOL_SEARCH_INFRA_TOOL_NAMES",
    "TURN_NOTES_HEADER",
    "TeamsChannel",
    "TelegramChannel",
    "TransportChannel",
    "VoiceChannel",
    "WebSocketChannel",
    "WhatsAppChannel",
    "WhatsAppPersonalChannel",
    "add_turn_note",
    "addressing_of",
    "split_turn_notes",
]

# ---------------------------------------------------------------------------
# Capability constants
# ---------------------------------------------------------------------------

SMS_CAPABILITIES = ChannelCapabilities(
    media_types=[ChannelMediaType.TEXT, ChannelMediaType.MEDIA],
    max_length=1600,
    supports_read_receipts=True,
    supports_media=True,
    # RFC Appendix A.1: another file is sent as its text form (§6.6).
    supported_media_types=["image/jpeg", "image/png", "image/gif"],
)

EMAIL_CAPABILITIES = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT,
        ChannelMediaType.RICH,
        ChannelMediaType.MEDIA,
    ],
    supports_threading=True,
    supports_rich_text=True,
    supports_media=True,
)

WHATSAPP_CAPABILITIES = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT,
        ChannelMediaType.RICH,
        ChannelMediaType.MEDIA,
        ChannelMediaType.LOCATION,
        ChannelMediaType.TEMPLATE,
    ],
    max_length=4096,
    supports_read_receipts=True,
    supports_reactions=True,
    supports_edit=True,
    supports_delete=True,
    supports_templates=True,
    supports_rich_text=True,
    supports_buttons=True,
    max_buttons=3,
    supports_quick_replies=True,
    supports_media=True,
)

WHATSAPP_PERSONAL_CAPABILITIES = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT,
        ChannelMediaType.RICH,
        ChannelMediaType.MEDIA,
        ChannelMediaType.AUDIO,
        ChannelMediaType.VIDEO,
        ChannelMediaType.LOCATION,
    ],
    max_length=4096,
    supports_read_receipts=True,
    supports_reactions=True,
    supports_edit=True,
    supports_delete=True,
    supports_media=True,
    supports_audio=True,
    supported_audio_formats=["audio/ogg", "audio/mp4"],
    supports_video=True,
    supported_video_formats=["video/mp4"],
    supports_typing=True,
)

MESSENGER_CAPABILITIES = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT,
        ChannelMediaType.RICH,
        ChannelMediaType.MEDIA,
        ChannelMediaType.TEMPLATE,
    ],
    max_length=2000,
    supports_read_receipts=True,
    supports_delete=True,
    supports_buttons=True,
    max_buttons=3,
    supports_quick_replies=True,
    supports_media=True,
)

TELEGRAM_CAPABILITIES = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT,
        ChannelMediaType.RICH,
        ChannelMediaType.MEDIA,
        ChannelMediaType.AUDIO,
        ChannelMediaType.VIDEO,
        ChannelMediaType.LOCATION,
    ],
    max_length=4096,
    supports_edit=True,
    supports_delete=True,
    supports_reactions=True,
    supports_media=True,
)

TEAMS_CAPABILITIES = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT,
        ChannelMediaType.RICH,
    ],
    max_length=28000,
    supports_threading=True,
    supports_reactions=True,
    supports_edit=True,
    supports_delete=True,
    supports_read_receipts=True,
    supports_rich_text=True,
)

DISCORD_CAPABILITIES = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT,
        ChannelMediaType.RICH,
        ChannelMediaType.MEDIA,
    ],
    max_length=2000,
    supports_threading=True,
    supports_reactions=True,
    supports_rich_text=True,
    supports_media=True,
)

BUZZ_CAPABILITIES = ChannelCapabilities(
    media_types=[ChannelMediaType.TEXT],
    max_length=65536,
    supports_threading=True,
    supports_reactions=True,
)

HTTP_CAPABILITIES = ChannelCapabilities(
    media_types=[ChannelMediaType.TEXT, ChannelMediaType.RICH],
)

RCS_CAPABILITIES = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT,
        ChannelMediaType.RICH,
        ChannelMediaType.MEDIA,
    ],
    max_length=8000,  # RCS supports longer messages
    supports_read_receipts=True,
    supports_typing=True,
    supports_rich_text=True,
    supports_buttons=True,
    supports_quick_replies=True,
    supports_cards=True,
    supports_media=True,
)

# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def SMSChannel(
    channel_id: str,
    *,
    provider: Any = None,
    from_number: str | None = None,
    default_country_code: str | None = None,
) -> TransportChannel:
    """Create an SMS transport channel.

    Numbers are compared and stored in E.164; *default_country_code*
    (``"1"``, ``"+44"``) places a national number a provider sends without it.
    """
    return TransportChannel(
        channel_id,
        ChannelType.SMS,
        provider=provider,
        capabilities=SMS_CAPABILITIES,
        country_code=default_country_code,
        defaults={"from_": from_number},
    )


def EmailChannel(
    channel_id: str,
    *,
    provider: Any = None,
    from_address: str | None = None,
) -> TransportChannel:
    """Create an Email transport channel.

    Addresses are compared and stored lower-case, without a display name
    (``Alice <Alice@X.com>`` is ``alice@x.com``).
    """
    return TransportChannel(
        channel_id,
        ChannelType.EMAIL,
        provider=provider,
        capabilities=EMAIL_CAPABILITIES,
        defaults={"from_": from_address, "subject": None},
    )


def WhatsAppChannel(
    channel_id: str,
    *,
    provider: Any = None,
    default_country_code: str | None = None,
) -> TransportChannel:
    """Create a WhatsApp transport channel.

    Numbers are compared and stored in E.164 (see :func:`SMSChannel`).
    """
    return TransportChannel(
        channel_id,
        ChannelType.WHATSAPP,
        provider=provider,
        capabilities=WHATSAPP_CAPABILITIES,
        country_code=default_country_code,
    )


def WhatsAppPersonalChannel(
    channel_id: str,
    *,
    provider: Any = None,
    default_country_code: str | None = None,
) -> TransportChannel:
    """Create a WhatsApp Personal transport channel (neonize).

    Numbers are compared and stored in E.164 (see :func:`SMSChannel`). A
    private chat is its sender's conversation; a group is the group's,
    answered in the group (its JID under ``phone_number``).
    """
    return TransportChannel(
        channel_id,
        ChannelType.WHATSAPP_PERSONAL,
        provider=provider,
        capabilities=WHATSAPP_PERSONAL_CAPABILITIES,
        country_code=default_country_code,
    )


def MessengerChannel(
    channel_id: str,
    *,
    provider: Any = None,
) -> TransportChannel:
    """Create a Facebook Messenger transport channel."""
    return TransportChannel(
        channel_id,
        ChannelType.MESSENGER,
        provider=provider,
        capabilities=MESSENGER_CAPABILITIES,
    )


def TelegramChannel(
    channel_id: str,
    *,
    provider: Any = None,
) -> TransportChannel:
    """Create a Telegram Bot transport channel."""
    return TransportChannel(
        channel_id,
        ChannelType.TELEGRAM,
        provider=provider,
        capabilities=TELEGRAM_CAPABILITIES,
    )


def TeamsChannel(
    channel_id: str,
    *,
    provider: Any = None,
) -> TransportChannel:
    """Create a Microsoft Teams transport channel."""
    return TransportChannel(
        channel_id,
        ChannelType.TEAMS,
        provider=provider,
        capabilities=TEAMS_CAPABILITIES,
    )


def DiscordChannel(
    channel_id: str,
    *,
    provider: Any = None,
) -> TransportChannel:
    """Create a Discord bot transport channel.

    The recipient key ``discord_channel_id`` resolves to the target Discord
    channel snowflake at delivery time.
    """
    return TransportChannel(
        channel_id,
        ChannelType.DISCORD,
        provider=provider,
        capabilities=DISCORD_CAPABILITIES,
        addressing=ChannelAddressing(
            Addressing.CHAT, recipient_key="discord_channel_id", chat_key="channel_id"
        ),
    )


def BuzzChannel(
    channel_id: str,
    *,
    provider: Any = None,
) -> TransportChannel:
    """Create a Buzz (Nostr relay) transport channel.

    The recipient key ``buzz_channel_id`` resolves to the target Buzz channel
    UUID at delivery time.
    """
    return TransportChannel(
        channel_id,
        ChannelType.BUZZ,
        provider=provider,
        capabilities=BUZZ_CAPABILITIES,
        addressing=ChannelAddressing(
            Addressing.CHAT, recipient_key="buzz_channel_id", chat_key="buzz_channel_id"
        ),
    )


def HTTPChannel(
    channel_id: str,
    *,
    provider: Any = None,
) -> TransportChannel:
    """Create an HTTP webhook transport channel."""
    return TransportChannel(
        channel_id,
        ChannelType.WEBHOOK,
        provider=provider,
        capabilities=HTTP_CAPABILITIES,
        # A recipient the provider forwards, if the binding names one, which
        # names no conversation: the provider posts to its configured URL,
        # and a room opened for an inbound message replies there.
        addressing=ChannelAddressing(Addressing.AS_WRITTEN, recipient_key="recipient_id"),
        requires_recipient=False,
    )


def RCSChannel(
    channel_id: str,
    *,
    provider: Any = None,
    fallback: bool = True,
    default_country_code: str | None = None,
) -> TransportChannel:
    """Create an RCS (Rich Communication Services) transport channel.

    Args:
        channel_id: Unique identifier for this channel.
        provider: RCS provider instance (e.g., TwilioRCSProvider).
        fallback: If True (default), allow SMS fallback when RCS unavailable.
        default_country_code: Places a national number (E.164, see
            :func:`SMSChannel`).

    Returns:
        A TransportChannel configured for RCS messaging.
    """
    return TransportChannel(
        channel_id,
        ChannelType.RCS,
        provider=provider,
        capabilities=RCS_CAPABILITIES,
        country_code=default_country_code,
        defaults={"fallback": fallback},
    )

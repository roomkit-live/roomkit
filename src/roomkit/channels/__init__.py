"""Channel implementations and transport-channel factories."""

from __future__ import annotations

from typing import Any

from roomkit.channels.transport import TransportChannel
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType

# ---------------------------------------------------------------------------
# Capability constants
# ---------------------------------------------------------------------------

SMS_CAPABILITIES = ChannelCapabilities(
    media_types=[ChannelMediaType.TEXT, ChannelMediaType.MEDIA],
    max_length=1600,
    supports_read_receipts=True,
    supports_media=True,
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
    supports_media=True,
    supported_media_types=["image/jpeg", "image/png", "image/webp"],
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
    supports_buttons=True,
    max_buttons=3,
    supports_quick_replies=True,
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
    supports_read_receipts=True,
    supports_rich_text=True,
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
) -> TransportChannel:
    """Create an SMS transport channel."""
    return TransportChannel(
        channel_id,
        ChannelType.SMS,
        provider=provider,
        capabilities=SMS_CAPABILITIES,
        recipient_key="phone_number",
        defaults={"from_": from_number},
    )


def EmailChannel(
    channel_id: str,
    *,
    provider: Any = None,
    from_address: str | None = None,
) -> TransportChannel:
    """Create an Email transport channel."""
    return TransportChannel(
        channel_id,
        ChannelType.EMAIL,
        provider=provider,
        capabilities=EMAIL_CAPABILITIES,
        recipient_key="email_address",
        defaults={"from_": from_address, "subject": None},
    )


def WhatsAppChannel(
    channel_id: str,
    *,
    provider: Any = None,
) -> TransportChannel:
    """Create a WhatsApp transport channel."""
    return TransportChannel(
        channel_id,
        ChannelType.WHATSAPP,
        provider=provider,
        capabilities=WHATSAPP_CAPABILITIES,
        recipient_key="phone_number",
    )


def WhatsAppPersonalChannel(
    channel_id: str,
    *,
    provider: Any = None,
) -> TransportChannel:
    """Create a WhatsApp Personal transport channel (neonize)."""
    return TransportChannel(
        channel_id,
        ChannelType.WHATSAPP_PERSONAL,
        provider=provider,
        capabilities=WHATSAPP_PERSONAL_CAPABILITIES,
        recipient_key="phone_number",
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
        recipient_key="facebook_user_id",
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
        recipient_key="teams_conversation_id",
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
        recipient_key="recipient_id",
    )


def RCSChannel(
    channel_id: str,
    *,
    provider: Any = None,
    fallback: bool = True,
) -> TransportChannel:
    """Create an RCS (Rich Communication Services) transport channel.

    Args:
        channel_id: Unique identifier for this channel.
        provider: RCS provider instance (e.g., TwilioRCSProvider).
        fallback: If True (default), allow SMS fallback when RCS unavailable.

    Returns:
        A TransportChannel configured for RCS messaging.
    """
    return TransportChannel(
        channel_id,
        ChannelType.RCS,
        provider=provider,
        capabilities=RCS_CAPABILITIES,
        recipient_key="phone_number",
        defaults={"fallback": fallback},
    )

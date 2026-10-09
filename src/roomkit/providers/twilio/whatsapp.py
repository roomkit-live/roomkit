"""Twilio WhatsApp provider — WhatsApp through Twilio's Messages API.

Twilio carries WhatsApp on the same Messages API as SMS: the same account,
signature and webhook, with both addresses written ``whatsapp:+15550000001``.
The channel stores the correspondent's number in E.164 (RFC §5.7); this
provider adds the prefix on the way out, and ``WhatsAppChannel`` drops it from
an inbound sender.
"""

from __future__ import annotations

from typing import Any

from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.twilio.config import TwilioConfig
from roomkit.providers.twilio.sms import TwilioSMSProvider, parse_twilio_webhook
from roomkit.providers.whatsapp.base import WhatsAppProvider

_SCHEME = "whatsapp:"


def _whatsapp_address(number: str) -> str:
    """*number* as Twilio addresses it on WhatsApp."""
    return number if number.startswith(_SCHEME) else f"{_SCHEME}{number}"


class TwilioWhatsAppProvider(WhatsAppProvider):
    """WhatsApp provider using Twilio's Messages API.

    ``config.from_number`` is the WhatsApp sender's number (E.164, with or
    without ``whatsapp:``); a ``messaging_service_sid`` holding a WhatsApp
    sender replaces it, as on SMS. Messages outside the 24-hour customer
    service window need an approved template, which Twilio refuses otherwise;
    the refusal comes back as a failed result.
    """

    def __init__(self, config: TwilioConfig) -> None:
        self._config = config
        self._messages = TwilioSMSProvider(config)

    @property
    def name(self) -> str:
        return "twilio_whatsapp"

    @property
    def from_number(self) -> str:
        return self._config.from_number

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        # The channel's telemetry reaches this provider; the Messages API
        # client records under it too.
        self._messages._telemetry = getattr(self, "_telemetry", None)  # noqa: SLF001
        return await self._messages.send(
            event,
            _whatsapp_address(to),
            from_=_whatsapp_address(self._config.from_number),
        )

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        timestamp: str | None = None,
        url: str | None = None,
    ) -> bool:
        """Verify a Twilio webhook signature (HMAC-SHA1), as for SMS."""
        return self._messages.verify_signature(payload, signature, timestamp, url)

    async def close(self) -> None:
        await self._messages.close()


def parse_twilio_whatsapp_webhook(payload: dict[str, Any], channel_id: str) -> InboundMessage:
    """Convert a Twilio WhatsApp webhook into an InboundMessage.

    The payload is the Messages API's, as for SMS; the sender comes as
    ``whatsapp:+15550000001``, which ``WhatsAppChannel`` reads as
    ``+15550000001`` (RFC §5.7).
    """
    return parse_twilio_webhook(payload, channel_id=channel_id)

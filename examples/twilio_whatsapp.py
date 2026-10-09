"""Twilio WhatsApp example — one Twilio number on SMS and on WhatsApp.

Twilio carries WhatsApp on the Messages API it uses for SMS: one account, one
webhook, and the WhatsApp addresses written ``whatsapp:+15145551234``. This
example parses an SMS and a WhatsApp webhook from the same customer and routes
them: each channel gets its own room, its binding naming the customer by the
same E.164 number, so the WhatsApp room replies through
``TwilioWhatsAppProvider`` (which sends to ``whatsapp:+1...``) and the SMS room
through ``TwilioSMSProvider``.

Offline demo: no agent answers, so nothing is sent and no network call is
made. To send for real, put your Twilio credentials and a WhatsApp-enabled
sender in TwilioConfig and attach an agent; messages outside WhatsApp's
24-hour customer service window need an approved template, which Twilio
refuses otherwise. In production, a web server receives Twilio's webhook,
verifies ``X-Twilio-Signature`` with ``provider.verify_signature`` and picks
the channel by the ``From`` prefix, as below.

Run with:
    uv run python examples/twilio_whatsapp.py
"""

from __future__ import annotations

import asyncio

from shared import setup_logging

from roomkit import RoomKit
from roomkit.channels import SMSChannel, WhatsAppChannel
from roomkit.providers.twilio import (
    TwilioConfig,
    TwilioSMSProvider,
    TwilioWhatsAppProvider,
    parse_twilio_webhook,
)

logger = setup_logging("twilio_whatsapp")

BUSINESS = "+15145551234"
CUSTOMER = "+15145559999"


def channel_for(payload: dict[str, str]) -> str:
    """Twilio posts SMS and WhatsApp to one webhook: the sender's prefix tells them apart."""
    return "whatsapp" if payload["From"].startswith("whatsapp:") else "sms"


async def main() -> None:
    config = TwilioConfig(account_sid="ACdemo", auth_token="demo-token", from_number=BUSINESS)
    kit = RoomKit()
    kit.register_channel(SMSChannel("sms", provider=TwilioSMSProvider(config)))
    kit.register_channel(WhatsAppChannel("whatsapp", provider=TwilioWhatsAppProvider(config)))

    webhooks = [
        {"MessageSid": "SM1", "From": CUSTOMER, "To": BUSINESS, "Body": "Hi by SMS"},
        {
            "MessageSid": "SM2",
            "From": f"whatsapp:{CUSTOMER}",
            "To": f"whatsapp:{BUSINESS}",
            "Body": "Hi on WhatsApp",
        },
    ]
    for payload in webhooks:
        message = parse_twilio_webhook(
            {**payload, "NumMedia": "0"}, channel_id=channel_for(payload)
        )
        result = await kit.process_inbound(message)
        assert result.event is not None
        binding = await kit.store.get_binding(result.event.room_id, message.channel_id)
        assert binding is not None
        logger.info(
            "%-8s from %-26s -> room %s, correspondent %s, replies to %s",
            message.channel_id,
            payload["From"],
            result.event.room_id[:8],
            binding.participant_id,
            binding.metadata.get("phone_number"),
        )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

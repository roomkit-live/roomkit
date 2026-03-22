# Transport Providers

Transport providers handle sending and receiving messages over external protocols. Each provider implements a channel-specific ABC.

## SMS

### Twilio

```python
from roomkit import RoomKit, SMSChannel
from roomkit.providers.twilio.sms import TwilioSMSProvider
from roomkit.providers.twilio.config import TwilioConfig

sms = SMSChannel("sms-twilio", provider=TwilioSMSProvider(TwilioConfig(
    account_sid="AC...",
    auth_token="...",
    from_number="+15551234567",
)))

kit = RoomKit()
kit.register_channel(sms)
```

### Telnyx

```python
from roomkit.providers.telnyx.sms import TelnyxSMSProvider
from roomkit.providers.telnyx.config import TelnyxConfig

sms = SMSChannel("sms-telnyx", provider=TelnyxSMSProvider(TelnyxConfig(
    api_key="KEY...",
    from_number="+15551234567",
)))
```

### Sinch

```python
from roomkit.providers.sinch.sms import SinchSMSProvider
from roomkit.providers.sinch.config import SinchConfig

sms = SMSChannel("sms-sinch", provider=SinchSMSProvider(SinchConfig(
    service_plan_id="...",
    api_token="...",
    from_number="+15551234567",
)))
```

### VoiceMeUp

```python
from roomkit.providers.voicemeup.sms import VoiceMeUpSMSProvider
from roomkit.providers.voicemeup.config import VoiceMeUpConfig

sms = SMSChannel("sms-vmu", provider=VoiceMeUpSMSProvider(VoiceMeUpConfig(
    username="...",
    password="...",
    from_number="+15551234567",
)))
```

### Webhook Parsing

Each SMS provider has a webhook parser:

```python
from roomkit.providers.twilio.sms import parse_twilio_webhook
from roomkit.providers.telnyx.sms import parse_telnyx_webhook
from roomkit.providers.sinch.sms import parse_sinch_webhook
from roomkit.providers.voicemeup.sms import VoiceMeUpSMSProvider  # use provider.parse_inbound()

# Or use the universal webhook parser
message = await kit.process_webhook(meta=request_data, channel_id="sms-twilio")
```

## RCS

```python
from roomkit import RCSChannel
from roomkit.providers.twilio.rcs import TwilioRCSProvider, TwilioRCSConfig

rcs = RCSChannel("rcs-main", provider=TwilioRCSProvider(TwilioRCSConfig(
    account_sid="AC...",
    auth_token="...",
    from_number="+15551234567",
)))
```

Also available via Telnyx: `TelnyxRCSProvider`, `TelnyxRCSConfig`.

## Email

### Elastic Email

```python
from roomkit import EmailChannel
from roomkit.providers.elasticemail.email import ElasticEmailProvider
from roomkit.providers.elasticemail.config import ElasticEmailConfig

email = EmailChannel("email-main", provider=ElasticEmailProvider(ElasticEmailConfig(
    api_key="...",
    from_email="support@example.com",
    from_name="Support Team",
)))
```

### SendGrid

```python
from roomkit.providers.sendgrid.email import SendGridEmailProvider
from roomkit.providers.sendgrid.config import SendGridConfig

email = EmailChannel("email-sg", provider=SendGridEmailProvider(SendGridConfig(
    api_key="SG...",
    from_email="support@example.com",
)))
```

## WhatsApp

### Business API (Cloud)

```python
from roomkit import WhatsAppChannel
from roomkit.providers.whatsapp.base import WhatsAppProvider

whatsapp = WhatsAppChannel("wa-business", provider=WhatsAppProvider(
    access_token="...",
    phone_number_id="...",
))
```

### Personal (neonize)

```python
from roomkit import WhatsAppPersonalChannel
from roomkit.providers.whatsapp.personal import WhatsAppPersonalProvider

whatsapp = WhatsAppPersonalChannel("wa-personal", provider=WhatsAppPersonalProvider())
```

Requires `pip install roomkit[whatsapp-personal]`. Uses the neonize library for multidevice protocol with typing indicators, read receipts, and media handling.

## Facebook Messenger

```python
from roomkit import MessengerChannel
from roomkit.providers.messenger.facebook import FacebookMessengerProvider
from roomkit.providers.messenger.config import MessengerConfig

messenger = MessengerChannel("messenger", provider=FacebookMessengerProvider(MessengerConfig(
    page_access_token="...",
    app_secret="...",
    verify_token="...",
)))
```

Webhook parser: `parse_messenger_webhook(request_data)`.

## Telegram

```python
from roomkit import TelegramChannel
from roomkit.providers.telegram.bot import TelegramBotProvider
from roomkit.providers.telegram.config import TelegramConfig

telegram = TelegramChannel("telegram", provider=TelegramBotProvider(TelegramConfig(
    bot_token="123456:ABC-DEF...",
)))
```

Webhook parser: `parse_telegram_webhook(request_data)`.

## Microsoft Teams

```python
from roomkit import TeamsChannel
from roomkit.providers.teams.bot_framework import BotFrameworkTeamsProvider
from roomkit.providers.teams.config import TeamsConfig

teams = TeamsChannel("teams", provider=BotFrameworkTeamsProvider(TeamsConfig(
    app_id="...",
    app_password="...",
)))
```

Features: proactive messaging, bot mention detection, reaction handling, conversation reference storage.

```python
from roomkit.providers.teams.webhook import parse_teams_webhook, is_bot_added

# Parse incoming Teams activity
activity = parse_teams_webhook(request_data)

# Check if bot was added to a conversation
if is_bot_added(activity):
    # Handle bot installation
    pass
```

## HTTP (Generic Webhook)

```python
from roomkit import HTTPChannel
from roomkit.providers.http.provider import WebhookHTTPProvider
from roomkit.providers.http.config import HTTPProviderConfig

http = HTTPChannel("webhook", provider=WebhookHTTPProvider(HTTPProviderConfig(
    url="https://api.example.com/messages",
    headers={"Authorization": "Bearer ..."},
)))
```

## WebSocket

WebSocket channels don't use a provider — they handle connections directly:

```python
from roomkit import WebSocketChannel

ws = WebSocketChannel("ws-client")

# Register a connection
ws.register_connection("conn-1", on_receive_callback)

# In production, connect to the framework
await kit.connect_websocket("ws-client", "conn-1", send_fn)
await kit.disconnect_websocket("ws-client", "conn-1")
```

## Phone Number Utilities

```python
from roomkit.providers.sms.phone import is_valid_phone, normalize_phone

is_valid_phone("+15551234567")   # True
normalize_phone("555-123-4567")  # "+15551234567"
```

## Delivery Status Tracking

Track delivery status for sent messages:

```python
from roomkit import DeliveryStatus

@kit.on_delivery_status
async def track_delivery(status: DeliveryStatus) -> None:
    if status.status == "failed":
        print(f"Message {status.message_id} failed: {status.error_message}")

# Process status webhooks from providers
await kit.process_delivery_status(status)
```

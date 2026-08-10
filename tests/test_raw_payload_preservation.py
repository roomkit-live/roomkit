"""The provider's payload survives parsing (RFC §5.2).

"Implementations MUST preserve `raw_payload` unmodified. This is the audit
trail and the source of truth for provider-specific data." A parser lifts the
handful of fields RoomKit models; delivery annotations, carrier fields and
anything a provider added last week live only in what it was sent.
"""

from __future__ import annotations

from roomkit import RoomKit
from roomkit.channels.websocket import WebSocketChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.providers.http.webhook import parse_http_webhook
from roomkit.providers.messenger.webhook import parse_messenger_webhook
from roomkit.providers.teams.webhook import parse_teams_webhook
from roomkit.providers.telegram.webhook import parse_telegram_webhook
from roomkit.providers.twilio.webhook import parse_twilio_payload


class TestParsersPreserveThePayload:
    def test_twilio_keeps_fields_it_does_not_model(self) -> None:
        payload = {
            "From": "+15551234567",
            "Body": "hello",
            "MessageSid": "SM123",
            # Nothing in RoomKit's model covers these.
            "SmsStatus": "received",
            "FromCity": "MONTREAL",
            "ApiVersion": "2010-04-01",
        }
        message = parse_twilio_payload(payload, "sms1")

        assert message.raw_payload == payload
        assert message.raw_payload["FromCity"] == "MONTREAL"
        assert message.provider_message_id == "SM123"

    def test_the_payload_is_copied_not_aliased(self) -> None:
        """Preserved "unmodified" means the caller cannot mutate it afterwards."""
        payload = {"From": "+1555", "Body": "hi", "MessageSid": "SM1"}
        message = parse_twilio_payload(payload, "sms1")

        payload["Body"] = "tampered"
        assert message.raw_payload["Body"] == "hi"

    def test_http_keeps_the_whole_body(self) -> None:
        payload = {"sender_id": "u1", "body": "hello", "custom_field": {"nested": True}}
        message = parse_http_webhook(payload, "http1")
        assert message.raw_payload["custom_field"] == {"nested": True}

    def test_telegram_keeps_the_update_envelope(self) -> None:
        """The message object alone would drop ``update_id``, which places the
        message in Telegram's own sequence."""
        payload = {
            "update_id": 987654,
            "message": {
                "message_id": 42,
                "chat": {"id": 111, "type": "private"},
                "from": {"id": 222, "is_bot": False, "first_name": "Ada"},
                "text": "hello",
            },
        }
        messages = parse_telegram_webhook(payload, "tg1")
        assert len(messages) == 1
        assert messages[0].raw_payload["update_id"] == 987654
        assert messages[0].provider_message_id == "42"

    def test_teams_keeps_the_activity(self) -> None:
        payload = {
            "type": "message",
            "id": "act-1",
            "text": "hello",
            "from": {"id": "u1", "name": "Ada"},
            "conversation": {"id": "c1", "conversationType": "personal"},
            "attachments": [{"contentType": "image/png"}],
        }
        messages = parse_teams_webhook(payload, "teams1")
        assert len(messages) == 1
        assert messages[0].raw_payload["attachments"] == [{"contentType": "image/png"}]
        assert messages[0].provider_message_id == "act-1"

    def test_messenger_keeps_each_entry_not_the_envelope(self) -> None:
        """A batched webhook must leave each message its own payload."""
        payload = {
            "entry": [
                {
                    "messaging": [
                        {
                            "sender": {"id": "s1"},
                            "recipient": {"id": "page"},
                            "timestamp": 1,
                            "message": {"mid": "m1", "text": "first"},
                        },
                        {
                            "sender": {"id": "s2"},
                            "recipient": {"id": "page"},
                            "timestamp": 2,
                            "message": {"mid": "m2", "text": "second"},
                        },
                    ]
                }
            ]
        }
        messages = parse_messenger_webhook(payload, "fb1")

        assert len(messages) == 2
        assert messages[0].raw_payload["message"]["mid"] == "m1"
        assert messages[1].raw_payload["message"]["mid"] == "m2"
        assert messages[0].provider_message_id == "m1"


class TestPayloadReachesTheTimeline:
    async def test_raw_payload_lands_on_the_stored_event(self) -> None:
        """End to end: the audit trail is only a guarantee if it survives the
        pipeline, not just the parser."""
        kit = RoomKit()
        kit.register_channel(WebSocketChannel("ws1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        await kit.process_inbound(
            InboundMessage(
                channel_id="ws1",
                sender_id="user1",
                content=TextContent(body="hello"),
                raw_payload={"MessageSid": "SM1", "FromCity": "MONTREAL"},
                provider_message_id="SM1",
            ),
            room_id="r1",
        )

        timeline = await kit.get_timeline("r1")
        message = timeline[-1]
        assert message.source.raw_payload == {"MessageSid": "SM1", "FromCity": "MONTREAL"}
        assert message.source.provider_message_id == "SM1"

    async def test_absent_payload_stays_empty(self) -> None:
        kit = RoomKit()
        kit.register_channel(WebSocketChannel("ws1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        await kit.process_inbound(
            InboundMessage(channel_id="ws1", sender_id="u", content=TextContent(body="hi")),
            room_id="r1",
        )

        timeline = await kit.get_timeline("r1")
        assert timeline[-1].source.raw_payload == {}

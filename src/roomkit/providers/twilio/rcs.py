"""Twilio RCS provider — sends RCS messages via the Twilio REST API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, SecretStr

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent
from roomkit.providers.rcs.base import RCSDeliveryResult, RCSProvider
from roomkit.providers.sms.meta import (
    extract_media_urls,
    extract_text_body,
)

if TYPE_CHECKING:
    import httpx


class TwilioRCSConfig(BaseModel):
    """Twilio RCS provider configuration."""

    account_sid: str
    auth_token: SecretStr
    messaging_service_sid: str  # Required for RCS (must be RCS-enabled)
    timeout: float = 10.0

    @property
    def api_url(self) -> str:
        return f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"


class TwilioRCSProvider(RCSProvider):
    """RCS provider using the Twilio REST API.

    Twilio RCS uses the same Messages API as SMS, but requires an RCS-enabled
    Messaging Service. When the recipient doesn't support RCS, Twilio can
    automatically fall back to SMS (unless disabled via fallback=False).
    """

    def __init__(self, config: TwilioRCSConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for TwilioRCSProvider. "
                "Install it with: pip install roomkit[httpx]"
            ) from exc
        self._config = config
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(timeout=config.timeout)

    @property
    def sender_id(self) -> str:
        return self._config.messaging_service_sid

    async def send(
        self,
        event: RoomEvent,
        to: str,
        *,
        fallback: bool = True,
    ) -> RCSDeliveryResult:
        """Send an RCS message via Twilio.

        Args:
            event: The room event containing the message content.
            to: Recipient phone number (E.164 format).
            fallback: If True, allow SMS fallback. If False, RCS only.

        Returns:
            Result with delivery info including channel used.
        """
        content = event.content
        body = extract_text_body(content)
        media_urls = extract_media_urls(content)

        if not body and not media_urls:
            return RCSDeliveryResult(success=False, error="empty_message")

        auth = (self._config.account_sid, self._config.auth_token.get_secret_value())

        # Twilio expects form-encoded data
        data: dict[str, str] = {
            "MessagingServiceSid": self._config.messaging_service_sid,
        }

        # For RCS-only (no fallback), prefix "to" with "rcs:"
        if fallback:
            data["To"] = to
        else:
            data["To"] = f"rcs:{to}"

        if body:
            data["Body"] = body

        # Add media URLs (Twilio supports up to 10)
        for i, url in enumerate(media_urls[:10]):
            data[f"MediaUrl{i}"] = url

        try:
            import time

            t0 = time.monotonic()
            resp = await self._client.post(
                self._config.api_url,
                auth=auth,
                data=data,
            )
            resp.raise_for_status()
            send_ms = (time.monotonic() - t0) * 1000
            response_data = resp.json()

            from roomkit.telemetry.noop import NoopTelemetryProvider

            _tel = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
            _tel.record_metric(
                "roomkit.delivery.send_ms",
                send_ms,
                unit="ms",
                attributes={"provider": "TwilioRCSProvider"},
            )
        except (
            self._httpx.TimeoutException,
            self._httpx.HTTPStatusError,
            self._httpx.HTTPError,
        ) as exc:
            from roomkit.providers.http_errors import handle_http_error, parse_twilio_error

            return handle_http_error(  # ty: ignore[invalid-return-type]
                exc,
                self._httpx,
                parse_400=parse_twilio_error,
                result_cls=RCSDeliveryResult,
            )

        # Twilio returns the channel used in the response
        # For now, we assume RCS unless we detect fallback from status callback
        return RCSDeliveryResult(
            success=True,
            provider_message_id=response_data.get("sid"),
            channel_used="rcs",
            fallback=False,
        )

    async def close(self) -> None:
        await self._client.aclose()


def parse_twilio_rcs_webhook(
    payload: dict[str, Any],
    channel_id: str,
) -> InboundMessage:
    """Convert a Twilio RCS webhook POST body into an InboundMessage.

    Note: Twilio sends webhooks as form-encoded data. Convert to dict first:
        payload = dict(await request.form())
    """
    from roomkit.providers.twilio.webhook import parse_twilio_payload

    return parse_twilio_payload(
        payload,
        channel_id,
        extra_metadata={
            "channel": payload.get("Channel", "rcs"),
            "messaging_service_sid": payload.get("MessagingServiceSid", ""),
        },
    )

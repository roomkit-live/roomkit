"""Twilio SMS provider — sends SMS via the Twilio REST API."""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.sms.base import SMSProvider
from roomkit.providers.sms.meta import (
    build_inbound_content,
    extract_media_urls,
    extract_text_body,
)
from roomkit.providers.twilio.config import TwilioConfig

if TYPE_CHECKING:
    import httpx


class TwilioSMSProvider(SMSProvider):
    """SMS provider using the Twilio REST API."""

    def __init__(self, config: TwilioConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for TwilioSMSProvider. "
                "Install it with: pip install roomkit[httpx]"
            ) from exc
        self._config = config
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(timeout=config.timeout)

    @property
    def from_number(self) -> str:
        return self._config.from_number

    async def send(self, event: RoomEvent, to: str, from_: str | None = None) -> ProviderResult:
        content = event.content
        body = extract_text_body(content)
        media_urls = extract_media_urls(content)

        if not body and not media_urls:
            return ProviderResult(success=False, error="empty_message")

        from_number = from_ or self._config.from_number

        # Twilio uses HTTP Basic auth
        auth = (self._config.account_sid, self._config.auth_token.get_secret_value())

        # Twilio expects form-encoded data
        data: dict[str, str] = {
            "To": to,
        }

        if body:
            data["Body"] = body

        # Add media URLs (Twilio supports up to 10)
        for i, url in enumerate(media_urls[:10]):
            data[f"MediaUrl{i}"] = url

        # Use MessagingServiceSid if provided, otherwise use From number
        if self._config.messaging_service_sid:
            data["MessagingServiceSid"] = self._config.messaging_service_sid
        else:
            data["From"] = from_number

        try:
            resp = await self._client.post(
                self._config.api_url,
                auth=auth,
                data=data,
            )
            resp.raise_for_status()
            response_data = resp.json()
        except self._httpx.TimeoutException:
            return ProviderResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                return ProviderResult(success=False, error="auth_error")
            if status == 429:
                return ProviderResult(success=False, error="rate_limit")
            if status == 400:
                # Try to extract Twilio error code
                try:
                    error_data = exc.response.json()
                    error_code = error_data.get("code", "invalid_request")
                    return ProviderResult(
                        success=False,
                        error=f"twilio_{error_code}",
                        metadata={"message": error_data.get("message", "")},
                    )
                except Exception:
                    return ProviderResult(success=False, error="invalid_request")
            return ProviderResult(success=False, error=f"http_{status}")
        except self._httpx.HTTPError as exc:
            return ProviderResult(success=False, error=str(exc))

        return ProviderResult(
            success=True,
            provider_message_id=response_data.get("sid"),
        )

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        timestamp: str | None = None,
        url: str | None = None,
    ) -> bool:
        """Verify a Twilio webhook signature using HMAC-SHA1.

        Args:
            payload: Raw request body bytes (form-encoded).
            signature: Value of the ``X-Twilio-Signature`` header.
            timestamp: Not used by Twilio (included for interface compatibility).
            url: The full URL that Twilio called (required for verification).

        Returns:
            True if the signature is valid, False otherwise.
        """
        if not url:
            return False

        # Parse form-encoded payload and sort parameters
        try:
            from urllib.parse import unquote_plus

            pairs = payload.decode().split("&")
            params = dict(pair.split("=", 1) for pair in pairs if "=" in pair)
            # URL decode the values
            params = {k: unquote_plus(v) for k, v in params.items()}
        except Exception:
            return False

        # Build the validation string: URL + sorted params
        validation_string = url
        for key in sorted(params.keys()):
            validation_string += key + params[key]

        # Compute HMAC-SHA1
        expected_sig = base64.b64encode(
            hmac.new(
                self._config.auth_token.get_secret_value().encode(),
                validation_string.encode(),
                hashlib.sha1,
            ).digest()
        ).decode()

        return hmac.compare_digest(expected_sig, signature)

    async def close(self) -> None:
        await self._client.aclose()


def parse_twilio_webhook(
    payload: dict[str, Any],
    channel_id: str,
) -> InboundMessage:
    """Convert a Twilio webhook POST body into an InboundMessage.

    Note: Twilio sends webhooks as form-encoded data. Convert to dict first:
        payload = dict(await request.form())
    """
    body = payload.get("Body", "")

    media: list[dict[str, str | None]] = []
    num_media = int(payload.get("NumMedia", "0"))
    for i in range(num_media):
        url = payload.get(f"MediaUrl{i}")
        if url:
            media.append(
                {
                    "url": url,
                    "mime_type": payload.get(f"MediaContentType{i}"),
                }
            )

    return InboundMessage(
        channel_id=channel_id,
        sender_id=payload.get("From", ""),
        content=build_inbound_content(body, media),
        external_id=payload.get("MessageSid"),
        idempotency_key=payload.get("MessageSid"),
        metadata={
            "to": payload.get("To", ""),
            "account_sid": payload.get("AccountSid", ""),
            "num_media": payload.get("NumMedia", "0"),
            "from_city": payload.get("FromCity"),
            "from_state": payload.get("FromState"),
            "from_country": payload.get("FromCountry"),
        },
    )

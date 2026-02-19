"""Sinch SMS provider — sends SMS via the Sinch REST API."""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.sinch.config import SinchConfig
from roomkit.providers.sms.base import SMSProvider
from roomkit.providers.sms.meta import (
    build_inbound_content,
    extract_media_urls,
    extract_text_body,
)

if TYPE_CHECKING:
    import httpx


class SinchSMSProvider(SMSProvider):
    """SMS provider using the Sinch REST API."""

    def __init__(self, config: SinchConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for SinchSMSProvider. "
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

        headers = {
            "Authorization": f"Bearer {self._config.api_token.get_secret_value()}",
            "Content-Type": "application/json",
        }

        # Sinch expects 'to' as an array
        payload: dict[str, Any] = {
            "from": from_number,
            "to": [to],
        }

        if media_urls:
            # Sinch MMS: type mt_media, body contains url(s) and optional message.
            # When multiple media URLs are provided, use the `urls` array field;
            # for a single URL the simple `url` field works too.
            payload["type"] = "mt_media"
            media_body: dict[str, Any] = {}
            if len(media_urls) == 1:
                media_body["url"] = media_urls[0]
            else:
                media_body["url"] = media_urls[0]
                media_body["urls"] = media_urls
            if body:
                media_body["message"] = body
            payload["body"] = media_body
        else:
            payload["body"] = body

        try:
            import time

            t0 = time.monotonic()
            resp = await self._client.post(
                self._config.api_url,
                headers=headers,
                json=payload,
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
                attributes={"provider": "SinchSMSProvider"},
            )
        except self._httpx.TimeoutException:
            return ProviderResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                return ProviderResult(success=False, error="auth_error")
            if status == 429:
                return ProviderResult(success=False, error="rate_limit")
            if status == 400:
                try:
                    error_data = exc.response.json()
                    error_code = error_data.get("code", "invalid_request")
                    return ProviderResult(
                        success=False,
                        error=f"sinch_{error_code}",
                        metadata={"message": error_data.get("text", "")},
                    )
                except Exception:
                    return ProviderResult(success=False, error="invalid_request")
            return ProviderResult(success=False, error=f"http_{status}")
        except self._httpx.HTTPError as exc:
            return ProviderResult(success=False, error=str(exc))

        return ProviderResult(
            success=True,
            provider_message_id=response_data.get("id"),
        )

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        timestamp: str | None = None,
    ) -> bool:
        """Verify a Sinch webhook signature using HMAC-SHA1.

        Args:
            payload: Raw request body bytes (JSON).
            signature: Value of the ``X-Sinch-Signature`` header.
            timestamp: Not used by Sinch (included for interface compatibility).

        Returns:
            True if the signature is valid, False otherwise.

        Raises:
            ValueError: If webhook_secret was not provided in config.
        """
        if not self._config.webhook_secret:
            raise ValueError(
                "webhook_secret must be provided in SinchConfig for signature verification"
            )

        try:
            expected_sig = base64.b64encode(
                hmac.new(
                    self._config.webhook_secret.get_secret_value().encode(),
                    payload,
                    hashlib.sha1,
                ).digest()
            ).decode()
            return hmac.compare_digest(expected_sig, signature)
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


def parse_sinch_webhook(
    payload: dict[str, Any],
    channel_id: str,
) -> InboundMessage:
    """Convert a Sinch SMS webhook POST body into an InboundMessage.

    Sinch inbound SMS webhook structure:
    {
        "id": "message-id",
        "from": "+15551234567",
        "to": "12345",
        "body": "Message text",
        "received_at": "2026-01-28T12:00:00.000Z",
        "operator_id": "...",
        "media": [{"url": "...", "mimeType": "image/jpeg"}],
        ...
    }
    """
    body = payload.get("body", "")

    media: list[dict[str, str | None]] = []
    for m in payload.get("media", []):
        if url := m.get("url"):
            media.append(
                {
                    "url": url,
                    "mime_type": m.get("mimeType"),
                }
            )

    return InboundMessage(
        channel_id=channel_id,
        sender_id=payload.get("from", ""),
        content=build_inbound_content(body, media),
        external_id=payload.get("id"),
        idempotency_key=payload.get("id"),
        metadata={
            "to": payload.get("to", ""),
            "received_at": payload.get("received_at"),
            "operator_id": payload.get("operator_id"),
            "client_reference": payload.get("client_reference"),
        },
    )

"""Telnyx SMS provider — sends SMS via the Telnyx REST API."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.sms.base import SMSProvider
from roomkit.providers.sms.meta import (
    build_inbound_content,
    extract_media_urls,
    extract_text_body,
)
from roomkit.providers.telnyx.config import TelnyxConfig

if TYPE_CHECKING:
    import httpx

_API_URL = "https://api.telnyx.com/v2/messages"


class TelnyxSMSProvider(SMSProvider):
    """SMS provider using the Telnyx REST API."""

    def __init__(self, config: TelnyxConfig, public_key: str | None = None) -> None:
        """Initialize the Telnyx SMS provider.

        Args:
            config: Telnyx configuration.
            public_key: Telnyx public key for webhook signature verification.
                Found in Mission Control Portal > Keys & Credentials > Public Key.
        """
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for TelnyxSMSProvider. "
                "Install it with: pip install roomkit[httpx]"
            ) from exc
        self._config = config
        self._public_key = public_key
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
            "Authorization": f"Bearer {self._config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "from": from_number,
            "to": to,
        }

        if body:
            payload["text"] = body

        if media_urls:
            payload["media_urls"] = media_urls

        if self._config.messaging_profile_id:
            payload["messaging_profile_id"] = self._config.messaging_profile_id

        try:
            resp = await self._client.post(_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except self._httpx.TimeoutException:
            return ProviderResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                return ProviderResult(success=False, error="auth_error")
            if status == 429:
                return ProviderResult(success=False, error="rate_limit")
            if status == 400:
                return ProviderResult(success=False, error="invalid_request")
            return ProviderResult(success=False, error=f"http_{status}")
        except self._httpx.HTTPError as exc:
            return ProviderResult(success=False, error=str(exc))

        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> ProviderResult:
        message_data = data.get("data", {})
        message_id = message_data.get("id")
        return ProviderResult(success=True, provider_message_id=message_id)

    async def close(self) -> None:
        await self._client.aclose()

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        timestamp: str | None = None,
    ) -> bool:
        """Verify a Telnyx webhook signature using ED25519.

        Args:
            payload: Raw request body bytes.
            signature: Value of the ``Telnyx-Signature-Ed25519`` header.
            timestamp: Value of the ``Telnyx-Timestamp`` header.

        Returns:
            True if the signature is valid, False otherwise.

        Raises:
            ValueError: If public_key was not provided to the constructor.
            ImportError: If PyNaCl is not installed.
        """
        if not self._public_key:
            raise ValueError(
                "public_key must be provided to TelnyxSMSProvider for signature verification"
            )

        if not timestamp:
            return False

        try:
            from nacl.signing import VerifyKey
        except ImportError as exc:
            raise ImportError(
                "PyNaCl is required for Telnyx signature verification. "
                "Install it with: pip install pynacl"
            ) from exc

        try:
            # Telnyx signs: timestamp|payload
            signed_payload = f"{timestamp}|".encode() + payload
            signature_bytes = base64.b64decode(signature)
            public_key_bytes = base64.b64decode(self._public_key)
            verify_key = VerifyKey(public_key_bytes)
            verify_key.verify(signed_payload, signature_bytes)
            return True
        except Exception:
            return False


def _is_telnyx_inbound(payload: dict[str, Any]) -> bool:
    """Check if a Telnyx webhook is an inbound message (internal use)."""
    event_data = payload.get("data", {})
    event_type = str(event_data.get("event_type", ""))
    direction = str(event_data.get("payload", {}).get("direction", ""))

    return event_type == "message.received" and direction == "inbound"


def parse_telnyx_webhook(
    payload: dict[str, Any],
    channel_id: str,
    *,
    strict: bool = True,
) -> InboundMessage:
    """Convert a Telnyx webhook POST body into an InboundMessage.

    Args:
        payload: The Telnyx webhook POST body as a dictionary.
        channel_id: The channel ID to associate with the message.
        strict: If True (default), raises ValueError for non-inbound webhooks.
            Set to False to skip validation (not recommended).

    Returns:
        An InboundMessage ready for process_inbound().

    Raises:
        ValueError: If strict=True and the webhook is not an inbound message.

    Example:
        Recommended: Use extract_sms_meta() for generic handling::

            from roomkit import extract_sms_meta

            @app.post("/webhooks/sms/{provider}")
            async def sms_webhook(provider: str, payload: dict):
                meta = extract_sms_meta(provider, payload)
                if meta.is_inbound:
                    await kit.process_inbound(meta.to_inbound("sms"))
                elif meta.is_status:
                    await kit.process_delivery_status(meta.to_status())
                return {"ok": True}
    """
    if strict and not _is_telnyx_inbound(payload):
        event_type = payload.get("data", {}).get("event_type", "unknown")
        direction = payload.get("data", {}).get("payload", {}).get("direction", "unknown")
        raise ValueError(
            f"Not an inbound message (event_type={event_type}, direction={direction}). "
            f"Use extract_sms_meta() with meta.is_inbound to filter webhooks."
        )

    data = payload["data"]["payload"]
    body = data.get("text", "")

    media: list[dict[str, str | None]] = []
    for m in data.get("media", []):
        if url := m.get("url"):
            media.append(
                {
                    "url": url,
                    "mime_type": m.get("content_type"),
                }
            )

    return InboundMessage(
        channel_id=channel_id,
        sender_id=data["from"]["phone_number"],
        content=build_inbound_content(body, media),
        external_id=data["id"],
        idempotency_key=data["id"],
        metadata={
            "destination_number": data["to"][0]["phone_number"],
            "received_at": data.get("received_at"),
        },
    )

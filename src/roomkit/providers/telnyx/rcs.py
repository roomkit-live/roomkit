"""Telnyx RCS provider — sends RCS messages via the Telnyx REST API."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, SecretStr

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent
from roomkit.providers.rcs.base import RCSDeliveryResult, RCSProvider
from roomkit.providers.sms.meta import (
    build_inbound_content,
    extract_media_urls,
    extract_text_body,
)

if TYPE_CHECKING:
    import httpx

_API_URL = "https://api.telnyx.com/v2/messages"
_RCS_CAPABILITY_URL = "https://api.telnyx.com/v2/messaging/rcs/capabilities"


class TelnyxRCSConfig(BaseModel):
    """Telnyx RCS provider configuration.

    Attributes:
        api_key: Telnyx API key (v2 key starting with KEY...).
        agent_id: RCS agent ID (obtained after agent onboarding/brand approval).
        messaging_profile_id: Optional messaging profile ID for webhooks.
        timeout: HTTP request timeout in seconds.
    """

    api_key: SecretStr
    agent_id: str
    messaging_profile_id: str | None = None
    timeout: float = 10.0


class TelnyxRCSProvider(RCSProvider):
    """RCS provider using the Telnyx REST API.

    Telnyx RCS uses the same /v2/messages endpoint as SMS, but requires an
    RCS agent_id as the sender. When the recipient doesn't support RCS,
    messages can fall back to SMS (if fallback=True).

    Example:
        config = TelnyxRCSConfig(
            api_key="KEY...",
            agent_id="your-rcs-agent-id",
        )
        provider = TelnyxRCSProvider(config)
        result = await provider.send(event, to="+14155551234")
    """

    def __init__(self, config: TelnyxRCSConfig, public_key: str | None = None) -> None:
        """Initialize the Telnyx RCS provider.

        Args:
            config: Telnyx RCS configuration.
            public_key: Telnyx public key for webhook signature verification.
                Found in Mission Control Portal > Keys & Credentials > Public Key.
        """
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for TelnyxRCSProvider. "
                "Install it with: pip install roomkit[httpx]"
            ) from exc
        self._config = config
        self._public_key = public_key
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(timeout=config.timeout)

    @property
    def sender_id(self) -> str:
        """RCS agent ID used as sender."""
        return self._config.agent_id

    async def send(
        self,
        event: RoomEvent,
        to: str,
        *,
        fallback: bool = True,
    ) -> RCSDeliveryResult:
        """Send an RCS message via Telnyx.

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

        headers = {
            "Authorization": f"Bearer {self._config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "from": self._config.agent_id,
            "to": to,
            "type": "RCS",
        }

        if body:
            payload["text"] = body

        if media_urls:
            payload["media_urls"] = media_urls

        if self._config.messaging_profile_id:
            payload["messaging_profile_id"] = self._config.messaging_profile_id

        # If no fallback, set auto_detect to false to force RCS-only
        if not fallback:
            payload["auto_detect"] = False

        try:
            resp = await self._client.post(_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except self._httpx.TimeoutException:
            return RCSDeliveryResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                return RCSDeliveryResult(success=False, error="auth_error")
            if status == 429:
                return RCSDeliveryResult(success=False, error="rate_limit")
            if status == 400:
                try:
                    error_data = exc.response.json()
                    errors = error_data.get("errors", [])
                    if errors:
                        error_code = errors[0].get("code", "invalid_request")
                        error_msg = errors[0].get("detail", "")
                    else:
                        error_code = "invalid_request"
                        error_msg = ""
                    return RCSDeliveryResult(
                        success=False,
                        error=f"telnyx_{error_code}",
                        metadata={"message": error_msg},
                    )
                except Exception:
                    return RCSDeliveryResult(success=False, error="invalid_request")
            return RCSDeliveryResult(success=False, error=f"http_{status}")
        except self._httpx.HTTPError as exc:
            return RCSDeliveryResult(success=False, error=str(exc))

        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> RCSDeliveryResult:
        """Parse Telnyx API response into RCSDeliveryResult."""
        message_data = data.get("data", {})
        message_id = message_data.get("id")
        message_type = message_data.get("type", "RCS")

        # Determine if fallback occurred based on message type
        channel_used = "rcs" if message_type == "RCS" else "sms"
        fallback_occurred = channel_used == "sms"

        return RCSDeliveryResult(
            success=True,
            provider_message_id=message_id,
            channel_used=channel_used,
            fallback=fallback_occurred,
        )

    async def check_capability(self, phone_number: str) -> bool:
        """Check if a phone number supports RCS.

        Args:
            phone_number: Phone number to check (E.164 format).

        Returns:
            True if the number supports RCS, False otherwise.
        """
        headers = {
            "Authorization": f"Bearer {self._config.api_key.get_secret_value()}",
        }

        url = f"{_RCS_CAPABILITY_URL}/{self._config.agent_id}/{phone_number}"

        try:
            resp = await self._client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except self._httpx.HTTPError:
            return False

        # Check if RCS is supported
        capabilities = data.get("data", {})
        return bool(capabilities.get("rcs_enabled", False))

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
                "public_key must be provided to TelnyxRCSProvider for signature verification"
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

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


def _is_telnyx_rcs_inbound(payload: dict[str, Any]) -> bool:
    """Check if a Telnyx RCS webhook is an inbound message (internal use)."""
    event_data = payload.get("data", {})
    event_type = str(event_data.get("event_type", ""))
    direction = str(event_data.get("payload", {}).get("direction", ""))

    return event_type == "message.received" and direction == "inbound"


def parse_telnyx_rcs_webhook(
    payload: dict[str, Any],
    channel_id: str,
    *,
    strict: bool = True,
) -> InboundMessage:
    """Convert a Telnyx RCS webhook POST body into an InboundMessage.

    Telnyx RCS webhooks use JSON format (unlike SMS which can be form-encoded).
    The webhook structure follows the same pattern as SMS but includes RCS-specific
    fields like agent_id.

    Args:
        payload: The webhook POST body as a dictionary.
        channel_id: The channel ID to associate with the message.
        strict: If True (default), raises ValueError for non-inbound webhooks.
            Set to False to skip validation (not recommended).

    Returns:
        An InboundMessage ready for process_inbound().

    Raises:
        ValueError: If strict=True and the webhook is not an inbound message.
    """
    if strict and not _is_telnyx_rcs_inbound(payload):
        event_type = payload.get("data", {}).get("event_type", "unknown")
        direction = payload.get("data", {}).get("payload", {}).get("direction", "unknown")
        raise ValueError(
            f"Not an inbound message (event_type={event_type}, direction={direction}). "
            f"Use extract_sms_meta() with meta.is_inbound to filter webhooks."
        )
    data = payload.get("data", {}).get("payload", {})

    # Extract text content
    body = data.get("text", "")

    # Extract media (RCS supports rich media)
    media: list[dict[str, str | None]] = []
    for m in data.get("media", []):
        if url := m.get("url"):
            media.append(
                {
                    "url": url,
                    "mime_type": m.get("content_type"),
                }
            )

    # RCS can also have user_file for file transfers
    user_file = data.get("user_file", {}).get("payload", {})
    if user_file and (file_uri := user_file.get("file_uri")):
        media.append(
            {
                "url": file_uri,
                "mime_type": user_file.get("mime_type"),
            }
        )

    # Get sender info
    from_data = data.get("from", {})
    sender = from_data.get("phone_number", "")

    # Get recipient info (the RCS agent)
    to_data = data.get("to", {})
    agent_id = to_data.get("agent_id", "")

    return InboundMessage(
        channel_id=channel_id,
        sender_id=sender,
        content=build_inbound_content(body, media),
        external_id=data.get("id"),
        idempotency_key=data.get("id"),
        metadata={
            "agent_id": agent_id,
            "agent_name": to_data.get("agent_name", ""),
            "received_at": data.get("received_at"),
            "type": data.get("type", "RCS"),
            # RCS-specific: suggestion responses
            "suggestion_response": data.get("suggestion_response"),
            # RCS-specific: location sharing
            "location": data.get("location"),
        },
    )

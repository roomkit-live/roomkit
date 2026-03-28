"""VoiceMeUp SMS provider — sends SMS via the VoiceMeUp REST API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.sms.base import SMSProvider
from roomkit.providers.sms.meta import (
    build_inbound_content,
    extract_media_urls,
    extract_text_body,
)
from roomkit.providers.voicemeup.config import VoiceMeUpConfig

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

_MAX_SEGMENT_LENGTH = 1000


# ---------------------------------------------------------------------------
# MMS Aggregation helpers (stateless — state lives on the provider instance)
# ---------------------------------------------------------------------------


@dataclass
class _PendingMMS:
    """Buffered first part of a split MMS webhook."""

    payload: dict[str, Any]
    text: str
    timestamp: float
    channel_id: str


def _is_mms_metadata_wrapper(url: str | None, mime_type: str | None) -> bool:
    """Check if attachment is a VoiceMeUp MMS metadata wrapper (not real media)."""
    if not url:
        return False
    if url.endswith(".mms.html"):
        return True
    return mime_type == "text/html" and ".mms." in url


def _make_correlation_key(payload: dict[str, Any]) -> str:
    """Create a key to correlate split MMS webhooks."""
    return (
        f"{payload.get('source_number', '')}:"
        f"{payload.get('destination_number', '')}:"
        f"{payload.get('datetime_transmission', '')}"
    )


def _build_inbound_message(payload: dict[str, Any], channel_id: str) -> InboundMessage:
    """Build an InboundMessage from a VoiceMeUp webhook payload."""
    body = payload.get("message", "")

    media: list[dict[str, str | None]] = []
    attachment_url = payload.get("attachment") or payload.get("attachment_url")
    if attachment_url:
        media.append(
            {
                "url": attachment_url,
                "mime_type": payload.get("attachment_mime_type") or payload.get("attachment_type"),
            }
        )

    return InboundMessage(
        channel_id=channel_id,
        sender_id=payload.get("source_number", ""),
        content=build_inbound_content(body, media),
        external_id=payload.get("sms_hash"),
        idempotency_key=payload.get("sms_hash"),
        metadata={
            "destination_number": payload.get("destination_number", ""),
            "direction": payload.get("direction", "inbound"),
            "datetime_transmission": payload.get("datetime_transmission", ""),
            "has_attachment": bool(media),
        },
    )


def _strip_plus(number: str) -> str:
    """Strip leading '+' from an E.164 phone number."""
    return number.lstrip("+")


# ---------------------------------------------------------------------------
# SMS Provider
# ---------------------------------------------------------------------------


class VoiceMeUpSMSProvider(SMSProvider):
    """SMS provider using the VoiceMeUp REST API.

    MMS aggregation state is per-instance, supporting multi-tenant
    deployments where each tenant has its own provider.
    """

    def __init__(self, config: VoiceMeUpConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for VoiceMeUpSMSProvider. "
                "Install it with: pip install roomkit[httpx]"
            ) from exc
        self._config = config
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(timeout=config.timeout)
        # Per-instance MMS aggregation state
        self._mms_buffer: dict[str, _PendingMMS] = {}
        self._mms_timeout_seconds: float = 5.0
        self._mms_on_timeout: Callable[[InboundMessage], Awaitable[None] | None] | None = None

    @property
    def from_number(self) -> str:
        return self._config.from_number

    def configure_mms(
        self,
        *,
        timeout_seconds: float = 5.0,
        on_timeout: Callable[[InboundMessage], Awaitable[None] | None] | None = None,
    ) -> None:
        """Configure MMS aggregation for this provider instance.

        Args:
            timeout_seconds: How long to wait for the second MMS part.
            on_timeout: Callback invoked with text-only message if image
                never arrives. If not set, orphaned messages are logged.
        """
        self._mms_timeout_seconds = timeout_seconds
        self._mms_on_timeout = on_timeout

    def parse_inbound(
        self,
        payload: dict[str, Any],
        channel_id: str,
    ) -> InboundMessage | None:
        """Parse a VoiceMeUp webhook using this instance's MMS buffer.

        Handles split MMS aggregation: buffers the first part (text +
        metadata wrapper) and merges it with the second (actual media).

        Returns:
            InboundMessage if ready to process, None if buffered.
        """
        attachment_url = payload.get("attachment") or payload.get("attachment_url")
        attachment_mime = payload.get("attachment_mime_type") or payload.get("attachment_type")

        # First part of split MMS — buffer it
        if _is_mms_metadata_wrapper(attachment_url, attachment_mime):
            key = _make_correlation_key(payload)
            self._mms_buffer[key] = _PendingMMS(
                payload=payload,
                text=payload.get("message", ""),
                timestamp=time.time(),
                channel_id=channel_id,
            )
            with contextlib.suppress(RuntimeError):
                task = asyncio.get_running_loop().create_task(self._handle_mms_timeout(key))
                task.add_done_callback(log_task_exception)
            logger.debug("VoiceMeUp MMS: buffered first part for %s", key)
            return None

        # Second part — merge with buffered first part
        key = _make_correlation_key(payload)
        if key in self._mms_buffer:
            pending = self._mms_buffer.pop(key)
            merged_payload = dict(payload)
            merged_payload["message"] = pending.text
            first_hash = pending.payload.get("sms_hash", "")
            second_hash = payload.get("sms_hash", "")
            merged_payload["sms_hash"] = f"{first_hash}+{second_hash}"
            logger.debug("VoiceMeUp MMS: merged text + media for %s", key)
            return _build_inbound_message(merged_payload, channel_id)

        # Regular SMS or standalone MMS
        return _build_inbound_message(payload, channel_id)

    async def _handle_mms_timeout(self, key: str) -> None:
        """Emit buffered message as text-only if timeout expires."""
        await asyncio.sleep(self._mms_timeout_seconds)

        if key not in self._mms_buffer:
            return

        pending = self._mms_buffer.pop(key)
        payload_copy = dict(pending.payload)
        for k in ("attachment", "attachment_url", "attachment_mime_type", "attachment_type"):
            payload_copy.pop(k, None)

        message = _build_inbound_message(payload_copy, pending.channel_id)

        if self._mms_on_timeout:
            logger.debug("VoiceMeUp MMS timeout: invoking callback for %s", key)
            result = self._mms_on_timeout(message)
            if asyncio.iscoroutine(result):
                await result
        else:
            logger.warning(
                "VoiceMeUp MMS timeout: discarding orphaned text from %s "
                "(set on_timeout via configure_mms to handle this)",
                pending.payload.get("source_number"),
            )

    async def send(self, event: RoomEvent, to: str, from_: str | None = None) -> ProviderResult:
        body = extract_text_body(event.content)
        media_urls = extract_media_urls(event.content)

        if not body and not media_urls:
            return ProviderResult(success=False, error="empty_message")

        from_number = _strip_plus(from_ or self._config.from_number)
        to_number = _strip_plus(to)

        if media_urls:
            return await self._send_message(
                body or "", to_number, from_number, attachment=media_urls[0]
            )

        segments = self._split_message(body)
        last_result: ProviderResult | None = None

        for segment in segments:
            last_result = await self._send_message(segment, to_number, from_number)
            if not last_result.success:
                return last_result

        if last_result is None:
            raise RuntimeError("No segments to send")
        return last_result

    async def _send_message(
        self, message: str, to: str, from_: str, *, attachment: str | None = None
    ) -> ProviderResult:
        url = f"{self._config.base_url}queue_sms"

        form_data: dict[str, str] = {
            "username": self._config.username,
            "auth_token": self._config.auth_token.get_secret_value(),
            "source_number": from_,
            "destination_number": to,
        }

        if message:
            form_data["message"] = message

        if attachment:
            form_data["attachment"] = attachment

        try:
            import time as _time

            t0 = _time.monotonic()
            resp = await self._client.post(url, data=form_data)
            resp.raise_for_status()
            send_ms = (_time.monotonic() - t0) * 1000
            data = resp.json()

            from roomkit.telemetry.noop import NoopTelemetryProvider

            _tel = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
            _tel.record_metric(
                "roomkit.delivery.send_ms",
                send_ms,
                unit="ms",
                attributes={"provider": "VoiceMeUpSMSProvider"},
            )
        except (
            self._httpx.TimeoutException,
            self._httpx.HTTPStatusError,
            self._httpx.HTTPError,
        ) as exc:
            from roomkit.providers.http_errors import handle_http_error

            return handle_http_error(exc, self._httpx)

        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> ProviderResult:
        details = data.get("response_details", {})
        status = details.get("response_status", "")

        if status == "error":
            messages = details.get("response_messages", {}).get("message", [])
            if isinstance(messages, dict):
                messages = [messages]
            code = messages[0].get("code", "unknown_error") if messages else "unknown_error"
            description = messages[0].get("_content", code) if messages else code
            return ProviderResult(success=False, error=code, metadata={"description": description})

        messages = details.get("response_messages", {}).get("message", [])
        if isinstance(messages, dict):
            messages = [messages]

        sms_id: str | None = None
        for msg in messages:
            if msg.get("code") == "queued_sms_hash":
                sms_id = msg.get("_content")
                break

        return ProviderResult(success=True, provider_message_id=sms_id)

    @staticmethod
    def _split_message(text: str) -> list[str]:
        if len(text) <= _MAX_SEGMENT_LENGTH:
            return [text]
        import textwrap

        return textwrap.wrap(
            text,
            width=_MAX_SEGMENT_LENGTH,
            break_long_words=True,
            break_on_hyphens=False,
        )

    async def close(self) -> None:
        await self._client.aclose()

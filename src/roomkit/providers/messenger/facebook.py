"""Facebook Messenger provider — sends messages via the Messenger Platform Send API."""

from __future__ import annotations

import hashlib
import hmac
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.messenger.base import MessengerProvider
from roomkit.providers.messenger.config import MessengerConfig

if TYPE_CHECKING:
    import httpx


class FacebookMessengerProvider(MessengerProvider):
    """Send messages via the Facebook Messenger Platform Send API."""

    def __init__(self, config: MessengerConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for FacebookMessengerProvider. "
                "Install it with: pip install roomkit[httpx]"
            ) from exc
        self._config = config
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(
            timeout=config.timeout,
        )

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        text = self._extract_text(event)
        if not text:
            return ProviderResult(success=False, error="empty_message")

        payload: dict[str, Any] = {
            "recipient": {"id": to},
            "messaging_type": "RESPONSE",
            "message": {"text": text},
        }

        try:
            import time

            t0 = time.monotonic()
            resp = await self._client.post(
                self._config.base_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._config.page_access_token.get_secret_value()}",
                },
            )
            resp.raise_for_status()
            send_ms = (time.monotonic() - t0) * 1000
            data = resp.json()

            from roomkit.telemetry.noop import NoopTelemetryProvider

            _tel = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
            _tel.record_metric(
                "roomkit.delivery.send_ms",
                send_ms,
                unit="ms",
                attributes={"provider": "FacebookMessengerProvider"},
            )
        except self._httpx.TimeoutException:
            return ProviderResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            return self._parse_error(exc)
        except self._httpx.HTTPError as exc:
            return ProviderResult(success=False, error=str(exc))

        return ProviderResult(
            success=True,
            provider_message_id=data.get("message_id"),
        )

    @staticmethod
    def _parse_error(exc: Any) -> ProviderResult:
        """Extract a Facebook Graph API error message when available."""
        try:
            body = exc.response.json()
            error = body.get("error", {})
            code = error.get("code", exc.response.status_code)
            message = error.get("message", "")
            return ProviderResult(
                success=False,
                error=f"graph_{code}",
                metadata={"message": message},
            )
        except Exception:
            return ProviderResult(
                success=False,
                error=f"http_{exc.response.status_code}",
            )

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify a Facebook Messenger webhook signature using HMAC-SHA256.

        Facebook sends an ``X-Hub-Signature-256`` header with format
        ``sha256=<hex_digest>``.  This method computes the expected
        HMAC-SHA256 of *payload* using the ``app_secret`` from config
        and performs a constant-time comparison.

        Args:
            payload: Raw request body bytes.
            signature: Value of the ``X-Hub-Signature-256`` header.

        Returns:
            True if the signature is valid, False otherwise.

        Raises:
            ValueError: If ``app_secret`` was not provided in config.
        """
        if not self._config.app_secret:
            raise ValueError(
                "app_secret must be provided in MessengerConfig for signature verification"
            )

        prefix = "sha256="
        if not signature.startswith(prefix):
            return False

        expected = hmac.new(
            self._config.app_secret.get_secret_value().encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature[len(prefix) :])

    @staticmethod
    def _extract_text(event: RoomEvent) -> str:
        content = event.content
        if isinstance(content, TextContent):
            return content.body
        if hasattr(content, "body"):
            return str(content.body)
        return ""

    async def close(self) -> None:
        await self._client.aclose()

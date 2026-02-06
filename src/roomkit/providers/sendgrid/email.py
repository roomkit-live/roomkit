"""SendGrid provider — sends emails via the SendGrid v3 API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RichContent, RoomEvent, TextContent
from roomkit.providers.email.base import EmailProvider
from roomkit.providers.sendgrid.config import SendGridConfig

if TYPE_CHECKING:
    import httpx


class SendGridProvider(EmailProvider):
    """Send-only email provider using the SendGrid v3 Mail Send API."""

    def __init__(self, config: SendGridConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for SendGridProvider. "
                "Install it with: pip install roomkit[httpx]"
            ) from exc
        self._config = config
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(
            timeout=config.timeout,
        )

    async def send(
        self,
        event: RoomEvent,
        to: str,
        from_: str | None = None,
        subject: str | None = None,
    ) -> ProviderResult:
        body_text, body_html = self._extract_body(event)
        if not body_text and not body_html:
            return ProviderResult(success=False, error="empty_message")

        from_email = from_ or self._config.from_email
        from_obj: dict[str, str] = {"email": from_email}
        if self._config.from_name:
            from_obj["name"] = self._config.from_name

        content: list[dict[str, str]] = []
        if body_text:
            content.append({"type": "text/plain", "value": body_text})
        if body_html:
            content.append({"type": "text/html", "value": body_html})

        payload: dict[str, Any] = {
            "personalizations": [{"to": [{"email": to}]}],
            "from": from_obj,
            "subject": subject or "",
            "content": content,
        }

        headers = {
            "Authorization": f"Bearer {self._config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        try:
            resp = await self._client.post(
                self._config.base_url,
                json=payload,
                headers=headers,
            )
            if resp.status_code == 429:
                return ProviderResult(success=False, error="rate_limited")
            resp.raise_for_status()
        except self._httpx.TimeoutException:
            return ProviderResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            error_msg = f"http_{exc.response.status_code}"
            try:
                body = exc.response.json()
                errors = body.get("errors", [])
                if errors:
                    error_msg = errors[0].get("message", error_msg)
            except Exception:  # nosec B110 — best-effort error body parsing
                pass
            return ProviderResult(success=False, error=error_msg)
        except self._httpx.HTTPError as exc:
            return ProviderResult(success=False, error=str(exc))

        message_id = resp.headers.get("X-Message-Id")
        return ProviderResult(success=True, provider_message_id=message_id)

    @staticmethod
    def _extract_body(event: RoomEvent) -> tuple[str, str]:
        """Return ``(body_text, body_html)`` from the event content."""
        content = event.content
        if isinstance(content, RichContent):
            return "", content.body
        if isinstance(content, TextContent):
            return content.body, ""
        if hasattr(content, "body"):
            return str(content.body), ""
        return "", ""

    async def close(self) -> None:
        await self._client.aclose()

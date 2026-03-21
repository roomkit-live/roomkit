"""Shared HTTP error handling for delivery providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from roomkit.models.delivery import ProviderResult

# ---------------------------------------------------------------------------
# Provider-specific 400-level parsers
#
# Each returns ``(error_string, metadata_dict)`` extracted from the response
# body, or *None* to fall back to the generic ``"invalid_request"`` default.
# ---------------------------------------------------------------------------

ErrorDetail = tuple[str, dict[str, Any]]
"""``(error_code, metadata)`` extracted from a provider's 400 response."""


def parse_twilio_error(body: dict[str, Any]) -> ErrorDetail:
    """Twilio: ``{"code": 21211, "message": "..."}``."""
    code = body.get("code", "invalid_request")
    return f"twilio_{code}", {"message": body.get("message", "")}


def parse_sinch_error(body: dict[str, Any]) -> ErrorDetail:
    """Sinch: ``{"code": "invalid_parameter", "text": "..."}``."""
    code = body.get("code", "invalid_request")
    return f"sinch_{code}", {"message": body.get("text", "")}


def parse_telnyx_error(body: dict[str, Any]) -> ErrorDetail:
    """Telnyx: ``{"errors": [{"code": "...", "detail": "..."}]}``."""
    errors = body.get("errors", [])
    if errors:
        first = errors[0]
        code = first.get("code", "invalid_request")
        return f"telnyx_{code}", {"message": first.get("detail", "")}
    return "invalid_request", {}


def parse_sendgrid_error(body: dict[str, Any]) -> ErrorDetail:
    """SendGrid: ``{"errors": [{"message": "..."}]}``."""
    errors = body.get("errors", [])
    if errors:
        msg = errors[0].get("message", "")
        if msg:
            return msg, {}
    return "invalid_request", {}


def handle_http_error(
    exc: Exception,
    httpx_module: Any,
    *,
    parse_400: Callable[[dict[str, Any]], ErrorDetail] | None = None,
    result_cls: type[ProviderResult] = ProviderResult,
) -> ProviderResult:
    """Convert an httpx exception to a *ProviderResult*.

    Provides consistent error categorization across all delivery providers:

    - ``TimeoutException`` → ``"timeout"``
    - ``HTTPStatusError 401`` → ``"auth_error"``
    - ``HTTPStatusError 429`` → ``"rate_limit"``
    - ``HTTPStatusError 400`` → provider-specific via *parse_400* callback,
      falling back to ``"invalid_request"``
    - Other ``HTTPStatusError`` → ``"http_{status_code}"``
    - Other ``HTTPError`` → ``str(exc)``

    Args:
        exc: The caught httpx exception.
        httpx_module: The ``httpx`` module reference (avoids top-level import).
        parse_400: Optional callback that receives the JSON-decoded response
            body for a 400 status and returns ``(error_string, metadata_dict)``.
            If *None* or if JSON decoding fails, falls back to
            ``"invalid_request"``.
        result_cls: The result class to instantiate (default ``ProviderResult``).
            Pass a subclass like ``RCSDeliveryResult`` when needed.
    """
    if isinstance(exc, httpx_module.TimeoutException):
        return result_cls(success=False, error="timeout")

    if isinstance(exc, httpx_module.HTTPStatusError):
        status: int = exc.response.status_code

        if status == 401:
            return result_cls(success=False, error="auth_error")
        if status == 429:
            return result_cls(success=False, error="rate_limit")
        if status == 400:
            return _handle_400(exc, parse_400, result_cls)

        return result_cls(success=False, error=f"http_{status}")

    if isinstance(exc, httpx_module.HTTPError):
        return result_cls(success=False, error=str(exc))

    return result_cls(success=False, error=str(exc))


def _handle_400(
    exc: Any,
    parse_400: Callable[[dict[str, Any]], ErrorDetail] | None,
    result_cls: type[ProviderResult],
) -> ProviderResult:
    """Extract a structured error from a 400 response."""
    if parse_400 is not None:
        try:
            body = exc.response.json()
            error, metadata = parse_400(body)
            return result_cls(success=False, error=error, metadata=metadata)
        except Exception:  # nosec B110 — best-effort body parsing
            pass
    return result_cls(success=False, error="invalid_request")

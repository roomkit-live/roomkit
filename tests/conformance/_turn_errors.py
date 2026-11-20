"""Synthetic application failures raised by per-room verbs (RFC §10.6.8)."""

from __future__ import annotations

from typing import Any

from roomkit.core.exceptions import RemoteTurnError
from roomkit.providers.ai.base import ProviderError
from roomkit.turns._error_codec import error_record


class _LocalDetail:
    def __str__(self) -> str:
        return "detail from worker"


class DetailError(Exception):
    """Rendering requires a non-JSON field available only at the sender."""

    def __init__(self, detail: _LocalDetail) -> None:
        super().__init__()
        self.detail = detail

    def __str__(self) -> str:
        return str(self.detail)


class HostProviderError(ProviderError):
    """The application's own provider error."""


ERROR_CLASSES = {"DetailError": DetailError, "HostProviderError": HostProviderError}


def failure(kind: str) -> Exception:
    if kind == "DetailError":
        return DetailError(_LocalDetail())
    if kind == "HostProviderError":
        return HostProviderError("provider unavailable", provider="fixture", status_code=429)
    raise ValueError(f"unknown fixture failure {kind!r}")


def inspect_failure(error: Exception) -> dict[str, Any]:
    record = error_record(error)
    classes = (DetailError, HostProviderError, ProviderError)
    return {
        "kind": record.kind.split(":")[-1],
        "message": str(error),
        "generic": isinstance(error, RemoteTurnError),
        "reads_as": next((cls.__name__ for cls in classes if isinstance(error, cls)), None),
        "bases": [base.split(":")[-1] for base in record.bases],
        "attributes": record.attributes,
    }

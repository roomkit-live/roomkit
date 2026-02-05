"""Identity and identification models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.enums import IdentificationStatus
from roomkit.models.hook import InjectedEvent


class Identity(BaseModel):
    """A resolved user identity."""

    id: str
    organization_id: str | None = None
    display_name: str | None = None
    email: str | None = None
    phone: str | None = None
    channel_addresses: dict[str, list[str]] = Field(default_factory=dict)
    external_id: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IdentityResult(BaseModel):
    """Result of identity resolution."""

    status: IdentificationStatus
    identity: Identity | None = None
    candidates: list[Identity] = Field(default_factory=list)
    address: str | None = None
    channel_type: str | None = None
    challenge_type: str | None = None
    message: str | None = None


class IdentityHookResult(BaseModel):
    """Result from an identity resolution hook."""

    status: IdentificationStatus
    identity: Identity | None = None
    display_name: str | None = None
    candidates: list[Identity] | None = None
    inject: InjectedEvent | None = None
    reason: str | None = None
    challenge_type: str | None = None
    message: str | None = None

    @classmethod
    def resolved(cls, identity: Identity) -> IdentityHookResult:
        """Resolved - we know who this is."""
        return cls(status=IdentificationStatus.IDENTIFIED, identity=identity)

    @classmethod
    def pending(
        cls,
        display_name: str | None = None,
        candidates: list[Identity] | None = None,
    ) -> IdentityHookResult:
        """Pending - create participant with status=pending. Advisor resolves later."""
        return cls(
            status=IdentificationStatus.PENDING,
            display_name=display_name,
            candidates=candidates,
        )

    @classmethod
    def challenge(cls, inject: InjectedEvent, message: str | None = None) -> IdentityHookResult:
        """Challenge - hold the message, ask the sender to self-identify."""
        return cls(
            status=IdentificationStatus.CHALLENGE_SENT,
            inject=inject,
            message=message,
        )

    @classmethod
    def reject(cls, reason: str = "Unknown sender") -> IdentityHookResult:
        """Reject - do not create room or participant."""
        return cls(status=IdentificationStatus.REJECTED, reason=reason)

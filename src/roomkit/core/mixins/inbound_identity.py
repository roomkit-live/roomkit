"""InboundIdentityMixin — identity resolution pipeline (RFC §7)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import HookTrigger, IdentificationStatus
from roomkit.models.identity import Identity, IdentityResult

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine
    from roomkit.identity.base import IdentityResolver
    from roomkit.models.context import RoomContext
    from roomkit.models.delivery import InboundMessage
    from roomkit.models.enums import ChannelType
    from roomkit.models.hook import InjectedEvent

logger = logging.getLogger("roomkit.framework")


class _IdentityBlockedError(Exception):
    """Internal signal: identity hook blocked the inbound message."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@runtime_checkable
class IdentityHost(Protocol):
    """Contract: capabilities a host class must provide for InboundIdentityMixin.

    Attributes provided by the host's ``__init__``:
        _identity_resolver: Pluggable identity resolver (or ``None`` to skip).
        _identity_channel_types: Channel types subject to identity resolution
            (``None`` means all channel types).
        _identity_timeout: Maximum seconds for resolver.resolve() before timeout.
        _hook_engine: Hook engine for identity hooks.

    Methods provided by InboundLockedMixin (or equivalent):
        _deliver_injected_events: Deliver events injected by identity hooks
            (e.g., challenge messages).
    """

    _identity_resolver: IdentityResolver | None
    _identity_channel_types: set[ChannelType] | None
    _identity_timeout: float
    _hook_engine: HookEngine

    async def _deliver_injected_events(
        self, injected: list[InjectedEvent], room_id: str, context: RoomContext
    ) -> None: ...


class InboundIdentityMixin(HelpersMixin):
    """Identity resolution pipeline for inbound messages (RFC §7).

    Host contract: :class:`IdentityHost`.
    """

    _identity_resolver: IdentityResolver | None
    _identity_channel_types: set[ChannelType] | None
    _identity_timeout: float
    _hook_engine: HookEngine

    # Cross-mixin call — implemented by InboundLockedMixin. Cannot use a
    # method stub because InboundIdentityMixin precedes InboundLockedMixin
    # in the MRO, and a stub would shadow the real implementation at runtime.
    _deliver_injected_events: Any  # see IdentityHost for typed signature

    async def _resolve_identity(
        self,
        event: Any,
        message: InboundMessage,
        channel: Channel,
        room_id: str,
        context: RoomContext,
        telemetry: Any,
    ) -> tuple[Any, Identity | None, IdentityResult | None]:
        """Run identity resolution pipeline (RFC §7).

        Returns:
            A tuple of (event, resolved_identity, pending_id_result). The event
            may be modified with participant_id stamped on the source.
        """
        resolver = self._identity_resolver
        should_resolve = resolver is not None and (
            self._identity_channel_types is None
            or channel.channel_type in self._identity_channel_types
        )
        if not should_resolve or resolver is None:
            return event, None, None

        from roomkit.telemetry.base import SpanKind
        from roomkit.telemetry.context import get_current_span

        identity_span = telemetry.start_span(
            SpanKind.INBOUND_PIPELINE,
            "framework.identity_resolution",
            parent_id=get_current_span(),
            room_id=room_id,
            channel_id=message.channel_id,
        )
        _identity_error: str | None = None
        resolved_identity: Identity | None = None
        pending_id_result: IdentityResult | None = None
        try:
            try:
                id_result = await asyncio.wait_for(
                    resolver.resolve(message, context),
                    timeout=self._identity_timeout,
                )
            except TimeoutError:
                logger.warning(
                    "Identity resolution timed out after %.1fs",
                    self._identity_timeout,
                    extra={"room_id": room_id, "channel_id": message.channel_id},
                )
                await self._emit_framework_event(
                    "identity_timeout",
                    room_id=room_id,
                    channel_id=message.channel_id,
                    data={"timeout": self._identity_timeout},
                )
                id_result = IdentityResult(status=IdentificationStatus.UNKNOWN)
                _identity_error = "timeout"

            # Backfill address and channel_type from the message if not set by resolver
            # This ensures identity hooks always have access to sender info
            updates = {}
            if id_result.address is None:
                updates["address"] = message.sender_id
            if id_result.channel_type is None:
                updates["channel_type"] = str(channel.channel_type)
            if updates:
                id_result = id_result.model_copy(update=updates)

            event, resolved_identity, pending_id_result = await self._apply_identity_result(
                id_result, event, room_id, context
            )
        except Exception as exc:
            _identity_error = str(exc)
            raise
        finally:
            if _identity_error:
                telemetry.end_span(identity_span, status="error", error_message=_identity_error)
            else:
                telemetry.end_span(
                    identity_span,
                    attributes={"identity_status": str(id_result.status)},
                )

        return event, resolved_identity, pending_id_result

    async def _apply_identity_result(
        self,
        id_result: IdentityResult,
        event: Any,
        room_id: str,
        context: RoomContext,
    ) -> tuple[Any, Identity | None, IdentityResult | None]:
        """Apply identity resolution result: dispatch hooks and return outcome."""
        resolved_identity: Identity | None = None
        pending_id_result: IdentityResult | None = None

        if id_result.status == IdentificationStatus.IDENTIFIED and id_result.identity:
            # Known identity — stamp participant_id; persist later
            event = event.model_copy(
                update={
                    "source": event.source.model_copy(
                        update={"participant_id": id_result.identity.id}
                    )
                }
            )
            resolved_identity = id_result.identity

        elif id_result.status in (
            IdentificationStatus.AMBIGUOUS,
            IdentificationStatus.PENDING,
        ):
            event, resolved_identity, pending_id_result = await self._handle_ambiguous_identity(
                id_result, event, room_id, context
            )

        elif id_result.status in (
            IdentificationStatus.UNKNOWN,
            IdentificationStatus.REJECTED,
        ):
            event, resolved_identity = await self._handle_unknown_identity(
                id_result, event, room_id, context
            )

        return event, resolved_identity, pending_id_result

    async def _handle_ambiguous_identity(
        self,
        id_result: IdentityResult,
        event: Any,
        room_id: str,
        context: RoomContext,
    ) -> tuple[Any, Identity | None, IdentityResult | None]:
        """Handle ambiguous/pending identity — run hooks, return outcome."""
        hook_result = await self._run_identity_hooks(
            room_id, HookTrigger.ON_IDENTITY_AMBIGUOUS, event, context, id_result
        )
        # Also fire regular async hooks for observation/logging
        await self._hook_engine.run_async_hooks(
            room_id, HookTrigger.ON_IDENTITY_AMBIGUOUS, event, context
        )
        if (
            hook_result
            and hook_result.status == IdentificationStatus.IDENTIFIED
            and hook_result.identity
        ):
            event = event.model_copy(
                update={
                    "source": event.source.model_copy(
                        update={"participant_id": hook_result.identity.id}
                    )
                }
            )
            return event, hook_result.identity, None
        if hook_result and hook_result.status == IdentificationStatus.CHALLENGE_SENT:
            if hook_result.inject:
                await self._deliver_injected_events([hook_result.inject], room_id, context)
            raise _IdentityBlockedError("identity_challenge_sent")
        if hook_result and hook_result.status == IdentificationStatus.REJECTED:
            raise _IdentityBlockedError(hook_result.reason or "identity_rejected")
        # No hook resolved it — mark for pending creation
        return event, None, id_result

    async def _handle_unknown_identity(
        self,
        id_result: IdentityResult,
        event: Any,
        room_id: str,
        context: RoomContext,
    ) -> tuple[Any, Identity | None]:
        """Handle unknown/rejected identity — run hooks, return resolved identity or None."""
        hook_result = await self._run_identity_hooks(
            room_id, HookTrigger.ON_IDENTITY_UNKNOWN, event, context, id_result
        )
        # Also fire regular async hooks for observation/logging
        await self._hook_engine.run_async_hooks(
            room_id, HookTrigger.ON_IDENTITY_UNKNOWN, event, context
        )
        if hook_result and hook_result.status == IdentificationStatus.REJECTED:
            raise _IdentityBlockedError(hook_result.reason or "unknown_sender")
        if (
            hook_result
            and hook_result.status == IdentificationStatus.IDENTIFIED
            and hook_result.identity
        ):
            event = event.model_copy(
                update={
                    "source": event.source.model_copy(
                        update={"participant_id": hook_result.identity.id}
                    )
                }
            )
            return event, hook_result.identity
        return event, None

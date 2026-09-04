"""RoomKit exception hierarchy."""

from __future__ import annotations

from typing import Any

from roomkit.models.delivery import ProviderResult


class RoomKitError(Exception):
    """Base exception for all RoomKit errors."""


class ProviderDeliveryError(RoomKitError):
    """Raised when a provider returns an explicit unsuccessful result.

    Providers use :class:`~roomkit.models.delivery.ProviderResult` for both
    accepted and rejected sends. Turning a negative result into an exception
    at the channel boundary lets the existing retry and circuit-breaker path
    treat it exactly like a transport exception while retaining the structured
    provider response for the caller.
    """

    def __init__(self, result: ProviderResult) -> None:
        message = result.error or "provider_send_failed"
        super().__init__(message)
        self.provider_result = result
        self.code = str(result.metadata.get("code") or result.error or "ProviderDeliveryFailed")
        declared_retryable = result.metadata.get("retryable")
        self.retryable = declared_retryable if isinstance(declared_retryable, bool) else True


class RoomNotFoundError(RoomKitError):
    """Room does not exist."""


class RoomClosedError(RoomKitError):
    """Room's status refuses new events (RFC §5.1).

    Raised by the APIs whose return value has no place for a refusal: direct
    injection, which returns the committed ``RoomEvent`` — returning an event
    marked DELIVERED for a write that never happened would be worse than
    raising — and ``regenerate_target``, which returns the event a regenerate
    would replay. The inbound path and ``regenerate_response``, whose result
    type can say so, return ``InboundResult(blocked=True, reason="room_closed")``
    instead.
    """


class ChannelNotFoundError(RoomKitError):
    """Channel binding not found in room."""


class ChannelNotRegisteredError(RoomKitError):
    """Channel type not registered."""


class ChannelAlreadyRegisteredError(RoomKitError):
    """A channel with this ID is already registered.

    Silently replacing a live channel would leave existing room bindings
    routing to an object the framework no longer knows about. Call
    ``unregister_channel()`` first to swap an implementation deliberately.
    """


class ParticipantNotFoundError(RoomKitError):
    """Participant not found in room."""


class IdentityNotFoundError(RoomKitError):
    """Identity not found."""


class SourceAlreadyAttachedError(RoomKitError):
    """Source already attached to channel."""


class SourceNotFoundError(RoomKitError):
    """Source not found for channel."""


class VoiceNotConfiguredError(RoomKitError):
    """Raised when voice operation attempted without configured provider."""


class VoiceBackendNotConfiguredError(RoomKitError):
    """Raised when voice backend operation attempted without configured backend."""


class VoiceSessionEndedError(RoomKitError):
    """Raised when a voice session is moved out of ENDED (RFC §12.1).

    ENDED is terminal. A participant who reconnects gets a new session — the
    old one's audio paths, recorders and lanes have already been released.
    """


class RoomNotAttachedError(RoomKitError):
    """Raised when a channel acts on a room it is no longer attached to.

    Detaching leaves the conference running for the humans in it, so backend
    callbacks keep arriving. Acting on them would reconnect a bot nobody asked
    for.
    """


class ParticipantNotAdmittedError(RoomKitError):
    """Raised when a room's participant is barred from what was asked for them.

    Distinct from :class:`ParticipantNotFoundError`, which says the room has
    never heard of them. This one says it has, and the answer is still no —
    ``BANNED`` is "removed and blocked" (RFC section 5.5), and a caller told
    "not found" would reasonably create the participant and try again.
    """


class ConferenceCapabilityError(RoomKitError):
    """Raised when a conference operation needs a capability the backend lacks.

    Refusing at the boundary rather than degrading silently: a moderation UI
    that offers an unmute the SFU will reject, or a recording that never
    materialises, is worse than a configuration that fails immediately.
    """


class ConferenceAlreadyAttachedError(RoomKitError):
    """Raised when a second conference channel is attached to a room.

    A conference maps 1:1 to a Room (RFC section 12.10.1, principle 2), and
    the attachment is where that is enforceable: a second conference channel
    is a second bot session, a second transcription of every utterance and a
    second AI voice speaking the same deliveries — duplicates the roster, the
    transcript and the meeting have no way to express. Re-attaching the
    *same* conference channel is an ordinary attach and is not refused.
    """


class ConferenceCloseError(RoomKitError):
    """Raised when a conference channel did not close all of its resources.

    Raised at the very end of ``ConferenceChannel.close()``, after every step
    has run. It names sessions that could not be taken out, joins or lanes
    retained past their budget, and backend or provider shutdown calls that
    failed. Sessions remain on the channel's books, where ``info()`` reports
    them; resources still used by an abandoned task remain alive until that
    task settles. Raised rather than summarised into a log because a clean
    return would misreport potentially live conference media as released.
    ``RoomKit.close()`` collects it into its ``ExceptionGroup`` instead of
    letting it stop the other channels' closes.

    ``issues`` carries the structured report the message was rendered from —
    one entry per step that failed, timed out, was abandoned, or left a
    resource retained — so operator tooling can match on component and
    status without parsing prose.
    """

    def __init__(self, message: str, *, issues: tuple[Any, ...] = ()) -> None:
        super().__init__(message)
        self.issues = issues

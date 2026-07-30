"""RoomKit exception hierarchy."""

from __future__ import annotations


class RoomKitError(Exception):
    """Base exception for all RoomKit errors."""


class RoomNotFoundError(RoomKitError):
    """Room does not exist."""


class ChannelNotFoundError(RoomKitError):
    """Channel binding not found in room."""


class ChannelNotRegisteredError(RoomKitError):
    """Channel type not registered."""


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


class ConferenceCloseError(RoomKitError):
    """Raised when closing a conference channel left a bot in a meeting.

    Raised at the very end of ``ConferenceChannel.close()``, after every step
    has run: the sessions it names could not be taken out of their
    conferences — a ``leave()`` the SFU refused, a backend that outlived its
    budget — and they are still on the channel's books, where ``info()``
    reports them. Raised rather than summarised into a log, because a close
    that returns cleanly while a bot may still be listening to a meeting
    reports the one thing the roster exists to never misstate.
    ``RoomKit.close()`` collects it into its ``ExceptionGroup`` instead of
    letting it stop the other channels' closes.
    """

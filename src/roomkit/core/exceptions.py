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

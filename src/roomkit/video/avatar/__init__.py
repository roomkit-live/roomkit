"""Video avatar — lip-synced talking head generation from audio."""

from __future__ import annotations

from roomkit.video.avatar.base import AvatarProvider
from roomkit.video.avatar.mock import MockAvatarProvider

__all__ = [
    "AvatarProvider",
    "MockAvatarProvider",
]

"""Abstract base classes for routing and transcoding."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roomkit.models.channel import ChannelBinding
from roomkit.models.event import EventContent


class ContentTranscoder(ABC):
    """Transcodes event content between channel capabilities."""

    @abstractmethod
    async def transcode(
        self,
        content: EventContent,
        source_binding: ChannelBinding,
        target_binding: ChannelBinding,
    ) -> EventContent | None:
        """Transcode content for the target channel's capabilities.

        Return ``None`` to signal that the content cannot be represented.
        """

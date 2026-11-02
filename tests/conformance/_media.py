"""The kit's readers of an author's media (roomkit-specs
``conformance/README.md``, "Setup"): ``setup.kit.media_resolver`` answers
each reference it names with its bytes, written in base64, recording each
call under ``media_resolver``, and ``setup.kit.attached_media`` names as an
event's media the media contents its metadata lists under ``key`` (RFC §6.4).
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from roomkit.models.event import MediaContent, RoomEvent

if TYPE_CHECKING:
    from tests.conformance._harness import Harness

# The members of each reader a scenario names.
MEDIA_RESOLVER = {"media"}
ATTACHED_MEDIA = {"key"}


def media_options(kit: Mapping[str, Any], harness: Harness) -> dict[str, Any]:
    """The kit's media keywords a scenario's ``setup.kit`` names, the resolver
    recording its calls into *harness*."""
    options: dict[str, Any] = {}
    if "media_resolver" in kit:
        options["media_resolver"] = _resolver(kit["media_resolver"]["media"], harness)
    if "attached_media" in kit:
        options["attached_media"] = _reader(kit["attached_media"]["key"])
    return options


def _resolver(media: Mapping[str, str], harness: Harness) -> Any:
    held = {url: base64.b64decode(data) for url, data in media.items()}

    async def resolve(content: MediaContent, room_id: str) -> bytes | None:
        harness.record_media_resolver(content.url, room_id)
        return held.get(content.url)

    return resolve


def _reader(key: str) -> Any:
    def read(event: RoomEvent) -> list[MediaContent]:
        found: list[MediaContent] = []
        for record in event.metadata.get(key, []):
            try:
                found.append(MediaContent.model_validate(record))
            except ValidationError:
                continue
        return found

    return read

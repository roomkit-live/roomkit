"""What CompactingMemory counts an event for (RMK-589): what the provider bills,
the same estimate as every other memory, never an image's URL or base64."""

from __future__ import annotations

import base64

from roomkit.memory.compacting import CompactingMemory
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.event import MediaContent
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event


def _image(url: str) -> object:
    return make_event(body="x").model_copy(
        update={"content": MediaContent(url=url, mime_type="image/png")}
    )


async def _summarized(events: list[object]) -> bool:
    """Whether a 3000-token window made the memory summarize *events*."""
    provider = MockAIProvider(responses=["summary"])
    memory = CompactingMemory(SlidingWindowMemory(), provider, 3000, min_events=1)
    room = RoomContext(room=Room(id="r1"), recent_events=events)

    await memory.retrieve("r1", make_event(body="now"), room, channel_id="a")

    return bool(provider.calls)


async def test_images_by_url_fill_the_window_as_the_provider_bills_them() -> None:
    # Six images cost ~6000 tokens, not the ~30 their URLs read as.
    images = [_image(f"https://example.com/{n}.png") for n in range(6)]

    assert await _summarized([*images, make_event(body="what do you see?")])


async def test_an_inline_image_does_not_count_its_base64() -> None:
    # 150 kB of base64 is one image (~1000 tokens), not 50 000 tokens of text.
    data_url = "data:image/png;base64," + base64.b64encode(b"\x00" * 150_000).decode()

    assert not await _summarized([_image(data_url), make_event(body="what is this?")])

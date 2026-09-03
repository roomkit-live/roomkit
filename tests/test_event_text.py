"""One event-text extraction, two fallbacks.

The transports read an event's text through
``roomkit.providers.utils.extract_event_text`` and send nothing when there is
none; the memory layer reads it through
``roomkit.memory.token_estimator.extract_event_text``, which builds on the same
extraction and renders text-less content instead, since it still costs tokens
and still belongs in a summary.
"""

from __future__ import annotations

from typing import Any

from roomkit.memory.token_estimator import extract_event_text as memory_text
from roomkit.models.enums import ChannelType
from roomkit.models.event import (
    EventSource,
    MediaContent,
    RichContent,
    RoomEvent,
    TemplateContent,
    TextContent,
)
from roomkit.providers.utils import extract_event_text as transport_text


def _event(content: Any) -> RoomEvent:
    return RoomEvent(
        room_id="room-1",
        source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
        content=content,
    )


class TestTransportText:
    def test_text_body(self) -> None:
        assert transport_text(_event(TextContent(body="hello"))) == "hello"

    def test_rich_body(self) -> None:
        assert transport_text(_event(RichContent(body="**hi**"))) == "**hi**"

    def test_template_without_body_is_empty_not_none(self) -> None:
        assert transport_text(_event(TemplateContent(template_id="welcome"))) == ""

    def test_media_is_empty(self) -> None:
        content = MediaContent(url="https://x.example/a.png", mime_type="image/png")
        assert transport_text(_event(content)) == ""


class TestMemoryText:
    def test_text_body_even_when_empty(self) -> None:
        assert memory_text(_event(TextContent(body=""))) == ""

    def test_rich_body_is_the_body_not_the_repr(self) -> None:
        assert memory_text(_event(RichContent(body="**hi**"))) == "**hi**"

    def test_media_is_rendered(self) -> None:
        content = MediaContent(url="https://x.example/a.png", mime_type="image/png")
        text = memory_text(_event(content))
        assert text == str(content)
        assert "https://x.example/a.png" in text

    def test_template_without_body_is_rendered(self) -> None:
        content = TemplateContent(template_id="welcome")
        assert memory_text(_event(content)) == str(content)

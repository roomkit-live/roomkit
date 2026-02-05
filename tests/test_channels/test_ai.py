"""Tests for AI channel."""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelMediaType, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIImagePart, AITextPart, AITool
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event, make_media_event


class TestAIChannel:
    async def test_on_event_generates_response(self) -> None:
        provider = MockAIProvider(responses=["AI says hello"])
        ch = AIChannel("ai1", provider=provider, system_prompt="Be helpful")
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="hello", channel_id="sms1")
        output = await ch.on_event(event, binding, ctx)
        assert output.responded is True
        assert len(output.response_events) == 1
        resp = output.response_events[0]
        assert isinstance(resp.content, TextContent)
        assert resp.content.body == "AI says hello"
        assert resp.chain_depth == 1

    async def test_chain_depth_increments(self) -> None:
        provider = MockAIProvider(responses=["reply"])
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="hi", chain_depth=3)
        output = await ch.on_event(event, binding, ctx)
        assert output.response_events[0].chain_depth == 4

    async def test_provider_records_calls(self) -> None:
        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider, system_prompt="test")
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="question")
        await ch.on_event(event, binding, ctx)
        assert len(provider.calls) == 1
        assert provider.calls[0].system_prompt == "test"

    async def test_deliver_is_noop(self) -> None:
        """Intelligence channels are not called via deliver by the router."""
        provider = MockAIProvider(responses=["should not reach"])
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="hello", channel_id="sms1")
        output = await ch.deliver(event, binding, ctx)
        assert output.responded is False
        assert len(provider.calls) == 0

    async def test_skips_own_events(self) -> None:
        """AI channel ignores events from itself to prevent self-loops."""
        provider = MockAIProvider(responses=["should not reach"])
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="my own response", channel_id="ai1")
        output = await ch.on_event(event, binding, ctx)
        assert output.responded is False
        assert len(provider.calls) == 0

    async def test_capabilities_with_vision_provider(self) -> None:
        """Vision provider includes MEDIA in capabilities."""
        provider = MockAIProvider(vision=True)
        ch = AIChannel("ai1", provider=provider)
        caps = ch.capabilities()
        assert ChannelMediaType.MEDIA in caps.media_types
        assert caps.supports_media is True

    async def test_capabilities_without_vision(self) -> None:
        """Non-vision provider excludes MEDIA."""
        provider = MockAIProvider(vision=False)
        ch = AIChannel("ai1", provider=provider)
        caps = ch.capabilities()
        assert ChannelMediaType.MEDIA not in caps.media_types
        assert caps.supports_media is False

    async def test_on_event_with_image(self) -> None:
        """Vision provider receives image in context."""
        provider = MockAIProvider(responses=["I see a cat"], vision=True)
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_media_event(
            url="https://example.com/cat.jpg",
            mime_type="image/jpeg",
            caption="Check this out!",
            channel_id="sms1",
        )
        output = await ch.on_event(event, binding, ctx)

        assert output.responded is True
        assert len(provider.calls) == 1

        # Verify multimodal content was passed
        call_ctx = provider.calls[0]
        assert len(call_ctx.messages) == 1
        msg = call_ctx.messages[0]
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], AITextPart)
        assert msg.content[0].text == "Check this out!"
        assert isinstance(msg.content[1], AIImagePart)
        assert msg.content[1].url == "https://example.com/cat.jpg"
        assert msg.content[1].mime_type == "image/jpeg"

    async def test_on_event_with_image_no_caption(self) -> None:
        """Vision provider receives image without caption."""
        provider = MockAIProvider(responses=["I see an image"], vision=True)
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_media_event(
            url="https://example.com/photo.png",
            mime_type="image/png",
            channel_id="sms1",
        )
        output = await ch.on_event(event, binding, ctx)

        assert output.responded is True
        call_ctx = provider.calls[0]
        msg = call_ctx.messages[0]
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1  # No text part since no caption
        assert isinstance(msg.content[0], AIImagePart)
        assert msg.content[0].url == "https://example.com/photo.png"

    async def test_non_vision_provider_gets_text_only(self) -> None:
        """Non-vision provider gets empty string for media content."""
        provider = MockAIProvider(responses=["ok"], vision=False)
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_media_event(
            url="https://example.com/image.jpg",
            mime_type="image/jpeg",
            channel_id="sms1",
        )
        output = await ch.on_event(event, binding, ctx)

        # Should still respond but with no content extracted
        assert output.responded is True
        call_ctx = provider.calls[0]
        # No messages since MediaContent has no text to extract
        assert len(call_ctx.messages) == 0

    async def test_composite_content_with_vision(self) -> None:
        """Vision provider receives composite content with text and images."""
        provider = MockAIProvider(responses=["Got multiple images"], vision=True)
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_media_event(
            url="https://example.com/img1.jpg",
            mime_type="image/jpeg",
            extra_urls=["https://example.com/img2.jpg"],
            body="Here are two photos",
            channel_id="sms1",
        )
        output = await ch.on_event(event, binding, ctx)

        assert output.responded is True
        call_ctx = provider.calls[0]
        msg = call_ctx.messages[0]
        assert isinstance(msg.content, list)
        # Text + 2 images
        assert len(msg.content) == 3
        assert isinstance(msg.content[0], AITextPart)
        assert msg.content[0].text == "Here are two photos"
        assert isinstance(msg.content[1], AIImagePart)
        assert msg.content[1].url == "https://example.com/img1.jpg"
        assert isinstance(msg.content[2], AIImagePart)
        assert msg.content[2].url == "https://example.com/img2.jpg"


class TestPerRoomConfiguration:
    """Tests for per-room AI configuration via binding metadata."""

    async def test_binding_metadata_overrides_system_prompt(self) -> None:
        """System prompt from binding metadata overrides channel default."""
        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider, system_prompt="Default prompt")
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"system_prompt": "Custom for this room"},
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="hello", channel_id="sms1")
        await ch.on_event(event, binding, ctx)

        assert len(provider.calls) == 1
        assert provider.calls[0].system_prompt == "Custom for this room"

    async def test_binding_metadata_overrides_temperature(self) -> None:
        """Temperature from binding metadata overrides channel default."""
        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider, temperature=0.7)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"temperature": 0.3},
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="hello", channel_id="sms1")
        await ch.on_event(event, binding, ctx)

        assert len(provider.calls) == 1
        assert provider.calls[0].temperature == 0.3

    async def test_binding_metadata_overrides_max_tokens(self) -> None:
        """Max tokens from binding metadata overrides channel default."""
        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider, max_tokens=1024)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"max_tokens": 2048},
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="hello", channel_id="sms1")
        await ch.on_event(event, binding, ctx)

        assert len(provider.calls) == 1
        assert provider.calls[0].max_tokens == 2048

    async def test_tools_passed_to_provider(self) -> None:
        """Tools from binding metadata are passed to provider."""
        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={
                "tools": [
                    {
                        "name": "search",
                        "description": "Search for information",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ]
            },
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="search for cats", channel_id="sms1")
        await ch.on_event(event, binding, ctx)

        assert len(provider.calls) == 1
        assert len(provider.calls[0].tools) == 1
        tool = provider.calls[0].tools[0]
        assert isinstance(tool, AITool)
        assert tool.name == "search"
        assert tool.description == "Search for information"
        assert tool.parameters["type"] == "object"

    async def test_multiple_tools_passed_to_provider(self) -> None:
        """Multiple tools from binding metadata are passed to provider."""
        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={
                "tools": [
                    {"name": "search", "description": "Search"},
                    {"name": "write_note", "description": "Write a note"},
                ]
            },
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="hello", channel_id="sms1")
        await ch.on_event(event, binding, ctx)

        assert len(provider.calls) == 1
        assert len(provider.calls[0].tools) == 2
        assert provider.calls[0].tools[0].name == "search"
        assert provider.calls[0].tools[1].name == "write_note"

    async def test_empty_tools_list_when_not_specified(self) -> None:
        """Tools list is empty when not specified in binding metadata."""
        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel("ai1", provider=provider)
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="hello", channel_id="sms1")
        await ch.on_event(event, binding, ctx)

        assert len(provider.calls) == 1
        assert provider.calls[0].tools == []

    async def test_channel_defaults_used_when_no_metadata(self) -> None:
        """Channel defaults are used when binding has no metadata overrides."""
        provider = MockAIProvider(responses=["ok"])
        ch = AIChannel(
            "ai1",
            provider=provider,
            system_prompt="Default",
            temperature=0.5,
            max_tokens=512,
        )
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(body="hello", channel_id="sms1")
        await ch.on_event(event, binding, ctx)

        assert len(provider.calls) == 1
        assert provider.calls[0].system_prompt == "Default"
        assert provider.calls[0].temperature == 0.5
        assert provider.calls[0].max_tokens == 512

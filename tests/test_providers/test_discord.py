"""Tests for the Discord bot provider and gateway source."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels import DiscordChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import (
    ChannelData,
    EventSource,
    MediaContent,
    RichContent,
    RoomEvent,
    TextContent,
)
from roomkit.providers.discord import (
    DiscordBotProvider,
    DiscordConfig,
    MockDiscordProvider,
)
from roomkit.sources.base import SourceStatus
from roomkit.sources.discord import default_message_parser, parse_discord_message

# =============================================================================
# Helpers
# =============================================================================


def _make_message(
    *,
    message_id: int = 123,
    content: str = "hello world",
    author_id: int = 42,
    author_bot: bool = False,
    attachments: list[Any] | None = None,
    reference_id: int | None = None,
) -> SimpleNamespace:
    """Build a duck-typed stand-in for a ``discord.Message``."""
    author = SimpleNamespace(id=author_id, name="alice", display_name="Alice", bot=author_bot)
    reference = SimpleNamespace(message_id=reference_id) if reference_id else None
    return SimpleNamespace(
        id=message_id,
        content=content,
        author=author,
        channel=SimpleNamespace(id=999, name="general"),
        guild=SimpleNamespace(id=777),
        attachments=attachments or [],
        reference=reference,
    )


def _make_attachment(
    *, url: str = "https://cdn.discord.com/a.png", content_type: str = "image/png"
) -> SimpleNamespace:
    return SimpleNamespace(url=url, content_type=content_type, filename="a.png", size=1024)


def _make_channel_mock(channel_id: int = 999, sent_id: int = 555) -> MagicMock:
    channel = MagicMock()
    channel.id = channel_id
    channel.send = AsyncMock(return_value=SimpleNamespace(id=sent_id))
    reacted = MagicMock()
    reacted.add_reaction = AsyncMock()
    channel.fetch_message = AsyncMock(return_value=reacted)
    return channel


def _make_source(*, connected: bool = True, channel: MagicMock | None = None) -> MagicMock:
    """Build a fake DiscordGatewaySource exposing a mock discord client."""
    ch = channel or _make_channel_mock()
    client = MagicMock()
    client.get_channel = MagicMock(return_value=ch)
    client.fetch_channel = AsyncMock(return_value=ch)
    source = MagicMock()
    source.status = SourceStatus.CONNECTED if connected else SourceStatus.STOPPED
    source.client = client
    return source


def _make_event(content: Any, **channel_data: Any) -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="discord-main", channel_type=ChannelType.DISCORD),
        content=content,
        channel_data=ChannelData(**channel_data) if channel_data else ChannelData(),
    )


# =============================================================================
# Config
# =============================================================================


class TestDiscordConfig:
    def test_requires_token(self) -> None:
        with pytest.raises(ValueError):
            DiscordConfig()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        cfg = DiscordConfig(bot_token="secret")
        assert cfg.bot_token.get_secret_value() == "secret"
        assert cfg.intents_message_content is True
        assert cfg.ignore_bots is True


# =============================================================================
# Inbound parsing
# =============================================================================


class TestParseDiscordMessage:
    def test_text_message(self) -> None:
        msg = parse_discord_message(_make_message(), "discord-main", bot_user_id=1)
        assert msg is not None
        assert msg.sender_id == "42"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "hello world"
        assert msg.external_id == "123"
        assert msg.idempotency_key == "123"

    def test_metadata(self) -> None:
        msg = parse_discord_message(_make_message(), "discord-main")
        assert msg is not None
        assert msg.metadata["guild_id"] == "777"
        assert msg.metadata["channel_id"] == "999"
        assert msg.metadata["channel_name"] == "general"
        assert msg.metadata["author_name"] == "Alice"
        assert msg.metadata["author_bot"] is False

    def test_attachment_becomes_media(self) -> None:
        msg = parse_discord_message(
            _make_message(content="look", attachments=[_make_attachment()]),
            "discord-main",
        )
        assert msg is not None
        assert isinstance(msg.content, MediaContent)
        assert msg.content.url == "https://cdn.discord.com/a.png"
        assert msg.content.mime_type == "image/png"
        assert msg.content.caption == "look"

    def test_multiple_attachments_recorded(self) -> None:
        atts = [_make_attachment(), _make_attachment(url="https://cdn.discord.com/b.png")]
        msg = parse_discord_message(_make_message(content="", attachments=atts), "discord-main")
        assert msg is not None
        assert msg.metadata["attachment_urls"] == [
            "https://cdn.discord.com/a.png",
            "https://cdn.discord.com/b.png",
        ]

    def test_skips_own_messages(self) -> None:
        msg = _make_message(author_id=99)
        assert parse_discord_message(msg, "discord-main", bot_user_id=99) is None

    def test_skips_other_bots_by_default(self) -> None:
        msg = _make_message(author_bot=True)
        assert parse_discord_message(msg, "discord-main", ignore_bots=True) is None

    def test_keeps_bots_when_allowed(self) -> None:
        msg = _make_message(author_bot=True)
        assert parse_discord_message(msg, "discord-main", ignore_bots=False) is not None

    def test_empty_message_is_skipped(self) -> None:
        msg = _make_message(content="", attachments=[])
        assert parse_discord_message(msg, "discord-main") is None

    def test_reply_maps_to_thread_id(self) -> None:
        msg = parse_discord_message(_make_message(reference_id=888), "discord-main")
        assert msg is not None
        assert msg.thread_id == "888"

    def test_default_parser_binds_channel(self) -> None:
        parser = default_message_parser("discord-main", ignore_bots=True)
        msg = parser(_make_message(), 1)
        assert msg is not None
        assert msg.channel_id == "discord-main"


# =============================================================================
# Mock provider
# =============================================================================


class TestMockDiscordProvider:
    async def test_send_records(self) -> None:
        provider = MockDiscordProvider()
        result = await provider.send(_make_event(TextContent(body="hi")), to="999")
        assert result.success
        assert provider.sent[0]["to"] == "999"

    async def test_reaction_records(self) -> None:
        provider = MockDiscordProvider()
        await provider.send_reaction("999", "123", "👍")
        assert provider.reactions == [{"channel_id": "999", "message_id": "123", "emoji": "👍"}]


# =============================================================================
# Bot provider (outbound via shared client)
# =============================================================================


class TestDiscordBotProvider:
    async def test_send_text(self) -> None:
        source = _make_source()
        provider = DiscordBotProvider(source)
        result = await provider.send(_make_event(TextContent(body="hi there")), to="999")
        assert result.success
        assert result.provider_message_id == "555"
        source.client.get_channel.assert_called_once_with(999)
        _, kwargs = source.client.get_channel.return_value.send.call_args
        assert kwargs["content"] == "hi there"

    async def test_send_rich_becomes_embed(self) -> None:
        source = _make_source()
        provider = DiscordBotProvider(source)
        await provider.send(_make_event(RichContent(body="**bold**")), to="999")
        _, kwargs = source.client.get_channel.return_value.send.call_args
        assert kwargs["embed"].description == "**bold**"

    async def test_send_media_url(self) -> None:
        source = _make_source()
        provider = DiscordBotProvider(source)
        content = MediaContent(
            url="https://cdn.discord.com/a.png", mime_type="image/png", caption="pic"
        )
        await provider.send(_make_event(content), to="999")
        _, kwargs = source.client.get_channel.return_value.send.call_args
        assert "https://cdn.discord.com/a.png" in kwargs["content"]

    async def test_reply_sets_reference(self) -> None:
        source = _make_source()
        provider = DiscordBotProvider(source)
        await provider.send(_make_event(TextContent(body="re"), thread_id="888"), to="999")
        _, kwargs = source.client.get_channel.return_value.send.call_args
        assert kwargs["reference"].message_id == 888

    async def test_not_connected_guard(self) -> None:
        source = _make_source(connected=False)
        provider = DiscordBotProvider(source)
        result = await provider.send(_make_event(TextContent(body="hi")), to="999")
        assert not result.success
        assert result.error == "discord_not_connected"

    async def test_channel_not_found(self) -> None:
        source = _make_source()
        source.client.get_channel = MagicMock(return_value=None)
        source.client.fetch_channel = AsyncMock(return_value=None)
        provider = DiscordBotProvider(source)
        result = await provider.send(_make_event(TextContent(body="hi")), to="999")
        assert not result.success
        assert result.error == "channel_not_found"

    async def test_send_reaction(self) -> None:
        source = _make_source()
        provider = DiscordBotProvider(source)
        await provider.send_reaction("999", "123", "🔥")
        channel = source.client.get_channel.return_value
        channel.fetch_message.assert_awaited_once_with(123)
        channel.fetch_message.return_value.add_reaction.assert_awaited_once_with("🔥")


# =============================================================================
# Inbound pipeline integration
# =============================================================================


class TestDiscordInboundIntegration:
    async def test_parsed_message_flows_through_pipeline(self) -> None:
        kit = RoomKit()
        kit.register_channel(DiscordChannel("discord-main", provider=MockDiscordProvider()))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "discord-main")

        inbound: InboundMessage = parse_discord_message(
            _make_message(content="ping"), "discord-main", bot_user_id=1
        )  # type: ignore[assignment]
        result = await kit.process_inbound(inbound)

        assert not result.blocked
        assert result.event is not None
        assert result.event.content.body == "ping"
        assert result.event.source.channel_type == ChannelType.DISCORD

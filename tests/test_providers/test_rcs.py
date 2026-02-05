"""Tests for RCS providers."""

from __future__ import annotations

import pytest

from roomkit import RCSChannel
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.providers.rcs.mock import MockRCSProvider
from tests.conftest import make_event, make_media_event


class TestMockRCSProvider:
    @pytest.mark.asyncio
    async def test_send_rcs_success(self) -> None:
        """Send RCS message successfully."""
        provider = MockRCSProvider()
        event = make_event(body="Hello RCS!")

        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        assert result.channel_used == "rcs"
        assert result.fallback is False
        assert result.provider_message_id is not None

    @pytest.mark.asyncio
    async def test_send_rcs_with_fallback(self) -> None:
        """Send RCS with SMS fallback when user doesn't support RCS."""
        provider = MockRCSProvider(simulate_fallback=True)
        event = make_event(body="Hello!")

        result = await provider.send(event, to="+15145559999", fallback=True)

        assert result.success is True
        assert result.channel_used == "sms"
        assert result.fallback is True

    @pytest.mark.asyncio
    async def test_send_rcs_no_fallback_when_disabled(self) -> None:
        """When fallback=False and user doesn't support RCS, still attempts RCS."""
        provider = MockRCSProvider(simulate_fallback=True)
        event = make_event(body="Hello!")

        # Even with simulate_fallback=True, if fallback=False is passed,
        # the mock respects it and returns RCS (simulating no fallback attempted)
        result = await provider.send(event, to="+15145559999", fallback=False)

        # Mock doesn't simulate failure, just fallback when both are True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_rcs_failure(self) -> None:
        """Simulate RCS send failure."""
        provider = MockRCSProvider(simulate_failure=True)
        event = make_event(body="Hello!")

        result = await provider.send(event, to="+15145559999")

        assert result.success is False
        assert result.error == "simulated_failure"

    @pytest.mark.asyncio
    async def test_send_rcs_with_media(self) -> None:
        """Send RCS message with media attachment."""
        provider = MockRCSProvider()
        event = make_media_event(
            url="https://example.com/image.jpg",
            mime_type="image/jpeg",
            caption="Check this out!",
        )

        result = await provider.send(event, to="+15145559999")

        assert result.success is True
        assert result.channel_used == "rcs"
        assert len(provider.calls) == 1
        assert provider.calls[0]["to"] == "+15145559999"

    @pytest.mark.asyncio
    async def test_check_capability(self) -> None:
        """Check RCS capability for a phone number."""
        provider = MockRCSProvider()
        assert await provider.check_capability("+15145559999") is True

        provider_no_rcs = MockRCSProvider(simulate_fallback=True)
        assert await provider_no_rcs.check_capability("+15145559999") is False

    def test_sender_id(self) -> None:
        """Provider has correct sender ID."""
        provider = MockRCSProvider(sender_id="my_brand_agent")
        assert provider.sender_id == "my_brand_agent"


class TestRCSChannel:
    def test_channel_capabilities(self) -> None:
        """RCS channel has correct capabilities."""
        provider = MockRCSProvider()
        channel = RCSChannel("rcs-main", provider=provider)

        caps = channel.capabilities()
        assert ChannelMediaType.TEXT in caps.media_types
        assert ChannelMediaType.RICH in caps.media_types
        assert ChannelMediaType.MEDIA in caps.media_types
        assert caps.supports_read_receipts is True
        assert caps.supports_buttons is True
        assert caps.supports_quick_replies is True
        assert caps.supports_media is True

    def test_channel_type(self) -> None:
        """RCS channel has correct type."""
        provider = MockRCSProvider()
        channel = RCSChannel("rcs-main", provider=provider)

        assert channel.channel_type == ChannelType.RCS

    def test_channel_with_fallback_disabled(self) -> None:
        """RCS channel can be created with fallback disabled."""
        provider = MockRCSProvider()
        channel = RCSChannel("rcs-main", provider=provider, fallback=False)

        # The fallback setting is stored in defaults
        assert channel._defaults.get("fallback") is False

    def test_channel_with_fallback_enabled(self) -> None:
        """RCS channel has fallback enabled by default."""
        provider = MockRCSProvider()
        channel = RCSChannel("rcs-main", provider=provider)

        assert channel._defaults.get("fallback") is True

"""Tests for the Microsoft Teams provider."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from roomkit.models.event import TextContent
from roomkit.providers.teams import (
    InMemoryConversationReferenceStore,
    MockTeamsProvider,
    TeamsConfig,
    is_bot_added,
    parse_teams_activity,
    parse_teams_reactions,
    parse_teams_webhook,
)
from tests.conftest import make_event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _personal_message_payload(
    text: str = "Hello from Teams",
    sender_id: str = "user-aad-1",
    sender_name: str = "Alice",
    bot_id: str = "bot-aad-1",
    conversation_id: str = "conv-1",
    activity_id: str = "act-1",
    service_url: str = "https://smba.trafficmanager.net/teams/",
    tenant_id: str = "tenant-abc",
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "message",
        "id": activity_id,
        "text": text,
        "from": {"id": sender_id, "name": sender_name},
        "recipient": {"id": bot_id, "name": "TestBot"},
        "conversation": {"id": conversation_id, "conversationType": "personal"},
        "serviceUrl": service_url,
        "channelData": {"tenant": {"id": tenant_id}},
    }
    payload.update(overrides)
    return payload


def _group_message_payload(
    text: str = "<at>TestBot</at> Hello from group",
    sender_id: str = "user-aad-2",
    sender_name: str = "Bob",
    bot_id: str = "bot-aad-1",
    conversation_id: str = "group-conv-1",
    activity_id: str = "act-2",
    service_url: str = "https://smba.trafficmanager.net/teams/",
    tenant_id: str = "tenant-abc",
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "message",
        "id": activity_id,
        "text": text,
        "from": {"id": sender_id, "name": sender_name},
        "recipient": {"id": bot_id, "name": "TestBot"},
        "conversation": {
            "id": conversation_id,
            "conversationType": "groupChat",
            "isGroup": True,
        },
        "serviceUrl": service_url,
        "channelData": {"tenant": {"id": tenant_id}},
        "entities": [
            {
                "type": "mention",
                "mentioned": {"id": bot_id, "name": "TestBot"},
                "text": "<at>TestBot</at>",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _conversation_update_payload(
    members_added: list[dict[str, str]] | None = None,
    bot_id: str = "bot-aad-1",
    conversation_id: str = "conv-1",
) -> dict[str, Any]:
    return {
        "type": "conversationUpdate",
        "id": "update-1",
        "conversation": {"id": conversation_id, "conversationType": "personal"},
        "recipient": {"id": bot_id, "name": "TestBot"},
        "from": {"id": "user-aad-1", "name": "Alice"},
        "membersAdded": members_added or [],
        "serviceUrl": "https://smba.trafficmanager.net/teams/",
        "channelData": {"tenant": {"id": "tenant-abc"}},
    }


# ---------------------------------------------------------------------------
# TeamsConfig
# ---------------------------------------------------------------------------


class TestTeamsConfig:
    def test_defaults(self) -> None:
        cfg = TeamsConfig(app_id="my-app", app_password="s3cret")
        assert cfg.tenant_id == "common"
        assert cfg.app_id == "my-app"

    def test_app_password_is_secret(self) -> None:
        cfg = TeamsConfig(app_id="my-app", app_password="s3cret")
        assert isinstance(cfg.app_password, SecretStr)
        assert cfg.app_password.get_secret_value() == "s3cret"
        # Ensure __repr__/str does not leak the secret
        assert "s3cret" not in repr(cfg)

    def test_custom_tenant_id(self) -> None:
        cfg = TeamsConfig(app_id="my-app", app_password="pw", tenant_id="my-tenant")
        assert cfg.tenant_id == "my-tenant"


# ---------------------------------------------------------------------------
# TeamsConfig — Certificate Auth
# ---------------------------------------------------------------------------


class TestTeamsConfigCertificateAuth:
    """Tests for certificate-based authentication config."""

    def test_valid_cert_config(self) -> None:
        cfg = TeamsConfig(
            app_id="my-app",
            certificate_thumbprint="AABBCCDD",
            certificate_private_key="-----BEGIN RSA PRIVATE KEY-----\nfake",
        )
        assert cfg.uses_certificate_auth is True
        assert cfg.certificate_thumbprint == "AABBCCDD"

    def test_cert_config_with_public_cert(self) -> None:
        cfg = TeamsConfig(
            app_id="my-app",
            certificate_thumbprint="AABBCCDD",
            certificate_private_key="-----BEGIN RSA PRIVATE KEY-----\nfake",
            certificate_public="-----BEGIN CERTIFICATE-----\npublic",
        )
        assert cfg.certificate_public == "-----BEGIN CERTIFICATE-----\npublic"

    def test_cert_config_with_custom_tenant(self) -> None:
        cfg = TeamsConfig(
            app_id="my-app",
            certificate_thumbprint="AABBCCDD",
            certificate_private_key="key-data",
            tenant_id="my-tenant",
        )
        assert cfg.tenant_id == "my-tenant"
        assert cfg.uses_certificate_auth is True

    def test_private_key_is_secret(self) -> None:
        cfg = TeamsConfig(
            app_id="my-app",
            certificate_thumbprint="AABBCCDD",
            certificate_private_key="super-secret-key",
        )
        assert isinstance(cfg.certificate_private_key, SecretStr)
        assert cfg.certificate_private_key.get_secret_value() == "super-secret-key"
        assert "super-secret-key" not in repr(cfg)

    def test_password_auth_backward_compat(self) -> None:
        cfg = TeamsConfig(app_id="my-app", app_password="s3cret")
        assert cfg.uses_certificate_auth is False
        assert cfg.app_password is not None
        assert cfg.app_password.get_secret_value() == "s3cret"

    def test_both_password_and_cert_raises(self) -> None:
        with pytest.raises(ValidationError, match="Cannot specify both"):
            TeamsConfig(
                app_id="my-app",
                app_password="pw",
                certificate_thumbprint="AABBCCDD",
                certificate_private_key="key-data",
            )

    def test_neither_password_nor_cert_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be provided"):
            TeamsConfig(app_id="my-app")

    def test_thumbprint_without_key_raises(self) -> None:
        with pytest.raises(ValidationError, match="certificate_private_key"):
            TeamsConfig(app_id="my-app", certificate_thumbprint="AABBCCDD")

    def test_key_without_thumbprint_raises(self) -> None:
        with pytest.raises(ValidationError, match="certificate_thumbprint"):
            TeamsConfig(app_id="my-app", certificate_private_key="key-data")

    def test_certificate_public_without_cert_auth_raises(self) -> None:
        with pytest.raises(ValidationError, match="certificate_public is only valid"):
            TeamsConfig(
                app_id="my-app",
                app_password="pw",
                certificate_public="-----BEGIN CERTIFICATE-----\npublic",
            )


# ---------------------------------------------------------------------------
# parse_teams_activity
# ---------------------------------------------------------------------------


class TestParseTeamsActivity:
    def test_parse_message_activity(self) -> None:
        payload = _personal_message_payload()
        result = parse_teams_activity(payload)

        assert result["activity_type"] == "message"
        assert result["conversation_id"] == "conv-1"
        assert result["conversation_type"] == "personal"
        assert result["is_group"] is False
        assert result["service_url"] == "https://smba.trafficmanager.net/teams/"
        assert result["tenant_id"] == "tenant-abc"
        assert result["sender_id"] == "user-aad-1"
        assert result["sender_name"] == "Alice"
        assert result["bot_id"] == "bot-aad-1"
        assert result["reply_to_id"] == ""
        assert result["members_added"] == []
        assert result["members_removed"] == []

    def test_parse_activity_with_reply_to_id(self) -> None:
        payload = _personal_message_payload(replyToId="parent-act-99")
        result = parse_teams_activity(payload)
        assert result["reply_to_id"] == "parent-act-99"

    def test_parse_conversation_update(self) -> None:
        payload = _conversation_update_payload(
            members_added=[{"id": "bot-aad-1"}, {"id": "user-aad-1"}],
        )
        result = parse_teams_activity(payload)

        assert result["activity_type"] == "conversationUpdate"
        assert result["members_added"] == ["bot-aad-1", "user-aad-1"]

    def test_parse_group_chat(self) -> None:
        payload = _group_message_payload()
        result = parse_teams_activity(payload)

        assert result["is_group"] is True
        assert result["conversation_type"] == "groupChat"

    def test_parse_channel_type_is_group(self) -> None:
        """conversationType='channel' should also be detected as group."""
        payload = _personal_message_payload()
        payload["conversation"]["conversationType"] = "channel"
        result = parse_teams_activity(payload)

        assert result["is_group"] is True

    def test_parse_empty_payload(self) -> None:
        result = parse_teams_activity({})

        assert result["activity_type"] == ""
        assert result["conversation_id"] == ""
        assert result["conversation_type"] == "personal"
        assert result["is_group"] is False
        assert result["service_url"] == ""
        assert result["tenant_id"] == ""
        assert result["sender_id"] == ""
        assert result["sender_name"] == ""
        assert result["bot_id"] == ""
        assert result["members_added"] == []
        assert result["members_removed"] == []


# ---------------------------------------------------------------------------
# is_bot_added
# ---------------------------------------------------------------------------


class TestIsBotAdded:
    def test_bot_added(self) -> None:
        payload = _conversation_update_payload(
            members_added=[{"id": "bot-aad-1"}],
        )
        assert is_bot_added(payload) is True

    def test_bot_added_among_multiple(self) -> None:
        payload = _conversation_update_payload(
            members_added=[{"id": "user-aad-2"}, {"id": "bot-aad-1"}],
        )
        assert is_bot_added(payload) is True

    def test_bot_not_added(self) -> None:
        payload = _conversation_update_payload(
            members_added=[{"id": "user-aad-2"}],
        )
        assert is_bot_added(payload) is False

    def test_not_conversation_update(self) -> None:
        payload = _personal_message_payload()
        assert is_bot_added(payload) is False

    def test_with_explicit_bot_id(self) -> None:
        payload = _conversation_update_payload(
            members_added=[{"id": "custom-bot-id"}],
        )
        assert is_bot_added(payload, bot_id="custom-bot-id") is True

    def test_with_explicit_bot_id_mismatch(self) -> None:
        payload = _conversation_update_payload(
            members_added=[{"id": "other-id"}],
        )
        assert is_bot_added(payload, bot_id="custom-bot-id") is False

    def test_empty_members_added(self) -> None:
        payload = _conversation_update_payload(members_added=[])
        assert is_bot_added(payload) is False


# ---------------------------------------------------------------------------
# parse_teams_webhook
# ---------------------------------------------------------------------------


class TestParseTeamsWebhook:
    def test_parse_personal_message(self) -> None:
        payload = _personal_message_payload(text="Hi bot!")
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert len(messages) == 1
        msg = messages[0]
        assert msg.channel_id == "teams-main"
        assert msg.sender_id == "user-aad-1"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hi bot!"
        assert msg.external_id == "act-1"
        assert msg.idempotency_key == "act-1"

    def test_parse_group_message_with_mention(self) -> None:
        payload = _group_message_payload(text="<at>TestBot</at> What time is it?")
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg.content, TextContent)
        # The <at>...</at> mention tag should be stripped
        assert msg.content.body == "What time is it?"

    def test_parse_non_message_ignored(self) -> None:
        payload = _conversation_update_payload(
            members_added=[{"id": "bot-aad-1"}],
        )
        messages = parse_teams_webhook(payload, channel_id="teams-main")
        assert messages == []

    def test_parse_empty_text_ignored(self) -> None:
        payload = _personal_message_payload(text="")
        messages = parse_teams_webhook(payload, channel_id="teams-main")
        assert messages == []

    def test_parse_whitespace_only_text_ignored(self) -> None:
        payload = _personal_message_payload(text="   ")
        messages = parse_teams_webhook(payload, channel_id="teams-main")
        assert messages == []

    def test_metadata_fields(self) -> None:
        payload = _personal_message_payload()
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert len(messages) == 1
        meta = messages[0].metadata
        assert meta["sender_name"] == "Alice"
        assert meta["conversation_id"] == "conv-1"
        assert meta["conversation_type"] == "personal"
        assert meta["is_group"] is False
        assert meta["bot_mentioned"] is True
        assert meta["service_url"] == "https://smba.trafficmanager.net/teams/"
        assert meta["tenant_id"] == "tenant-abc"

    def test_bot_mentioned_in_personal(self) -> None:
        """In personal chats the bot is always implicitly addressed."""
        payload = _personal_message_payload()
        # Personal chat without explicit entities -- bot_mentioned is still True
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert messages[0].metadata["bot_mentioned"] is True

    def test_bot_mentioned_in_group_with_entity(self) -> None:
        payload = _group_message_payload()
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert messages[0].metadata["bot_mentioned"] is True

    def test_bot_not_mentioned_in_group(self) -> None:
        """Group message without a @mention of the bot."""
        payload = _group_message_payload(text="Just chatting")
        # Remove entities so there's no mention
        payload["entities"] = []
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert len(messages) == 1
        assert messages[0].metadata["bot_mentioned"] is False

    def test_mention_stripping(self) -> None:
        """<at>BotName</at> tags are removed from group messages."""
        payload = _group_message_payload(text="<at>TestBot</at> do something")
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert len(messages) == 1
        assert "<at>" not in messages[0].content.body  # type: ignore[union-attr]
        assert messages[0].content.body == "do something"  # type: ignore[union-attr]

    def test_mention_only_message_in_group_ignored(self) -> None:
        """A group message that is ONLY an @mention (no other text) is ignored."""
        payload = _group_message_payload(text="<at>TestBot</at> ")
        messages = parse_teams_webhook(payload, channel_id="teams-main")
        assert messages == []

    def test_personal_message_preserves_at_tags(self) -> None:
        """In personal chats, <at> tags are not stripped (no group stripping)."""
        payload = _personal_message_payload(text="<at>SomeUser</at> hello")
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert len(messages) == 1
        # Personal chats do NOT strip <at> tags
        assert "<at>SomeUser</at>" in messages[0].content.body  # type: ignore[union-attr]

    def test_thread_id_from_reply_to_id(self) -> None:
        """replyToId is mapped to InboundMessage.thread_id."""
        payload = _personal_message_payload(replyToId="parent-act-1")
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert len(messages) == 1
        assert messages[0].thread_id == "parent-act-1"
        assert messages[0].metadata["reply_to_id"] == "parent-act-1"

    def test_no_reply_to_id(self) -> None:
        """Without replyToId, thread_id is None."""
        payload = _personal_message_payload()
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert len(messages) == 1
        assert messages[0].thread_id is None
        assert messages[0].metadata["reply_to_id"] == ""

    def test_empty_reply_to_id(self) -> None:
        """An empty-string replyToId is treated as no thread."""
        payload = _personal_message_payload(replyToId="")
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        assert len(messages) == 1
        assert messages[0].thread_id is None


# ---------------------------------------------------------------------------
# parse_teams_reactions
# ---------------------------------------------------------------------------


class TestParseTeamsReactions:
    def test_reactions_added(self) -> None:
        payload = {
            "type": "messageReaction",
            "from": {"id": "user-1", "name": "Alice"},
            "replyToId": "target-act-1",
            "reactionsAdded": [{"type": "like"}, {"type": "heart"}],
        }
        results = parse_teams_reactions(payload)

        assert len(results) == 2
        assert results[0]["action"] == "add"
        assert results[0]["emoji"] == "like"
        assert results[0]["sender_id"] == "user-1"
        assert results[0]["sender_name"] == "Alice"
        assert results[0]["target_activity_id"] == "target-act-1"
        assert results[1]["emoji"] == "heart"

    def test_reactions_removed(self) -> None:
        payload = {
            "type": "messageReaction",
            "from": {"id": "user-2", "name": "Bob"},
            "replyToId": "target-act-2",
            "reactionsRemoved": [{"type": "laugh"}],
        }
        results = parse_teams_reactions(payload)

        assert len(results) == 1
        assert results[0]["action"] == "remove"
        assert results[0]["emoji"] == "laugh"
        assert results[0]["sender_id"] == "user-2"

    def test_both_added_and_removed(self) -> None:
        payload = {
            "type": "messageReaction",
            "from": {"id": "user-1", "name": "Alice"},
            "replyToId": "target-act-3",
            "reactionsAdded": [{"type": "like"}],
            "reactionsRemoved": [{"type": "heart"}],
        }
        results = parse_teams_reactions(payload)

        assert len(results) == 2
        assert results[0]["action"] == "add"
        assert results[1]["action"] == "remove"

    def test_non_reaction_activity_returns_empty(self) -> None:
        payload = _personal_message_payload()
        results = parse_teams_reactions(payload)
        assert results == []

    def test_empty_reaction_lists(self) -> None:
        payload = {
            "type": "messageReaction",
            "from": {"id": "user-1", "name": "Alice"},
            "replyToId": "target-act-4",
            "reactionsAdded": [],
            "reactionsRemoved": [],
        }
        results = parse_teams_reactions(payload)
        assert results == []


# ---------------------------------------------------------------------------
# InMemoryConversationReferenceStore
# ---------------------------------------------------------------------------


class TestInMemoryConversationReferenceStore:
    async def test_save_and_get(self) -> None:
        store = InMemoryConversationReferenceStore()
        ref = {"conversation": {"id": "conv-1"}, "serviceUrl": "https://example.com"}
        await store.save("conv-1", ref)

        result = await store.get("conv-1")
        assert result == ref

    async def test_get_missing(self) -> None:
        store = InMemoryConversationReferenceStore()
        result = await store.get("nonexistent")
        assert result is None

    async def test_delete(self) -> None:
        store = InMemoryConversationReferenceStore()
        await store.save("conv-1", {"data": "value"})
        await store.delete("conv-1")

        result = await store.get("conv-1")
        assert result is None

    async def test_delete_missing_is_noop(self) -> None:
        store = InMemoryConversationReferenceStore()
        # Should not raise
        await store.delete("nonexistent")

    async def test_list_all(self) -> None:
        store = InMemoryConversationReferenceStore()
        ref_a = {"conversation": {"id": "a"}}
        ref_b = {"conversation": {"id": "b"}}
        await store.save("a", ref_a)
        await store.save("b", ref_b)

        all_refs = await store.list_all()
        assert len(all_refs) == 2
        assert all_refs["a"] == ref_a
        assert all_refs["b"] == ref_b

    async def test_list_all_empty(self) -> None:
        store = InMemoryConversationReferenceStore()
        all_refs = await store.list_all()
        assert all_refs == {}

    async def test_overwrite(self) -> None:
        store = InMemoryConversationReferenceStore()
        await store.save("conv-1", {"version": 1})
        await store.save("conv-1", {"version": 2})

        result = await store.get("conv-1")
        assert result == {"version": 2}

    async def test_list_all_returns_copy(self) -> None:
        """Mutating the returned dict should not affect the store."""
        store = InMemoryConversationReferenceStore()
        await store.save("conv-1", {"data": "original"})

        all_refs = await store.list_all()
        all_refs["conv-1"] = {"data": "tampered"}

        result = await store.get("conv-1")
        assert result == {"data": "original"}


# ---------------------------------------------------------------------------
# MockTeamsProvider
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BotFrameworkTeamsProvider.create_personal_conversation
# ---------------------------------------------------------------------------


class TestCreatePersonalConversation:
    """Tests for proactive 1:1 personal conversation creation."""

    def _make_provider(self) -> Any:
        """Build a BotFrameworkTeamsProvider with a mocked adapter."""
        from unittest.mock import AsyncMock, MagicMock

        from roomkit.providers.teams import BotFrameworkTeamsProvider

        config = TeamsConfig(app_id="bot-app-id", app_password="pw", tenant_id="default-tenant")
        provider = BotFrameworkTeamsProvider.__new__(BotFrameworkTeamsProvider)
        provider._config = config
        provider._conversation_store = InMemoryConversationReferenceStore()
        provider._adapter = MagicMock()
        provider._adapter.create_conversation = AsyncMock()
        return provider

    async def test_happy_path(self) -> None:
        """Adapter returns a valid ID — reference stored, ID returned."""
        from unittest.mock import MagicMock

        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.id = "personal-conv-123"
        provider._adapter.create_conversation.return_value = mock_response

        conv_id = await provider.create_personal_conversation(
            service_url="https://smba.trafficmanager.net/amer/",
            user_id="29:user-aad-id",
        )

        assert conv_id == "personal-conv-123"
        ref = await provider.conversation_store.get("personal-conv-123")
        assert ref is not None

    async def test_failure_no_response(self) -> None:
        """Adapter returns None — RuntimeError raised."""
        provider = self._make_provider()
        provider._adapter.create_conversation.return_value = None

        with pytest.raises(RuntimeError, match="Failed to create personal conversation"):
            await provider.create_personal_conversation(
                service_url="https://smba.trafficmanager.net/amer/",
                user_id="29:user-aad-id",
            )

    async def test_failure_no_id(self) -> None:
        """Adapter returns response without an ID — RuntimeError raised."""
        from unittest.mock import MagicMock

        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.id = None
        provider._adapter.create_conversation.return_value = mock_response

        with pytest.raises(RuntimeError, match="Failed to create personal conversation"):
            await provider.create_personal_conversation(
                service_url="https://smba.trafficmanager.net/amer/",
                user_id="29:user-aad-id",
            )

    async def test_custom_tenant_id(self) -> None:
        """Explicit tenant_id is forwarded to ConversationParameters."""
        from unittest.mock import MagicMock

        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.id = "conv-custom-tenant"
        provider._adapter.create_conversation.return_value = mock_response

        await provider.create_personal_conversation(
            service_url="https://smba.trafficmanager.net/amer/",
            user_id="29:user-aad-id",
            tenant_id="custom-tenant",
        )

        call_args = provider._adapter.create_conversation.call_args
        params = call_args[0][1]
        assert params.tenant_id == "custom-tenant"

    async def test_default_tenant_id(self) -> None:
        """When no tenant_id override, config.tenant_id is used."""
        from unittest.mock import MagicMock

        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.id = "conv-default-tenant"
        provider._adapter.create_conversation.return_value = mock_response

        await provider.create_personal_conversation(
            service_url="https://smba.trafficmanager.net/amer/",
            user_id="29:user-aad-id",
        )

        call_args = provider._adapter.create_conversation.call_args
        params = call_args[0][1]
        assert params.tenant_id == "default-tenant"

    async def test_is_group_false(self) -> None:
        """ConversationParameters.is_group must be False for 1:1 chats."""
        from unittest.mock import MagicMock

        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.id = "conv-personal"
        provider._adapter.create_conversation.return_value = mock_response

        await provider.create_personal_conversation(
            service_url="https://smba.trafficmanager.net/amer/",
            user_id="29:user-aad-id",
        )

        call_args = provider._adapter.create_conversation.call_args
        params = call_args[0][1]
        assert params.is_group is False
        assert params.members[0].id == "29:user-aad-id"

    async def test_send_after_create(self) -> None:
        """send() works immediately after create_personal_conversation()."""
        from unittest.mock import MagicMock

        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.id = "conv-roundtrip"
        provider._adapter.create_conversation.return_value = mock_response

        conv_id = await provider.create_personal_conversation(
            service_url="https://smba.trafficmanager.net/amer/",
            user_id="29:user-aad-id",
        )

        # Verify the stored reference is retrievable for send()
        ref = await provider.conversation_store.get(conv_id)
        assert ref is not None
        # ConversationAccount serializes with camelCase keys
        assert ref["conversation"]["id"] == "conv-roundtrip"
        assert ref["conversation"]["isGroup"] is False


# ---------------------------------------------------------------------------
# MockTeamsProvider
# ---------------------------------------------------------------------------


class TestMockTeamsProvider:
    async def test_send_records(self) -> None:
        provider = MockTeamsProvider()
        event = make_event(body="Test message")

        await provider.send(event, to="conv-1")

        assert len(provider.sent) == 1
        assert provider.sent[0]["event"] is event
        assert provider.sent[0]["to"] == "conv-1"

    async def test_send_returns_success(self) -> None:
        provider = MockTeamsProvider()
        event = make_event(body="Another message")

        result = await provider.send(event, to="conv-2")

        assert result.success is True
        assert result.provider_message_id is not None
        assert len(result.provider_message_id) > 0

    async def test_send_multiple(self) -> None:
        provider = MockTeamsProvider()
        e1 = make_event(body="first")
        e2 = make_event(body="second")

        await provider.send(e1, to="conv-1")
        await provider.send(e2, to="conv-2")

        assert len(provider.sent) == 2
        assert provider.sent[0]["to"] == "conv-1"
        assert provider.sent[1]["to"] == "conv-2"

    async def test_provider_name(self) -> None:
        provider = MockTeamsProvider()
        assert provider.name == "MockTeamsProvider"


# ---------------------------------------------------------------------------
# BotFrameworkTeamsProvider.send — threading
# ---------------------------------------------------------------------------


class TestBotFrameworkSendThreading:
    """Tests that send() sets reply_to_id on the Activity when thread_id is present."""

    def _make_provider(self) -> Any:
        from unittest.mock import AsyncMock, MagicMock

        from roomkit.providers.teams import BotFrameworkTeamsProvider

        config = TeamsConfig(app_id="bot-app-id", app_password="pw", tenant_id="default-tenant")
        provider = BotFrameworkTeamsProvider.__new__(BotFrameworkTeamsProvider)
        provider._config = config
        provider._conversation_store = InMemoryConversationReferenceStore()
        provider._adapter = MagicMock()
        provider._adapter.continue_conversation = AsyncMock()
        return provider

    async def test_send_with_thread_id(self) -> None:
        """When event has channel_data.thread_id, Activity gets reply_to_id."""
        from unittest.mock import MagicMock

        from roomkit.models.event import ChannelData

        provider = self._make_provider()
        await provider.conversation_store.save("conv-1", {"conversation": {"id": "conv-1"}})

        event = make_event(
            body="threaded reply",
            channel_data=ChannelData(thread_id="parent-act-1"),
        )

        # Capture the callback to inspect the Activity
        captured_activity = None

        async def _capture_callback(ref, callback, app_id):  # noqa: ARG001
            mock_turn = MagicMock()
            mock_response = MagicMock()
            mock_response.id = "response-1"
            mock_turn.send_activity = AsyncMock(return_value=mock_response)
            await callback(mock_turn)
            nonlocal captured_activity
            captured_activity = mock_turn.send_activity.call_args[0][0]

        from unittest.mock import AsyncMock

        provider._adapter.continue_conversation = _capture_callback

        result = await provider.send(event, to="conv-1")
        assert result.success is True
        assert captured_activity is not None
        assert captured_activity.reply_to_id == "parent-act-1"

    async def test_send_without_thread_id(self) -> None:
        """When event has no thread_id, Activity does not set reply_to_id."""
        from unittest.mock import MagicMock

        provider = self._make_provider()
        await provider.conversation_store.save("conv-1", {"conversation": {"id": "conv-1"}})

        event = make_event(body="no thread")

        captured_activity = None

        async def _capture_callback(ref, callback, app_id):  # noqa: ARG001
            mock_turn = MagicMock()
            mock_response = MagicMock()
            mock_response.id = "response-2"
            mock_turn.send_activity = AsyncMock(return_value=mock_response)
            await callback(mock_turn)
            nonlocal captured_activity
            captured_activity = mock_turn.send_activity.call_args[0][0]

        from unittest.mock import AsyncMock

        provider._adapter.continue_conversation = _capture_callback

        result = await provider.send(event, to="conv-1")
        assert result.success is True
        assert captured_activity is not None
        assert captured_activity.reply_to_id is None

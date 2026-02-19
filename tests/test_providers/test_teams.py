"""Tests for the Microsoft Teams provider."""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from roomkit.models.event import TextContent
from roomkit.providers.teams import (
    InMemoryConversationReferenceStore,
    MockTeamsProvider,
    TeamsConfig,
    is_bot_added,
    parse_teams_activity,
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
        assert result["members_added"] == []
        assert result["members_removed"] == []

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

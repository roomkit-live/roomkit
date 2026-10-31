"""Tests for all Pydantic data models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from roomkit.models import (
    Access,
    AudioContent,
    ChannelBinding,
    ChannelCapabilities,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelOutput,
    ChannelType,
    CompositeContent,
    DeliveryError,
    EventDirection,
    EventSource,
    EventStatus,
    EventType,
    FrameworkEvent,
    HookResult,
    IdentificationStatus,
    Identity,
    IdentityHookResult,
    InboundMessage,
    InboundResult,
    InjectedEvent,
    LocationContent,
    MediaContent,
    Observation,
    Participant,
    ParticipantRole,
    RateLimit,
    RichContent,
    Room,
    RoomContext,
    RoomEvent,
    RoomStatus,
    RoomTimers,
    SystemContent,
    Task,
    TemplateContent,
    TextContent,
    VideoContent,
)
from roomkit.models.channel import RetryPolicy
from roomkit.models.event import ToolCallContent
from roomkit.store.sqlite import SQLiteStore

# -- Content models --


class TestMediaContent:
    def test_create(self) -> None:
        c = MediaContent(url="https://example.com/img.png", mime_type="image/png")
        assert c.type == "media"
        assert c.filename is None


class TestAudioContent:
    def test_defaults(self) -> None:
        c = AudioContent(url="https://example.com/audio.ogg")
        assert c.mime_type == "audio/ogg"


class TestVideoContent:
    def test_defaults(self) -> None:
        c = VideoContent(url="https://example.com/video.mp4")
        assert c.mime_type == "video/mp4"


class TestCompositeContent:
    def test_create(self) -> None:
        parts = [
            TextContent(body="hello"),
            MediaContent(url="https://example.com/img.png", mime_type="image/png"),
        ]
        c = CompositeContent(parts=parts)
        assert c.type == "composite"
        assert len(c.parts) == 2


# -- Discriminated union --


class TestEventContentDiscrimination:
    def test_text_roundtrip(self) -> None:
        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
            content=TextContent(body="hi"),
        )
        assert event.content.type == "text"

    def test_rich_roundtrip(self) -> None:
        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.EMAIL),
            content=RichContent(body="<b>hi</b>", format="html"),
        )
        assert event.content.type == "rich"


# -- RoomEvent --


class TestRoomEvent:
    def test_defaults(self) -> None:
        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
            content=TextContent(body="msg"),
        )
        assert event.type == EventType.MESSAGE
        assert event.status == EventStatus.PENDING
        assert event.index == 0
        assert event.chain_depth == 0
        assert event.id  # auto-generated

    def test_custom_fields(self) -> None:
        event = RoomEvent(
            id="custom-id",
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.AI),
            content=TextContent(body="response"),
            type=EventType.SYSTEM,
            chain_depth=2,
        )
        assert event.id == "custom-id"
        assert event.chain_depth == 2

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            RoomEvent(  # type: ignore[call-arg]
                source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
                content=TextContent(body="msg"),
            )

    async def test_channel_data_keeps_the_keys_its_channel_writes(self, tmp_path: Path) -> None:
        """RFC §5.4: a channel's own data (an SMS's ``segments``) travels on
        the event and is stored with it, as roomkit-rs's map keeps it
        (RMK-1021 item 3). ``ChannelData`` dropped every key it did not name."""
        store = SQLiteStore(tmp_path / "rooms.db")
        await store.create_room(Room(id="r1"))
        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
            content=TextContent(body="msg"),
            channel_data={"provider": "twilio", "segments": 2, "encoding": "GSM-7"},
        )
        stored = await store.add_event(event)
        read = await store.get_event(stored.id)
        await store.close()
        assert read is not None
        data = read.channel_data.model_dump()
        assert (data["provider"], data["segments"], data["encoding"]) == ("twilio", 2, "GSM-7")


# -- Channel models --


class TestChannelBinding:
    def test_defaults(self) -> None:
        b = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        assert b.category == ChannelCategory.TRANSPORT
        assert b.direction == ChannelDirection.BIDIRECTIONAL
        assert b.access == Access.READ_WRITE
        assert b.muted is False
        assert b.visibility == "all"


class TestChannelCapabilities:
    def test_defaults(self) -> None:
        c = ChannelCapabilities()
        assert c.media_types == [ChannelMediaType.TEXT]
        assert c.max_length is None


class TestChannelOutput:
    def test_empty_factory(self) -> None:
        o = ChannelOutput.empty()
        assert o.responded is False
        assert o.response_events == []
        assert o.tasks == []
        assert o.observations == []

    def test_with_response(self) -> None:
        ev = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ai1", channel_type=ChannelType.AI),
            content=TextContent(body="reply"),
        )
        o = ChannelOutput(responded=True, response_events=[ev])
        assert o.responded is True
        assert len(o.response_events) == 1


# -- Participant --


class TestParticipant:
    def test_create(self) -> None:
        p = Participant(id="p1", room_id="r1", channel_id="ch1")
        assert p.role == ParticipantRole.MEMBER
        assert p.display_name is None
        assert p.connected_via == ["ch1"]

    def test_primary_channel_is_included_first_without_duplicates(self) -> None:
        p = Participant(
            id="p1",
            room_id="r1",
            channel_id="ch1",
            connected_via=["ch2", "ch1", "ch2"],
        )
        assert p.connected_via == ["ch1", "ch2"]


# -- Identity --


class TestIdentityHookResult:
    def test_resolved(self) -> None:
        identity = Identity(id="id1")
        r = IdentityHookResult.resolved(identity)
        assert r.status == IdentificationStatus.IDENTIFIED
        assert r.identity is not None

    def test_pending(self) -> None:
        r = IdentityHookResult.pending(display_name="waiting")
        assert r.status == IdentificationStatus.PENDING
        assert r.display_name == "waiting"

    def test_challenge(self) -> None:
        from roomkit.models.hook import InjectedEvent

        inject = InjectedEvent(
            event=RoomEvent(
                room_id="r1",
                source=EventSource(channel_id="sys", channel_type=ChannelType.WEBHOOK),
                content=TextContent(body="verify yourself"),
            )
        )
        r = IdentityHookResult.challenge(inject, "check your phone")
        assert r.status == IdentificationStatus.CHALLENGE_SENT
        assert r.inject is not None

    def test_reject(self) -> None:
        r = IdentityHookResult.reject("not allowed")
        assert r.status == IdentificationStatus.REJECTED


# -- Room --


class TestRoom:
    def test_defaults(self) -> None:
        r = Room(id="r1")
        assert r.status == RoomStatus.ACTIVE
        assert r.event_count == 0
        assert r.latest_index is None
        assert r.closed_at is None

    def test_custom(self) -> None:
        r = Room(id="r1", status=RoomStatus.CLOSED, metadata={"key": "val"})
        assert r.metadata["key"] == "val"


# -- Hook --


class TestHookResult:
    def test_allow(self) -> None:
        r = HookResult.allow()
        assert r.action == "allow"
        assert r.injected_events == []

    def test_block(self) -> None:
        r = HookResult.block("spam detected")
        assert r.action == "block"
        assert r.reason == "spam detected"

    def test_modify(self) -> None:
        ev = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
            content=TextContent(body="modified"),
        )
        r = HookResult.modify(ev)
        assert r.action == "modify"
        assert r.event is not None


class TestInjectedEvent:
    def test_create(self) -> None:
        ev = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="sys", channel_type=ChannelType.WEBHOOK),
            content=SystemContent(body="notice"),
        )
        ie = InjectedEvent(event=ev, target_channel_ids=["ch1"])
        assert ie.target_channel_ids == ["ch1"]


# -- Task / Observation --


class TestTask:
    def test_create(self) -> None:
        t = Task(id="t1", room_id="r1", title="Follow up")
        assert t.status == "pending"


class TestObservation:
    def test_create(self) -> None:
        o = Observation(id="o1", room_id="r1", channel_id="ai1", content="sentiment: positive")
        assert o.confidence == 1.0


# -- Delivery --


class TestInboundMessage:
    def test_create(self) -> None:
        m = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            content=TextContent(body="hi"),
        )
        assert m.idempotency_key is None


class TestInboundResult:
    def test_blocked(self) -> None:
        r = InboundResult(blocked=True, reason="spam")
        assert r.event is None


class TestDeliveryResult:
    def test_an_unmarked_error_reports_as_retryable(self) -> None:
        """Matching what the retry loop does with an error that says nothing."""
        assert DeliveryError(code="Boom", message="boom").retryable is True


# -- Context --


class TestRoomContext:
    def test_other_channels(self) -> None:
        b1 = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        b2 = ChannelBinding(channel_id="ch2", room_id="r1", channel_type=ChannelType.WEBSOCKET)
        ctx = RoomContext(room=Room(id="r1"), bindings=[b1, b2])
        others = ctx.other_channels("ch1")
        assert len(others) == 1
        assert others[0].channel_id == "ch2"

    def test_channels_by_type(self) -> None:
        b1 = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        b2 = ChannelBinding(channel_id="ch2", room_id="r1", channel_type=ChannelType.SMS)
        b3 = ChannelBinding(channel_id="ch3", room_id="r1", channel_type=ChannelType.AI)
        ctx = RoomContext(room=Room(id="r1"), bindings=[b1, b2, b3])
        sms = ctx.channels_by_type(ChannelType.SMS)
        assert len(sms) == 2

    def test_get_binding(self) -> None:
        b1 = ChannelBinding(channel_id="ch1", room_id="r1", channel_type=ChannelType.SMS)
        ctx = RoomContext(room=Room(id="r1"), bindings=[b1])
        assert ctx.get_binding("ch1") is not None
        assert ctx.get_binding("missing") is None


# -- FrameworkEvent --


class TestFrameworkEvent:
    def test_create(self) -> None:
        fe = FrameworkEvent(type="delivery_succeeded", room_id="r1")
        assert fe.channel_id is None
        assert fe.data == {}


# -- Validation constraints (Changeset 1) --


class TestRoomFieldConstraints:
    def test_negative_event_count(self) -> None:
        with pytest.raises(ValidationError):
            Room(id="r1", event_count=-1)

    def test_negative_latest_index(self) -> None:
        with pytest.raises(ValidationError):
            Room(id="r1", latest_index=-1)


class TestRoomTimersConstraints:
    def test_zero_is_valid(self) -> None:
        t = RoomTimers(inactive_after_seconds=0, closed_after_seconds=0)
        assert t.inactive_after_seconds == 0
        assert t.closed_after_seconds == 0


class TestRoomEventFieldConstraints:
    def test_negative_index(self) -> None:
        with pytest.raises(ValidationError):
            RoomEvent(
                room_id="r1",
                source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
                content=TextContent(body="hi"),
                index=-1,
            )

    def test_negative_chain_depth(self) -> None:
        with pytest.raises(ValidationError):
            RoomEvent(
                room_id="r1",
                source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
                content=TextContent(body="hi"),
                chain_depth=-1,
            )


class TestMediaContentConstraints:
    def test_negative_size_bytes(self) -> None:
        with pytest.raises(ValidationError):
            MediaContent(url="https://example.com/f.png", mime_type="image/png", size_bytes=-1)

    def test_invalid_url(self) -> None:
        with pytest.raises(ValidationError):
            MediaContent(url="ftp://example.com/f.png", mime_type="image/png")

    def test_valid_url(self) -> None:
        c = MediaContent(url="https://example.com/f.png", mime_type="image/png")
        assert c.url == "https://example.com/f.png"


class TestLocationContentConstraints:
    def test_boundary_values(self) -> None:
        c = LocationContent(latitude=90.0, longitude=-180.0)
        assert c.latitude == 90.0
        assert c.longitude == -180.0


class TestAudioContentConstraints:
    def test_negative_duration(self) -> None:
        with pytest.raises(ValidationError):
            AudioContent(url="https://example.com/a.ogg", duration_seconds=-1.0)

    def test_invalid_url(self) -> None:
        with pytest.raises(ValidationError):
            AudioContent(url="ftp://example.com/a.ogg")


class TestVideoContentConstraints:
    def test_negative_duration(self) -> None:
        with pytest.raises(ValidationError):
            VideoContent(url="https://example.com/v.mp4", duration_seconds=-1.0)

    def test_invalid_url(self) -> None:
        with pytest.raises(ValidationError):
            VideoContent(url="ftp://example.com/v.mp4")

    def test_invalid_thumbnail_url(self) -> None:
        with pytest.raises(ValidationError):
            VideoContent(url="https://example.com/v.mp4", thumbnail_url="ftp://thumb.png")

    def test_none_thumbnail_url_ok(self) -> None:
        v = VideoContent(url="https://example.com/v.mp4", thumbnail_url=None)
        assert v.thumbnail_url is None


class TestCompositeContentConstraints:
    def test_empty_parts(self) -> None:
        with pytest.raises(ValidationError, match="at least one part"):
            CompositeContent(parts=[])

    def test_nesting_depth_exceeds_limit(self) -> None:
        """Build 6 levels of nesting (> max of 5) using raw dicts to bypass
        intermediate validation."""
        # Build nested dict structure: 6 CompositeContent levels + a text leaf
        inner: dict[str, object] = {"type": "text", "body": "leaf"}
        for _ in range(6):
            inner = {"type": "composite", "parts": [inner]}
        with pytest.raises(ValidationError, match="nesting depth"):
            CompositeContent.model_validate(inner)

    def test_nesting_counts_through_an_edit(self) -> None:
        """An edit's new content stands in the edit's place: a composite
        holding an edit of a 5-level composite is 6 levels deep."""
        inner: dict[str, object] = {"type": "text", "body": "leaf"}
        for _ in range(5):
            inner = {"type": "composite", "parts": [inner]}
        edit = {"type": "edit", "target_event_id": "evt_1", "new_content": inner}
        with pytest.raises(ValidationError, match="nesting depth"):
            CompositeContent.model_validate({"type": "composite", "parts": [edit]})

    def test_nesting_within_limit(self) -> None:
        """5 levels should be accepted."""
        inner = CompositeContent(parts=[TextContent(body="leaf")])
        for _ in range(3):
            inner = CompositeContent(parts=[inner])
        # 5 levels deep — should be fine
        c = CompositeContent(parts=[inner])
        assert c.type == "composite"


class TestChannelBindingConstraints:
    def test_negative_last_read_index(self) -> None:
        with pytest.raises(ValidationError):
            ChannelBinding(
                channel_id="ch1",
                room_id="r1",
                channel_type=ChannelType.SMS,
                last_read_index=-1,
            )


class TestChannelCapabilitiesConstraints:
    def test_zero_max_length(self) -> None:
        with pytest.raises(ValidationError):
            ChannelCapabilities(max_length=0)

    def test_zero_max_buttons(self) -> None:
        with pytest.raises(ValidationError):
            ChannelCapabilities(max_buttons=0)

    def test_zero_max_media_size_bytes(self) -> None:
        with pytest.raises(ValidationError):
            ChannelCapabilities(max_media_size_bytes=0)


class TestObservationConfidenceConstraint:
    def test_confidence_above_one(self) -> None:
        with pytest.raises(ValidationError):
            Observation(id="o1", room_id="r1", channel_id="ai1", content="test", confidence=1.5)

    def test_confidence_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            Observation(id="o1", room_id="r1", channel_id="ai1", content="test", confidence=-0.1)


class TestHookResultValidation:
    def test_modify_without_event_raises(self) -> None:
        with pytest.raises(ValidationError, match="modify"):
            HookResult(action="modify")

    def test_block_without_reason_raises(self) -> None:
        with pytest.raises(ValidationError, match="block"):
            HookResult(action="block")

    def test_block_classmethod_requires_reason(self) -> None:
        r = HookResult.block("a reason")
        assert r.reason == "a reason"

    def test_allow_no_validation_error(self) -> None:
        r = HookResult(action="allow")
        assert r.action == "allow"


class TestEventSourceDirection:
    def test_an_event_travels_one_way(self) -> None:
        """An event is inbound or outbound, never both (RFC §5.2)."""
        with pytest.raises(ValidationError):
            EventSource(channel_id="sms", channel_type=ChannelType.SMS, direction="bidirectional")

    def test_a_channel_direction_value_is_read_as_the_event_s(self) -> None:
        source = EventSource(
            channel_id="sms", channel_type=ChannelType.SMS, direction=ChannelDirection.OUTBOUND
        )
        assert source.direction is EventDirection.OUTBOUND


class TestRecordTimestamps:
    """A record's timestamps hold their instant in UTC (RFC §1.4)."""

    def test_a_timestamp_without_an_offset_is_read_as_utc(self) -> None:
        room = Room(id="r1", created_at=datetime(2026, 1, 1, 12, 0))
        assert room.created_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def test_a_timestamp_with_an_offset_is_held_in_utc(self) -> None:
        paris = timezone(timedelta(hours=2))
        room = Room(id="r1", created_at=datetime(2026, 1, 1, 14, 0, tzinfo=paris))
        assert (room.created_at.utcoffset(), room.created_at.hour) == (timedelta(0), 12)

    def test_a_timestamp_is_written_in_utc(self) -> None:
        room = Room(id="r1", created_at=datetime(2026, 1, 1, 12, 0, 30, 250000))
        assert room.model_dump(mode="json")["created_at"] == "2026-01-01T12:00:30.250000Z"


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (LocationContent, {"latitude": -91.0, "longitude": 0.0}),
        (LocationContent, {"latitude": 91.0, "longitude": 0.0}),
        (LocationContent, {"latitude": 0.0, "longitude": -181.0}),
        (LocationContent, {"latitude": 0.0, "longitude": 181.0}),
        (RateLimit, {"max_per_minute": -1.0}),
        (RateLimit, {"max_per_second": 0.0}),
        (RetryPolicy, {"max_retries": -1}),
        (RetryPolicy, {"base_delay_seconds": 0.0}),
        (RetryPolicy, {"max_delay_seconds": 0.0}),
        (RoomTimers, {"closed_after_seconds": -5}),
        (RoomTimers, {"inactive_after_seconds": -1}),
    ],
)
def test_a_value_out_of_its_range_is_refused(model: type, values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        model(**values)


@pytest.mark.parametrize(
    ("content", "kind"),
    [
        (TextContent(body="hello"), "text"),
        (RichContent(body="**bold**"), "rich"),
        (LocationContent(latitude=45.5, longitude=-73.6), "location"),
        (SystemContent(body="User joined"), "system"),
        (TemplateContent(template_id="welcome_v1"), "template"),
        (ToolCallContent(tool_name="web_search", tool_id="call-1"), "tool_call"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_content_names_its_kind_where_it_is_stored_and_sent(content: object, kind: str) -> None:
    assert content.model_dump(mode="json")["type"] == kind  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("model", "field", "default"),
    [
        (RichContent(body="**bold**"), "format", "markdown"),  # RFC §5.3
        (TemplateContent(template_id="welcome_v1"), "language", "en"),  # RFC §5.3
        (ToolCallContent(tool_name="web_search", tool_id="call-1"), "status", "pending"),  # §5.3
        (RetryPolicy(), "exponential_base", 2.0),  # RFC §13.2
    ],
    ids=["rich-format", "template-language", "tool-call-status", "retry-base"],
)
def test_the_defaults_the_rfc_sets(model: object, field: str, default: object) -> None:
    """Values a stored or sent content takes when its author left them out."""
    assert getattr(model, field) == default

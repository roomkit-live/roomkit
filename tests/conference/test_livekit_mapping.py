"""What the LiveKit backend promises, checked without LiveKit installed.

The ``livekit`` extra is optional, so a contract that could only be checked on a
machine that installed it would go unchecked in CI — and the translations are
the part most worth checking, because they are where a grant becomes a
permission and an attribute becomes an identity. Nothing here imports the SDK.

The behaviour that needs a constructed backend lives in
``test_livekit_backend.py``; the behaviour that needs a real SFU lives in
``test_livekit_live.py``.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from roomkit.conference._livekit_mapping import (
    CAMERA,
    MICROPHONE,
    SCREEN_SHARE,
    asserted_attributes,
    capabilities_for,
    codec_for_buffer_type,
    participant_record,
    publish_source_names,
    quality_label,
    require_publishable_pcm,
    rtc_participant_kind_name,
    rtc_track_kind_name,
    rtc_track_source_name,
    track_kind_for,
    track_record,
    video_grant_kwargs,
)
from roomkit.conference.base import ConferenceBackend
from roomkit.conference.livekit import LiveKitConferenceBackend, LiveKitConfig
from roomkit.conference.models import ConferenceCapability, ConferenceGrants, TrackKind
from roomkit.voice.base import AudioChunk

BACKEND_SURFACE = (
    "name",
    "capabilities",
    "ensure_room",
    "close_room",
    "mint_access",
    "list_participants",
    "remove_participant",
    "mute_track",
    "unmute_track",
    "join_as_bot",
    "leave",
    "subscribe_track",
    "unsubscribe_track",
    "publish_audio",
    "publish_video",
    "close",
)


class TestConformance:
    """The interface is implemented, and implemented here."""

    def test_the_backend_is_concrete(self) -> None:
        assert LiveKitConferenceBackend.__abstractmethods__ == frozenset()

    def test_the_whole_interface_is_implemented_by_this_class(self) -> None:
        """Defined on the backend itself, not inherited.

        A member left to the ABC would be abstract, and one picked up from
        somewhere else would not be this backend's behaviour.
        """
        for member in BACKEND_SURFACE:
            assert member in LiveKitConferenceBackend.__dict__, member

    def test_the_interface_is_the_one_the_abc_declares(self) -> None:
        assert set(BACKEND_SURFACE) == ConferenceBackend.__abstractmethods__

    def test_no_method_is_a_stub(self) -> None:
        """No ``NotImplementedError`` anywhere in the backend.

        The card's definition of done says all sixteen methods are implemented
        with none of them raising it, and a refusal this backend means — video
        publishing — is a declared missing capability rather than a stub. Read
        off the source because that is the claim: not "it does not raise today",
        but "there is no stub in here".
        """
        assert "NotImplementedError" not in inspect.getsource(LiveKitConferenceBackend)

    def test_the_module_imports_without_the_sdk(self) -> None:
        """Reached by getting here at all, and worth naming.

        Everything above depends on the module being importable without
        ``livekit``, which is what the deferred import in ``_import_livekit``
        buys and what keeps this file running in CI.
        """
        assert LiveKitConfig().url is None


class TestCapabilities:
    def test_wired_events_are_declared(self) -> None:
        capabilities = capabilities_for(remote_unmute=False, sip_gateway=False)

        assert ConferenceCapability.SCREEN_SHARE in capabilities
        assert ConferenceCapability.ACTIVE_SPEAKER in capabilities
        assert ConferenceCapability.CONNECTION_QUALITY in capabilities

    @pytest.mark.parametrize(
        "capability",
        [
            ConferenceCapability.VIDEO_PUBLISH,
            ConferenceCapability.E2EE,
            ConferenceCapability.EGRESS_RECORDING,
        ],
    )
    def test_what_is_not_wired_is_not_declared(self, capability: ConferenceCapability) -> None:
        """LiveKit can do all three. This backend has not branched any of them,
        and a capability is a statement about the backend, not the product.
        """
        assert capability not in capabilities_for(remote_unmute=True, sip_gateway=True)

    def test_server_side_opt_ins_are_declared_only_when_configured(self) -> None:
        off = capabilities_for(remote_unmute=False, sip_gateway=False)
        on = capabilities_for(remote_unmute=True, sip_gateway=True)

        assert ConferenceCapability.REMOTE_UNMUTE not in off
        assert ConferenceCapability.SIP_GATEWAY not in off
        assert ConferenceCapability.REMOTE_UNMUTE in on
        assert ConferenceCapability.SIP_GATEWAY in on


class TestGrantTranslation:
    def test_a_permissive_human_may_publish_every_source(self) -> None:
        kwargs = video_grant_kwargs("room-1", ConferenceGrants(), publish_data=True)

        assert kwargs["room"] == "room-1"
        assert kwargs["room_join"] is True
        assert kwargs["can_publish"] is True
        assert kwargs["can_publish_sources"] == [MICROPHONE, CAMERA, SCREEN_SHARE]
        assert kwargs["can_subscribe"] is True

    def test_a_listening_bot_publishes_nothing_and_asks_for_no_source_list(self) -> None:
        """An empty source list would supersede ``can_publish`` on LiveKit's
        side, and what it then means is the server's to decide. The boundary
        says "may not publish" in the one way that has a single reading.
        """
        kwargs = video_grant_kwargs("room-1", ConferenceGrants.for_bot(), publish_data=False)

        assert kwargs["can_publish"] is False
        assert "can_publish_sources" not in kwargs
        assert kwargs["can_subscribe"] is True

    def test_a_speaking_bot_may_publish_a_microphone_and_nothing_else(self) -> None:
        kwargs = video_grant_kwargs(
            "room-1", ConferenceGrants.for_bot(speaks=True), publish_data=False
        )

        assert kwargs["can_publish_sources"] == [MICROPHONE]

    def test_a_speak_only_bot_does_not_ask_to_subscribe(self) -> None:
        """Selective subscription reaches the token: a channel with nothing to
        consume the tracks it would receive asks for none of them.
        """
        kwargs = video_grant_kwargs(
            "room-1",
            ConferenceGrants.for_bot(speaks=True, listens=False),
            publish_data=False,
        )

        assert kwargs["can_subscribe"] is False

    def test_an_observer_is_hidden(self) -> None:
        kwargs = video_grant_kwargs("room-1", ConferenceGrants.observer(), publish_data=False)

        assert kwargs["hidden"] is True

    def test_a_visible_participant_does_not_send_a_hidden_claim(self) -> None:
        kwargs = video_grant_kwargs("room-1", ConferenceGrants(), publish_data=True)

        assert "hidden" not in kwargs
        assert "room_admin" not in kwargs

    def test_moderation_becomes_room_admin(self) -> None:
        kwargs = video_grant_kwargs("room-1", ConferenceGrants(moderate=True), publish_data=True)

        assert kwargs["room_admin"] is True

    def test_the_bot_is_denied_data_and_a_human_is_not(self) -> None:
        """The framework configured the bot and knows it publishes no data; it
        did not configure the integrator's client application and does not know
        what that needs.
        """
        bot = video_grant_kwargs("room-1", ConferenceGrants(), publish_data=False)
        human = video_grant_kwargs("room-1", ConferenceGrants(), publish_data=True)

        assert bot["can_publish_data"] is False
        assert human["can_publish_data"] is True

    def test_nobody_is_granted_metadata_writes(self) -> None:
        """What keeps ``asserted_attributes`` truthful: a participant that
        cannot write its own attributes cannot manufacture an address.
        """
        kwargs = video_grant_kwargs("room-1", ConferenceGrants(), publish_data=True)

        assert "can_update_own_metadata" not in kwargs

    def test_screen_share_audio_is_never_granted(self) -> None:
        """RoomKit's grant covers a screen share, and sharing a tab's sound is a
        separate publish right no RoomKit grant asks for.
        """
        assert "screen_share_audio" not in publish_source_names(ConferenceGrants())


class TestTrackKind:
    @pytest.mark.parametrize(
        ("kind", "source", "expected"),
        [
            ("KIND_AUDIO", "SOURCE_MICROPHONE", TrackKind.AUDIO),
            ("KIND_VIDEO", "SOURCE_CAMERA", TrackKind.VIDEO),
            ("KIND_VIDEO", "SOURCE_SCREENSHARE", TrackKind.SCREEN_SHARE),
        ],
    )
    def test_kind_and_source_together_decide(
        self, kind: str, source: str, expected: TrackKind
    ) -> None:
        assert track_kind_for(kind, source) is expected

    def test_the_audio_of_a_screen_share_is_audio(self) -> None:
        """It is speech to transcribe and sound to record. Calling it a screen
        share would route it away from both.
        """
        assert track_kind_for("KIND_AUDIO", "SOURCE_SCREENSHARE_AUDIO") is TrackKind.AUDIO

    def test_an_unclassifiable_track_is_refused(self) -> None:
        with pytest.raises(ValueError, match="neither audio nor video"):
            track_kind_for("KIND_UNKNOWN", "SOURCE_UNKNOWN")

    def test_a_data_track_is_refused(self) -> None:
        with pytest.raises(ValueError):
            track_kind_for(rtc_track_kind_name("DATA"), rtc_track_source_name("UNKNOWN"))


class TestDialects:
    """LiveKit names the same enum two ways, and only one of them is ours."""

    def test_the_server_api_track_names_are_brought_over(self) -> None:
        assert rtc_track_kind_name("AUDIO") == "KIND_AUDIO"
        assert rtc_track_source_name("SCREEN_SHARE") == "SOURCE_SCREENSHARE"

    def test_a_sip_participant_is_recognised_from_either_protocol(self) -> None:
        """The one that carries weight: provenance is decided on the realtime
        name, so a control-plane spelling left untranslated would make a
        dial-in's caller number unasserted.
        """
        assert rtc_participant_kind_name("SIP") == "PARTICIPANT_KIND_SIP"

    def test_an_unknown_name_is_passed_through_rather_than_guessed(self) -> None:
        assert rtc_track_kind_name("KIND_HOLOGRAM") == "KIND_HOLOGRAM"


class TestProvenance:
    """Which attributes LiveKit itself vouches for (RFC section 12.10.2)."""

    def test_a_dial_ins_sip_attributes_are_asserted(self) -> None:
        asserted = asserted_attributes(
            "PARTICIPANT_KIND_SIP",
            {"sip.phoneNumber": "+15145550123", "sip.callID": "abc"},
        )

        assert asserted == {"sip.phoneNumber": "+15145550123", "sip.callID": "abc"}

    def test_a_browser_participant_gets_an_empty_assertion_not_a_null_one(self) -> None:
        """Empty says "this backend distinguishes, and asserts nothing here".
        Null would say it cannot distinguish at all, and would cost a dial-in its
        identity.
        """
        assert asserted_attributes("PARTICIPANT_KIND_STANDARD", {"phone_number": "+1"}) == {}

    def test_a_client_cannot_launder_an_address_through_the_sip_prefix(self) -> None:
        """The prefix is not the proof — the kind is, and the server sets it. A
        standard participant writing ``sip.phoneNumber`` on itself is asserting
        nothing, which is what keeps it from reaching someone else's Identity.
        """
        assert asserted_attributes("PARTICIPANT_KIND_STANDARD", {"sip.phoneNumber": "+1"}) == {}

    def test_a_dial_ins_non_sip_attributes_stay_unasserted(self) -> None:
        asserted = asserted_attributes(
            "PARTICIPANT_KIND_SIP", {"sip.phoneNumber": "+1", "phone_number": "+2"}
        )

        assert "phone_number" not in asserted


class TestParticipantRecord:
    def _info(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "identity": "p-alice",
            "sid": "PA_123",
            "kind_name": "PARTICIPANT_KIND_STANDARD",
            "name": "Alice",
            "metadata": "",
            "attributes": {},
            "connected_at": None,
        }
        base.update(overrides)
        return base

    def test_the_identity_is_the_participant_id(self) -> None:
        """Rule 2 of RFC section 12.10.2, satisfied by LiveKit carrying the
        value the framework minted rather than by a mapping table here.
        """
        record = participant_record(**self._info())  # type: ignore[arg-type]

        assert record.participant_id == "p-alice"

    def test_provider_fields_sit_under_a_prefix_so_they_cannot_shadow(self) -> None:
        record = participant_record(
            **self._info(attributes={"name": "not mine to overwrite"})  # type: ignore[arg-type]
        )

        assert record.metadata["name"] == "not mine to overwrite"
        assert record.metadata["livekit.name"] == "Alice"
        assert record.metadata["livekit.sid"] == "PA_123"

    def test_the_server_asserts_the_sid_and_the_kind(self) -> None:
        record = participant_record(**self._info())  # type: ignore[arg-type]

        assert record.asserted_metadata == {
            "livekit.sid": "PA_123",
            "livekit.kind": "PARTICIPANT_KIND_STANDARD",
        }

    def test_a_token_supplied_name_is_surfaced_without_being_asserted(self) -> None:
        """This backend did not necessarily mint the token it came from."""
        record = participant_record(**self._info(metadata="{}"))  # type: ignore[arg-type]

        assert "livekit.name" in record.metadata
        assert "livekit.name" not in (record.asserted_metadata or {})
        assert "livekit.metadata" not in (record.asserted_metadata or {})

    def test_a_dial_in_carries_its_caller_number_in_both_maps(self) -> None:
        record = participant_record(
            **self._info(  # type: ignore[arg-type]
                kind_name="PARTICIPANT_KIND_SIP",
                attributes={"sip.phoneNumber": "+15145550123"},
            )
        )

        assert record.metadata["sip.phoneNumber"] == "+15145550123"
        assert (record.asserted_metadata or {})["sip.phoneNumber"] == "+15145550123"

    def test_an_empty_name_and_metadata_add_no_keys(self) -> None:
        record = participant_record(**self._info(name="", metadata=""))  # type: ignore[arg-type]

        assert "livekit.name" not in record.metadata
        assert "livekit.metadata" not in record.metadata

    def test_a_reported_join_time_is_taken_over_the_default(self) -> None:
        joined = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

        record = participant_record(**self._info(connected_at=joined))  # type: ignore[arg-type]

        assert record.connected_at == joined

    def test_no_reported_join_time_leaves_the_default(self) -> None:
        record = participant_record(**self._info(connected_at=None))  # type: ignore[arg-type]

        assert record.connected_at.tzinfo is not None


class TestTrackRecord:
    def test_the_publication_sid_is_the_track_id(self) -> None:
        record = track_record(
            sid="TR_123",
            room_id="room-1",
            participant_id="p-alice",
            kind_name="KIND_AUDIO",
            source_name="SOURCE_MICROPHONE",
            muted=False,
        )

        assert record.id == "TR_123"
        assert record.room_id == "room-1"
        assert record.participant_id == "p-alice"
        assert record.kind is TrackKind.AUDIO
        assert record.metadata == {"sid": "TR_123", "source": "SOURCE_MICROPHONE"}

    def test_a_muted_publication_is_recorded_muted(self) -> None:
        record = track_record(
            sid="TR_123",
            room_id="room-1",
            participant_id="p-alice",
            kind_name="KIND_AUDIO",
            source_name="SOURCE_MICROPHONE",
            muted=True,
            name="mic",
            mime_type="audio/opus",
        )

        assert record.muted is True
        assert record.metadata["name"] == "mic"
        assert record.metadata["mime_type"] == "audio/opus"


class TestQualityLabel:
    @pytest.mark.parametrize(
        ("name", "label"),
        [
            ("QUALITY_EXCELLENT", "excellent"),
            ("QUALITY_GOOD", "good"),
            ("QUALITY_POOR", "poor"),
            ("QUALITY_LOST", "lost"),
        ],
    )
    def test_a_level_becomes_a_label(self, name: str, label: str) -> None:
        assert quality_label(name) == label

    def test_an_unknown_level_is_no_report_at_all(self) -> None:
        """Forwarding "unknown" as a quality would put a word on a dashboard
        that means less than the absence of a report.
        """
        assert quality_label("QUALITY_UNKNOWN") is None


class TestVideoCodec:
    @pytest.mark.parametrize(
        ("buffer_type", "codec"),
        [("I420", "raw_yuv420p"), ("NV12", "raw_nv12"), ("RGB24", "raw_rgb24")],
    )
    def test_a_buffer_layout_becomes_a_roomkit_codec(self, buffer_type: str, codec: str) -> None:
        assert codec_for_buffer_type(buffer_type) == codec

    def test_a_layout_roomkit_has_no_codec_for_is_refused(self) -> None:
        """Converting it here would be media-plane work in the wrong place."""
        with pytest.raises(ValueError, match="no codec for"):
            codec_for_buffer_type("ARGB")


class TestPublishableChunk:
    def test_decoded_pcm_is_accepted(self) -> None:
        require_publishable_pcm(AudioChunk(data=b"\x00\x00", format="pcm_s16le"))
        require_publishable_pcm(AudioChunk(data=b"\x00\x00", format="pcm"))

    def test_an_encoded_chunk_is_refused(self) -> None:
        """Encoding belongs to the backend: a caller choosing the wire format
        defeats the boundary (RFC section 12.10.3).
        """
        with pytest.raises(ValueError, match="expects decoded PCM"):
            require_publishable_pcm(AudioChunk(data=b"\x00", format="opus"))

    def test_another_pcm_width_is_refused_rather_than_reinterpreted(self) -> None:
        """LiveKit's frame is 16-bit signed. Handing it float samples would not
        fail, it would publish noise into a conference.
        """
        with pytest.raises(ValueError, match="16-bit signed"):
            require_publishable_pcm(AudioChunk(data=b"\x00" * 4, format="pcm_f32le"))

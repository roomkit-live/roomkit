"""Conference data models (RFC §12.10.2).

These are declarations, so the tests here are deliberately thin: they cover the
defaults and helpers that *encode a decision*, and skip the ones that merely
restate a field list. Behavioural coverage lives in the backend and channel
suites.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from roomkit import (
    BotSession,
    ConferenceAccess,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceInterruptionConfig,
    ConferenceInterruptionScope,
    ConferenceParticipant,
    ConferenceRecordingConfig,
    ConferenceRecordingMode,
    ConferenceTrack,
    TrackKind,
)
from roomkit.voice.interruption import InterruptionStrategy


class TestGrants:
    def test_defaults_are_permissive(self) -> None:
        """Deliberate: the common case works unconfigured, and narrowing is the
        integrator's call (§17.7 is a SHOULD). Flipping these to
        deny-by-default requires changing the specification first.
        """
        grants = ConferenceGrants()

        assert grants.publish_audio is True
        assert grants.publish_video is True
        assert grants.publish_screen_share is True
        assert grants.subscribe is True

    def test_privileged_flags_are_off_by_default(self) -> None:
        grants = ConferenceGrants()

        assert grants.moderate is False
        assert grants.hidden is False

    def test_observer_is_subscribe_only_and_hidden(self) -> None:
        """The Observer participation pattern (§12.10.6)."""
        grants = ConferenceGrants.observer()

        assert grants.subscribe is True
        assert grants.hidden is True
        assert not grants.publish_audio
        assert not grants.publish_video
        assert not grants.publish_screen_share

    def test_a_bot_asks_for_nothing_it_was_not_configured_to_do(self) -> None:
        """The counterpart of the permissive defaults above: those exist because
        the framework cannot know what a human will do. It knows exactly what it
        configured the bot to do.
        """
        grants = ConferenceGrants.for_bot()

        assert grants.subscribe is True
        assert grants.publish_audio is False
        assert grants.publish_video is False
        assert grants.publish_screen_share is False

    def test_a_speaking_bot_may_publish_audio_and_nothing_more(self) -> None:
        grants = ConferenceGrants.for_bot(speaks=True)

        assert grants.publish_audio is True
        assert grants.publish_video is False
        assert grants.publish_screen_share is False

    def test_a_bot_that_consumes_nothing_does_not_ask_to_subscribe(self) -> None:
        """Subscribing is the one privilege that was granted unconditionally,
        on the reasoning that receiving other participants' tracks is the bot's
        whole reason to be there. It is not: a channel with nothing to consume
        them subscribes to no track at all, and the grant is then permission to
        receive every participant's media for nobody to read.
        """
        grants = ConferenceGrants.for_bot(speaks=True, listens=False)

        assert grants.subscribe is False
        assert grants.publish_audio is True

    def test_a_bot_is_not_hidden_by_default(self) -> None:
        """Hiding is a disclosure choice, not a privilege — §17.7 leaves it to
        the integrator, so least privilege must not decide it either way.
        """
        assert ConferenceGrants.for_bot().hidden is False

    def test_an_observer_is_a_silent_bot_that_is_also_invisible(self) -> None:
        assert ConferenceGrants.observer() == replace(ConferenceGrants.for_bot(), hidden=True)


class TestCapabilities:
    def test_flags_compose(self) -> None:
        caps = ConferenceCapability.SCREEN_SHARE | ConferenceCapability.ACTIVE_SPEAKER

        assert ConferenceCapability.SCREEN_SHARE in caps
        assert ConferenceCapability.ACTIVE_SPEAKER in caps
        assert ConferenceCapability.EGRESS_RECORDING not in caps

    def test_none_is_the_empty_capability_set(self) -> None:
        caps = ConferenceCapability.NONE

        assert ConferenceCapability.VIDEO_PUBLISH not in caps

    def test_remote_unmute_is_a_separate_capability_from_muting(self) -> None:
        """Muting is always available; unmuting someone else's microphone is a
        privacy decision that SFUs commonly refuse by default.
        """
        assert ConferenceCapability.REMOTE_UNMUTE in ConferenceCapability


class TestTrack:
    def test_track_carries_both_room_and_participant(self) -> None:
        """Frame callbacks receive only a track, and one backend instance
        serves many rooms — without room_id the frames are not routable.
        """
        track = ConferenceTrack(
            id="tr-1",
            room_id="room-1",
            participant_id="p-alice",
            kind=TrackKind.AUDIO,
        )

        assert track.room_id == "room-1"
        assert track.participant_id == "p-alice"
        assert track.muted is False
        assert track.metadata == {}

    def test_track_kinds(self) -> None:
        assert [k.value for k in TrackKind] == ["audio", "video", "screen_share"]


class TestParticipant:
    def test_metadata_carries_provider_attributes(self) -> None:
        """For a participant the framework did not name, metadata is where the
        resolvable address lives — the caller number of a PSTN dial-in.
        """
        participant = ConferenceParticipant(
            participant_id="sip_15551234",
            metadata={"sip.phoneNumber": "+15551234"},
        )

        assert participant.metadata["sip.phoneNumber"] == "+15551234"
        assert participant.tracks == []

    def test_a_backend_says_nothing_about_provenance_unless_it_can(self) -> None:
        """Null by default, and the default is a backend that has not been
        written to distinguish. Believing it would be believing a guess.
        """
        participant = ConferenceParticipant(
            participant_id="sip_15551234",
            metadata={"sip.phoneNumber": "+15551234"},
        )

        assert participant.asserted_metadata is None

    def test_asserted_metadata_is_the_subset_the_sfu_vouches_for(self) -> None:
        participant = ConferenceParticipant(
            participant_id="sip_15551234",
            metadata={"sip.phoneNumber": "+15551234", "nickname": "bob"},
            asserted_metadata={"sip.phoneNumber": "+15551234"},
        )

        assert participant.asserted_metadata == {"sip.phoneNumber": "+15551234"}
        assert "nickname" not in participant.asserted_metadata

    def test_connected_at_defaults_to_now(self) -> None:
        before = datetime.now(UTC)

        participant = ConferenceParticipant(participant_id="p-alice")

        assert before <= participant.connected_at <= datetime.now(UTC)


class TestAccess:
    def test_access_holds_opaque_credentials(self) -> None:
        expiry = datetime.now(UTC) + timedelta(hours=1)

        access = ConferenceAccess(url="wss://sfu.example", token="opaque", expires_at=expiry)

        assert access.token == "opaque"
        assert access.expires_at == expiry
        assert access.provider_data == {}

    def test_expiry_is_optional(self) -> None:
        assert ConferenceAccess(url="wss://sfu.example", token="t").expires_at is None

    def test_token_never_appears_in_repr(self) -> None:
        """Access tokens are credentials and must not be logged. The generated
        repr is the easy way to leak one — through a log line or a traceback.
        """
        access = ConferenceAccess(url="wss://sfu.example", token="super-secret-jwt")

        rendered = repr(access)

        assert "super-secret-jwt" not in rendered
        assert "wss://sfu.example" in rendered


class TestBotSession:
    def test_identity_is_what_makes_self_exclusion_possible(self) -> None:
        bot = BotSession(id="bs-1", room_id="room-1", identity="roomkit-bot")

        assert bot.identity == "roomkit-bot"
        assert bot.room_id == "room-1"


class TestInterruptionConfig:
    def test_defaults_to_immediate_and_anyone(self) -> None:
        config = ConferenceInterruptionConfig()

        assert config.strategy is InterruptionStrategy.IMMEDIATE
        assert config.scope is ConferenceInterruptionScope.ANY
        assert config.allowlist == []

    def test_allowlist_scope_carries_participants(self) -> None:
        config = ConferenceInterruptionConfig(
            scope=ConferenceInterruptionScope.ALLOWLIST,
            allowlist=["p-moderator"],
        )

        assert config.allowlist == ["p-moderator"]

    def test_scopes(self) -> None:
        assert [s.value for s in ConferenceInterruptionScope] == ["any", "none", "allowlist"]


class TestRecordingConfig:
    def test_defaults_to_framework_recording(self) -> None:
        """The path that always works: no backend capability, functions against
        the mock, and the file lands where the implementation writes it.
        Delegating to the SFU is opt-in and exists only for composed video.
        """
        config = ConferenceRecordingConfig()

        assert config.mode is ConferenceRecordingMode.FRAMEWORK
        assert config.format == "wav"

    def test_egress_is_selectable(self) -> None:
        config = ConferenceRecordingConfig(mode=ConferenceRecordingMode.EGRESS, format="mp4")

        assert config.mode is ConferenceRecordingMode.EGRESS

"""Conference data models (RFC §12.10.2).

These are declarations, so the tests here are deliberately thin: they cover the
defaults and helpers that *encode a decision*, and skip the ones that merely
restate a field list. Behavioural coverage lives in the backend and channel
suites.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from roomkit import (
    ConferenceAccess,
    ConferenceGrants,
    ConferenceInterruptionConfig,
    ConferenceInterruptionScope,
    ConferenceParticipant,
    ConferenceRecordingConfig,
    TrackKind,
)
from roomkit.voice.interruption import InterruptionStrategy

_NOTHING = ConferenceGrants(
    publish_audio=False, publish_video=False, publish_screen_share=False, subscribe=False
)


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

    def test_the_sound_of_a_screen_share_is_off_by_default(self) -> None:
        """The one publish right that is not permissive: a caller that never
        names it mints no credential carrying it (§12.10.2).
        """
        assert ConferenceGrants().publish_screen_share_audio is False

    def test_the_earlier_fields_keep_their_positions(self) -> None:
        """The field is appended, so a caller passing the others positionally
        is not silently re-pointed.
        """
        grants = ConferenceGrants(False, False, False, False, True, True)

        assert grants.moderate is True
        assert grants.hidden is True
        assert grants.publish_screen_share_audio is False

    @pytest.mark.parametrize(
        ("make", "options", "granted"),
        [
            # The counterpart of the permissive defaults above: those exist because
            # the framework cannot know what a human will do. It knows exactly what
            # it configured the bot to do. Hiding is a disclosure choice, not a
            # privilege (§17.7 leaves it to the integrator), so least privilege
            # does not decide it: a bot is not hidden.
            pytest.param(ConferenceGrants.for_bot, {}, {"subscribe": True}, id="listening-bot"),
            pytest.param(
                ConferenceGrants.for_bot,
                {"speaks": True},
                {"subscribe": True, "publish_audio": True},
                id="speaking-bot",
            ),
            # Receiving the others' tracks is not the bot's reason to be there by
            # itself: a channel with nothing to consume them subscribes to no
            # track, and the grant would be permission to receive every
            # participant's media for nobody to read.
            pytest.param(
                ConferenceGrants.for_bot,
                {"speaks": True, "listens": False},
                {"publish_audio": True},
                id="speak-only-bot",
            ),
            # The Observer participation pattern (§12.10.6): a silent bot that is
            # also invisible.
            pytest.param(
                ConferenceGrants.observer, {}, {"subscribe": True, "hidden": True}, id="observer"
            ),
        ],
    )
    def test_a_bot_asks_for_nothing_it_was_not_configured_to_do(
        self,
        make: Callable[..., ConferenceGrants],
        options: dict[str, bool],
        granted: dict[str, bool],
    ) -> None:
        assert make(**options) == replace(_NOTHING, **granted)


class TestTrack:
    def test_track_kinds(self) -> None:
        assert [k.value for k in TrackKind] == ["audio", "video", "screen_share"]


class TestParticipant:
    def test_a_backend_says_nothing_about_provenance_unless_it_can(self) -> None:
        """Null by default, and the default is a backend that has not been
        written to distinguish. Believing it would be believing a guess.
        """
        participant = ConferenceParticipant(
            participant_id="sip_15551234",
            metadata={"sip.phoneNumber": "+15551234"},
        )

        assert participant.asserted_metadata is None


class TestAccess:
    def test_token_never_appears_in_repr(self) -> None:
        """Access tokens are credentials and must not be logged. The generated
        repr is the easy way to leak one — through a log line or a traceback.
        """
        access = ConferenceAccess(url="wss://sfu.example", token="super-secret-jwt")

        rendered = repr(access)

        assert "super-secret-jwt" not in rendered
        assert "wss://sfu.example" in rendered


class TestInterruptionConfig:
    def test_defaults_to_immediate_and_anyone(self) -> None:
        config = ConferenceInterruptionConfig()

        assert config.strategy is InterruptionStrategy.IMMEDIATE
        assert config.scope is ConferenceInterruptionScope.ANY
        assert config.allowlist == []

    def test_scopes(self) -> None:
        assert [s.value for s in ConferenceInterruptionScope] == ["any", "none", "allowlist"]


class TestRecordingConfig:
    def test_defaults_to_wav(self) -> None:
        """The framework records the tracks the bot subscribes to: no backend
        capability, functions against the mock (RFC §12.10.8)."""
        config = ConferenceRecordingConfig()

        assert config.format == "wav"
